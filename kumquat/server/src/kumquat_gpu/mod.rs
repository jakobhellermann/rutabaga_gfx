// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap as Map;
use std::collections::BTreeSet as Set;
use std::os::fd::AsRawFd;
use std::os::raw::c_void;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use log::error;
use magma_gpu::protocols::ipc::KumquatStream;
use magma_gpu::protocols::kumquat_gpu_protocol::*;
use std::sync::atomic::{AtomicU64, Ordering};

use magma_gpu::util::create_event_pair;

/// Number of completed GPU snapshots. A resource with
/// `created_epoch == SNAPSHOT_EPOCH` was created after the most recent
/// snapshot and belongs to no restore point yet. Each slot records the era of
/// its own snapshot in `SlotSnapshotState::epoch`; the restore-time
/// post-checkpoint diagnostics compare resources against that.
static SNAPSHOT_EPOCH: AtomicU64 = AtomicU64::new(0);
use magma_gpu::util::AsBorrowedDescriptor;
use magma_gpu::util::AsRawDescriptor;
use magma_gpu::util::Error as MagmaGpuError;
use magma_gpu::util::EventSignaler;
use magma_gpu::util::FromRawDescriptor;
use magma_gpu::util::Handle as MagmaGpuHandle;
use magma_gpu::util::MemoryMapping;
use magma_gpu::util::OwnedDescriptor;
use magma_gpu::util::SharedMemory;
use magma_gpu::util::Tube;
use magma_gpu::util::MAGMA_GPU_HANDLE_TYPE_MEM_DMABUF;
use magma_gpu::util::MAGMA_GPU_HANDLE_TYPE_MEM_SHM;
use remain::sorted;
use rutabaga_gfx::calculate_capset_mask;
use rutabaga_gfx::ResourceCreate3D;
use rutabaga_gfx::ResourceCreateBlob;
use rutabaga_gfx::Rutabaga;
use rutabaga_gfx::RutabagaBuilder;
use rutabaga_gfx::RutabagaError;
use rutabaga_gfx::RutabagaFence;
use rutabaga_gfx::RutabagaFenceHandler;
use rutabaga_gfx::RutabagaHandle;
use rutabaga_gfx::RutabagaIovec;
use rutabaga_gfx::RutabagaWsi;
use rutabaga_gfx::Transfer3D;
use rutabaga_gfx::VulkanInfo as RutabagaVulkanInfo;
use rutabaga_gfx::RUTABAGA_BLOB_MEM_GUEST;
use rutabaga_gfx::RUTABAGA_FLAG_FENCE;
use rutabaga_gfx::RUTABAGA_FLAG_FENCE_HOST_SHAREABLE;
use rutabaga_gfx::RUTABAGA_MAP_ACCESS_RW;
use rutabaga_gfx::RUTABAGA_MAP_CACHE_CACHED;
use thiserror::Error;

const SNAPSHOT_DIR: &str = "/tmp/";

/// `STREAM_BLOB_FLAG_CREATE_GUEST_HANDLE` (virtio-gpu spec / gfxstream's
/// stream_renderer.h); not modelled as a rutabaga constant.
const BLOB_FLAG_CREATE_GUEST_HANDLE: u32 = 0x0008;
/// `UDMABUF_CREATE` from linux/udmabuf.h: `_IOW('u', 0x42, struct udmabuf_create)`.
/// `struct udmabuf_create` is 24 bytes (u32 memfd, u32 flags, u64 offset, u64 size).
const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;
const UDMABUF_FLAGS_CLOEXEC: u32 = 1 << 0;

/// Arguments of the `UDMABUF_CREATE` ioctl (linux/udmabuf.h).
#[repr(C)]
#[derive(Default)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

/// Converts a sealed memfd into a dma-buf whose pages alias the memfd's.
///
/// gfxstream imports guest-handle blobs into the host VkDeviceMemory via
/// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`, so the descriptor handed
/// to rutabaga must be a real dma-buf, not a plain memfd. The udmabuf kernel
/// interface provides that conversion without a GPU round-trip; the memfd's
/// pages stay shared with any mapping of it.
fn udmabuf_from_memfd(memfd_fd: u32, size: u64) -> Result<OwnedDescriptor, MagmaGpuError> {
    let dev = std::fs::OpenOptions::new()
        .read(true)
        .open("/dev/udmabuf")
        .map_err(MagmaGpuError::IoError)?;

    let create = UdmabufCreate {
        memfd: memfd_fd,
        flags: UDMABUF_FLAGS_CLOEXEC,
        offset: 0,
        size,
    };

    // SAFETY: `dev` is a valid /dev/udmabuf file descriptor and `create` is a
    // valid, correctly laid-out udmabuf_create struct referenced for the
    // duration of the ioctl; the kernel only reads it and returns a new fd.
    let ret = unsafe { libc::ioctl(dev.as_raw_fd(), UDMABUF_CREATE, &create) };
    if ret < 0 {
        return Err(MagmaGpuError::IoError(std::io::Error::last_os_error()));
    }

    // SAFETY: the ioctl returned a fresh, owning dma-buf file descriptor.
    Ok(unsafe { OwnedDescriptor::from_raw_descriptor(ret) })
}

/// Dups the descriptor of a response handle so kumquat can keep a reference
/// to the client-visible memory after the handle itself is sent to the
/// client.
fn descriptor_try_clone(handle: &MagmaGpuHandle) -> Option<OwnedDescriptor> {
    handle.os_handle.try_clone().ok()
}

/// Backing memory for GUEST guest-handle blobs: the memfd provides the mapping
/// (the stand-in for guest kernel memory), the udmabuf is the dma-buf handed
/// to rutabaga/gfxstream for the host-side import, and the response handle is
/// what the client mmaps as its blob memory.
struct GuestBlobBacking {
    mapping: MemoryMapping,
    iovecs: Vec<RutabagaIovec>,
    guest_handle: MagmaGpuHandle,
    response_handle: MagmaGpuHandle,
    /// Dup of the dma-buf, kept so the descriptor can be re-registered after
    /// a snapshot restore (the original is consumed by the first mapping).
    dma_buf_dup: OwnedDescriptor,
}

/// Creates the backing for a `BLOB_MEM_GUEST` + `CREATE_GUEST_HANDLE` blob.
///
/// There is no guest kernel to allocate and export the blob's memory, so
/// kumquat plays that role itself (mirroring the ResourceCreate3d flow):
/// allocate a sealed memfd, export it as a dma-buf via /dev/udmabuf, pass the
/// dma-buf to rutabaga as the blob's guest handle and attach the memfd mapping
/// as the resource backing.
fn create_guest_blob_backing(size: u64) -> Result<GuestBlobBacking, MagmaGpuError> {
    let descriptor: OwnedDescriptor = SharedMemory::new("kumquat-guest-blob", size)?.into();

    // udmabuf requires the memfd to be sealed against shrinking so the pages
    // it pins cannot disappear.
    // SAFETY: `descriptor` is a valid memfd; F_ADD_SEALS only sets the seal
    // flags on it and fails if they cannot be applied.
    let ret = unsafe {
        libc::fcntl(
            descriptor.as_raw_descriptor(),
            libc::F_ADD_SEALS,
            libc::F_SEAL_SHRINK,
        )
    };
    if ret < 0 {
        return Err(MagmaGpuError::IoError(std::io::Error::last_os_error()));
    }

    let dma_buf = udmabuf_from_memfd(
        // SAFETY: borrowed only for the ioctl call; `descriptor` stays valid
        // and owned here, the fd is not closed by this conversion.
        descriptor.as_raw_descriptor() as u32,
        size,
    )?;
    let dma_buf_dup = dma_buf.try_clone()?;

    let clone = descriptor.try_clone()?;
    let mapping = MemoryMapping::from_safe_descriptor(
        clone,
        size as usize,
        RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
    )?;
    let rutabaga_mapping = mapping.as_raw_mapping();

    let iovecs = vec![RutabagaIovec {
        base: rutabaga_mapping.ptr as *mut c_void,
        len: size as usize,
    }];

    let handle = MagmaGpuHandle {
        // SAFETY: `dma_buf` is an owned dma-buf fd that gfxstream takes over;
        // no other code retains it after the move into the handle.
        os_handle: dma_buf,
        handle_type: MAGMA_GPU_HANDLE_TYPE_MEM_DMABUF,
    };

    // The client mmaps the fd from the create response (its stand-in for the
    // guest kernel's GEM object). A dup of the memfd aliases the same pages as
    // the dma-buf gfxstream imports.
    let response_handle = MagmaGpuHandle {
        os_handle: descriptor.try_clone()?,
        handle_type: MAGMA_GPU_HANDLE_TYPE_MEM_SHM,
    };

    Ok(GuestBlobBacking {
        mapping,
        iovecs,
        guest_handle: handle,
        response_handle,
        dma_buf_dup,
    })
}

#[sorted]
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum KumquatGpuError {
    #[error("Mesa Error {0}")]
    MagmaGpuError(MagmaGpuError),
    #[error("Rutabaga Error {0}")]
    RutabagaError(RutabagaError),
    #[error("Snapshot Error {0}")]
    SnapshotError(String),
}

impl From<MagmaGpuError> for KumquatGpuError {
    fn from(e: MagmaGpuError) -> KumquatGpuError {
        KumquatGpuError::MagmaGpuError(e)
    }
}

impl From<RutabagaError> for KumquatGpuError {
    fn from(e: RutabagaError) -> KumquatGpuError {
        KumquatGpuError::RutabagaError(e)
    }
}

pub type KumquatGpuResult<T> = std::result::Result<T, KumquatGpuError>;

pub struct KumquatGpuConnection {
    stream: KumquatStream,
}

/// A blob whose fd was exported to the client. The dup is kept so the
/// client-visible mapping can be re-attached to the restored gfxstream
/// resource after a snapshot restore.
struct ExportedBlob {
    ctx_id: u32,
    blob_id: u32,
    size: u64,
    descriptor: OwnedDescriptor,
    /// Dup of the dma-buf handed to gfxstream for the host-side import
    /// (GUEST guest-handle blobs only). Re-registered before a restore so the
    /// replayed vkAllocateMemory can import it again.
    dma_buf: Option<OwnedDescriptor>,
    /// Mapping of `descriptor`, created at re-attach time and kept alive for
    /// as long as the gfxstream resource uses the address. Shared: slot
    /// archives hold `Arc` clones so the mapping outlives any one archive.
    mapping: Option<Arc<MemoryMapping>>,
}

pub struct KumquatGpuResource {
    attached_contexts: Set<u32>,
    /// SHM backing mapping (3D resources). Shared with slot archives via
    /// `Arc`: archiving duplicates the resource per slot, but the mapping
    /// must stay alive exactly once (RAII munmap on the last reference).
    mapping: Option<Arc<MemoryMapping>>,
    exported: Option<ExportedBlob>,
    /// Snapshot generation the resource was created in. Used by the restore
    /// re-attach logging to spot resources from the discarded timeline.
    created_epoch: u64,
    /// Backed by `create_guest_blob_backing` memfd memory that the guest has
    /// mapped (the `/memfd:kumquat-guest-blob` areas). The restore write
    /// guard mprotects these mappings read-only: any write from this process
    /// during a restore corrupts the checkpoint state the guest is about to
    /// be resumed with.
    guest_blob: bool,
}

impl KumquatGpuResource {
    /// Duplicate the resource for a slot archive: mappings are shared via
    /// `Arc` (the original is dropped by the caller), descriptors are fresh
    /// dups, `attached_contexts` is overwritten by the caller from the slot's
    /// snapshot bookkeeping.
    fn try_clone_for_archive(&self) -> KumquatGpuResult<KumquatGpuResource> {
        Ok(KumquatGpuResource {
            attached_contexts: self.attached_contexts.clone(),
            mapping: self.mapping.clone(),
            exported: self
                .exported
                .as_ref()
                .map(|exported| -> KumquatGpuResult<ExportedBlob> {
                    Ok(ExportedBlob {
                        ctx_id: exported.ctx_id,
                        blob_id: exported.blob_id,
                        size: exported.size,
                        descriptor: exported
                            .descriptor
                            .try_clone()
                            .map_err(MagmaGpuError::IoError)?,
                        dma_buf: exported
                            .dma_buf
                            .as_ref()
                            .map(|d| d.try_clone().map_err(MagmaGpuError::IoError))
                            .transpose()?,
                        mapping: exported.mapping.clone(),
                    })
                })
                .transpose()?,
            created_epoch: self.created_epoch,
            guest_blob: self.guest_blob,
        })
    }
}

pub struct FenceData {
    pub pending_fences: Map<u64, EventSignaler>,
}

pub type FenceState = Arc<Mutex<FenceData>>;

pub fn create_fence_handler(fence_state: FenceState) -> RutabagaFenceHandler {
    RutabagaFenceHandler::new(move |completed_fence: RutabagaFence| {
        let mut state = fence_state.lock().unwrap();
        match state.pending_fences.entry(completed_fence.fence_id) {
            Entry::Occupied(o) => {
                let (_, event) = o.remove_entry();
                event.signal().unwrap();
            }
            Entry::Vacant(_) => {
                // This is fine, since an actual fence doesn't create emulated sync
                // entry
            }
        }
    })
}

/// Host-side bookkeeping of one savestate slot's last snapshot.
#[derive(Default)]
struct SlotSnapshotState {
    /// Context attachments at this slot's last snapshot.
    snapshot_contexts: Map<u32, Set<u32>>,
    /// Snapshot resources unreferenced by the guest since that snapshot;
    /// kept alive so a restore from this slot can bring them back.
    archived_resources: Map<u32, KumquatGpuResource>,
    /// SNAPSHOT_EPOCH of the era this slot's snapshot opens: resources with
    /// `created_epoch >= epoch` postdate the snapshot (diagnostics).
    epoch: u64,
}

pub struct KumquatGpu {
    rutabaga: Rutabaga,
    fence_state: FenceState,
    id_allocator: u32,
    resources: Map<u32, KumquatGpuResource>,
    /// Snapshot bookkeeping per savestate slot.
    slot_states: Map<u32, SlotSnapshotState>,
}

impl KumquatGpu {
    pub fn new(capset_names: String, renderer_features: String) -> KumquatGpuResult<KumquatGpu> {
        let capset_mask = calculate_capset_mask(capset_names.as_str().split(":"));
        if capset_mask == 0 {
            return Err(MagmaGpuError::Unsupported.into());
        }

        let fence_state = Arc::new(Mutex::new(FenceData {
            pending_fences: Default::default(),
        }));

        let fence_handler = create_fence_handler(fence_state.clone());

        let renderer_features_opt = if renderer_features.is_empty() {
            None
        } else {
            Some(renderer_features)
        };

        let rutabaga = RutabagaBuilder::new(capset_mask, fence_handler)
            .set_use_external_blob(true)
            // Metal cannot export device memory, so host visible memory is
            // shared memory the host imports as a host pointer instead.
            .set_use_system_blob(cfg!(target_vendor = "apple"))
            .set_use_egl(true)
            .set_wsi(RutabagaWsi::Surfaceless)
            .set_renderer_features(renderer_features_opt)
            .build()?;

        Ok(KumquatGpu {
            rutabaga,
            fence_state,
            id_allocator: 0,
            resources: Default::default(),
            slot_states: Default::default(),
        })
    }

    pub fn allocate_id(&mut self) -> u32 {
        self.id_allocator += 1;
        self.id_allocator
    }
}

impl KumquatGpuConnection {
    pub fn new(connection: Tube) -> KumquatGpuConnection {
        KumquatGpuConnection {
            stream: KumquatStream::new(connection),
        }
    }

    pub fn process_command(&mut self, kumquat_gpu: &mut KumquatGpu) -> KumquatGpuResult<bool> {
        let mut hung_up = false;
        let protocols = self.stream.read()?;

        for protocol in protocols {
            match protocol {
                KumquatGpuProtocol::GetNumCapsets => {
                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_NUM_CAPSETS,
                        payload: kumquat_gpu.rutabaga.get_num_capsets(),
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::GetCapsetInfo(capset_index) => {
                    let (capset_id, version, size) =
                        kumquat_gpu.rutabaga.get_capset_info(capset_index)?;

                    let resp = kumquat_gpu_protocol_resp_capset_info {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_CAPSET_INFO,
                            ..Default::default()
                        },
                        capset_id,
                        version,
                        size,
                        ..Default::default()
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::GetCapset(cmd) => {
                    let capset = kumquat_gpu
                        .rutabaga
                        .get_capset(cmd.capset_id, cmd.capset_version)?;

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_CAPSET,
                        payload: capset
                            .len()
                            .try_into()
                            .map_err(MagmaGpuError::TryFromIntError)?,
                    };

                    self.stream
                        .write(KumquatGpuProtocolWrite::CmdWithData(resp, capset))?;
                }
                KumquatGpuProtocol::CtxCreate(cmd) => {
                    let context_id = kumquat_gpu.allocate_id();
                    let context_name: Option<String> =
                        String::from_utf8(cmd.debug_name.to_vec()).ok();

                    kumquat_gpu.rutabaga.create_context(
                        context_id,
                        cmd.context_init,
                        context_name.as_deref(),
                    )?;

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_CONTEXT_CREATE,
                        payload: context_id,
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::CtxDestroy(ctx_id) => {
                    kumquat_gpu.rutabaga.destroy_context(ctx_id)?;
                }
                KumquatGpuProtocol::CtxAttachResource(cmd) => {
                    kumquat_gpu
                        .rutabaga
                        .context_attach_resource(cmd.ctx_id, cmd.resource_id)
                        .map_err(|error| {
                            eprintln!(
                                "kumquat: attach ctx {} resource {} failed: {error:?} (wrapper_present={})",
                                cmd.ctx_id,
                                cmd.resource_id,
                                kumquat_gpu.resources.contains_key(&cmd.resource_id)
                            );
                            error
                        })?;
                }
                KumquatGpuProtocol::CtxDetachResource(cmd) => {
                    kumquat_gpu
                        .rutabaga
                        .context_detach_resource(cmd.ctx_id, cmd.resource_id)
                        .map_err(|error| {
                            eprintln!(
                                "kumquat: detach ctx {} resource {} failed: {error:?} (wrapper_present={})",
                                cmd.ctx_id,
                                cmd.resource_id,
                                kumquat_gpu.resources.contains_key(&cmd.resource_id)
                            );
                            error
                        })?;

                    let mut resource = kumquat_gpu
                        .resources
                        .remove(&cmd.resource_id)
                        .ok_or(RutabagaError::InvalidResourceId)?;

                    resource.attached_contexts.remove(&cmd.ctx_id);
                    if resource.attached_contexts.is_empty() {
                        if resource.mapping.is_some() {
                            kumquat_gpu
                                .rutabaga
                                .detach_backing(cmd.resource_id)
                                .map_err(|error| {
                                    eprintln!(
                                        "kumquat: detach backing resource {} failed: {error:?}",
                                        cmd.resource_id
                                    );
                                    error
                                })?;
                        }

                        kumquat_gpu
                            .rutabaga
                            .unref_resource(cmd.resource_id)
                            .map_err(|error| {
                                eprintln!(
                                    "kumquat: unref resource {} failed: {error:?}",
                                    cmd.resource_id
                                );
                                error
                            })?;
                        // Archive the resource for every slot whose snapshot
                        // still knows it, so each restore point can bring
                        // back exactly its own set.
                        for slot_state in kumquat_gpu.slot_states.values_mut() {
                            if let Some(contexts) =
                                slot_state.snapshot_contexts.get(&cmd.resource_id)
                            {
                                let mut archived = resource.try_clone_for_archive()?;
                                archived.attached_contexts = contexts.clone();
                                slot_state
                                    .archived_resources
                                    .insert(cmd.resource_id, archived);
                            }
                        }
                    } else {
                        kumquat_gpu.resources.insert(cmd.resource_id, resource);
                    }
                }
                KumquatGpuProtocol::ResourceCreate3d(cmd) => {
                    let resource_create_3d = ResourceCreate3D {
                        target: cmd.target,
                        format: cmd.format,
                        bind: cmd.bind,
                        width: cmd.width,
                        height: cmd.height,
                        depth: cmd.depth,
                        array_size: cmd.array_size,
                        last_level: cmd.last_level,
                        nr_samples: cmd.nr_samples,
                        flags: cmd.flags,
                    };

                    let size = cmd.size as usize;
                    let descriptor: OwnedDescriptor =
                        SharedMemory::new("rutabaga_server", size as u64)?.into();

                    let clone = descriptor.try_clone().map_err(MagmaGpuError::IoError)?;
                    let mut vecs: Vec<RutabagaIovec> = Vec::new();

                    let mapping = MemoryMapping::from_safe_descriptor(
                        clone,
                        size,
                        RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
                    )?;
                    let rutabaga_mapping = mapping.as_raw_mapping();

                    vecs.push(RutabagaIovec {
                        base: rutabaga_mapping.ptr as *mut c_void,
                        len: size,
                    });

                    let resource_id = kumquat_gpu.allocate_id();

                    kumquat_gpu
                        .rutabaga
                        .resource_create_3d(resource_id, resource_create_3d)?;

                    kumquat_gpu.rutabaga.attach_backing(resource_id, vecs)?;
                    kumquat_gpu.resources.insert(
                        resource_id,
                        KumquatGpuResource {
                            attached_contexts: Default::default(),
                            mapping: Some(Arc::new(mapping)),
                            exported: None,
                            created_epoch: SNAPSHOT_EPOCH.load(Ordering::SeqCst),
                            guest_blob: false,
                        },
                    );

                    kumquat_gpu
                        .rutabaga
                        .context_attach_resource(cmd.ctx_id, resource_id)?;

                    let resp = kumquat_gpu_protocol_resp_resource_create {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_RESOURCE_CREATE,
                            ..Default::default()
                        },
                        resource_id,
                        ..Default::default()
                    };

                    self.stream.write(KumquatGpuProtocolWrite::CmdWithHandle(
                        resp,
                        MagmaGpuHandle {
                            os_handle: descriptor,
                            handle_type: MAGMA_GPU_HANDLE_TYPE_MEM_SHM,
                        },
                    ))?;
                }
                KumquatGpuProtocol::TransferToHost3d(cmd, emulated_fence) => {
                    let resource_id = cmd.resource_id;

                    let transfer = Transfer3D {
                        x: cmd.box_.x,
                        y: cmd.box_.y,
                        z: cmd.box_.z,
                        w: cmd.box_.w,
                        h: cmd.box_.h,
                        d: cmd.box_.d,
                        level: cmd.level,
                        stride: cmd.stride,
                        layer_stride: cmd.layer_stride,
                        offset: cmd.offset,
                    };

                    kumquat_gpu
                        .rutabaga
                        .transfer_write(cmd.ctx_id, resource_id, transfer, None)?;

                    let signaler: EventSignaler = emulated_fence.try_into()?;
                    signaler.signal()?;
                }
                KumquatGpuProtocol::TransferFromHost3d(cmd, emulated_fence) => {
                    let resource_id = cmd.resource_id;

                    let transfer = Transfer3D {
                        x: cmd.box_.x,
                        y: cmd.box_.y,
                        z: cmd.box_.z,
                        w: cmd.box_.w,
                        h: cmd.box_.h,
                        d: cmd.box_.d,
                        level: cmd.level,
                        stride: cmd.stride,
                        layer_stride: cmd.layer_stride,
                        offset: cmd.offset,
                    };

                    kumquat_gpu
                        .rutabaga
                        .transfer_read(cmd.ctx_id, resource_id, transfer, None)?;

                    let signaler: EventSignaler = emulated_fence.try_into()?;
                    signaler.signal()?;
                }
                KumquatGpuProtocol::CmdSubmit3d(cmd, mut cmd_buf, fence_ids) => {
                    kumquat_gpu.rutabaga.submit_command(
                        cmd.ctx_id,
                        &mut cmd_buf[..],
                        &fence_ids[..],
                    )?;

                    if cmd.flags & RUTABAGA_FLAG_FENCE != 0 {
                        let fence_id = kumquat_gpu.allocate_id() as u64;
                        let fence = RutabagaFence {
                            flags: cmd.flags,
                            fence_id,
                            ctx_id: cmd.ctx_id,
                            ring_idx: cmd.ring_idx,
                        };

                        let mut fence_descriptor_opt: Option<MagmaGpuHandle> = None;
                        let actual_fence = cmd.flags & RUTABAGA_FLAG_FENCE_HOST_SHAREABLE != 0;
                        if !actual_fence {
                            // This end signals when the fence retires; the
                            // guest waits, so it gets the other half.
                            let (signaler, waiter) = create_event_pair()?;
                            let emulated_fence: MagmaGpuHandle = waiter.into();

                            fence_descriptor_opt = Some(emulated_fence);
                            let mut fence_state = kumquat_gpu.fence_state.lock().unwrap();
                            fence_state.pending_fences.insert(fence_id, signaler);
                        }

                        kumquat_gpu.rutabaga.create_fence(fence)?;

                        if actual_fence {
                            fence_descriptor_opt =
                                Some(kumquat_gpu.rutabaga.export_fence(fence_id)?);
                            kumquat_gpu.rutabaga.destroy_fences(&[fence_id])?;
                        }

                        let fence_descriptor = fence_descriptor_opt
                            .ok_or(MagmaGpuError::WithContext("No fence descriptor"))?;

                        let resp = kumquat_gpu_protocol_resp_cmd_submit_3d {
                            hdr: kumquat_gpu_protocol_ctrl_hdr {
                                type_: KUMQUAT_GPU_PROTOCOL_RESP_CMD_SUBMIT_3D,
                                ..Default::default()
                            },
                            fence_id,
                            handle_type: fence_descriptor.handle_type,
                            ..Default::default()
                        };

                        self.stream.write(KumquatGpuProtocolWrite::CmdWithHandle(
                            resp,
                            fence_descriptor,
                        ))?;
                    }
                }
                KumquatGpuProtocol::ResourceCreateBlob(cmd) => {
                    let resource_id = kumquat_gpu.allocate_id();

                    let resource_create_blob = ResourceCreateBlob {
                        blob_mem: cmd.blob_mem,
                        blob_flags: cmd.blob_flags,
                        blob_id: cmd.blob_id,
                        size: cmd.size,
                    };

                    // GUEST blobs with CREATE_GUEST_HANDLE: there is no guest
                    // kernel to allocate/export the blob's memory, so kumquat
                    // provides it (sealed memfd, exported as dma-buf). gfxstream
                    // dereferences the handle unconditionally for this blob type
                    // (virtio_gpu_resource.cpp: ManagedDescriptor) — passing None
                    // segfaults the backend. All other blob types carry their own
                    // or no backing and take None like before.
                    let backing = if cmd.blob_mem == RUTABAGA_BLOB_MEM_GUEST
                        && (cmd.blob_flags & BLOB_FLAG_CREATE_GUEST_HANDLE) != 0
                    {
                        Some(create_guest_blob_backing(cmd.size as u64)?)
                    } else {
                        None
                    };

                    let guest_blob = backing.is_some();
                    let (iovecs, handle, response_handle, mapping, dma_buf_dup) = match backing {
                        Some(GuestBlobBacking {
                            mapping,
                            iovecs,
                            guest_handle,
                            response_handle,
                            dma_buf_dup,
                        }) => (
                            Some(iovecs),
                            Some(RutabagaHandle::from(guest_handle)),
                            Some(response_handle),
                            Some(mapping),
                            Some(dma_buf_dup),
                        ),
                        None => (None, None, None, None, None),
                    };

                    kumquat_gpu.rutabaga.resource_create_blob(
                        cmd.ctx_id,
                        resource_id,
                        resource_create_blob,
                        iovecs,
                        handle,
                    )?;

                    // GUEST blob memory belongs to the guest: the response
                    // carries a dup of the guest memory (the client mmaps it),
                    // and gfxstream holds the dma-buf for the vkAllocateMemory
                    // import. export_blob would fail here (the resource has no
                    // blob memory of its own) and must not be used.
                    let handle = match response_handle {
                        Some(handle) => handle,
                        None => {
                            let handle = kumquat_gpu.rutabaga.export_blob(resource_id)?;
                            MagmaGpuHandle::try_from(handle)?
                        }
                    };

                    // Keep a dup of the fd handed to the client: after a
                    // snapshot restore the gfxstream resource must be
                    // re-attached to this client-visible memory (e.g. the ASG
                    // ring) instead of a fresh allocation.
                    let exported = match descriptor_try_clone(&handle) {
                        Some(descriptor) => Some(ExportedBlob {
                            ctx_id: cmd.ctx_id,
                            blob_id: cmd.blob_id as u32,
                            size: cmd.size as u64,
                            descriptor,
                            dma_buf: dma_buf_dup,
                            mapping: None,
                        }),
                        None => None,
                    };

                    let mut vk_info: RutabagaVulkanInfo = Default::default();
                    if let Ok(vulkan_info) = kumquat_gpu.rutabaga.vulkan_info(resource_id) {
                        vk_info = vulkan_info;
                    }

                    kumquat_gpu.resources.insert(
                        resource_id,
                        KumquatGpuResource {
                            attached_contexts: Set::from([cmd.ctx_id]),
                            mapping: mapping.map(Arc::new),
                            exported,
                            created_epoch: SNAPSHOT_EPOCH.load(Ordering::SeqCst),
                            guest_blob,
                        },
                    );

                    let resp = kumquat_gpu_protocol_resp_resource_create {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_RESOURCE_CREATE,
                            ..Default::default()
                        },
                        resource_id,
                        handle_type: handle.handle_type,
                        vulkan_info: VulkanInfo {
                            memory_idx: vk_info.memory_idx,
                            device_id: DeviceId {
                                device_uuid: vk_info.device_id.device_uuid,
                                driver_uuid: vk_info.device_id.driver_uuid,
                            },
                        },
                    };

                    self.stream
                        .write(KumquatGpuProtocolWrite::CmdWithHandle(resp, handle))?;

                    kumquat_gpu
                        .rutabaga
                        .context_attach_resource(cmd.ctx_id, resource_id)?;
                }
                KumquatGpuProtocol::SnapshotSave => {
                    kumquat_gpu.rutabaga.snapshot(Path::new(SNAPSHOT_DIR))?;

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_OK_SNAPSHOT,
                        payload: 0,
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::SnapshotRestore => {
                    // Re-attach lives in KumquatGpu::rutabaga_restore — the
                    // SIGUSR2 path bypasses this protocol arm.
                    kumquat_gpu.rutabaga.restore(Path::new(SNAPSHOT_DIR))?;

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_OK_SNAPSHOT,
                        payload: 0,
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::OkNoData => {
                    hung_up = true;
                }
                _ => {
                    error!("Unsupported protocol {protocol:?}");
                    return Err(MagmaGpuError::Unsupported.into());
                }
            };
        }

        Ok(hung_up)
    }
}

impl AsBorrowedDescriptor for KumquatGpuConnection {
    fn as_borrowed_descriptor(&self) -> &OwnedDescriptor {
        self.stream.as_borrowed_descriptor()
    }
}

impl KumquatGpu {
    pub fn rutabaga_snapshot(
        &mut self,
        directory: &std::path::Path,
        slot: u32,
    ) -> KumquatGpuResult<()> {
        let epoch = SNAPSHOT_EPOCH.load(Ordering::SeqCst);
        eprintln!(
            "kumquat: [detect] snapshot begin (slot {slot}) epoch {epoch}, {} resources live",
            self.resources.len()
        );
        self.rutabaga.snapshot(directory)?;
        let state = self.slot_states.entry(slot).or_default();
        state.snapshot_contexts = self
            .resources
            .iter()
            .map(|(id, resource)| (*id, resource.attached_contexts.clone()))
            .collect();
        // A re-snapshot of this slot supersedes its old archive; other
        // slots keep theirs.
        state.archived_resources.clear();
        // Everything created from now on belongs to a later era than this
        // slot's snapshot.
        SNAPSHOT_EPOCH.fetch_add(1, Ordering::SeqCst);
        state.epoch = SNAPSHOT_EPOCH.load(Ordering::SeqCst);
        eprintln!(
            "kumquat: [detect] snapshot done (slot {slot}), next epoch {}",
            state.epoch
        );
        Ok(())
    }

    pub fn rutabaga_restore(
        &mut self,
        directory: &std::path::Path,
        slot: u32,
    ) -> KumquatGpuResult<()> {
        let Some(state) = self.slot_states.get_mut(&slot) else {
            return Err(KumquatGpuError::SnapshotError(format!(
                "no snapshot has been taken in slot {slot} this session"
            )));
        };
        let epoch = state.epoch;
        let post_checkpoint: Vec<u32> = self
            .resources
            .iter()
            .filter(|(_, r)| r.created_epoch >= epoch)
            .map(|(id, _)| *id)
            .collect();
        if !post_checkpoint.is_empty() {
            eprintln!(
                "kumquat: [detect] restore (slot {slot}) against epoch {epoch}: {} resource(s) created AFTER this slot's snapshot: {:?} — these do not exist at the restore point",
                post_checkpoint.len(),
                post_checkpoint
            );
        }
        // Split the slot's bookkeeping out (contexts by clone, archives by
        // value) so the cross-slot archiving below can borrow slot_states.
        let contexts = state.snapshot_contexts.clone();
        let archived = std::mem::take(&mut state.archived_resources);

        // Drop the resources this slot does not know. A plain retain would
        // silently lose them for OTHER slots: a resource created between two
        // snapshots is referenced by the later slot only, and an earlier
        // restore's retain removed it without archiving — the later slot's
        // restore then found its context id missing and failed with
        // InvalidResourceId. Archive each dropped resource for every slot
        // whose contexts still know it, exactly like the unref path.
        let dropped: Vec<u32> = self
            .resources
            .keys()
            .filter(|id| !contexts.contains_key(*id))
            .copied()
            .collect();
        for id in &dropped {
            let resource = &self.resources[id];
            for other in self.slot_states.values_mut() {
                if let Some(other_contexts) = other.snapshot_contexts.get(id) {
                    let mut copy = resource.try_clone_for_archive()?;
                    copy.attached_contexts = other_contexts.clone();
                    other.archived_resources.insert(*id, copy);
                }
            }
        }
        for id in &dropped {
            self.resources.remove(id);
        }

        for (id, resource) in archived {
            self.resources.insert(id, resource);
        }
        for (id, other_contexts) in &contexts {
            let resource = self
                .resources
                .get_mut(id)
                .ok_or(RutabagaError::InvalidResourceId)?;
            resource.attached_contexts = other_contexts.clone();
        }
        // Re-attach the client-visible blob mappings (e.g. the ASG ring the
        // client keeps mapping) BEFORE the restore: the gfxstream frontend
        // consumes the registered mappings while recreating the resources.
        for (resource_id, res) in self.resources.iter_mut() {
            let Some(exported) = &mut res.exported else {
                continue;
            };
            // Re-register a fresh dup of the dma-buf descriptor on every
            // restore: the replayed vkAllocateMemory consumes it each time.
            if let Some(dma_buf) = &exported.dma_buf {
                if let Ok(dup) = dma_buf.try_clone() {
                    use magma_gpu::util::IntoRawDescriptor;
                    rutabaga_gfx::reattach_blob_descriptor(
                        exported.ctx_id,
                        exported.blob_id as u64,
                        dup.into_raw_descriptor(),
                        MAGMA_GPU_HANDLE_TYPE_MEM_DMABUF,
                    );
                }
            }
            if exported.mapping.is_none() {
                let Ok(clone) = exported.descriptor.try_clone() else {
                    continue;
                };
                let clone_fd = clone.as_raw_descriptor();
                let clone_flags = unsafe { libc::fcntl(clone_fd, libc::F_GETFL) };
                // Capture identity NOW: from_safe_descriptor consumes (and on
                // failure closes) the descriptor, so /proc reads must happen
                // before the mmap attempt.
                let fd_kind = std::fs::read_link(format!("/proc/self/fd/{clone_fd}"))
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|e| format!("readlink failed: {e}"));
                let fdinfo = std::fs::read_to_string(format!("/proc/self/fdinfo/{clone_fd}"))
                    .map(|s| s.lines().take(6).collect::<Vec<_>>().join("; "))
                    .unwrap_or_else(|e| format!("fdinfo failed: {e}"));
                match MemoryMapping::from_safe_descriptor(
                    clone,
                    exported.size as usize,
                    RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
                ) {
                    Ok(mapping) => {
                        let raw = mapping.as_raw_mapping();
                        eprintln!(
                            "kumquat: reattach resource {resource_id} ctx {} blob {} at {:#x} (fd {clone_fd} flags {clone_flags:#x})",
                            exported.ctx_id, exported.blob_id, raw.ptr
                        );
                        exported.mapping = Some(Arc::new(mapping));
                    }
                    Err(e) => {
                        // TODO: hack
                        // mmap gives EPERM for PROT_WRITE on an O_RDONLY fd —
                        // log the fd access mode so RO descriptors are visible.
                        if format!("{e:?}").contains("PermissionDenied")
                            && fd_kind.contains("dmabuf")
                        {
                            // The kernel refuses CPU mappings for VRAM-only
                            // images (e.g. swapchain dma-bufs). These blobs
                            // never had a host-side mapping on the normal
                            // path; the descriptor re-registration above is
                            // what the replay needs. Verified benign.
                            eprintln!(
                                "kumquat: blob {blob} (resource {resource_id}) not CPU-mappable ({fd_kind}) — descriptor re-registered, no host mapping needed",
                                blob = exported.blob_id
                            );
                        } else {
                            eprintln!(
                                "kumquat: re-attach mapping failed for resource {resource_id} ctx {} blob {}: {e} fd={clone_fd} flags={clone_flags:#x} kind={fd_kind} fdinfo=[{fdinfo}]",
                                exported.ctx_id, exported.blob_id,
                            );
                        }
                        continue;
                    }
                }
            }
            // Re-register on every restore: the external object manager
            // consumes the mapping on use, and the mapping itself stays alive
            // in `exported`.
            let mapping = exported.mapping.as_ref().unwrap();
            let raw = mapping.as_raw_mapping();
            rutabaga_gfx::reattach_blob_mapping(
                exported.ctx_id,
                exported.blob_id as u32,
                raw.ptr as *mut c_void,
                RUTABAGA_MAP_CACHE_CACHED,
            );
        }

        // Write guard: GUEST blob memory is guest-owned snapshot state. While
        // the restore replays, the only legal writer is valo's CPU restore
        // through its own mapping (the guest itself is frozen). Any write from
        // THIS process would silently corrupt what the guest resumes with — so
        // fail loudly instead: mprotect our mappings of guest blob memory
        // read-only for the duration of the replay, letting an offending write
        // SIGSEGV at the culprit with a backtrace. The ring mappings are NOT
        // guarded: the ASG host legitimately writes ring state during restore
        // (value-equal to the checkpoint content). Writes through separate
        // host-driver mappings of the dma-buf are NOT covered by this guard.
        let mut guarded: Vec<(usize, usize)> = Vec::new();
        for res in self.resources.values() {
            if !res.guest_blob {
                continue;
            }
            if let Some(mapping) = &res.mapping {
                let raw = mapping.as_raw_mapping();
                guarded.push((raw.ptr as usize, raw.size as usize));
            }
            if let Some(exported) = &res.exported {
                if let Some(mapping) = &exported.mapping {
                    let raw = mapping.as_raw_mapping();
                    guarded.push((raw.ptr as usize, raw.size as usize));
                }
            }
        }
        for (ptr, size) in &guarded {
            let ret = unsafe { libc::mprotect((*ptr) as *mut c_void, *size, libc::PROT_READ) };
            if ret != 0 {
                return Err(MagmaGpuError::IoError(std::io::Error::last_os_error()).into());
            }
        }
        // Self-test for the write guard (see comment above): deliberately
        // write into the first guarded mapping to prove the guard crashes
        // loudly instead of corrupting silently. Only with
        // KUMQUAT_WRITE_GUARD_PROBE=1.
        if !guarded.is_empty()
            && std::env::var("KUMQUAT_WRITE_GUARD_PROBE")
                .map(|v| !matches!(v.as_str(), "" | "0" | "false" | "no"))
                .unwrap_or(false)
        {
            let (ptr, _size) = guarded[0];
            eprintln!("kumquat: write guard probe: writing to {ptr:#x}");
            // SAFETY: deliberate write into the guarded mapping.
            unsafe { std::ptr::write_volatile(ptr as *mut u8, 0x42) };
            eprintln!("kumquat: write guard probe: write did NOT crash — guard is broken!");
        }
        let restore_result = self.rutabaga.restore(directory);
        for (ptr, size) in &guarded {
            // SAFETY: same address/size that were mprotected above; restoring
            // the original read/write access unconditionally, also on error.
            let ret = unsafe {
                libc::mprotect(
                    (*ptr) as *mut c_void,
                    *size,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
            if ret != 0 {
                eprintln!(
                    "kumquat: write guard: mprotect RW failed at {ptr:#x}: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
        restore_result?;
        self.reattach_backings();
        Ok(())
    }

    /// Rutabaga::restore() rebuilds resources without backing (the iovec
    /// pointers are VMM state). Our SHM mappings survive — re-bind them so the
    /// restored resources point at the same memory as before.
    fn reattach_backings(&mut self) {
        for (resource_id, res) in &self.resources {
            if let Some(mapping) = &res.mapping {
                let raw = mapping.as_raw_mapping();
                let vecs = vec![RutabagaIovec {
                    base: raw.ptr as *mut c_void,
                    len: raw.size as usize,
                }];
                if let Err(e) = self.rutabaga.attach_backing(*resource_id, vecs) {
                    eprintln!(
                        "kumquat: re-attach backing for resource {resource_id} failed: {e:?}"
                    );
                }
            }
        }
    }
}
