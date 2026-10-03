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
use magma_gpu::util::create_event_pair;
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
    /// Mapping of `descriptor`, created at re-attach time and kept alive for
    /// as long as the gfxstream resource uses the address.
    mapping: Option<MemoryMapping>,
}

pub struct KumquatGpuResource {
    attached_contexts: Set<u32>,
    mapping: Option<MemoryMapping>,
    exported: Option<ExportedBlob>,
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

pub struct KumquatGpu {
    rutabaga: Rutabaga,
    fence_state: FenceState,
    id_allocator: u32,
    resources: Map<u32, KumquatGpuResource>,
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
                        .context_attach_resource(cmd.ctx_id, cmd.resource_id)?;
                }
                KumquatGpuProtocol::CtxDetachResource(cmd) => {
                    kumquat_gpu
                        .rutabaga
                        .context_detach_resource(cmd.ctx_id, cmd.resource_id)?;

                    let mut resource = kumquat_gpu
                        .resources
                        .remove(&cmd.resource_id)
                        .ok_or(RutabagaError::InvalidResourceId)?;

                    resource.attached_contexts.remove(&cmd.ctx_id);
                    if resource.attached_contexts.is_empty() {
                        if resource.mapping.is_some() {
                            kumquat_gpu.rutabaga.detach_backing(cmd.resource_id)?;
                        }

                        kumquat_gpu.rutabaga.unref_resource(cmd.resource_id)?;
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
                            mapping: Some(mapping),
                            exported: None,
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

                    let (iovecs, handle, response_handle, mapping) = match backing {
                        Some(GuestBlobBacking {
                            mapping,
                            iovecs,
                            guest_handle,
                            response_handle,
                        }) => (
                            Some(iovecs),
                            Some(RutabagaHandle::from(guest_handle)),
                            Some(response_handle),
                            Some(mapping),
                        ),
                        None => (None, None, None, None),
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
                            mapping,
                            exported,
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
    pub fn rutabaga_snapshot(&self, directory: &std::path::Path) -> KumquatGpuResult<()> {
        self.rutabaga.snapshot(directory)?;
        Ok(())
    }

    pub fn rutabaga_restore(&mut self, directory: &std::path::Path) -> KumquatGpuResult<()> {
        // Re-attach the client-visible blob mappings (e.g. the ASG ring the
        // client keeps mapping) BEFORE the restore: the gfxstream frontend
        // consumes the registered mappings while recreating the resources.
        for (resource_id, res) in self.resources.iter_mut() {
            let Some(exported) = &mut res.exported else {
                continue;
            };
            if exported.mapping.is_some() {
                continue;
            }
            let Ok(clone) = exported.descriptor.try_clone() else {
                continue;
            };
            match MemoryMapping::from_safe_descriptor(
                clone,
                exported.size as usize,
                RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
            ) {
                Ok(mapping) => {
                    let raw = mapping.as_raw_mapping();
                    eprintln!(
                        "kumquat: reattach resource {resource_id} ctx {} blob {} at {:#x}",
                        exported.ctx_id, exported.blob_id, raw.ptr
                    );
                    #[cfg(feature = "gfxstream")]
                    rutabaga_gfx::reattach_blob_mapping(
                        exported.ctx_id,
                        exported.blob_id,
                        raw.ptr as *mut c_void,
                        RUTABAGA_MAP_CACHE_CACHED,
                    );
                    exported.mapping = Some(mapping);
                }
                Err(e) => {
                    eprintln!("kumquat: re-attach mapping failed: {e:?}");
                }
            }
        }

        self.rutabaga.restore(directory)?;
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
