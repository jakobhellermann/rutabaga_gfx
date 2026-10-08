// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

mod kumquat;
mod kumquat_gpu;

use std::io::Write;
use std::os::fd::FromRawFd;
use std::path::PathBuf;

use clap::Parser;
use kumquat::GpuRequest;
use kumquat::KumquatBuilder;
use magma_gpu::util::FromRawDescriptor;
use magma_gpu::util::IntoRawDescriptor;
use magma_gpu::util::OwnedDescriptor;
use magma_gpu::util::WritePipe;

use crate::kumquat_gpu::KumquatGpuError;
use crate::kumquat_gpu::KumquatGpuResult;

#[derive(Parser, Debug)]
#[command(version = "1.71", about = None, long_about = None)]
struct Args {
    /// Colon-separated list of virtio-gpu capsets.  For example,
    /// "--capset-names=gfxstream-vulkan:cross-domain"
    #[arg(long, default_value = "gfxstream-vulkan")]
    capset_names: String,

    /// Path to the emulated virtio-gpu socket.
    #[arg(long, default_value = "/tmp/kumquat-gpu-0")]
    gpu_socket_path: String,

    /// Opaque renderer specific features
    #[arg(long, default_value = "")]
    renderer_features: String,

    /// An OS-specific pipe descriptor to the parent process
    #[arg(long, default_value = "0")]
    pipe_descriptor: i64,

    /// Write one completion byte after each signal-triggered GPU operation.
    #[arg(long)]
    result_fd: Option<i32>,

    #[arg(long, default_value = "/tmp/kumquat-snapshot")]
    snapshot_dir: PathBuf,

    /// Base directory for savestate slots: slot N snapshots into
    /// `<base>/<N>/gpu`. When set it replaces the single-shot
    /// `--snapshot-dir`, which only serves slot 0 (legacy/manual runs).
    #[arg(long)]
    snapshot_base: Option<PathBuf>,
}

/// Block the GPU request signals and return a nonblocking signalfd reporting
/// them. Blocking is required for signalfd delivery and replaces the default
/// disposition (terminate); threads spawned later inherit the mask.
fn create_signal_fd() -> std::result::Result<OwnedDescriptor, magma_gpu::util::Error> {
    // SAFETY: the sigset calls only mutate the local set; signalfd returns a
    // fresh descriptor owned from here on.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGUSR1);
        libc::sigaddset(&mut set, libc::SIGUSR2);
        if libc::sigprocmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) != 0 {
            return Err(magma_gpu::util::Error::IoError(
                std::io::Error::last_os_error(),
            ));
        }
        let fd = libc::signalfd(-1, &set, libc::SFD_CLOEXEC | libc::SFD_NONBLOCK);
        if fd < 0 {
            return Err(magma_gpu::util::Error::IoError(
                std::io::Error::last_os_error(),
            ));
        }
        Ok(OwnedDescriptor::from_raw_descriptor(fd))
    }
}

fn main() -> KumquatGpuResult<()> {
    let args = Args::parse();

    // gfxstream's Vulkan snapshot capture (VkReconstruction: handles, queues,
    // call log) is default-off and read at renderer init — without it the
    // snapshot carries no Vulkan state and post-restore submits crash the
    // decoder ("Failed to unbox VkQueue").
    // SAFETY: single-threaded startup, no other threads read the environment.
    unsafe { std::env::set_var("ANDROID_GFXSTREAM_CAPTURE_VK_SNAPSHOT", "1") };

    // Must run before the renderer spawns threads: they inherit the mask.
    let signal_fd = create_signal_fd()?;

    let mut kumquat = KumquatBuilder::new()
        .set_capset_names(args.capset_names)
        .set_gpu_socket((!args.gpu_socket_path.is_empty()).then_some(args.gpu_socket_path))
        .set_renderer_features(args.renderer_features)
        .set_signal_fd(signal_fd)
        .build()?;

    if args.pipe_descriptor != 0 {
        // SAFETY: the caller passed a descriptor it opened and handed over.
        let descriptor = unsafe {
            OwnedDescriptor::from_raw_descriptor(args.pipe_descriptor.into_raw_descriptor())
        };
        let write_pipe = WritePipe::new(descriptor);
        write_pipe.write(&1u64.to_le_bytes())?;
    }

    let mut result_pipe = args.result_fd.map(|fd| {
        // SAFETY: the launcher passes an owned descriptor that this process
        // alone closes; it is not used by the virtio-gpu transport.
        unsafe { std::fs::File::from_raw_fd(fd) }
    });

    loop {
        kumquat.run()?;
        while let Some(request) = kumquat.take_gpu_request() {
            let (op, slot) = match request {
                GpuRequest::Snapshot(slot) => ("snapshot", slot),
                GpuRequest::Restore(slot) => ("restore", slot),
            };
            println!("kumquat: {op} requested (slot {slot})");
            let (result, ok_byte, err_byte) = match request {
                GpuRequest::Snapshot(slot) => (
                    slot_directory(args.snapshot_base.as_deref(), &args.snapshot_dir, slot)
                        .and_then(|dir| {
                            std::fs::create_dir_all(&dir)
                                .map_err(magma_gpu::util::Error::IoError)
                                .map_err(Into::into)
                                .and_then(|()| kumquat.rutabaga_snapshot(&dir, slot))
                        }),
                    b'S',
                    b's',
                ),
                GpuRequest::Restore(slot) => (
                    slot_directory(args.snapshot_base.as_deref(), &args.snapshot_dir, slot)
                        .and_then(|dir| kumquat.rutabaga_restore(&dir, slot)),
                    b'R',
                    b'r',
                ),
            };
            let dir = slot_directory(args.snapshot_base.as_deref(), &args.snapshot_dir, slot)
                .map(|d| d.display().to_string());
            match &result {
                Ok(()) => {
                    println!(
                        "kumquat: {op} (slot {slot}) done from {}",
                        dir.unwrap_or_default()
                    )
                }
                Err(e) => eprintln!("kumquat: {op} (slot {slot}) failed: {e:?}"),
            };
            send_result(
                &mut result_pipe,
                if result.is_ok() { ok_byte } else { err_byte },
            )?;
        }
    }
}

fn send_result(pipe: &mut Option<std::fs::File>, status: u8) -> KumquatGpuResult<()> {
    if let Some(pipe) = pipe {
        pipe.write_all(&[status])
            .map_err(magma_gpu::util::Error::IoError)?;
    }
    Ok(())
}

/// Resolve the snapshot directory for a slot request. With
/// `--snapshot-base` every slot lives in `<base>/<slot>/gpu` (valo's
/// `$VALO_HOME/slots` layout); the legacy `--snapshot-dir` only serves
/// slot 0. Takes the fields disjointly instead of `&Args` because the
/// builder moves `args.renderer_features`.
fn slot_directory(
    snapshot_base: Option<&std::path::Path>,
    snapshot_dir: &std::path::Path,
    slot: u32,
) -> KumquatGpuResult<PathBuf> {
    match snapshot_base {
        Some(base) => Ok(base.join(slot.to_string()).join("gpu")),
        None if slot == 0 => Ok(snapshot_dir.to_path_buf()),
        None => Err(KumquatGpuError::SnapshotError(format!(
            "slot {slot} requested without --snapshot-base \
             (legacy --snapshot-dir only serves slot 0)"
        ))),
    }
}
