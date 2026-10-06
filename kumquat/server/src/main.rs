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
            let (result, ok_byte, err_byte) = match request {
                GpuRequest::Snapshot => {
                    println!("kumquat: snapshot requested");
                    (
                        std::fs::create_dir_all(&args.snapshot_dir)
                            .map_err(magma_gpu::util::Error::IoError)
                            .map_err(Into::into)
                            .and_then(|()| kumquat.rutabaga_snapshot(&args.snapshot_dir)),
                        b'S',
                        b's',
                    )
                }
                GpuRequest::Restore => {
                    println!("kumquat: restore requested");
                    (kumquat.rutabaga_restore(&args.snapshot_dir), b'R', b'r')
                }
            };
            match &result {
                Ok(()) => println!(
                    "kumquat: {} done from {}",
                    match request {
                        GpuRequest::Snapshot => "snapshot",
                        GpuRequest::Restore => "restore",
                    },
                    args.snapshot_dir.display()
                ),
                Err(e) => eprintln!("kumquat: {} failed: {e:?}", {
                    match request {
                        GpuRequest::Snapshot => "snapshot",
                        GpuRequest::Restore => "restore",
                    }
                }),
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
