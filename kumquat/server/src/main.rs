// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

mod kumquat;
mod kumquat_gpu;

use std::io::Write;
use std::os::fd::FromRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

static SNAPSHOT_REQUESTED: AtomicBool = AtomicBool::new(false);
static RESTORE_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_sigusr1(_: libc::c_int) {
    SNAPSHOT_REQUESTED.store(true, Ordering::SeqCst);
}

extern "C" fn on_sigusr2(_: libc::c_int) {
    RESTORE_REQUESTED.store(true, Ordering::SeqCst);
}

use clap::Parser;
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

fn main() -> KumquatGpuResult<()> {
    let args = Args::parse();

    // gfxstream's Vulkan snapshot capture (VkReconstruction: handles, queues,
    // call log) is default-off and read at renderer init — without it the
    // snapshot carries no Vulkan state and post-restore submits crash the
    // decoder ("Failed to unbox VkQueue").
    // SAFETY: single-threaded startup, no other threads read the environment.
    unsafe { std::env::set_var("ANDROID_GFXSTREAM_CAPTURE_VK_SNAPSHOT", "1") };

    unsafe {
        libc::signal(libc::SIGUSR1, on_sigusr1 as *const () as usize);
        libc::signal(libc::SIGUSR2, on_sigusr2 as *const () as usize);
    }

    let mut kumquat = KumquatBuilder::new()
        .set_capset_names(args.capset_names)
        .set_gpu_socket((!args.gpu_socket_path.is_empty()).then_some(args.gpu_socket_path))
        .set_renderer_features(args.renderer_features)
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
        if SNAPSHOT_REQUESTED.swap(false, Ordering::SeqCst) {
            println!("kumquat: snapshot requested");
            let result = std::fs::create_dir_all(&args.snapshot_dir)
                .map_err(magma_gpu::util::Error::IoError)
                .map_err(Into::into)
                .and_then(|()| kumquat.rutabaga_snapshot(&args.snapshot_dir));
            match &result {
                Ok(()) => println!(
                    "kumquat: snapshot written to {}",
                    args.snapshot_dir.display()
                ),
                Err(e) => eprintln!("kumquat: snapshot failed: {e:?}"),
            };
            send_result(&mut result_pipe, if result.is_ok() { b'S' } else { b's' })?;
        }
        if RESTORE_REQUESTED.swap(false, Ordering::SeqCst) {
            println!("kumquat: restore requested");
            let result = kumquat.rutabaga_restore(&args.snapshot_dir);
            match &result {
                Ok(()) => println!("kumquat: restore done from {}", args.snapshot_dir.display()),
                Err(e) => eprintln!("kumquat: restore failed: {e:?}"),
            };
            send_result(&mut result_pipe, if result.is_ok() { b'R' } else { b'r' })?;
        }
        kumquat.run()?;
    }
}

fn send_result(pipe: &mut Option<std::fs::File>, status: u8) -> KumquatGpuResult<()> {
    if let Some(pipe) = pipe {
        pipe.write_all(&[status])
            .map_err(magma_gpu::util::Error::IoError)?;
    }
    Ok(())
}
