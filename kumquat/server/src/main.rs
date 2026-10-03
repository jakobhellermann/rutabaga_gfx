// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

mod kumquat;
mod kumquat_gpu;

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
}

fn main() -> KumquatGpuResult<()> {
    let args = Args::parse();

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

    loop {
        if SNAPSHOT_REQUESTED.swap(false, Ordering::SeqCst) {
            println!("kumquat: snapshot requested");
            let dir = kumquat_gpu_snapshot_dir();
            let _ = std::fs::create_dir_all(&dir);
            match kumquat.rutabaga_snapshot(std::path::Path::new(&dir)) {
                Ok(()) => println!("kumquat: snapshot written to {}", dir),
                Err(e) => println!("kumquat: snapshot failed: {:?}", e),
            }
        }
        if RESTORE_REQUESTED.swap(false, Ordering::SeqCst) {
            println!("kumquat: restore requested");
            let dir = kumquat_gpu_snapshot_dir();
            match kumquat.rutabaga_restore(std::path::Path::new(&dir)) {
                Ok(()) => println!("kumquat: restore done from {}", dir),
                Err(e) => println!("kumquat: restore failed: {:?}", e),
            }
        }
        kumquat.run()?;
    }
}

fn kumquat_gpu_snapshot_dir() -> String {
    "/tmp/kumquat-snapshot".to_string()
}
