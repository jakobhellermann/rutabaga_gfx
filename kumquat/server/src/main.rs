// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

mod control;
mod kumquat;
mod kumquat_gpu;

use std::io::Write;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use clap::Parser;
use kumquat::KumquatBuilder;
use magma_gpu::util::FromRawDescriptor;
use magma_gpu::util::IntoRawDescriptor;
use magma_gpu::util::OwnedDescriptor;
use magma_gpu::util::WritePipe;

use crate::kumquat_gpu::KumquatGpuResult;

struct StderrLogger;

/// Log lines stderr refused, e.g. on a full disk; a panic would take the renderer down with them.
static DROPPED_LINES: AtomicU64 = AtomicU64::new(0);

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let mut stderr = std::io::stderr().lock();
        let dropped = DROPPED_LINES.load(Ordering::Relaxed);
        if dropped > 0
            && writeln!(
                stderr,
                "kumquat: ERROR: {dropped} log lines lost to failed writes"
            )
            .is_ok()
        {
            DROPPED_LINES.fetch_sub(dropped, Ordering::Relaxed);
        }
        if writeln!(stderr, "kumquat: {}: {}", record.level(), record.args()).is_err() {
            DROPPED_LINES.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush(&self) {}
}

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

    /// A SOCK_SEQPACKET socket descriptor for snapshot/restore commands from the parent process
    #[arg(long)]
    control_fd: Option<i32>,
}

fn main() -> KumquatGpuResult<()> {
    log::set_logger(&StderrLogger).expect("no other logger is installed");
    log::set_max_level(log::LevelFilter::Info);
    let args = Args::parse();

    let mut kumquat = KumquatBuilder::new()
        .set_capset_names(args.capset_names)
        .set_gpu_socket((!args.gpu_socket_path.is_empty()).then_some(args.gpu_socket_path))
        .set_renderer_features(args.renderer_features)
        .set_control(args.control_fd.map(|fd| {
            // SAFETY: the parent passed a descriptor it opened and handed over.
            unsafe { OwnedDescriptor::from_raw_descriptor(fd) }
        }))
        .build()?;

    if args.pipe_descriptor != 0 {
        // SAFETY: the caller passed a descriptor it opened and handed over.
        let descriptor = unsafe {
            OwnedDescriptor::from_raw_descriptor(args.pipe_descriptor.into_raw_descriptor())
        };
        let write_pipe = WritePipe::new(descriptor);
        write_pipe.write(&1u64.to_le_bytes())?;
    }

    while kumquat.run()? {}
    Ok(())
}
