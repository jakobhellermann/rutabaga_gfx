// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap as Map;
use std::collections::VecDeque;
use std::path::PathBuf;

use magma_gpu::util::AsBorrowedDescriptor;
use magma_gpu::util::AsRawDescriptor;
use magma_gpu::util::Error as MagmaGpuError;
use magma_gpu::util::Listener;
use magma_gpu::util::OwnedDescriptor;
use magma_gpu::util::WaitContext;
use magma_gpu::util::WaitTimeout;

use crate::kumquat_gpu::KumquatGpu;
use crate::kumquat_gpu::KumquatGpuConnection;
use crate::kumquat_gpu::KumquatGpuResult;

enum KumquatConnection {
    GpuListener,
    GpuConnection(Box<KumquatGpuConnection>),
}

/// Snapshot or restore request delivered over the blocked SIGUSR1/2 signalfd.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRequest {
    Snapshot,
    Restore,
}

/// Reserved wait-context id for the signalfd; connection ids never reach it.
const SIGNAL_CONNECTION_ID: u64 = u64::MAX;

pub struct Kumquat {
    connection_id: u64,
    wait_ctx: WaitContext,
    kumquat_gpu_opt: Option<KumquatGpu>,
    gpu_listener_opt: Option<Listener>,
    connections: Map<u64, KumquatConnection>,
    signal_fd: OwnedDescriptor,
    pending_requests: VecDeque<GpuRequest>,
}

impl Kumquat {
    pub fn rutabaga_snapshot(&mut self, directory: &std::path::Path) -> KumquatGpuResult<()> {
        if let Some(gpu) = &mut self.kumquat_gpu_opt {
            gpu.rutabaga_snapshot(directory)?;
        }
        Ok(())
    }

    pub fn rutabaga_restore(&mut self, directory: &std::path::Path) -> KumquatGpuResult<()> {
        if let Some(gpu) = &mut self.kumquat_gpu_opt {
            gpu.rutabaga_restore(directory)?;
        }
        Ok(())
    }

    /// Oldest signalfd request not yet handled by the main loop.
    pub fn take_gpu_request(&mut self) -> Option<GpuRequest> {
        self.pending_requests.pop_front()
    }

    fn drain_signal_fd(&mut self) -> KumquatGpuResult<()> {
        loop {
            // SAFETY: info is a plain C struct; a successful read fills it.
            let mut info: libc::signalfd_siginfo = unsafe { std::mem::zeroed() };
            let size = std::mem::size_of_val(&info);
            let read = unsafe {
                libc::read(
                    self.signal_fd.as_raw_descriptor(),
                    (&mut info as *mut libc::signalfd_siginfo).cast(),
                    size,
                )
            };
            if read == size as isize {
                let request = match i32::try_from(info.ssi_signo).ok() {
                    Some(libc::SIGUSR1) => GpuRequest::Snapshot,
                    Some(libc::SIGUSR2) => GpuRequest::Restore,
                    _ => continue,
                };
                // Blocked standard signals coalesce, so the queue holds at
                // most one entry per request kind.
                self.pending_requests.push_back(request);
                continue;
            }
            if read < 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                if error.raw_os_error() == Some(libc::EAGAIN) {
                    return Ok(());
                }
                return Err(MagmaGpuError::IoError(error).into());
            }
            // signalfd never returns partial records; a zero read would mean
            // the (never closed) fd went away — treat as drained.
            return Ok(());
        }
    }

    pub fn run(&mut self) -> KumquatGpuResult<()> {
        let events = self.wait_ctx.wait(WaitTimeout::NoTimeout)?;
        for event in events {
            if event.connection_id == SIGNAL_CONNECTION_ID {
                self.drain_signal_fd()?;
                continue;
            }
            let mut hung_up = false;
            match self.connections.entry(event.connection_id) {
                Entry::Occupied(mut o) => {
                    let connection = o.get_mut();
                    match connection {
                        KumquatConnection::GpuListener => {
                            if let Some(ref listener) = self.gpu_listener_opt {
                                // A failed accept is transient — skip it, keep serving.
                                let Ok(stream) = listener.accept() else { continue };
                                self.connection_id += 1;
                                let new_gpu_conn = KumquatGpuConnection::new(stream);
                                self.wait_ctx.add(
                                    self.connection_id,
                                    new_gpu_conn.as_borrowed_descriptor(),
                                )?;
                                self.connections.insert(
                                    self.connection_id,
                                    KumquatConnection::GpuConnection(Box::new(new_gpu_conn)),
                                );
                            }
                        }
                        KumquatConnection::GpuConnection(ref mut gpu_conn) => {
                            if event.readable {
                                if let Some(ref mut kumquat_gpu) = self.kumquat_gpu_opt {
                                    // A client dying mid-protocol (ECONNRESET etc.) drops that
                                    // connection — it must not take the server down with it.
                                    hung_up = match gpu_conn.process_command(kumquat_gpu) {
                                        Ok(processed) => !processed && event.hung_up,
                                        Err(e) => {
                                            eprintln!(
                                                "kumquat: client connection error: {e}"
                                            );
                                            true
                                        }
                                    };
                                }
                            }

                            if hung_up {
                                self.wait_ctx.delete(gpu_conn.as_borrowed_descriptor())?;
                                o.remove_entry();
                            }
                        }
                    }
                }
                Entry::Vacant(_) => {
                    return Err(MagmaGpuError::WithContext("no connection found").into())
                }
            }
        }

        Ok(())
    }
}

pub struct KumquatBuilder {
    capset_names_opt: Option<String>,
    gpu_socket_opt: Option<String>,
    renderer_features_opt: Option<String>,
    signal_fd_opt: Option<OwnedDescriptor>,
}

impl KumquatBuilder {
    pub fn new() -> KumquatBuilder {
        KumquatBuilder {
            capset_names_opt: None,
            gpu_socket_opt: None,
            renderer_features_opt: None,
            signal_fd_opt: None,
        }
    }

    pub fn set_capset_names(mut self, capset_names: String) -> KumquatBuilder {
        self.capset_names_opt = Some(capset_names);
        self
    }

    pub fn set_gpu_socket(mut self, gpu_socket_opt: Option<String>) -> KumquatBuilder {
        self.gpu_socket_opt = gpu_socket_opt;
        self
    }

    pub fn set_renderer_features(mut self, renderer_features_opt: String) -> KumquatBuilder {
        self.renderer_features_opt = Some(renderer_features_opt);
        self
    }

    pub fn set_signal_fd(mut self, signal_fd: OwnedDescriptor) -> KumquatBuilder {
        self.signal_fd_opt = Some(signal_fd);
        self
    }

    pub fn build(self) -> KumquatGpuResult<Kumquat> {
        let connection_id: u64 = 0;
        let mut wait_ctx = WaitContext::new()?;
        let mut kumquat_gpu_opt: Option<KumquatGpu> = None;
        let mut gpu_listener_opt: Option<Listener> = None;
        let mut connections: Map<u64, KumquatConnection> = Default::default();

        if let Some(gpu_socket) = self.gpu_socket_opt {
            // Remove path if it exists
            let path = PathBuf::from(&gpu_socket);
            let _ = std::fs::remove_file(&path);

            // Should not panic, since main.rs always calls set_capset_names and
            // set_renderer_features, even with the empty string.
            kumquat_gpu_opt = Some(KumquatGpu::new(
                self.capset_names_opt.unwrap(),
                self.renderer_features_opt.unwrap(),
            )?);

            let gpu_listener = Listener::bind(path)?;
            wait_ctx.add(connection_id, gpu_listener.as_borrowed_descriptor())?;
            connections.insert(connection_id, KumquatConnection::GpuListener);
            gpu_listener_opt = Some(gpu_listener);
        }

        // Should not panic, since main.rs always calls set_signal_fd before
        // build().
        let signal_fd = self.signal_fd_opt.unwrap();
        wait_ctx.add(SIGNAL_CONNECTION_ID, &signal_fd)?;

        Ok(Kumquat {
            connection_id,
            wait_ctx,
            kumquat_gpu_opt,
            gpu_listener_opt,
            connections,
            signal_fd,
            pending_requests: Default::default(),
        })
    }
}
