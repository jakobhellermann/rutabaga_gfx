// Copyright 2026 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Commands from the process that started kumquat, over a `SOCK_SEQPACKET` socket passed with
//! `--control-fd`. Every message starts with a little-endian `u32` type.

use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;

use magma_gpu::util::AsBorrowedDescriptor;
use magma_gpu::util::Error as MagmaGpuError;
use magma_gpu::util::OwnedDescriptor;
use magma_gpu::util::Tube;
use rustix::net::sockopt;
use rustix::net::SocketType;

use crate::kumquat_gpu::KumquatGpuResult;
use crate::kumquat_gpu::ObjectIdentity;

/// Parent to kumquat: snapshot into the directory given as payload.
pub const KUMQUAT_CONTROL_SNAPSHOT: u32 = 0x1;
/// Parent to kumquat: restore from the directory given as payload. Answered with OK, one REMAP
/// per blob resource whose memory the restore replaced, then END.
pub const KUMQUAT_CONTROL_RESTORE: u32 = 0x2;
/// Parent to kumquat: list the blob resources whose memory a client holds. Answered with OK,
/// one RESOURCE per resource, then END.
pub const KUMQUAT_CONTROL_LIST: u32 = 0x3;
/// Kumquat to parent: the GPU socket accepts connections.
pub const KUMQUAT_CONTROL_READY: u32 = 0x100;
/// Kumquat to parent: the last request succeeded.
pub const KUMQUAT_CONTROL_RESP_OK: u32 = 0x101;
/// Kumquat to parent: the last request failed, the payload is the UTF-8 error.
pub const KUMQUAT_CONTROL_RESP_ERR: u32 = 0x102;
/// Kumquat to parent: `u32` resource id, `u64` device and `u64` inode of its exported memory.
pub const KUMQUAT_CONTROL_RESOURCE: u32 = 0x103;
/// Kumquat to parent: `u32` resource id, with the descriptor of its new memory attached.
pub const KUMQUAT_CONTROL_REMAP: u32 = 0x104;
/// Kumquat to parent: ends a list of RESOURCE or REMAP messages.
pub const KUMQUAT_CONTROL_END: u32 = 0x105;

const MAX_MESSAGE_SIZE: usize = 4 + 4096;

pub enum ControlRequest {
    Snapshot(PathBuf),
    Restore(PathBuf),
    List,
}

pub struct Control {
    tube: Tube,
}

impl Control {
    pub fn new(socket: OwnedDescriptor) -> KumquatGpuResult<Control> {
        let socket_type = sockopt::socket_type(&socket).map_err(MagmaGpuError::from)?;
        if socket_type != SocketType::SEQPACKET {
            return Err(MagmaGpuError::WithContext("control socket must be SOCK_SEQPACKET").into());
        }
        Ok(Control {
            tube: Tube::try_from(socket)?,
        })
    }

    /// The next request, or `None` once the parent closed its end.
    pub fn receive(&self) -> KumquatGpuResult<Option<ControlRequest>> {
        let mut message = [0u8; MAX_MESSAGE_SIZE];
        let (len, descriptors) = self.tube.receive(&mut message)?;
        if len == 0 {
            return Ok(None);
        }
        if !descriptors.is_empty() || len < 4 {
            return Err(MagmaGpuError::WithContext("malformed control request").into());
        }
        let kind = u32::from_le_bytes(message[..4].try_into().unwrap());
        let directory = PathBuf::from(std::ffi::OsString::from_vec(message[4..len].to_vec()));
        match kind {
            KUMQUAT_CONTROL_SNAPSHOT => Ok(Some(ControlRequest::Snapshot(directory))),
            KUMQUAT_CONTROL_RESTORE => Ok(Some(ControlRequest::Restore(directory))),
            KUMQUAT_CONTROL_LIST => Ok(Some(ControlRequest::List)),
            _ => Err(MagmaGpuError::WithContext("unknown control request").into()),
        }
    }

    pub fn send_ready(&self) -> KumquatGpuResult<()> {
        self.send(KUMQUAT_CONTROL_READY, &[])
    }

    pub fn send_ok(&self) -> KumquatGpuResult<()> {
        self.send(KUMQUAT_CONTROL_RESP_OK, &[])
    }

    pub fn send_err(&self, error: &str) -> KumquatGpuResult<()> {
        self.send(KUMQUAT_CONTROL_RESP_ERR, error.as_bytes())
    }

    pub fn send_resource(
        &self,
        resource_id: u32,
        identity: ObjectIdentity,
    ) -> KumquatGpuResult<()> {
        let mut payload = resource_id.to_le_bytes().to_vec();
        payload.extend_from_slice(&identity.dev.to_le_bytes());
        payload.extend_from_slice(&identity.ino.to_le_bytes());
        self.send(KUMQUAT_CONTROL_RESOURCE, &payload)
    }

    pub fn send_remap(&self, resource_id: u32, memory: OwnedDescriptor) -> KumquatGpuResult<()> {
        self.send_with(
            KUMQUAT_CONTROL_REMAP,
            &resource_id.to_le_bytes(),
            vec![memory],
        )
    }

    pub fn send_end(&self) -> KumquatGpuResult<()> {
        self.send(KUMQUAT_CONTROL_END, &[])
    }

    fn send(&self, kind: u32, payload: &[u8]) -> KumquatGpuResult<()> {
        self.send_with(kind, payload, Vec::new())
    }

    fn send_with(
        &self,
        kind: u32,
        payload: &[u8],
        descriptors: Vec<OwnedDescriptor>,
    ) -> KumquatGpuResult<()> {
        let mut message = kind.to_le_bytes().to_vec();
        message.extend_from_slice(payload);
        self.tube.send(&message, descriptors)?;
        Ok(())
    }
}

impl AsBorrowedDescriptor for Control {
    fn as_borrowed_descriptor(&self) -> &OwnedDescriptor {
        self.tube.as_borrowed_descriptor()
    }
}
