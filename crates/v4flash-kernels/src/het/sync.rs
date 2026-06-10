//! Cross-device synchronization helpers.
//!
//! The **load-bearing rule** (from `project_peer_copy_stream_rule.md`):
//! `hipMemcpyPeerAsync` must be queued on the **source device's stream**.
//! Queueing on the destination's stream silently returns zero bytes on
//! Strix/9070 setups. All peer copies here enforce that by taking a
//! `src_stream` argument.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, Stream};

/// Queue an async peer copy from `src` to `dst`. The copy runs on
/// `src_stream` (which must belong to `src`'s device).
///
/// In M13.1 the caller follows this with `src_stream.synchronize()` plus
/// any required state-sync; M13.4 will replace those syncs with HIP
/// events recorded on `src_stream` and waited on by the destination's
/// compute stream.
pub fn peer_push_f32(
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    src_stream: &Stream,
) -> eyre::Result<()> {
    if src.device_id() == dst.device_id() {
        return Err(eyre!(
            "peer_push: same-device copy (src={}, dst={})",
            src.device_id(),
            dst.device_id()
        ));
    }
    if src_stream.device_id() != src.device_id() {
        return Err(eyre!(
            "peer_push: src_stream device {} != src buffer device {}",
            src_stream.device_id(),
            src.device_id()
        ));
    }
    src.copy_to_peer_async(dst, src_stream)
}

/// i32 variant of [`peer_push_f32`]. Same semantics, same rules.
pub fn peer_push_i32(
    src: &DeviceBuffer<i32>,
    dst: &mut DeviceBuffer<i32>,
    src_stream: &Stream,
) -> eyre::Result<()> {
    if src.device_id() == dst.device_id() {
        return Err(eyre!(
            "peer_push: same-device copy (src={}, dst={})",
            src.device_id(),
            dst.device_id()
        ));
    }
    if src_stream.device_id() != src.device_id() {
        return Err(eyre!(
            "peer_push: src_stream device {} != src buffer device {}",
            src_stream.device_id(),
            src.device_id()
        ));
    }
    src.copy_to_peer_async(dst, src_stream)
}

pub fn peer_push_u8(
    src: &DeviceBuffer<u8>,
    dst: &mut DeviceBuffer<u8>,
    src_stream: &Stream,
) -> eyre::Result<()> {
    if src.device_id() == dst.device_id() {
        return Err(eyre!(
            "peer_push: same-device copy (src={}, dst={})",
            src.device_id(),
            dst.device_id()
        ));
    }
    if src_stream.device_id() != src.device_id() {
        return Err(eyre!(
            "peer_push: src_stream device {} != src buffer device {}",
            src_stream.device_id(),
            src.device_id()
        ));
    }
    src.copy_to_peer_async(dst, src_stream)
}
