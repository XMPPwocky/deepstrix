//! Raw perfetto-protobuf emitter for device-time tracks.
//!
//! tracing-perfetto handles host-side spans (when forward_layer calls a
//! kernel-launch wrapper). To see when the **GPU** actually executes a
//! kernel — which is the only way to verify dGPU/iGPU overlap visually
//! — we post-process the EventPool's HIP-timing pairs at token end into
//! perfetto packets on separate tracks (one per device-stream).
//!
//! Wire format reference: the perfetto Trace message is `repeated
//! TracePacket = 1`. Each TracePacket contains a `track_descriptor` (to
//! declare a track) or a `track_event` + timestamp (to draw a slice).
//! The encoder here implements just the fields we need; if more become
//! necessary, drop into the full schema at
//! https://perfetto.dev/docs/reference/trace-packet-proto.

use std::fs::File;
use std::io::Write;
use std::sync::Mutex;

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_hip::{Device, Event, Stream};

pub const TYPE_SLICE_BEGIN: u32 = 1;
pub const TYPE_SLICE_END: u32 = 2;

fn write_varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push(((v & 0x7f) | 0x80) as u8);
        v >>= 7;
    }
    buf.push(v as u8);
}
fn write_field_varint(buf: &mut Vec<u8>, field: u32, v: u64) {
    write_varint(buf, ((field as u64) << 3) | 0);
    write_varint(buf, v);
}
fn write_field_string(buf: &mut Vec<u8>, field: u32, s: &str) {
    write_varint(buf, ((field as u64) << 3) | 2);
    write_varint(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}
fn write_field_message(buf: &mut Vec<u8>, field: u32, body: &[u8]) {
    write_varint(buf, ((field as u64) << 3) | 2);
    write_varint(buf, body.len() as u64);
    buf.extend_from_slice(body);
}

fn encode_track_descriptor(uuid: u64, name: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(name.len() + 16);
    write_field_varint(&mut buf, 1, uuid); // uuid
    write_field_string(&mut buf, 2, name); // name
    buf
}

fn encode_track_event(event_type: u32, name: Option<&str>, track_uuid: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(name.map(|s| s.len() + 16).unwrap_or(16));
    write_field_varint(&mut buf, 9, event_type as u64); // type
    write_field_varint(&mut buf, 11, track_uuid); // track_uuid
    if let Some(s) = name {
        write_field_string(&mut buf, 23, s); // name
    }
    buf
}

fn encode_packet_descriptor(td: &[u8], seq_id: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(td.len() + 16);
    write_field_varint(&mut buf, 10, seq_id as u64); // trusted_packet_sequence_id
    write_field_message(&mut buf, 60, td); // track_descriptor
    buf
}

fn encode_packet_event(timestamp_ns: u64, event: &[u8], seq_id: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(event.len() + 32);
    write_field_varint(&mut buf, 8, timestamp_ns); // timestamp
    write_field_varint(&mut buf, 10, seq_id as u64); // trusted_packet_sequence_id
    write_field_message(&mut buf, 11, event); // track_event
    buf
}

fn encode_trace(packets: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    for p in packets {
        write_field_message(&mut buf, 1, p); // Trace.packet (repeated)
    }
    buf
}

/// HIP-event anchor for one stream: lets us convert a stream-local
/// `elapsed_ms_from_anchor` into a host wall-time `ns` for perfetto.
pub struct Anchor {
    pub event: Event,
    pub host_ns: u64,
}

impl Anchor {
    /// Record a timing event on `stream`, wait for it to actually fire,
    /// then capture the host wall-time. `host_ns` thus reflects when the
    /// anchor event physically completed on the GPU — not when it was
    /// queued. This matters across streams: queue depths differ at
    /// startup, so a queue-time capture leaves each track on its own
    /// uncorrelated wall-clock origin and the perfetto UI shows the
    /// tracks visibly shifted.
    pub fn new(stream: &Stream, device: Device) -> eyre::Result<Self> {
        device.set_current()?;
        let event = Event::new()?;
        event.record(stream)?;
        stream.synchronize()?;
        let host_ns = now_ns();
        Ok(Self { event, host_ns })
    }

    pub fn ns_for(&self, evt: &Event) -> eyre::Result<u64> {
        let ms = Event::elapsed_ms(&self.event, evt)?;
        Ok(self.host_ns + (ms as f64 * 1_000_000.0) as u64)
    }
}

/// Per-stream track on the perfetto timeline.
pub struct Track {
    pub uuid: u64,
    pub anchor: Anchor,
}

/// Device-time perfetto exporter. Writes raw Trace protobuf packets to
/// the file alongside tracing-perfetto's host-time packets (same
/// writer); the parser concatenates them transparently.
pub struct DeviceTimingExporter {
    writer: Mutex<File>,
    seq_id: u32,
    pub dgpu_compute: Track,
    pub dgpu_xfer: Track,
    pub igpu_compute: Track,
    pub igpu_xfer: Track,
}

impl DeviceTimingExporter {
    pub fn open(
        path: impl AsRef<std::path::Path>,
        dgpu: Device,
        dgpu_compute_stream: &Stream,
        dgpu_xfer_stream: &Stream,
        igpu: Device,
        igpu_compute_stream: &Stream,
        igpu_xfer_stream: &Stream,
    ) -> eyre::Result<Self> {
        let path = path.as_ref();
        let _ = std::fs::remove_file(path);
        let file = File::create(path)
            .wrap_err_with(|| format!("create perfetto trace file at {}", path.display()))?;

        // Use a fixed seq_id so all packets we emit share a sequence
        // (perfetto requires this for the descriptor → event linkage).
        let seq_id: u32 = 0x6485f51d;
        let dgpu_compute = Track {
            uuid: 0x44504755_0000_0001,
            anchor: Anchor::new(dgpu_compute_stream, dgpu)?,
        };
        let dgpu_xfer = Track {
            uuid: 0x44504755_0000_0002,
            anchor: Anchor::new(dgpu_xfer_stream, dgpu)?,
        };
        let igpu_compute = Track {
            uuid: 0x49504755_0000_0001,
            anchor: Anchor::new(igpu_compute_stream, igpu)?,
        };
        let igpu_xfer = Track {
            uuid: 0x49504755_0000_0002,
            anchor: Anchor::new(igpu_xfer_stream, igpu)?,
        };

        let writer = Mutex::new(file);
        let mut this = Self {
            writer,
            seq_id,
            dgpu_compute,
            dgpu_xfer,
            igpu_compute,
            igpu_xfer,
        };
        this.declare_track(this.dgpu_compute.uuid, "dgpu.compute (device)")?;
        this.declare_track(this.dgpu_xfer.uuid, "dgpu.xfer (device)")?;
        this.declare_track(this.igpu_compute.uuid, "igpu.compute (device)")?;
        this.declare_track(this.igpu_xfer.uuid, "igpu.xfer (device)")?;
        Ok(this)
    }

    /// Re-record the per-stream anchors and capture fresh host wall-times.
    /// Bounds dGPU/iGPU clock drift in long traces — each call resets the
    /// reference frame, so subsequent events on a track are measured from
    /// the most recent anchor on that same track. Call once per
    /// `forward_token` (after emit_slice for the current token's events
    /// completes) to bound drift to a single token's duration.
    pub fn re_anchor(
        &mut self,
        dgpu: Device,
        dgpu_compute_stream: &Stream,
        dgpu_xfer_stream: &Stream,
        igpu: Device,
        igpu_compute_stream: &Stream,
        igpu_xfer_stream: &Stream,
    ) -> eyre::Result<()> {
        self.dgpu_compute.anchor = Anchor::new(dgpu_compute_stream, dgpu)?;
        self.dgpu_xfer.anchor = Anchor::new(dgpu_xfer_stream, dgpu)?;
        self.igpu_compute.anchor = Anchor::new(igpu_compute_stream, igpu)?;
        self.igpu_xfer.anchor = Anchor::new(igpu_xfer_stream, igpu)?;
        Ok(())
    }

    fn declare_track(&mut self, uuid: u64, name: &str) -> eyre::Result<()> {
        let td = encode_track_descriptor(uuid, name);
        let pkt = encode_packet_descriptor(&td, self.seq_id);
        let trace = encode_trace(&[pkt]);
        self.writer.lock().unwrap().write_all(&trace)?;
        Ok(())
    }

    /// Emit a slice on `track` from `(start_event, end_event)`. The
    /// caller must have already ensured `end_event` has completed
    /// (e.g. via the EventPool harvest's sync on the last event).
    pub fn emit_slice(
        &self,
        track: &Track,
        name: &str,
        start_event: &Event,
        end_event: &Event,
    ) -> eyre::Result<()> {
        let start_ns = track.anchor.ns_for(start_event)?;
        let end_ns = track.anchor.ns_for(end_event)?;
        if end_ns < start_ns {
            return Err(eyre!(
                "perfetto emit_slice: end_ns {} < start_ns {} for {}",
                end_ns, start_ns, name
            ));
        }
        let begin = encode_track_event(TYPE_SLICE_BEGIN, Some(name), track.uuid);
        let end = encode_track_event(TYPE_SLICE_END, None, track.uuid);
        let begin_pkt = encode_packet_event(start_ns, &begin, self.seq_id);
        let end_pkt = encode_packet_event(end_ns, &end, self.seq_id);
        let trace = encode_trace(&[begin_pkt, end_pkt]);
        self.writer.lock().unwrap().write_all(&trace)?;
        Ok(())
    }
}

fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
