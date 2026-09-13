//! Regression test for the `device_id`-is-inert footgun.
//!
//! `hipMalloc` / `hipStreamCreate` bind to the device that is CURRENT on the calling thread.
//! Constructors that merely RECORDED a `device_id` therefore placed objects wherever ambient
//! `hipSetDevice` last pointed, while the recorded id claimed otherwise. That id is load-bearing:
//! the copy paths compare `src.device_id != dst.device_id` to route local vs peer transfers, so a
//! mismatch yields zeros/garbage with no error. It cost a multi-hour misdiagnosis (an expert pool
//! allocated on the dGPU while the iGPU kernel read it).
//!
//! These pin the guard's contract: `device_id` is authoritative, and the caller's ambient device
//! is left untouched.

use v4flash_hip::{Device, DeviceBuffer, Stream};

fn two_devices() -> Option<(Device, Device)> {
    let all = Device::all().ok()?;
    if all.len() < 2 { return None; }
    Some((all[0], all[1]))
}

fn current() -> i32 {
    let mut d = -1;
    // Safety: plain query into a local.
    unsafe { v4flash_hip::sys::hipGetDevice(&mut d) };
    d
}

#[test]
fn buffer_records_requested_device_and_restores_ambient() {
    let Some((a, b)) = two_devices() else {
        eprintln!("skip: needs 2 HIP devices");
        return;
    };
    a.set_current().expect("set_current a");
    assert_eq!(current(), a.id, "precondition: ambient is a");

    // Allocate on b while ambient is a — the case that used to land on a.
    let buf = DeviceBuffer::<f32>::new(b.id, 1024).expect("alloc on b");
    assert_eq!(buf.device_id(), b.id, "buffer must record the device it was asked for");
    assert_eq!(current(), a.id, "constructor must restore the caller's ambient device");

    // And the same-device path must not disturb anything either.
    let buf_a = DeviceBuffer::<f32>::new(a.id, 1024).expect("alloc on a");
    assert_eq!(buf_a.device_id(), a.id);
    assert_eq!(current(), a.id);
}

#[test]
fn stream_records_requested_device_and_restores_ambient() {
    let Some((a, b)) = two_devices() else {
        eprintln!("skip: needs 2 HIP devices");
        return;
    };
    a.set_current().expect("set_current a");
    let s = Stream::new(b.id).expect("stream on b");
    assert_eq!(current(), a.id, "Stream::new must restore the caller's ambient device");
    drop(s);
    let s2 = Stream::new_with_priority(b.id, 0).expect("prio stream on b");
    assert_eq!(current(), a.id, "Stream::new_with_priority must restore ambient");
    drop(s2);
}
