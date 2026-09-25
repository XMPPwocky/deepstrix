//! `copy_to_pinned_async` on a `new_non_blocking` stream (the router readback
//! batch, `het::batch_scratch::BatchDgpuScratch::rb_stream`): the bytes land at
//! the requested offset once the stream is synchronized, slice views copy only
//! their window, and an out-of-range destination is an error, not a write.

use v4flash_hip::{Device, DeviceBuffer, PinnedBuffer, Stream};

#[test]
fn pinned_readback_on_non_blocking_stream() {
    let Ok(devices) = Device::all() else {
        eprintln!("skip: no HIP devices");
        return;
    };
    for (i, &dev) in devices.iter().enumerate() {
        // The route runs with the pager's device (the iGPU) possibly current:
        // make ANOTHER device current, so the copy must follow its stream.
        let other = devices[(i + 1) % devices.len()];
        other.set_current().expect("set_current other");
        let src: Vec<i32> = (0..64).map(|i| i * 7 - 100).collect();
        let mut d = DeviceBuffer::<i32>::new(dev.id, src.len()).expect("alloc");
        d.copy_from_host(&src).expect("h2d");
        let s = Stream::new_non_blocking(dev.id).expect("non-blocking stream");
        let mut pin = PinnedBuffer::<i32>::new(80).expect("pinned");
        // Whole buffer at offset 3, then a 10-element window at offset 70.
        d.copy_to_pinned_async(&mut pin, 3, &s).expect("d2h whole");
        d.slice_view(20, 10).copy_to_pinned_async(&mut pin, 70, &s).expect("d2h window");
        s.synchronize().expect("sync");
        let h = pin.as_slice();
        assert_eq!(&h[..3], &[0, 0, 0], "device {}: bytes before the offset untouched", dev.id);
        assert_eq!(&h[3..67], &src[..], "device {}: whole copy", dev.id);
        assert_eq!(&h[70..80], &src[20..30], "device {}: window copy", dev.id);
        // 64 elements at offset 17 would end at 81 > 80.
        assert!(d.copy_to_pinned_async(&mut pin, 17, &s).is_err(), "out of range must fail");
        s.synchronize().expect("sync");
    }
}
