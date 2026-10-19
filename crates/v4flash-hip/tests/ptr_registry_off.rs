//! `V41_PTR_REGISTRY=0`: no registration and no checks (its own process, so the STATIC knob is read with the env
//! set). CPU only: no HIP call is made.
use std::panic::Location;

use v4flash_hip::ptr_registry;

#[test]
fn knob_off_disables_every_check() {
    std::env::set_var("V41_PTR_REGISTRY", "0");
    assert!(!ptr_registry::enabled());
    let garbage = 0x4c00_0000_0000_1000usize;
    let here = Location::caller();
    assert!(ptr_registry::check_device("t", garbage, 64, 1, here).is_ok());
    assert!(ptr_registry::check_pinned("t", garbage, 64, here).is_ok());
    assert!(ptr_registry::check_host("t", garbage, 64, here).is_ok());
    assert!(ptr_registry::lookup(garbage).is_none());
    assert_eq!(ptr_registry::live_count(), 0);
}
