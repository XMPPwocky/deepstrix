//! `V41_PTR_REGISTRY=off`: no registration and no checks, and the mode cannot be turned on live (nothing is
//! registered). Its own process, so the STATIC registration switch is read with the env set. CPU only: no HIP call.
use std::panic::Location;

use v4flash_hip::ptr_registry::{self, Mode};

#[test]
fn knob_off_disables_every_check() {
    std::env::set_var("V41_PTR_REGISTRY", "off");
    assert!(!ptr_registry::enabled());
    assert_eq!(ptr_registry::mode(), Mode::Off);
    let garbage = 0x4c00_0000_0000_1000usize;
    let here = Location::caller();
    assert!(ptr_registry::check_device("t", garbage, 64, 1, here).is_ok());
    assert!(ptr_registry::check_pinned("t", garbage, 64, 1, here).is_ok());
    assert!(ptr_registry::check_host("t", garbage, 64, 1, here).is_ok());
    assert!(ptr_registry::lookup(garbage).is_none());
    assert_eq!(ptr_registry::live_count(), 0);
    assert_eq!(ptr_registry::set_mode(Mode::Enforce), Mode::Off, "registration was off at startup: checks stay off");
    assert!(ptr_registry::check_device("t", garbage, 64, 1, here).is_ok());
    assert_eq!(ptr_registry::misses(), 0);
}
