//! `V41_PTR_REGISTRY` unset = `warn` (the production default for the first deploy): a miss is reported and the call
//! proceeds; a live flip to `enforce` refuses the same miss. Its own process (the mode is process-wide). CPU only: the
//! checks are exercised directly, no HIP call is made.
use std::panic::Location;

use v4flash_hip::ptr_registry::{self, Mode};

#[test]
fn warn_reports_and_proceeds_then_enforce_refuses() {
    std::env::remove_var("V41_PTR_REGISTRY");
    assert!(ptr_registry::enabled());
    assert_eq!(ptr_registry::mode(), Mode::Warn, "the default");
    let garbage = 0x4c00_0000_0000_1000usize;
    let here = Location::caller();
    let before = ptr_registry::misses();
    assert!(ptr_registry::check_device("t", garbage, 64, 1, here).is_ok(), "warn: the call proceeds");
    assert!(ptr_registry::check_pinned("t", garbage, 64, 1, here).is_ok());
    assert_eq!(ptr_registry::misses(), before + 2, "both misses counted (the report is rate-limited, the count is not)");
    assert!(ptr_registry::device_operand_ok("t", garbage, 64, 1, here).is_err(), "the plain answer is still a miss");
    assert_eq!(ptr_registry::set_mode(Mode::Enforce), Mode::Enforce);
    let e = ptr_registry::check_device("t", garbage, 64, 1, here).unwrap_err().to_string();
    assert!(e.contains("not inside any live allocation"), "{e}");
    assert_eq!(ptr_registry::set_mode(Mode::Off), Mode::Off);
    assert!(ptr_registry::check_device("t", garbage, 64, 1, here).is_ok(), "off: no check");
}
