# docs/GLOSSARY.md "pin / release" vs "protect": inside box 2, a set of
# experts that must not be evicted (an in-flight request's picks, a parked
# request's picks) is a PROTECT set. "pin" is the hub<->box-2 pin protocol
# only (KB #37 was a protect set called `parked_pins`). The field
# `ExpertShard::pinned` is renamed by hand (the word `pinned` is the
# protocol's everywhere else in remote_experts.rs).
SPECIAL = {
    'ExtraPins': 'ExtraProtect',
    'parked_pins': 'parked_protect',
    'push_parked_pins': 'push_parked_protect',
    'cur_pins': 'cur_protect',
    'nested_park_keeps_the_parked_requests_pins': 'nested_park_keeps_the_parked_requests_protected',
}
EXCLUDE = set()


def rename(w):
    return SPECIAL.get(w)
