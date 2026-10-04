# docs/GLOSSARY.md "raw region / raw window": in the arena a decoder layer's
# raw window slides through slack, it is not a ring (KB #25, #28, #42 were
# raw-window counter bugs filed as "decoder ring" bugs). Snapshot metadata has
# no key with "ring" in it; `decoder_rings_empty` infers from window fields.
SPECIAL = {
    'decoder_rings_empty': 'decoder_windows_empty',
    'clear_decoder_rings_for_checkpoint': 'clear_decoder_windows_for_checkpoint',
}
EXCLUDE = set()


def rename(w):
    return SPECIAL.get(w)
