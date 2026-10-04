# docs/GLOSSARY.md "drafter" (owner's choice, review §10 Q2 option a): the
# engine's drafting role is the drafter. `mtp` stays only where it is an
# interface: the checkpoint's `mtp.N.*` tensor names, the V41_MTP_* env knobs
# and the d_mtp/i_mtp telemetry fields (all string literals, which
# rename_idents.py never edits), and test FILE names that gate scripts run by
# name (tests/mtp_*.rs).
import re

EXCLUDE = {
    'MTP_MODEL_PATH',  # tests/q4_k_matvec.rs: really V4-Flash's MTP GGUF
}

SPECIAL = {
    'MTP_DRAFT_TOP_M': 'DRAFT_TOP_M',
    'MTP': 'DRAFTER',
    'Mtp': 'Drafter',
    'mtp': 'drafter',
}

MOVES = [
    ('crates/v4flash-kernels/src/het/mtp.rs', 'crates/v4flash-kernels/src/het/drafter.rs'),
]


def rename(w):
    if w in SPECIAL:
        return SPECIAL[w]
    if re.fullmatch(r'MTP_[A-Z0-9_]+', w):
        return 'DRAFT_' + w[4:]
    if re.fullmatch(r'Mtp[A-Z]\w*', w):
        return 'Drafter' + w[3:]
    if re.search(r'(?:^|_)mtp(?:_|$)', w) and w == w.lower():
        return re.sub(r'(?:(?<=^)|(?<=_))mtp(?=_|$)', 'drafter', w)
    return None
