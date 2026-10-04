# docs/GLOSSARY.md "job / chunk / LM window / layer group / unit / replay":
# under layer-major prefill (V41_LM_PREFILL, production) one call runs one
# UNIT -- a (layer group, sub-chunk) forward -- and returns rows only when a
# window closes; "chunk" named the pre-LM implementation (review S8).
SPECIAL = {'prefill_job_chunk': 'prefill_job_unit'}
EXCLUDE = set()


def rename(w):
    return SPECIAL.get(w)
