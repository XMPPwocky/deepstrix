# Review patches applied to the COPIES in review/ (the engineer's files are untouched):
#  1. cand_shared.hip: instantiate the production b=3 B-variant (missing from the engineer's file)
#     plus b=6/7 for the tail.
#  2. harness.cpp: C1_XS_SCALE env to put gate/up in a non-saturating range for the swiglu clamp,
#     and a clamp-saturation census printed with the range line.
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

p = 'cand_shared.hip'
s = open(p).read()
add = """

// ---- REVIEW additions (review/ copy only). Production b per lane is 1..5, but the engineer
// instantiated no tB3 B-variant. Added for the b=3 check, plus 6/7 for the tail.
SHARED_A(shared_gateup_swiglu_tB6, 6)
SHARED_A(shared_gateup_swiglu_tB7, 7)
SHARED_B(shared_gateup_swiglu_q8_tB3_r1, 3, 1)
SHARED_B(shared_gateup_swiglu_q8_tB3_r2, 3, 2)
SHARED_B(shared_gateup_swiglu_q8_tB6_r1, 6, 1)
SHARED_B(shared_gateup_swiglu_q8_tB7_r1, 7, 1)
"""
if 'REVIEW additions' not in s:
    open(p, 'w').write(s + add)
    print('cand_shared.hip patched')

p = 'harness.cpp'
s = open(p).read()
old = "    for (auto& v : xs) v = ((rnd() >> 8) * (1.0f / 16777216.0f)) * 0.1f + 1e-3f;\n}"
new = """    const float xsc = getenv("C1_XS_SCALE") ? (float)atof(getenv("C1_XS_SCALE")) : 1.0f;  // REVIEW
    for (auto& v : xs) v = (((rnd() >> 8) * (1.0f / 16777216.0f)) * 0.1f + 1e-3f) * xsc;
}"""
if 'C1_XS_SCALE' not in s:
    assert s.count(old) == 1
    s = s.replace(old, new)
    old2 = """        float mx = 0; for (float v : r_mid) mx = std::max(mx, fabsf(v));
        printf("shared b=%u: |mid| max %.4g (clamp %.1f)\\n", b, mx, clamp);"""
    new2 = """        float mx = 0; size_t nsat = 0, nzero = 0, nsmall = 0;
        for (float v : r_mid) { mx = std::max(mx, fabsf(v)); nsat += fabsf(v) == 100.0f; nzero += v == 0.0f; nsmall += fabsf(v) > 0 && fabsf(v) < 50.0f; }
        printf("shared b=%u: |mid| max %.4g (clamp %.1f)  REVIEW census: n=%zu |mid|==100: %zu  ==0: %zu  0<|mid|<50: %zu\\n", b, mx, clamp, r_mid.size(), nsat, nzero, nsmall);
        { std::vector<float> gg = kb::d2h(gate_o, nmid); float gm = 0; for (float v : gg) gm = std::max(gm, fabsf(v)); printf("shared b=%u: REVIEW |gate| max %.4g\\n", b, gm); }"""
    assert s.count(old2) == 1
    s = s.replace(old2, new2)
    open(p, 'w').write(s)
    print('harness.cpp patched')
