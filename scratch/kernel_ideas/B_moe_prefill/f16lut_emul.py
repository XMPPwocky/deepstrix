"""CPU emulation of the proposed f16-WMMA MXFP4 dequant (scale folded into the perm LUT).

Checks, for every E8M0 exponent byte e8 in E_LO..E_HI and all 16 nibbles, that the f16 bit pattern built by
  LUT build (2 ops/block) -> perm(sel=nibble&7) -> OR sign<<4 -> 2 spread perms
equals the exact half-precision value kv * 2^(e8-128), kv = {0,1,2,3,4,6,8,12}[n&7] * (+-1)
(kv = 2*e2m1, the production kernels' integer scale; scale = e8m0_half_to_f32_pair = 2^(e8-128)).

usage: python3 -I f16lut_emul.py
"""
import struct

KV = [0, 1, 2, 3, 4, 6, 8, 12]


def perm(src0, src1, sel):
    """v_perm_b32 subset: selector bytes 0..3 pick src1 bytes, 4..7 pick src0 bytes, 0x0C -> 0x00."""
    comb = (src0 << 32) | src1
    out = 0
    for i in range(4):
        c = (sel >> (8 * i)) & 0xFF
        b = 0 if c == 0x0C else (comb >> (8 * c)) & 0xFF
        out |= b << (8 * i)
    return out


def lut_consts():
    """f16 hi bytes of kv * 2^-14 ... the s=-14 base table (all bytes >= 0, idx0 = 0)."""
    hb = []
    for kv in KV:
        if kv == 0:
            hb.append(0)
            continue
        h = struct.unpack('<H', struct.pack('<e', kv * 2.0 ** -14))[0]
        assert h & 0xFF == 0, (kv, hex(h))      # low byte is zero for every e2m1 magnitude
        hb.append(h >> 8)
    lo = hb[0] | hb[1] << 8 | hb[2] << 16 | hb[3] << 24
    hi = hb[4] | hb[5] << 8 | hb[6] << 16 | hb[7] << 24
    return lo, hi


def build_lut(e8, lo0, hi0):
    s = e8 - 128
    assert -14 <= s <= 12 - 3, s                 # kv<=12 -> keep inside f16 normal range
    off = (s + 14) * 4                           # +4 per octave in the hi byte (exp<<2)
    add = (off * 0x01010101) & 0xFFFFFFFF
    lo = ((lo0 + add) & 0xFFFFFFFF) & 0xFFFFFF00  # idx0 (zero) stays zero
    hi = (hi0 + add) & 0xFFFFFFFF
    return lo, hi


def dequant4(x, lo, hi):
    """x: 4 nibbles in the low nibble of each byte. Returns two u32 = f16 pairs (w0,w1), (w2,w3)."""
    s = x & 0x07070707
    hb = perm(hi, lo, s)
    hb |= (x & 0x08080808) << 4
    a = perm(0, hb, 0x010C000C)
    b = perm(0, hb, 0x030C020C)
    return a, b


def main():
    lo0, hi0 = lut_consts()
    print('base LUT words lo=%08x hi=%08x' % (lo0, hi0))
    bad = 0
    checked = 0
    for e8 in range(100, 140):                   # observed 119..123; margin both sides
        try:
            lo, hi = build_lut(e8, lo0, hi0)
        except AssertionError:
            continue
        for n in range(16):
            x = n | (n << 8) | (n << 16) | (n << 24)
            a, b = dequant4(x, lo, hi)
            kv = KV[n & 7] * (-1 if n & 8 else 1)
            want = kv * 2.0 ** (e8 - 128)
            for word in (a, b):
                for half in (word & 0xFFFF, word >> 16):
                    got = struct.unpack('<e', struct.pack('<H', half))[0]
                    checked += 1
                    if got != want and not (got == 0 and want == 0):
                        bad += 1
                        if bad < 10:
                            print('MISMATCH e8=%d n=%d got=%r want=%r' % (e8, n, got, want))
    print('checked %d f16 values over e8=100..139 (valid range), mismatches: %d' % (checked, bad))
    # distinct nibbles per byte lane too
    x = 0x0F37A912 & 0x0F0F0F0F | 0x08040201 & 0x0F0F0F0F
    lo, hi = build_lut(121, lo0, hi0)
    a, b = dequant4(0x0F37A912 & 0x0F0F0F0F, lo, hi)
    vals = [struct.unpack('<e', struct.pack('<H', h))[0] for h in (a & 0xFFFF, a >> 16, b & 0xFFFF, b >> 16)]
    nib = [(0x0F37A912 >> (8 * i)) & 0xF for i in range(4)]
    exp = [(KV[n & 7] * (-1 if n & 8 else 1)) * 2.0 ** (121 - 128) for n in nib]
    print('lane order ok:', vals == exp, vals, exp)
    raise SystemExit(1 if bad else 0)


main()
