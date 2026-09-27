"""Convert the oracle's gen2 dump (torch .pt zips) to raw little-endian files for
tests/dspark_parity.rs, without torch. Output dir: <dump>/parity/
  mh.bin        f32 [T, 3, 5120]   mean over hc copies of the residual AFTER layers 36/37/38
  tokens.bin    i32 [T]
  argmax.bin    i32 [T]            main greedy token for position j+1
  ref_<name>.bin i32 [steps, 1 + 5 + 5]  per record: i, drafts[5], greedy[5]
  ref_<name>_conf.bin f32 [steps, 5]
"""
import array, json, os, sys, zipfile
dump = sys.argv[1]
out = os.path.join(dump, "parity"); os.makedirs(out, exist_ok=True)
def raw(path, code):
    z = zipfile.ZipFile(path)
    member = [n for n in z.namelist() if n.split('/')[-2:] == ['data', '0']][0]
    a = array.array(code); a.frombytes(z.read(member)); return a
def mh_rows(dumpdir, T, positions=None):
    layers = [raw(os.path.join(dumpdir, f"layer_{l:02d}_residual.pt"), 'f') for l in (36, 37, 38)]
    D, H = 5120, 4
    res = array.array('f')
    for p in (range(T) if positions is None else positions):
        for L in layers:
            base = p * H * D
            res.extend((L[base + d] + L[base + D + d] + L[base + 2 * D + d] + L[base + 3 * D + d]) * 0.25 for d in range(D))
    return res
if len(sys.argv) > 2 and sys.argv[2] == "--check-main200":
    got = mh_rows(dump, 1006, [200])
    want = array.array('f'); want.frombytes(open(os.path.join(dump, "mtp_ref", "main_hidden.bin"), 'rb').read())
    md = max(abs(a - b) for a, b in zip(got, want)); mx = max(abs(b) for b in want)
    print(f"check vs mtp_ref/main_hidden.bin pos 200: n={len(got)}/{len(want)} max|diff|={md:.3e} max|ref|={mx:.3e}")
    sys.exit(0)
ids = json.load(open(os.path.join(dump, "tokens.json"))); T = len(ids)
open(os.path.join(out, "mh.bin"), "wb").write(mh_rows(dump, T).tobytes())
array.array('i', ids).tofile(open(os.path.join(out, "tokens.bin"), "wb"))
am = raw(os.path.join(dump, "main_argmax.pt"), 'i'); assert len(am) == T
open(os.path.join(out, "argmax.bin"), "wb").write(am.tobytes())
for f in sorted(os.listdir(dump)):
    if not (f.startswith("dspark_accept_") and f.endswith(".json")): continue
    d = json.load(open(os.path.join(dump, f)))
    if "records" not in d: continue
    name = f[len("dspark_accept_"):-5]
    r = array.array('i'); c = array.array('f')
    for rec in d["records"]:
        r.append(rec["i"]); r.extend(rec["drafts"]); r.extend(rec["greedy"]); c.extend(rec["conf"])
    open(os.path.join(out, f"ref_{name}.bin"), "wb").write(r.tobytes())
    open(os.path.join(out, f"ref_{name}_conf.bin"), "wb").write(c.tobytes())
    print(f"ref {name}: {len(d['records'])} steps, start {d['records'][0]['i']}, E {d['expected_tokens'][-1]:.3f}, cfg {d.get('experiment')}")
print(f"T={T}; wrote {out}")
