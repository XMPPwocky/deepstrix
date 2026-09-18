N_HEAD=64
SWA=128
IMG_W=512
RATIOS=[0,0]+[2]*18+[1]*20
CED_START=20
rows=512
def cap_keys(rows, nkv, ced=True):
    legacy = rows*N_HEAD*3072
    if nkv==0: return legacy
    replay_rows=min((SWA+1)//2, rows)
    need=0
    for l,r in enumerate(RATIOS):
        if r==0: continue
        keys = IMG_W + -(-nkv//r)
        b = replay_rows if (ced and l>=CED_START) else rows
        need=max(need, b*N_HEAD*keys)
    return max(need, legacy)
for T in (0, 8192, 32768, 100000, 131072):
    k=cap_keys(rows,T)
    print(f"ctx={T:>7}  keys={k:>13,}  f16 bytes={k*2/2**20:8.1f} MiB")
# comp_kv (already existing, not my change): 3 ratio-2 stores + 1 ratio-1 store, f16 512-wide
for T in (8192,32768,100000,131072):
    b=(3*((T+1)//2)+T)*512*2
    print(f"ctx={T:>7} comp_kv = {b/2**20:.1f} MiB")
# decode scratch
print("decode attn_scores 82176:", 64*82176*4/2**20, "MiB; 131200:", 64*131200*4/2**20,"MiB")
# dead indexer scratch freed (v41)
print("R1 indexer_scores@82176:", 512*82176*4/2**20, "MiB -> 0; arena falls back to q_len:", 512*32768*4/2**20)
print("attn_active_comp_kv:", 512*512*512*2/2**20, "MiB -> 0")
mc=(82176+4095)//4096; gc=4096//512; ng=(mc+gc-1)//gc
print("indexer_topk_scratch:", 512*((mc+ng)*512)*4/2**20, "MiB -> 0; indexer_q:", 512*32*128*4/2**20)
