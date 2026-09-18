import numpy as np, gguf, gguf.quants as q
np.random.seed(0)
# 1) a 2-row x 64-col matrix of E2M1-representable values with a per-32 power-of-2 scale
E2M1=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)
rows,cols=2,64; nb=cols//32
codes=np.random.randint(0,16,size=(rows,cols)).astype(np.uint8)      # e2m1 code incl sign bit 3
expo=np.random.randint(120,130,size=(rows,nb)).astype(np.uint8)      # e8m0 exponents (2^(e-127))
val=np.where(codes&8, -E2M1[codes&7], E2M1[codes&7]).astype(np.float32)
scale=np.exp2(expo.astype(np.float32)-127.0)
f=(val.reshape(rows,nb,32)*scale[:,:,None]).reshape(rows,cols)
# 2) HF layout: packed [rows, cols/2] with element 2i in the LOW nibble; scale [rows, nb] e8m0 bytes
hf_packed=(codes[:,0::2] | (codes[:,1::2]<<4)).astype(np.uint8)
hf_scale=expo
# 3) my repack -> llama.cpp block_mxfp4: per 32-block: [e8m0 byte][16 bytes: elems 0..15 LOW nibbles, 16..31 HIGH]
def repack(packed,scale):
    r_,h=packed.shape; c=h*2; nb=c//32
    lo=packed&0x0F; hi=packed>>4
    el=np.empty((r_,c),dtype=np.uint8); el[:,0::2]=lo; el[:,1::2]=hi    # unpack to element order
    blk=el.reshape(r_,nb,32)
    qs=(blk[:,:,:16] | (blk[:,:,16:]<<4)).astype(np.uint8)              # llama.cpp nibble order
    out=np.concatenate([scale.reshape(r_,nb,1), qs],axis=2)             # 17 B per block
    return out.reshape(r_,nb*17)
mine=repack(hf_packed,hf_scale)
# 4) reference: gguf-py quantising the floats (its scale choice may legitimately differ; compare DEQUANT values, and bytes where they agree)
ref=q.quantize(f, gguf.GGMLQuantizationType.MXFP4)
back_mine=q.dequantize(mine.reshape(rows,-1), gguf.GGMLQuantizationType.MXFP4)
back_ref =q.dequantize(ref, gguf.GGMLQuantizationType.MXFP4)
print("  my repack dequantises to the HF values exactly:", np.array_equal(back_mine.reshape(rows,cols), f))
print("  gguf-py quantize dequantises to the same values:", np.array_equal(back_ref.reshape(rows,cols), f))
print("  bytes identical to gguf-py's own packing:", np.array_equal(mine, ref.reshape(rows,-1)), "(may differ only by scale choice on ambiguous blocks)")
print("  block bytes:", mine.shape[1]//nb, "(expect 17)")
