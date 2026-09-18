p='/home/claude-code/deepstrix/crates/v4flash-kernels/src/het/expert_pager.rs'
s=open(p).read()
def rep(old,new,cnt=1):
    global s
    assert s.count(old)==cnt,(s.count(old),old[:90])
    s=s.replace(old,new,cnt)

SPARSE_HIT = """            if let Some(&slot) = self.slot_of.get(&key) {
                self.touch(slot);
                self.remap[id as usize] = -(slot as i32) - 1;
                continue;
            }"""
SPARSE_HIT_NEW = """            if let Some(&slot) = self.slot_of.get(&key) {
                self.touch(slot);
                // ABSOLUTE pool slot, NOT a window-relative one (that is what
                // `write_window_remap` writes into the same field). The group
                // builder's arrays are sized to `sparse_group_bound() ==
                // n_slots` and its `n_expert` argument is a BUFFER LIMIT, not a
                // guard: a slot at or above it is dropped SILENTLY while the
                // reducer still claims its zeroed partial row.
                assert!(
                    slot < self.n_slots,
                    "expert pager: L{layer} e{id} resident at slot {slot} >= pool size {}; the \\
                     sparse remap carries ABSOLUTE slots and the group arrays only cover the pool",
                    self.n_slots
                );
                self.remap[id as usize] = -(slot as i32) - 1;
                continue;
            }"""
assert s.count(SPARSE_HIT)==2
s=s.replace(SPARSE_HIT,SPARSE_HIT_NEW,2)

# ensure_batched miss write
rep("""            self.slot_of.insert((layer, id), slot);
            self.touch(slot);
            self.remap[id as usize] = -(slot as i32) - 1;
        }""",
"""            self.slot_of.insert((layer, id), slot);
            self.touch(slot);
            // ABSOLUTE pool slot; see the resident-hit branch above.
            assert!(
                slot < self.n_slots,
                "expert pager: L{layer} e{id} paged into slot {slot} >= pool size {}",
                self.n_slots
            );
            self.remap[id as usize] = -(slot as i32) - 1;
        }""")

# ensure() miss write
rep("""            self.slot_key[slot as usize] = Some(key);
            self.touch(slot);
            self.remap[id as usize] = -(slot as i32) - 1;
        }""",
"""            self.slot_key[slot as usize] = Some(key);
            self.touch(slot);
            // ABSOLUTE pool slot; see the resident-hit branch above.
            assert!(
                slot < self.n_slots,
                "expert pager: L{layer} e{id} paged into slot {slot} >= pool size {}",
                self.n_slots
            );
            self.remap[id as usize] = -(slot as i32) - 1;
        }""")

# write_window_remap: WINDOW-RELATIVE slot space
rep("""        let base = self.window_base(w);
        for r in self.remap.iter_mut() {
            *r = 0;
        }
        for &(e, sl) in assign {
            self.remap[e as usize] = -((sl - base) as i32) - 1;
        }""",
"""        let base = self.window_base(w);
        let width = self.window_width(w);
        for r in self.remap.iter_mut() {
            *r = 0;
        }
        for &(e, sl) in assign {
            // This field carries a WINDOW-RELATIVE slot here and an ABSOLUTE
            // pool slot in `ensure` -- two slot spaces in one i32. A slot from
            // outside this window would underflow `sl - base` (usize) into a
            // huge id, or alias another expert's slot inside the window.
            assert!(
                sl >= base && sl < base + width,
                "expert pager: window remap for expert {e} got slot {sl}, outside window {w} \\
                 [{base}, {}); this remap is WINDOW-RELATIVE, `ensure`'s is ABSOLUTE",
                base + width
            );
            assert!(
                (e as usize) < self.remap.len(),
                "expert pager: window remap for expert id {e} >= remap length {}",
                self.remap.len()
            );
            self.remap[e as usize] = -((sl - base) as i32) - 1;
        }""")
open(p,'w').write(s)
print("ok")
