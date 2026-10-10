//! Layer-outer prefill schedule (P2a of docs/v41/LM_LAYER_OUTER_PREFILL_DESIGN.md, sec 3.1).
//!
//! PURE LOGIC, no GPU: the lane-unit table of a layer-major window and the software-pipeline op order the driver
//! (`HeterogeneousEngine::prefill_job_lm_pass`) executes. Kept separate so the invariants can be simulated on the CPU.
//!
//! A window is cut by `plan_chunk` into sub-chunks `(start, end, lane_cut)`; each gives up to two LANE UNITS (lane A =
//! rows `[start, start+cut)`, lane B = `[start+cut, end)`; an empty lane B is dropped, so the unit count `U` can be odd).
//! A quantum runs layers `[a, b)` over ALL units as a flat task list `T[t] = (layer, unit)`, `t = (layer-a)*U + unit`.
//! Scratch lane of task `t` is `t % 2` (alternating, whichever sub-chunk it belongs to). The op order is the existing
//! steady state generalised: warm `Pre(0) Pre(1)`; for `t >= 2`: `Post(t-2) Pre(t)`; drain `Post(N-2) Post(N-1)`.
//! At `U = 2` this IS the existing `post_A(L) pre_A(L+1) post_B(L) pre_B(L+1)` loop.

/// One lane unit of a layer-major window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneUnit {
    /// Index of the sub-chunk this unit belongs to.
    pub sub: usize,
    /// 0 = lane A of its sub-chunk, 1 = lane B.
    pub lane: usize,
    /// Row offset inside the WINDOW (the `lm_store` / index-store row of its first row).
    pub row0: usize,
    /// Absolute token index (into the job's tokens) of its first row.
    pub tok0: usize,
    pub rows: usize,
}

/// Lane units of a window: `subs` = `(start, end, lane_cut)` as planned by `lm_plan_window` (token indices, `lane_cut` =
/// rows in lane A), `win_start` = the window's first token index. Empty lane units are dropped.
pub fn units(subs: &[(usize, usize, usize)], win_start: usize) -> Vec<LaneUnit> {
    let mut out = Vec::with_capacity(subs.len() * 2);
    for (k, &(s, e, cut)) in subs.iter().enumerate() {
        let b = e - s;
        debug_assert!(cut <= b, "lane cut {cut} exceeds the sub-chunk of {b} rows");
        if cut > 0 {
            out.push(LaneUnit { sub: k, lane: 0, row0: s - win_start, tok0: s, rows: cut });
        }
        if b - cut > 0 {
            out.push(LaneUnit { sub: k, lane: 1, row0: s - win_start + cut, tok0: s + cut, rows: b - cut });
        }
    }
    out
}

/// One executor op. `t` indexes the flat task list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Pre(usize),
    Post(usize),
}

/// Op order for `n_tasks` tasks (depth-2 software pipeline). `has_post(t)` is false for tasks with no MoE (the
/// `KvSourceOnly` layer 20): they only get `Pre`.
pub fn ops(n_tasks: usize, has_post: impl Fn(usize) -> bool) -> Vec<Op> {
    let mut v = Vec::with_capacity(n_tasks * 2);
    let post = |v: &mut Vec<Op>, t: usize| {
        if has_post(t) {
            v.push(Op::Post(t));
        }
    };
    for t in 0..n_tasks.min(2) {
        v.push(Op::Pre(t));
    }
    for t in 2..n_tasks {
        post(&mut v, t - 2);
        v.push(Op::Pre(t));
    }
    // Drain the (up to) two tasks still in flight.
    for t in n_tasks.saturating_sub(2)..n_tasks {
        post(&mut v, t);
    }
    v
}

/// `(layer, unit)` of task `t` in a quantum over layers starting at `a` with `u_count` units.
#[inline]
pub fn task(t: usize, a: usize, u_count: usize) -> (usize, usize) {
    (a + t / u_count, t % u_count)
}

/// The layer ranges (quanta) of a window pass over encoder layers `0..split` plus the one-layer `KvSourceOnly` quantum
/// at `split`: `quantum` layers each (the last encoder quantum may be shorter).
pub fn quanta(split: usize, quantum: usize) -> Vec<std::ops::Range<usize>> {
    let q = quantum.max(1);
    let mut v = Vec::new();
    let mut a = 0;
    while a < split {
        let b = (a + q).min(split);
        v.push(a..b);
        a = b;
    }
    v.push(split..split + 1);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny deterministic PRNG (the crate has no rand dependency for tests).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next() as usize) % (hi - lo + 1)
        }
    }

    /// Simulate the op list of one quantum and check every invariant of the design (sec 3.1, sec 9).
    fn simulate(a: usize, b: usize, u_count: usize, kv_only_last: bool) {
        let layers = b - a;
        let n = layers * u_count;
        let has_post = |t: usize| !(kv_only_last && task(t, a, u_count).0 == b - 1);
        let v = ops(n, has_post);
        // Pre exactly once per task, Post exactly once per task that has one.
        let mut pre_at: HashMap<usize, usize> = HashMap::new();
        let mut post_at: HashMap<usize, usize> = HashMap::new();
        for (i, op) in v.iter().enumerate() {
            match *op {
                Op::Pre(t) => assert!(pre_at.insert(t, i).is_none(), "Pre({t}) twice"),
                Op::Post(t) => assert!(post_at.insert(t, i).is_none(), "Post({t}) twice"),
            }
        }
        assert_eq!(pre_at.len(), n);
        assert_eq!(post_at.len(), (0..n).filter(|&t| has_post(t)).count());
        // Per-task order and the depth-2 rule: pre(t) after post(t-2); at most 2 tasks pre'd-but-not-post'd at any time.
        let mut in_flight = 0usize;
        let mut max_in_flight = 0usize;
        // residual flow: layers completed per unit (the store version). Task (L,u) needs version L-a (or host seed at a=0).
        let mut done_layers = vec![0usize; u_count];
        let mut last_pre_unit_of_layer: HashMap<usize, usize> = HashMap::new();
        for (i, op) in v.iter().enumerate() {
            match *op {
                Op::Pre(t) => {
                    let (l, u) = task(t, a, u_count);
                    if t >= 2 && has_post(t - 2) {
                        assert!(post_at[&(t - 2)] < i, "Pre({t}) before Post({})", t - 2);
                    }
                    // own rows of the previous layer must be finished (post + store_out) before this pre reads the store
                    assert_eq!(done_layers[u], l - a, "Pre(L{l},u{u}) reads a stale/future residual");
                    // I2: within a layer, units run in increasing order.
                    if let Some(&prev) = last_pre_unit_of_layer.get(&l) {
                        assert_eq!(prev + 1, u, "layer {l}: unit order {prev} -> {u}");
                    } else {
                        assert_eq!(u, 0, "layer {l} must start at unit 0");
                    }
                    last_pre_unit_of_layer.insert(l, u);
                    if !has_post(t) {
                        done_layers[u] += 1; // KvSourceOnly: residual untouched, layer counted as passed
                    } else {
                        in_flight += 1;
                        max_in_flight = max_in_flight.max(in_flight);
                    }
                }
                Op::Post(t) => {
                    let (_, u) = task(t, a, u_count);
                    assert!(pre_at[&t] < i, "Post({t}) before its Pre");
                    in_flight -= 1;
                    done_layers[u] += 1;
                }
            }
        }
        assert_eq!(in_flight, 0, "drain must leave nothing in flight");
        assert!(max_in_flight <= 2, "depth {max_in_flight} > 2");
        for u in 0..u_count {
            assert_eq!(done_layers[u], layers, "unit {u} did not complete all layers");
        }
    }

    #[test]
    fn schedule_is_legal_for_any_unit_count_including_odd() {
        let mut rng = Rng(0xC0FFEE);
        for _ in 0..400 {
            let u_count = rng.range(2, 16); // includes odd U (an empty lane B drops a unit)
            let a = rng.range(0, 18);
            let b = a + rng.range(1, 6);
            simulate(a, b, u_count, false);
        }
        // KvSourceOnly (no post) one-layer quantum, and a KvSourceOnly tail layer of a longer one.
        for u_count in 2..12 {
            simulate(20, 21, u_count, true);
            simulate(18, 21, u_count, true);
        }
    }

    #[test]
    fn u_equals_two_is_the_existing_steady_state() {
        // post_A(L) pre_A(L+1) post_B(L) pre_B(L+1) with tasks t = 2L + lane.
        let v = ops(6, |_| true);
        assert_eq!(
            v,
            vec![
                Op::Pre(0), Op::Pre(1),
                Op::Post(0), Op::Pre(2),
                Op::Post(1), Op::Pre(3),
                Op::Post(2), Op::Pre(4),
                Op::Post(3), Op::Pre(5),
                Op::Post(4), Op::Post(5),
            ]
        );
    }

    #[test]
    fn units_follow_the_plan_cuts_and_drop_empty_lane_b() {
        // two sub-chunks of 1024 (cut 512), one tiny (3 rows, cut 3 => empty B), one of 700 (cut 350).
        let subs = [(100, 1124, 512), (1124, 2148, 512), (2148, 2151, 3), (2151, 2851, 350)];
        let us = units(&subs, 100);
        assert_eq!(us.len(), 7);
        assert_eq!(us[0], LaneUnit { sub: 0, lane: 0, row0: 0, tok0: 100, rows: 512 });
        assert_eq!(us[1], LaneUnit { sub: 0, lane: 1, row0: 512, tok0: 612, rows: 512 });
        assert_eq!(us[4], LaneUnit { sub: 2, lane: 0, row0: 2048, tok0: 2148, rows: 3 });
        assert_eq!(us[5], LaneUnit { sub: 3, lane: 0, row0: 2051, tok0: 2151, rows: 350 });
        // rows tile the window with no gap or overlap, in position order
        let mut next = 0;
        for u in &us {
            assert_eq!(u.row0, next);
            next += u.rows;
        }
        assert_eq!(next, 2851 - 100);
    }

    #[test]
    fn quanta_cover_every_encoder_layer_once_then_the_source_only_layer() {
        for q in 1..=8 {
            let v = quanta(20, q);
            let mut next = 0;
            for r in &v[..v.len() - 1] {
                assert_eq!(r.start, next);
                assert!(r.end - r.start <= q);
                next = r.end;
            }
            assert_eq!(next, 20);
            assert_eq!(*v.last().unwrap(), 20..21);
        }
        assert_eq!(quanta(20, 4).len(), 6);
    }
}
