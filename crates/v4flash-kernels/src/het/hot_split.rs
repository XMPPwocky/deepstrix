//! HOT SPLIT (docs/v41/HOT_SPLIT_DESIGN.md): the pure parts of the
//! interleaved placement -- which box owns each expert of a layer -- and the
//! leg choice for replicated picks. No statics, no device, no config: the
//! hub's `expert_pager::hot_set` feeds them the decayed pick counts and the
//! previous placement, and the route feeds the leg choice one lane-layer's
//! counts. Host-tested here; the design's sections are cited as `D<n>`.

/// Where an expert of a layer lives (D0.3). `B2` = box 2's cold tail (its LRU);
/// `B1` = box 1 owns and computes it; `B2Head` = box 2 computes it and the hub
/// keeps it PINNED there; `Rep` = resident on both, the route picks the leg.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Side {
    B2 = 0,
    B1 = 1,
    B2Head = 2,
    Rep = 3,
}

impl Side {
    pub fn from_u8(v: u8) -> Side {
        match v {
            1 => Side::B1,
            2 => Side::B2Head,
            3 => Side::Rep,
            _ => Side::B2,
        }
    }
    /// Box 2 is the home (`expert_pager::partition_box2`): true for `B2` and
    /// `B2Head`; a `Rep` pick goes to box 2 only by the route's leg choice.
    pub fn box2_home(self) -> bool {
        matches!(self, Side::B2 | Side::B2Head)
    }
    /// Box 1 must hold it resident (owned or replicated).
    pub fn box1_resident(self) -> bool {
        matches!(self, Side::B1 | Side::Rep)
    }
    /// The hub keeps it pinned on box 2 (D4: the KEEP set).
    pub fn keep(self) -> bool {
        matches!(self, Side::B2Head | Side::Rep)
    }
}

/// D2's sizes and targets for one refresh.
#[derive(Clone, Copy, Debug)]
pub struct IlParams {
    /// Box 1's owned + replicated ids per layer (`N1`).
    pub n1: usize,
    /// Replicated ids per layer (`K`, at most `n1`).
    pub k: usize,
    /// Box 2's pinned head share per layer (`P2`).
    pub p2: usize,
    /// Box 1's target share of the layer's non-replicated mass.
    pub target: f64,
    /// Balance tolerance around `target`.
    pub tol: f64,
    /// Moves per layer per refresh: box-1 newcomers (region fills and swaps
    /// both read on box 1) and replicated-set newcomers.
    pub moves: usize,
    /// Rank hysteresis of the region (the hot set's `V41_B1_HOT_HYST`).
    pub hyst: usize,
    /// Rank hysteresis of the replicated set.
    pub rep_hyst: usize,
}

/// One layer's refresh result.
#[derive(Clone, Debug, Default)]
pub struct LayerPlan {
    pub sides: Vec<Side>,
    /// Box 1's share of the non-replicated mass after the refresh.
    pub share1: f64,
    /// Pair swaps made by the balance step.
    pub swaps: usize,
    /// Ids box 1 must now hold that it did not have to before (B1 or Rep, from B2/B2Head).
    pub b1_new: Vec<u32>,
    /// Ids the hub must now keep pinned on box 2 that it did not before (B2Head or Rep, from B2/B1).
    pub b2_new: Vec<u32>,
}

/// Hottest first, ties by id (the sim's `rank_by_counts`).
fn order(counts: &[u32]) -> Vec<usize> {
    let mut v: Vec<usize> = (0..counts.len()).collect();
    v.sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    v
}

/// D2: the sticky interleave for one layer. `prev` is the layer's previous
/// placement (all `B2` / `B1` from today's top-K hot set on a policy switch).
pub fn interleave_layer(counts: &[u32], prev: &[Side], p: &IlParams) -> LayerPlan {
    let ne = counts.len();
    assert_eq!(prev.len(), ne);
    let n1 = p.n1.min(ne);
    let k = p.k.min(n1);
    let ord = order(counts);
    let mut rank = vec![ne; ne];
    for (i, &e) in ord.iter().enumerate() {
        rank[e] = i;
    }
    let mut side = vec![Side::B2; ne];
    let mut moves_left = p.moves;

    // 1. Replicated set: incumbents within the band, strongest first; then
    //    newcomers from the top ranks, capped by the moves; then, if still
    //    short, the strongest departing incumbents (the set stays `k` wide
    //    while it migrates, as the hot set's change cap does).
    let mut rep: Vec<usize> = Vec::with_capacity(k);
    for &e in &ord {
        if rep.len() == k {
            break;
        }
        if prev[e] == Side::Rep && counts[e] > 0 && rank[e] < k + p.rep_hyst {
            rep.push(e);
        }
    }
    // Replicated newcomers have their own cap (`moves`, each one a box-2 read
    // and pin); one box 1 does not hold yet also spends a box-1 read.
    let mut rep_left = p.moves;
    for &e in ord.iter().take(k) {
        if rep.len() == k || counts[e] == 0 {
            break;
        }
        if !rep.contains(&e) && rep_left > 0 && (prev[e].box1_resident() || moves_left > 0) {
            rep.push(e);
            rep_left -= 1;
            if !prev[e].box1_resident() {
                moves_left -= 1;
            }
        }
    }
    if rep.len() < k {
        for &e in &ord {
            if rep.len() == k {
                break;
            }
            if prev[e] == Side::Rep && !rep.contains(&e) {
                rep.push(e);
            }
        }
    }
    for &e in &rep {
        side[e] = Side::Rep;
    }

    // 2./3. The region and its sides. Non-replicated ranks only.
    let cap1 = n1 - rep.len();
    let cap2 = p.p2;
    let region = cap1 + cap2;
    let nonrep: Vec<usize> = ord.iter().copied().filter(|&e| side[e] != Side::Rep).collect();
    let mut nr_rank = vec![ne; ne];
    for (i, &e) in nonrep.iter().enumerate() {
        nr_rank[e] = i;
    }
    let mass: f64 = nonrep.iter().map(|&e| counts[e] as f64).sum::<f64>().max(1.0);
    let (mut n_b1, mut n_b2) = (0usize, 0usize);
    let mut m1 = 0f64;
    // Incumbents hottest first: keep their side while within the band and
    // their side has room; an ex-replicated id goes to box 1 if there is room.
    for &e in &nonrep {
        if counts[e] == 0 || nr_rank[e] >= region + p.hyst {
            continue;
        }
        match prev[e] {
            Side::B1 | Side::Rep if n_b1 < cap1 => {
                side[e] = Side::B1;
                n_b1 += 1;
                m1 += counts[e] as f64;
            }
            Side::B2Head | Side::Rep if n_b2 < cap2 => {
                side[e] = Side::B2Head;
                n_b2 += 1;
            }
            _ => {}
        }
    }
    // Newcomers into the vacancies, hottest first, to the side further below
    // its target. A box-1 newcomer is a read on box 1: it spends a move.
    for &e in &nonrep {
        if n_b1 == cap1 && n_b2 == cap2 {
            break;
        }
        if counts[e] == 0 || nr_rank[e] >= region {
            break;
        }
        if side[e] != Side::B2 {
            continue;
        }
        let want1 = m1 / mass < p.target;
        let b1_ok = n_b1 < cap1 && (prev[e].box1_resident() || moves_left > 0);
        if (want1 || n_b2 == cap2) && b1_ok {
            if !prev[e].box1_resident() {
                moves_left -= 1;
            }
            side[e] = Side::B1;
            n_b1 += 1;
            m1 += counts[e] as f64;
        } else if n_b2 < cap2 {
            side[e] = Side::B2Head;
            n_b2 += 1;
        }
    }

    // 4. Balance: pair swaps that bring box 1's share closest to the target,
    //    only while outside the tolerance and only if the swap improves it.
    let mut swaps = 0usize;
    while moves_left > 0 {
        let dev = m1 / mass - p.target;
        if dev.abs() <= p.tol {
            break;
        }
        let ones: Vec<usize> = (0..ne).filter(|&e| side[e] == Side::B1).collect();
        let twos: Vec<usize> = (0..ne).filter(|&e| side[e] == Side::B2Head).collect();
        // Among the pairs that IMPROVE the deviation: one that lands within
        // the tolerance with the smallest moved mass (the fewest hot ids
        // exposed); else the one that improves most. Ties: lower ids.
        let mut best_in: Option<(f64, usize, usize)> = None;
        let mut best_any: Option<(f64, usize, usize)> = None;
        let better = |cur: Option<(f64, usize, usize)>, key: f64, a: usize, b: usize| match cur {
            None => true,
            Some((k, ba, bb)) => key < k - 1e-12 || ((key - k).abs() <= 1e-12 && (a, b) < (ba, bb)),
        };
        for &a in &ones {
            for &b in &twos {
                let after = (m1 - counts[a] as f64 + counts[b] as f64) / mass - p.target;
                if after.abs() + 1e-12 >= dev.abs() {
                    continue;
                }
                if after.abs() <= p.tol {
                    let moved = counts[a] as f64 + counts[b] as f64;
                    if better(best_in, moved, a, b) {
                        best_in = Some((moved, a, b));
                    }
                } else if better(best_any, after.abs(), a, b) {
                    best_any = Some((after.abs(), a, b));
                }
            }
        }
        let best = best_in.or(best_any);
        let Some((_, a, b)) = best else { break };
        side[a] = Side::B2Head;
        side[b] = Side::B1;
        m1 += counts[b] as f64 - counts[a] as f64;
        swaps += 1;
        moves_left -= 1;
    }

    let mut b1_new = Vec::new();
    let mut b2_new = Vec::new();
    for e in 0..ne {
        if side[e].box1_resident() && !prev[e].box1_resident() {
            b1_new.push(e as u32);
        }
        if side[e].keep() && !prev[e].keep() {
            b2_new.push(e as u32);
        }
    }
    LayerPlan { sides: side, share1: m1 / mass, swaps, b1_new, b2_new }
}

/// D3.1 cost constants (ms unless noted; `HOT_SPLIT_SIM.md` 2, the 10-03 fit).
#[derive(Clone, Copy, Debug)]
pub struct LegCosts {
    pub ig0: f64,
    pub ig_d: f64,
    pub ig_n: f64,
    pub ig_b: f64,
    /// Box-2 service: fixed / per distinct / per row, in us.
    pub s0_us: f64,
    pub s_d_us: f64,
    pub s_b_us: f64,
    /// Link (rtt - service) by lane rows 1..=6, us; larger rows use the last.
    pub link_us: [f64; 6],
    /// Host cost of OPENING a box-2 leg (submit + post + poll, ms), charged
    /// only when the lane-layer has no other box-2 pick. Not in the sim's
    /// greedy (its DES charged it after the fact): without it a replicated pick
    /// goes to box 2 whenever box 1 has any other work, even at one row.
    pub open_ms: f64,
}

impl Default for LegCosts {
    fn default() -> Self {
        LegCosts {
            ig0: 0.100,
            ig_d: 0.0774,
            ig_n: 0.0046,
            ig_b: 0.0397,
            s0_us: 116.0,
            s_d_us: 99.9,
            s_b_us: 15.6,
            link_us: [58.0, 87.0, 118.0, 276.0, 295.0, 318.0],
            open_ms: 0.2,
        }
    }
}

impl LegCosts {
    fn ig(&self, b: usize, n: usize, d: usize) -> f64 {
        self.ig0 + if d > 0 { self.ig_d * d as f64 + self.ig_n * n as f64 } else { 0.0 } + self.ig_b * b as f64
    }
    fn b2(&self, b: usize, d: usize) -> f64 {
        if d == 0 {
            return 0.0;
        }
        let link = self.link_us[b.clamp(1, 6) - 1];
        (link + self.s0_us + self.s_d_us * d as f64 + self.s_b_us * b as f64) / 1000.0
    }
    /// Box 2's leg with `d` distinct, `d0` of them already there before the
    /// replicated picks (the open cost applies only to a leg we open).
    fn b2_leg(&self, b: usize, d: usize, d0: usize) -> f64 {
        if d == 0 {
            return 0.0;
        }
        self.b2(b, d) + if d0 == 0 { self.open_ms } else { 0.0 }
    }
}

/// D3.1: assign one lane-layer's replicated distinct ids to the leg that
/// finishes first (the sim's `route_replicated`). `rep` = (expert, picks in
/// this lane-layer); `n1/d1` and `n2/d2` = the box-1 / box-2 picks and
/// distinct ids already assigned. Returns, per entry of `rep` in its given
/// order, true = box 2. Pure: the same picks always give the same legs.
pub fn choose_legs(c: &LegCosts, b: usize, rep: &[(u32, u32)], mut n1: usize, mut d1: usize, mut n2: usize, mut d2: usize) -> Vec<bool> {
    let mut idx: Vec<usize> = (0..rep.len()).collect();
    // Hottest first, ties by id.
    idx.sort_unstable_by(|&x, &y| rep[y].1.cmp(&rep[x].1).then(rep[x].0.cmp(&rep[y].0)));
    let mut out = vec![false; rep.len()];
    let d2_0 = d2;
    for i in idx {
        let kp = rep[i].1 as usize;
        let t1 = c.ig(b, n1 + kp, d1 + 1).max(c.b2_leg(b, d2, d2_0));
        let t2 = c.ig(b, n1, d1).max(c.b2_leg(b, d2 + 1, d2_0));
        if t1 <= t2 {
            n1 += kp;
            d1 += 1;
        } else {
            n2 += kp;
            d2 += 1;
            out[i] = true;
        }
    }
    let _ = n2;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zipf-like counts over `ne` ids: id i gets ~ 10000 / (i + 1).
    fn zipf(ne: usize) -> Vec<u32> {
        (0..ne).map(|i| (100_000 / (i + 1)) as u32).collect()
    }

    fn params() -> IlParams {
        IlParams { n1: 103, k: 10, p2: 60, target: 0.60, tol: 0.02, moves: 3, hyst: 100, rep_hyst: 5 }
    }

    fn top_k(counts: &[u32], k: usize) -> Vec<Side> {
        let ord = order(counts);
        let mut s = vec![Side::B2; counts.len()];
        for &e in ord.iter().take(k) {
            s[e] = Side::B1;
        }
        s
    }

    fn count(s: &[Side], x: Side) -> usize {
        s.iter().filter(|&&v| v == x).count()
    }

    /// Sizes never exceed the caps; the region holds the top ranks.
    #[test]
    fn sizes_hold() {
        let c = zipf(384);
        let mut prev = top_k(&c, 103);
        for _ in 0..30 {
            let p = interleave_layer(&c, &prev, &params());
            let r = count(&p.sides, Side::Rep);
            assert!(r <= 10);
            assert!(count(&p.sides, Side::B1) + r <= 103);
            assert!(count(&p.sides, Side::B2Head) <= 60);
            prev = p.sides;
        }
        assert_eq!(count(&prev, Side::Rep), 10);
        assert_eq!(count(&prev, Side::B1), 93);
        assert_eq!(count(&prev, Side::B2Head), 60);
        // The replicated set is the top 10.
        assert!((0..10).all(|e| prev[e] == Side::Rep));
    }

    /// From today's top-103, the share converges to the target within the
    /// tolerance in a bounded number of refreshes, and then stops moving.
    #[test]
    fn converges_then_stops() {
        let c = zipf(384);
        let p0 = params();
        let mut prev = top_k(&c, 103);
        let first = interleave_layer(&c, &prev, &p0);
        assert!(first.swaps > 0, "today's top-103 is box-1 heavy: the first refresh swaps");
        // Iterate to the fixed point (the replicated set fills at `moves` per
        // refresh, the balance swaps at `moves` per refresh).
        let mut plan = first;
        let mut refreshes = 1;
        loop {
            let next = interleave_layer(&c, &plan.sides, &p0);
            if next.sides == plan.sides {
                break;
            }
            plan = next;
            refreshes += 1;
            assert!(refreshes < 20, "no fixed point: share {}", plan.share1);
        }
        assert!((plan.share1 - p0.target).abs() <= p0.tol, "share {}", plan.share1);
        assert_eq!(count(&plan.sides, Side::Rep), 10);
        // Stable: same counts -> no further change.
        let again = interleave_layer(&c, &plan.sides, &p0);
        assert_eq!(again.swaps, 0);
        assert!(again.b1_new.is_empty() && again.b2_new.is_empty());
    }

    /// Each refresh reads at most `moves` new ids on box 1 per layer.
    #[test]
    fn box1_reads_capped() {
        let c = zipf(384);
        let mut prev = top_k(&c, 103);
        for _ in 0..20 {
            let p = interleave_layer(&c, &prev, &params());
            assert!(p.b1_new.len() <= 3, "{} box-1 newcomers", p.b1_new.len());
            prev = p.sides;
        }
    }

    /// Small count jitter inside the tolerance changes nothing (no flapping).
    #[test]
    fn jitter_is_ignored() {
        let c = zipf(384);
        let mut prev = top_k(&c, 103);
        for _ in 0..20 {
            prev = interleave_layer(&c, &prev, &params()).sides;
        }
        let mut c2 = c.clone();
        // Swap the counts of adjacent ranks across the head: a strict
        // alternation would flip the owners of every pair.
        for i in (10..80).step_by(2) {
            c2.swap(i, i + 1);
        }
        let p = interleave_layer(&c2, &prev, &params());
        assert!(p.swaps <= 1, "{} swaps on jitter", p.swaps);
        let changed = (0..384).filter(|&e| p.sides[e] != prev[e]).count();
        assert!(changed <= 2, "{changed} ids changed on jitter");
    }

    /// A new hot id enters: it takes a vacancy only by displacing the weakest
    /// incumbent out of the band; dead ids (count 0) leave.
    #[test]
    fn dead_ids_leave_and_new_hot_ids_enter() {
        let c = zipf(384);
        let mut prev = top_k(&c, 103);
        for _ in 0..20 {
            prev = interleave_layer(&c, &prev, &params()).sides;
        }
        let mut c2 = c.clone();
        // Id 50 dies; id 300 becomes the 15th hottest.
        c2[50] = 0;
        c2[300] = c[14];
        let p = interleave_layer(&c2, &prev, &params());
        assert_eq!(p.sides[50], Side::B2, "a dead id leaves");
        assert_ne!(p.sides[300], Side::B2, "a hot newcomer gets a side");
    }

    /// The way back: `top` again from an interleave is just today's hot set,
    /// so only the interleave direction is tested for bounded churn here;
    /// the placement never holds more than `n1` on box 1 at any point.
    #[test]
    fn n1_shrinks_safely() {
        let c = zipf(384);
        let mut prev = top_k(&c, 103);
        for _ in 0..20 {
            prev = interleave_layer(&c, &prev, &params()).sides;
        }
        let mut p = params();
        p.n1 = 80;
        let plan = interleave_layer(&c, &prev, &p);
        assert!(count(&plan.sides, Side::B1) + count(&plan.sides, Side::Rep) <= 80);
    }

    /// The leg choice: alone, replicated picks stay on box 1 (box 2 would be a
    /// whole round trip); behind a busy iGPU with box 2 already in play, they go
    /// to box 2; the result does not depend on the input order.
    #[test]
    fn legs() {
        let c = LegCosts::default();
        // One row, little else on box 1, nothing on box 2: opening a leg costs
        // more than the iGPU time, so it stays local.
        assert_eq!(choose_legs(&c, 1, &[(1, 1)], 1, 1, 0, 0), vec![false]);
        // Box 1 with nothing else: local.
        assert_eq!(choose_legs(&c, 2, &[(1, 2), (2, 1)], 0, 0, 0, 0), vec![false, false]);
        // Box 2 already serving 2 distinct, box 1 busy with 14: the hottest moves.
        let v = choose_legs(&c, 4, &[(1, 3), (2, 2)], 20, 14, 4, 2);
        assert!(v.iter().any(|&x| x), "{v:?}");
        // Order independence.
        let w = choose_legs(&c, 4, &[(2, 2), (1, 3)], 20, 14, 4, 2);
        assert_eq!(v[0], w[1]);
        assert_eq!(v[1], w[0]);
    }

    #[test]
    fn side_predicates() {
        assert!(Side::B2.box2_home() && Side::B2Head.box2_home());
        assert!(!Side::B1.box2_home() && !Side::Rep.box2_home());
        assert!(Side::Rep.keep() && Side::B2Head.keep() && !Side::B1.keep() && !Side::B2.keep());
        for s in [Side::B2, Side::B1, Side::B2Head, Side::Rep] {
            assert_eq!(Side::from_u8(s as u8), s);
        }
    }
}
