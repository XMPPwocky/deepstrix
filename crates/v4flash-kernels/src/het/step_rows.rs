//! The rows of one arena step and what each row depends on inside the step
//! (docs/v41/MS_DSPARK_STREAMS_DESIGN.md section 1).
//!
//! A decode step is a batch of rows. Plain multistream, a DSpark verify block,
//! several streams' blocks in one step and (later) tree speculation differ only
//! in which EARLIER rows of the batch a row reads at attention time:
//!
//!   * plain multistream: none -- a row reads its stream's committed KV only;
//!   * a DSpark chain: the previous row of its stream (the draft before it);
//!   * a tree: its parent node, and through it every ancestor.
//!
//! The router, MoE, mHC, head and sampler are per row and never look at it.
//! What does: the KV tables (which rows a row's attention sees, where its
//! appends go), the lane ordering (a later lane may enter a layer only after
//! the rows it depends on wrote that layer), and the host's accept (which rows'
//! KV a stream keeps). `StepRows` carries the dependency explicitly, one
//! `parent` per row, and those three derive it from here instead of inferring
//! it from repeated slot ids.
//!
//! A row's position is not stored: it is its stream's committed position plus
//! its `depth`. Siblings (two rows with one parent) are expressible -- a tree
//! -- and refused by the kernels, which read each stream's ancestors as a
//! contiguous range of rows (`is_chain_layout`).
use color_eyre::eyre::{self, eyre};

use super::kv_arena::ARENA_ROWS_PER_STREAM;

/// One row of an arena step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepRow {
    /// The arena slot (stream) whose KV this row reads and appends to.
    pub slot: u32,
    /// The EARLIER row of the same slot this row continues (its in-batch
    /// dependency); `None` = a ROOT: the stream's next token at its committed
    /// position, which depends on committed KV only.
    pub parent: Option<u16>,
}

/// A step's rows, validated (`new`): rows are in a topological order (a
/// row's parent precedes it), a parent belongs to the same slot, every slot
/// has exactly one root, and no slot runs more than `ARENA_ROWS_PER_STREAM`
/// rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepRows {
    rows: Vec<StepRow>,
    /// `rows[i].slot`, for the callers that index slots by row.
    slots: Vec<u32>,
    /// 0 for a root, `depth(parent) + 1` otherwise: the row's offset from its
    /// stream's committed position.
    depth: Vec<u16>,
}

impl StepRows {
    /// Validate `rows` (see the type).
    pub fn new(rows: Vec<StepRow>) -> eyre::Result<Self> {
        if rows.len() > u16::MAX as usize {
            return Err(eyre!("step rows: {} rows", rows.len()));
        }
        let mut depth: Vec<u16> = Vec::with_capacity(rows.len());
        // (slot, roots, rows) per slot seen so far: a step has few slots.
        let mut per_slot: Vec<(u32, u32, u32)> = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let d = match r.parent {
                None => 0,
                Some(p) => {
                    let p = p as usize;
                    if p >= i {
                        return Err(eyre!("step rows: row {i}'s parent {p} does not precede it"));
                    }
                    if rows[p].slot != r.slot {
                        return Err(eyre!("step rows: row {i} (slot {}) depends on row {p} of slot {}", r.slot, rows[p].slot));
                    }
                    depth[p] + 1
                }
            };
            depth.push(d);
            match per_slot.iter_mut().find(|e| e.0 == r.slot) {
                Some(e) => {
                    e.1 += u32::from(r.parent.is_none());
                    e.2 += 1;
                }
                None => per_slot.push((r.slot, u32::from(r.parent.is_none()), 1)),
            }
        }
        for &(slot, roots, n) in &per_slot {
            if roots != 1 {
                return Err(eyre!("step rows: slot {slot} has {roots} roots (one: its next token)"));
            }
            if n > ARENA_ROWS_PER_STREAM {
                return Err(eyre!("step rows: slot {slot} runs {n} rows (at most {ARENA_ROWS_PER_STREAM})"));
            }
        }
        let slots = rows.iter().map(|r| r.slot).collect();
        Ok(Self { rows, slots, depth })
    }

    /// One root per slot, no dependencies: a plain multistream step.
    pub fn plain(slots: &[u32]) -> eyre::Result<Self> {
        Self::new(slots.iter().map(|&slot| StepRow { slot, parent: None }).collect())
    }

    /// Per stream a root and then `extra` rows, each continuing the one before
    /// it: DSpark verify blocks (`extra` = K drafts; 0 = a plain row).
    pub fn chains(blocks: &[(u32, usize)]) -> eyre::Result<Self> {
        let mut rows = Vec::with_capacity(blocks.iter().map(|b| 1 + b.1).sum());
        for &(slot, extra) in blocks {
            rows.push(StepRow { slot, parent: None });
            for _ in 0..extra {
                let p = rows.len() - 1;
                rows.push(StepRow { slot, parent: Some(p as u16) });
            }
        }
        Self::new(rows)
    }

    /// A slot list with repeats, read the way the arena used to read one: each
    /// run of one slot is a root and then a chain. A slot that comes back after
    /// another slot's rows is refused (it would be a second root). TESTS ONLY:
    /// the tables tests compare run lists against what they always computed;
    /// nothing else may infer a dependency from slot repetition.
    #[cfg(test)]
    pub(crate) fn chains_from_runs(slots: &[u32]) -> eyre::Result<Self> {
        let rows = slots
            .iter()
            .enumerate()
            .map(|(i, &slot)| StepRow { slot, parent: (i > 0 && slots[i - 1] == slot).then(|| (i - 1) as u16) })
            .collect();
        Self::new(rows)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn rows(&self) -> &[StepRow] {
        &self.rows
    }

    /// Every row's slot, in row order.
    pub fn slots(&self) -> &[u32] {
        &self.slots
    }

    pub fn slot(&self, i: usize) -> u32 {
        self.rows[i].slot
    }

    pub fn parent(&self, i: usize) -> Option<usize> {
        self.rows[i].parent.map(usize::from)
    }

    /// Row `i`'s offset from its stream's committed position.
    pub fn depth(&self, i: usize) -> usize {
        self.depth[i] as usize
    }

    /// Row `i`'s ancestors, nearest first.
    pub fn ancestors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(self.parent(i), move |&p| self.parent(p))
    }

    /// Whether row `i` reads what row `j` writes (`j` is an ancestor of `i`).
    pub fn depends_on(&self, i: usize, j: usize) -> bool {
        self.ancestors(i).any(|a| a == j)
    }

    /// Every non-root row continues the row right before it. Then each
    /// stream's rows are contiguous, a stream's `j`-th row sits at depth `j`,
    /// and its ancestors are exactly the stream's rows before it: what the
    /// arena's tables and kernels compute (contiguous ranges). A tree is not.
    pub fn is_chain_layout(&self) -> bool {
        self.rows.iter().enumerate().all(|(i, r)| r.parent.is_none_or(|p| p as usize + 1 == i))
    }

    /// Some row at index `>= cut` depends on a row before `cut`: a lane cut
    /// there must be ORDERED (the later lane enters each layer only after the
    /// earlier one has written it). Checking parents suffices: a row's nearest
    /// ancestor below the cut is the parent of a row at or above it.
    pub fn crosses(&self, cut: usize) -> bool {
        self.rows.iter().enumerate().skip(cut).any(|(_, r)| r.parent.is_some_and(|p| (p as usize) < cut))
    }

    /// Per slot, in the order of their roots: `(slot, root row, rows of the
    /// slot)`.
    pub fn streams(&self) -> Vec<(u32, usize, usize)> {
        self.rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.parent.is_none())
            .map(|(i, r)| (r.slot, i, self.rows_of(r.slot)))
            .collect()
    }

    /// How many rows `slot` runs in this step.
    pub fn rows_of(&self, slot: u32) -> usize {
        self.slots.iter().filter(|&&s| s == slot).count()
    }

    /// The row index of `slot`'s root.
    pub fn root_of(&self, slot: u32) -> Option<usize> {
        self.rows.iter().position(|r| r.slot == slot && r.parent.is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(slot: u32, parent: Option<u16>) -> StepRow {
        StepRow { slot, parent }
    }

    #[test]
    fn builders_agree_on_todays_layout() {
        let a = StepRows::chains_from_runs(&[4, 4, 4, 1, 7, 7]).unwrap();
        let b = StepRows::chains(&[(4, 2), (1, 0), (7, 1)]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.slots(), &[4, 4, 4, 1, 7, 7]);
        assert_eq!((0..6).map(|i| a.depth(i)).collect::<Vec<_>>(), vec![0, 1, 2, 0, 0, 1]);
        assert!(a.is_chain_layout());
        assert_eq!(a.rows_of(4), 3);
        assert_eq!(a.root_of(7), Some(4));
        let p = StepRows::plain(&[3, 0, 5]).unwrap();
        assert_eq!(p, StepRows::chains_from_runs(&[3, 0, 5]).unwrap());
        assert!(p.is_chain_layout() && (0..=3).all(|c| !p.crosses(c)));
    }

    #[test]
    fn refusals() {
        // A parent must precede its row.
        assert!(StepRows::new(vec![row(0, Some(0))]).is_err());
        assert!(StepRows::new(vec![row(0, None), row(0, Some(2)), row(0, Some(1))]).is_err());
        // A parent belongs to the same slot.
        assert!(StepRows::new(vec![row(0, None), row(1, Some(0))]).is_err());
        // Exactly one root per slot.
        assert!(StepRows::new(vec![row(0, None), row(0, None)]).is_err());
        assert!(StepRows::chains_from_runs(&[2, 3, 2]).is_err());
        // At most ARENA_ROWS_PER_STREAM rows per slot.
        assert!(StepRows::chains(&[(0, ARENA_ROWS_PER_STREAM as usize - 1)]).is_ok());
        assert!(StepRows::chains(&[(0, ARENA_ROWS_PER_STREAM as usize)]).is_err());
    }

    #[test]
    fn a_tree_is_expressible_but_not_a_chain_layout() {
        // slot 0: root r0; r1, r2 children of r0 (siblings); r3 child of r2.
        let t = StepRows::new(vec![row(0, None), row(0, Some(0)), row(0, Some(0)), row(0, Some(2)), row(9, None)]).unwrap();
        assert!(!t.is_chain_layout());
        assert_eq!((0..5).map(|i| t.depth(i)).collect::<Vec<_>>(), vec![0, 1, 1, 2, 0]);
        assert_eq!(t.ancestors(3).collect::<Vec<_>>(), vec![2, 0]);
        assert!(t.depends_on(3, 0) && t.depends_on(3, 2) && !t.depends_on(3, 1) && !t.depends_on(2, 1));
        // A cut between the siblings: r2 depends on r0 across it.
        assert!(t.crosses(2));
        // A cut before the other stream's root: nothing depends across it.
        assert!(!t.crosses(4));
    }

    #[test]
    fn streams_lists_each_slot_by_its_root() {
        let r = StepRows::chains(&[(3, 2), (5, 0), (1, 1)]).unwrap();
        assert_eq!(r.streams(), vec![(3, 0, 3), (5, 3, 1), (1, 4, 2)]);
        assert_eq!(r.root_of(1), Some(4));
        assert_eq!(r.root_of(9), None);
    }

    /// For every layout today's arena could express (runs of slots), `crosses`
    /// is exactly the old rule "the rows either side of the cut share a slot".
    #[test]
    fn crosses_is_the_old_slot_rule_on_chain_layouts() {
        let mut checked = 0;
        for n in 1..=9usize {
            // Run lengths as compositions of n into parts of <= 8, slots 0, 1, 2, ...
            let mut stack: Vec<Vec<usize>> = vec![vec![]];
            while let Some(parts) = stack.pop() {
                let s: usize = parts.iter().sum();
                if s == n {
                    let slots: Vec<u32> = parts.iter().enumerate().flat_map(|(k, &len)| std::iter::repeat_n(k as u32, len)).collect();
                    let r = StepRows::chains_from_runs(&slots).unwrap();
                    for cut in 1..n {
                        assert_eq!(r.crosses(cut), slots[cut - 1] == slots[cut], "slots {slots:?} cut {cut}");
                        checked += 1;
                    }
                    continue;
                }
                for len in 1..=(n - s).min(ARENA_ROWS_PER_STREAM as usize) {
                    let mut p = parts.clone();
                    p.push(len);
                    stack.push(p);
                }
            }
        }
        assert!(checked > 1000, "checked {checked}");
    }
}
