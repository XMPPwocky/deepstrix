#!/usr/bin/env python3
"""LIVE placement policies with explicit residency on both boxes, replayed
refresh by refresh (docs/v41/HOT_SPLIT_DESIGN.md rev 3, slice 1 = a44685b).

`LiveSplit(mode='top')`      today's hot set (`hot_set::refresh_top`: top-103,
                             hysteresis 100, <= 3 newcomers/layer); newly owned
                             ids are read on demand on box 1 (b1_page_misses).
`LiveSplit(mode='interleave')` the design: `hot_split.interleave_layer` per
                             layer at every refresh, from the same decayed
                             counts, starting from today's incumbents (the
                             hot-set snapshot), with
  * MOVING ids (home box changed within the last 4 refreshes, cleared when the
    destination holds it) and the holder fallback `serve_on_box2`;
  * box-1 pre-warm of the refresh's box-1 newcomers, `prewarm_step` per step,
    resident from the next step boundary;
  * KEEP (B2H + REP) pinned on box 2: a KEEP word pins a resident id at the
    next step without a read; a non-resident one is read in the background,
    `keep_reads_step` per step, then pinned; a decode pick of a KEEP id that
    box 2 serves pins it;
  * REP picks: `hot_split.choose_legs` (when both boxes hold the id).

Box 1: per layer an LRU of `b1_slots` (production decode LRU 5,041 / 40 = 126)
in which the owned (B1/REP) ids are never evicted; de-owned ids stay as spares
until evicted. Box 2: per layer `b2_slots` (119) = pinned (KEEP held + a pin-
ledger stand-in) + an LRU of the rest. The ledger stand-in, both modes alike:
at each refresh the hottest box-2-home non-KEEP ids that box 2 currently
HOLDS are pinned, up to `pin_total` pins per layer in all (production
~b2_pinned / 40).

The replay calls begin_step / classify / observe (replay.run's live path).
"""
from collections import OrderedDict

import hot_split as hsp
import simlib

NL, NE = simlib.N_LAYER, simlib.N_EXPERT
MOVING_REFRESHES = 4


class LiveSplit:
    def __init__(self, mode='top', refresh_times=None, snapshot=None, refresh_start=0, il=None, b1_slots=126,
                 b2_slots=119, pin_total=105, holder=True, prewarm_step=2, keep_reads_step=30, legs=None,
                 per_layer=103, hyst=100, max_change=3, name=None, record_refresh=True):
        self.mode = mode
        self.name = name or mode
        self.hs = simlib.HotSet(per_layer=per_layer, hyst=hyst, max_change=max_change)
        if snapshot:
            self.hs.load(snapshot)
        self.refresh_times = refresh_times or []
        self.ri = refresh_start
        self.il = il or hsp.IlParams()
        self.b1_slots, self.b2_slots, self.pin_total = b1_slots, b2_slots, pin_total
        self.holder, self.prewarm_step, self.keep_reads_step = holder, prewarm_step, keep_reads_step
        self.legs = legs  # hot_split.LegCosts or None (REP -> leg choice needs it)
        self.side = [[hsp.B2] * NE for _ in range(NL)]
        self.il_active = False
        self.epoch = 0
        self.moved_at = [[0] * NE for _ in range(NL)]
        # box 1: per layer LRU (OrderedDict id -> True), owned ids never evicted
        self.b1 = [OrderedDict() for _ in range(NL)]
        # box 2: pinned ids per layer, split by reason; LRU of the rest
        self.pin_keep = [set() for _ in range(NL)]
        self.pin_led = [set() for _ in range(NL)]
        self.b2 = [OrderedDict() for _ in range(NL)]
        self.prewarm_q = []   # FIFO of (layer, e)
        self.prewarm_land = []  # (layer, e) admitted at the next step boundary
        self.keep_q = []      # FIFO of (layer, e) KEEP words for ids box 2 does not hold
        self.keep_land = []   # resident KEEP ids pinned at the next step
        self._keep_q_set = set()
        self.si = 0
        self.refreshes = 0
        self.log = [] if record_refresh else None
        self.step_ctr = {'holder_b1': 0, 'holder_b2': 0, 'prewarm_admits': 0, 'keep_reads': 0, 'b1_demand': 0,
                         'b2_miss_raw': 0}
        self.cur = None  # per-step counters
        if self.hs.warm:
            self._init_from_hotset()

    # ------------------------------------------------------------ setup
    def _init_from_hotset(self):
        """Warm start from the hot-set snapshot: today's owned ids resident on
        box 1, box 2 holding its hottest home ids (pinned up to pin_total, the
        rest of its slots filled by the next ranks)."""
        for l in range(NL):
            own = self.hs.own[l]
            self.side[l] = [hsp.B1 if own[e] else hsp.B2 for e in range(NE)]
            cnt = self.hs.counts[l]
            ordr = sorted(range(NE), key=lambda e: (-cnt[e], e))
            b1 = self.b1[l]
            for e in ordr:
                if own[e]:
                    b1[e] = True
            # spares: the next-ranked non-owned ids fill box 1's spare slots
            for e in ordr:
                if len(b1) >= self.b1_slots:
                    break
                if not own[e]:
                    b1[e] = True
            b2ids = [e for e in ordr if not own[e]][:self.b2_slots]
            self.pin_led[l] = set(b2ids[:self.pin_total])
            lru = self.b2[l]
            for e in reversed(b2ids[self.pin_total:]):
                lru[e] = True
            lru_items = list(lru.keys())
            lru.clear()
            for e in reversed(lru_items):
                lru[e] = True

    # ------------------------------------------------------------ residency helpers
    def b1_has(self, l, e):
        return e in self.b1[l]

    def b2_pinned(self, l, e):
        return e in self.pin_keep[l] or e in self.pin_led[l]

    def b2_holds(self, l, e):
        return self.b2_pinned(l, e) or e in self.b2[l]

    def _b2_lru_cap(self, l):
        return self.b2_slots - len(self.pin_keep[l]) - len(self.pin_led[l])

    def _b2_trim(self, l):
        lru = self.b2[l]
        cap = max(0, self._b2_lru_cap(l))
        while len(lru) > cap:
            lru.popitem(last=False)

    def _b1_admit(self, l, e):
        b1 = self.b1[l]
        if e in b1:
            b1.move_to_end(e)
            return
        b1[e] = True
        if len(b1) > self.b1_slots:
            side = self.side[l]
            for x in b1:
                if not hsp.box1_resident(side[x]) and x != e:
                    del b1[x]
                    break

    def _b2_pin_keep(self, l, e):
        """Pin a KEEP id box 2 holds (or just read)."""
        self.b2[l].pop(e, None)
        self.pin_led[l].discard(e)
        self.pin_keep[l].add(e)
        self._b2_trim(l)

    def moving(self, l, e):
        at = self.moved_at[l][e]
        return at != 0 and self.epoch - at < MOVING_REFRESHES

    # ------------------------------------------------------------ refresh
    def _refresh(self):
        hs = self.hs
        if hs.total < hs.min_picks:
            return
        warm = hs.warm
        if self.mode == 'top':
            before = [list(o) for o in hs.own] if warm else None
            hs.refresh()  # halves the counts
            self.refreshes += 1
            for l in range(NL):
                own = hs.own[l]
                self.side[l] = [hsp.B1 if own[e] else hsp.B2 for e in range(NE)]
            changed = sum(1 for l in range(NL) for e in range(NE) if before and before[l][e] != hs.own[l][e])
            if self.log is not None:
                self.log.append({'si': self.si, 'changed': changed})
            self._ledger()
            return
        # interleave
        first = not self.il_active
        self.epoch += 1
        tot = {'share': 0.0, 'swaps': 0, 'b1_new': 0, 'b2_new': 0, 'keep': 0, 'within': 0, 'changed': 0}
        new_pw = []
        for l in range(NL):
            cnt = hs.counts[l]
            if not warm:
                prev = [hsp.B2] * NE
            elif first:
                prev = [hsp.B1 if hs.own[l][e] else hsp.B2 for e in range(NE)]
            else:
                prev = self.side[l]
            plan = hsp.interleave_layer(cnt, prev, self.il)
            sides = plan.sides
            for e in range(NE):
                s = sides[e]
                if warm and hsp.box2_home(s) != hsp.box2_home(prev[e]):
                    self.moved_at[l][e] = self.epoch
                if s != prev[e]:
                    tot['changed'] += 1
                # KEEP bookkeeping: an id leaving KEEP is released into box 2's LRU
                if hsp.keep(prev[e]) and not hsp.keep(s) and e in self.pin_keep[l]:
                    self.pin_keep[l].discard(e)
                    self.b2[l][e] = True
                hs.own[l][e] = hsp.box1_resident(s)
                cnt[e] //= 2
            self.side[l] = sides
            tot['share'] += plan.share1
            tot['swaps'] += plan.swaps
            tot['b1_new'] += len(plan.b1_new)
            tot['b2_new'] += len(plan.b2_new)
            tot['within'] += abs(plan.share1 - self.il.target) <= self.il.tol
            if warm:
                new_pw.extend((l, e) for e in plan.b1_new)
            ks = [e for e in range(NE) if hsp.keep(sides[e])]
            tot['keep'] += len(ks)
            # KEEP fill: resident ids pin at the next step, the others queue a read
            for e in ks:
                if e in self.pin_keep[l]:
                    continue
                if e in self.pin_led[l] or e in self.b2[l]:
                    self.keep_land.append((l, e))
                elif (l, e) not in self._keep_q_set:
                    self.keep_q.append((l, e))
                    self._keep_q_set.add((l, e))
            self._b2_trim(l)
        # pre-warm queue: drop entries box 1 no longer needs, append the new ones
        self.prewarm_q = [(l, e) for (l, e) in self.prewarm_q if hsp.box1_resident(self.side[l][e])] + new_pw
        hs.total //= 2
        hs.warm = True
        self.il_active = True
        self.refreshes += 1
        if self.log is not None:
            tot['share'] /= NL
            tot['si'] = self.si
            tot['prewarm_queued'] = len(self.prewarm_q)
            tot['keep_queued'] = len(self.keep_q)
            self.log.append(tot)
        self._ledger()

    def prefill_touch(self, l, e):
        """A prefill chunk used e (the optional pollution model): box 2's LRU
        admits its box-2-home ids."""
        if hsp.box2_home(self.side[l][e]) and not self.b2_pinned(l, e):
            lru = self.b2[l]
            lru[e] = True
            lru.move_to_end(e)
            self._b2_trim(l)

    def _ledger(self):
        """Pin-ledger stand-in: the hottest box-2-home non-KEEP ids box 2 holds,
        up to pin_total pins per layer (KEEP pins count)."""
        for l in range(NL):
            cnt = self.hs.counts[l]
            side = self.side[l]
            room = max(0, self.pin_total - len(self.pin_keep[l]))
            cand = [e for e in range(NE) if hsp.box2_home(side[e]) and not hsp.keep(side[e])
                    and (e in self.b2[l] or e in self.pin_led[l])]
            cand.sort(key=lambda e: (-cnt[e], e))
            new = set(cand[:room])
            lru = self.b2[l]
            for e in self.pin_led[l] - new:
                lru[e] = True  # released: back to the LRU (most recent end)
            for e in new - self.pin_led[l]:
                lru.pop(e, None)
            self.pin_led[l] = new
            self._b2_trim(l)

    # ------------------------------------------------------------ step hooks
    def begin_step(self, si, t_unix):
        self.si = si
        # admissions queued last step land at this step boundary
        cur = {'prewarm_admits': 0, 'keep_pins': 0, 'keep_reads': 0, 'holder_b1': 0, 'holder_b2': 0, 'refresh': 0}
        for (l, e) in self.prewarm_land:
            if hsp.box1_resident(self.side[l][e]):
                self._b1_admit(l, e)
                self.moved_at[l][e] = 0
                cur['prewarm_admits'] += 1
        self.prewarm_land = []
        for (l, e) in self.keep_land:
            if hsp.keep(self.side[l][e]):
                if not self.b2_holds(l, e):
                    cur['keep_reads'] += 1
                self._b2_pin_keep(l, e)
                if hsp.box2_home(self.side[l][e]):
                    self.moved_at[l][e] = 0
                cur['keep_pins'] += 1
        self.keep_land = []
        while self.ri < len(self.refresh_times) and self.refresh_times[self.ri] <= t_unix:
            self._refresh()
            cur['refresh'] += 1
            self.ri += 1
        if self.mode == 'interleave':
            # queue this step's pre-warm reads (land at the next boundary)
            k = 0
            while self.prewarm_q and k < self.prewarm_step:
                l, e = self.prewarm_q.pop(0)
                if e in self.b1[l] or not hsp.box1_resident(self.side[l][e]):
                    if e in self.b1[l]:
                        self.moved_at[l][e] = 0
                    continue
                self.prewarm_land.append((l, e))
                k += 1
            # KEEP background reads (non-resident ids), keep_reads_step per step
            k = 0
            while self.keep_q and k < self.keep_reads_step:
                l, e = self.keep_q.pop(0)
                self._keep_q_set.discard((l, e))
                if not hsp.keep(self.side[l][e]) or e in self.pin_keep[l]:
                    continue
                self.keep_land.append((l, e))
                k += 1
        self.cur = cur

    def observe(self, l, e):
        self.hs.note(l, e)

    def classify(self, si, l, rows, subst, b, keep_miss):
        """One lane-layer: returns (n1, d1, n2, d2, m1, m2, n_rep). keep_miss(si, l, e)
        -> whether a pool-model box-2 miss is a demand read (calibrated thinning)."""
        cnt = {}
        for row in rows:
            for e in row:
                if 0 <= e < NE:
                    cnt[e] = cnt.get(e, 0) + 1
        side = self.side[l]
        n1 = d1 = n2 = d2 = m1 = m2 = 0
        rep = []
        il = self.mode == 'interleave'
        cur = self.cur
        for e, k in cnt.items():
            s = side[e]
            if s == hsp.REP:
                on1 = e in self.b1[l]
                on2 = self.b2_pinned(l, e)
                if on1 and on2 and self.legs is not None:
                    rep.append((e, k))
                    continue
                to2 = on2 and not on1
            else:
                home2 = hsp.box2_home(s)
                to2 = home2
                if il and self.holder and self.moving(l, e):
                    b1h = e in self.b1[l]
                    b2h = self.b2_pinned(l, e)
                    if not home2 and not b1h and b2h:
                        to2 = True
                        cur['holder_b2'] += 1
                    elif home2 and not b2h and b1h:
                        to2 = False
                        cur['holder_b1'] += 1
            if to2:
                n2 += k
                d2 += 1
                m2 += self._b2_use(l, e, s, e in subst, si, keep_miss)
            else:
                n1 += k
                d1 += 1
                if e in self.b1[l]:
                    self.b1[l].move_to_end(e)
                    if il and not hsp.box2_home(s):
                        self.moved_at[l][e] = 0
                else:
                    if e not in subst:
                        m1 += 1
                    self._b1_admit(l, e)
                    if not hsp.box2_home(s):
                        self.moved_at[l][e] = 0
        n_rep = 0
        if rep:
            legs = hsp.choose_legs(self.legs, b, rep, n1, d1, n2, d2)
            for (e, k), to2 in zip(rep, legs):
                n_rep += k
                if to2:
                    n2 += k
                    d2 += 1
                    self.b2[l].pop(e, None)  # pinned already
                else:
                    n1 += k
                    d1 += 1
                    self.b1[l].move_to_end(e)
        return n1, d1, n2, d2, m1, m2, n_rep

    def _b2_use(self, l, e, s, is_subst, si, keep_miss):
        """Box 2 serves e: 1 if it is a (thinned) demand read."""
        if self.b2_pinned(l, e):
            if hsp.box2_home(s):
                self.moved_at[l][e] = 0
            return 0
        lru = self.b2[l]
        miss = 0
        if e in lru:
            lru.move_to_end(e)
        else:
            if not is_subst and keep_miss(si, l, e):
                miss = 1
            lru[e] = True
        if hsp.keep(s):
            # a decode pick of a KEEP id grants its pin
            self._b2_pin_keep(l, e)
            if hsp.box2_home(s):
                self.moved_at[l][e] = 0
        else:
            self._b2_trim(l)
        return miss

    # ------------------------------------------------------------ the static-policy interface (unused here)
    def owner1(self, l, e):
        return hsp.box1_resident(self.side[l][e])

    def replicated(self, l, e):
        return self.side[l][e] == hsp.REP

    def share_now(self):
        return None
