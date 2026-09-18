import sys, pickle, collections, os
sys.path.insert(0, os.path.expanduser('~/scripts'))
import importlib.util
spec = importlib.util.spec_from_file_location("apg", os.path.expanduser('~/scripts/analyze_pftrace_gaps.py'))
apg = importlib.util.module_from_spec(spec); spec.loader.exec_module(apg)

TRACE='/tmp/claude-1000/-home-claude-code-deepstrix/e6bbeb19-ceab-4950-b666-df1b6c344139/scratchpad/verify.pftrace'
CACHE='/tmp/claude-1000/-home-claude-code-deepstrix/e6bbeb19-ceab-4950-b666-df1b6c344139/scratchpad/slices.pkl'
if os.path.exists(CACHE):
    tracks, sl_by_track = pickle.load(open(CACHE,'rb'))
else:
    tracks, events = apg.extract(TRACE)
    by = collections.defaultdict(list)
    for ev in events: by[ev[1]].append(ev)
    sl_by_track = {}
    for uuid, evs in by.items():
        evs.sort(key=lambda e: e[0])
        sl_by_track[uuid] = apg.pair_slices(evs)
    pickle.dump((tracks, sl_by_track), open(CACHE,'wb'))
print("tracks:")
for u,n in tracks.items(): print(f"  {u:#x} {n}  slices={len(sl_by_track.get(u,[]))}")
