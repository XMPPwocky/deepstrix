"""Per-request decomposition of prefill box-2 requests (hub_req joined to b2_req/b2_write) per window class.
usage: join3.py HUBEXTRACT LOG WINFILE SPLITJSON B2FILE[,B2FILE...]
class: steady = nmiss<=MAXS & no decode ; heavy = nmiss>=MINH & no decode"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

MAXS = int(sys.argv[6]) if len(sys.argv) > 6 else 100
MINH = int(sys.argv[7]) if len(sys.argv) > 7 else 800
H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
B, W = {}, {}
for fn in sys.argv[5].split(','):
    _, RB = load(fn, {'b2_req', 'b2_write'})
    for b in RB['b2_req']:
        B[(int(b['seq']), int(b['t_frame']))] = b
    for w in RB['b2_write']:
        W[int(w['seq'])] = w
print('b2_req', len(B), 'b2_write', len(W))
CL = defaultdict(list)   # class -> list of (hub req, window label)
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096:
        continue
    lo, hi = H.raw(a), H.raw(b)
    rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
    c = 'steady' if s['nmiss'] <= MAXS else ('heavy' if s['nmiss'] >= MINH else 'mid')
    for r in rs:
        CL[c].append(r)
    CL[c + '_n'].append(lab)


def row(P, k, unit=1.0):
    v = P[k]
    if not v:
        return 'n/a'
    return f"{pct(v, .1) / unit:7.2f} {pct(v, .5) / unit:7.2f} {sum(v) / len(v) / unit:7.2f} {pct(v, .9) / unit:7.2f}"


for c in ('steady', 'mid', 'heavy'):
    rs = CL.get(c)
    if not rs:
        continue
    print(f"\n===== class {c}: windows {CL[c + '_n']}  N={len(rs)} requests")
    P = defaultdict(list)
    nj = 0
    for h in rs:
        bb = B.get((int(h['seq']), int(h['t2_b2'])))
        ww = W.get(int(h['seq']))
        o = H.off(h['t1'])
        P['rtt'].append((h['t4'] - h['t_submit']) / 1e6)
        P['hub_pre'].append((h['t1'] - h['t_submit']) / 1e6)
        P['submit_call'].append((h['t_submit_end'] - h['t_submit']) / 1e6)
        P['wire_out'].append((h['t2_b2'] - h['t1'] - o) / 1e6)
        P['wire_back'].append((h['t4'] - h['t3_b2'] + o) / 1e6)
        P['b2_in_to_out'].append((h['t3_b2'] - h['t2_b2']) / 1e6)
        P['hub_t4_to_exit'].append((h['t_wait_exit'] - h['t4']) / 1e6)
        P['exposed_wait'].append((h['t_wait_exit'] - h['t_wait_enter']) / 1e6)
        P['blocked_flag'].append(h['blocked'])
        P['mb_out'].append(h['bytes_out'] / 1e6)
        P['mb_in'].append(h['bytes_in'] / 1e6)
        if P['wire_out'][-1] > 0.3:
            P['gbs_out'].append(h['bytes_out'] / 1e9 / (P['wire_out'][-1] / 1e3))
        if P['wire_back'][-1] > 0.3:
            P['gbs_back'].append(h['bytes_in'] / 1e9 / (P['wire_back'][-1] / 1e3))
        if bb:
            nj += 1
            P['b2_queue'].append((bb['t_dequeue'] - bb['t_frame']) / 1e6)
            P['b2_pre_run'].append((bb['t_run_start'] - bb['t_dequeue']) / 1e6)
            P['b2_merge'].append((bb['t_merge_end'] - bb['t_dequeue']) / 1e6)
            P['b2_hints'].append((bb['t_hints_end'] - bb['t_merge_end']) / 1e6)
            P['b2_ensure_etc'].append((bb['t_run_start'] - bb['t_hints_end']) / 1e6)
            P['b2_run'].append((bb['t_run_end'] - bb['t_run_start']) / 1e6)
            P['b2_d2h'].append((bb['t_d2h_end'] - bb['t_run_end']) / 1e6)
            P['b2_d2h_to_ready'].append((bb['t_ready'] - bb['t_d2h_end']) / 1e6)
            P['b2_ready_to_t3'].append((h['t3_b2'] - bb['t_ready']) / 1e6)
            P['b2_page_us_ms'].append(bb['page_us'] / 1e3)
            P['b2_n_miss'].append(bb['n_miss'])
            P['b2_merged'].append(bb['merged'])
            P['b2_exec_gpu'].append(bb['exec_gpu_us'] / 1e3)
            P['b2_exec_h2d'].append(bb['exec_h2d_us'] / 1e3)
            P['b2_gateup_wait'].append(bb.get('gateup_wait_us', float('nan')) / 1e3)
            P['b2_down_wait'].append(bb.get('down_wait_us', float('nan')) / 1e3)
            P['b2_depth'].append(bb['depth_on_take'])
            P['b2_idle_before'].append(bb['idle_before_us'] / 1e3)
            P['b2_server'].append(bb['server_us'] / 1e3)
        if ww:
            P['b2_write_dur'].append((ww['t_written'] - ww['t3']) / 1e6)
    print(f"joined {nj}/{len(rs)}.  columns: p10 p50 mean p90 (ms unless noted)")
    for k in ['rtt', 'submit_call', 'hub_pre', 'wire_out', 'b2_queue', 'b2_pre_run', 'b2_merge', 'b2_hints', 'b2_ensure_etc', 'b2_run', 'b2_d2h', 'b2_d2h_to_ready', 'b2_ready_to_t3',
              'b2_in_to_out', 'b2_write_dur', 'wire_back', 'hub_t4_to_exit', 'exposed_wait', 'blocked_flag', 'b2_page_us_ms', 'b2_n_miss', 'b2_merged', 'b2_exec_gpu', 'b2_exec_h2d', 'b2_gateup_wait', 'b2_down_wait',
              'b2_depth', 'b2_idle_before', 'b2_server', 'mb_out', 'mb_in', 'gbs_out', 'gbs_back']:
        print(f"  {k:<16} {row(P, k)}")
