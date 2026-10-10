"""Window-level class means. usage: agg3.py SPLITJSON [MAXS MINH]"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import pct
R = json.load(open(sys.argv[1]))
MAXS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MINH = int(sys.argv[3]) if len(sys.argv) > 3 else 800
C = {}
for r in R:
    if r['kind'] != 'W' or r['rows'] != 4096:
        continue
    if r['ndec'] > 0:
        c = 'decode-interleaved'
    elif r['nmiss'] <= MAXS:
        c = 'steady'
    elif r['nmiss'] >= MINH:
        c = 'heavy'
    else:
        c = 'mid'
    C.setdefault(c, []).append(r)
print(f"{'class':<20}{'N':>3}{'wall':>7}{'(p10-p90)':>13}{'rows/s':>7}{'blocked':>8}{'busy':>6}{'G0prep':>7}{'other':>6}{'head+tail':>10}  hub nmiss mean")
for c, v in C.items():
    m = lambda k: sum(x[k] for x in v) / len(v)
    w = [x['wall'] for x in v]
    print(f"{c:<20}{len(v):3d}{m('wall'):7.2f}  {pct(w, .1):5.2f}-{pct(w, .9):5.2f}{4096 / m('wall'):7.0f}{m('expo'):8.2f}{m('busy'):6.2f}{m('prep'):7.2f}{m('other'):6.2f}{m('head') + m('tail'):10.2f}  {m('nmiss'):.0f}")
rep = [r for r in R if r['kind'] == 'R']
for r in rep:
    print('replay', r['lab'], 'wall', round(r['wall'], 2), 'blocked', round(r['expo'], 2), 'nreq', r['n'], 'miss', r['nmiss'])
