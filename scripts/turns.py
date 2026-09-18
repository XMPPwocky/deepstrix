# Multi-turn driver: turn 1 = agentic_req.json; turn 2 = same + assistant reply + new user msg.
# Prints per-turn wall/tok/s and dumps replies. Usage: python3 turns.py <tag>
import json, subprocess, time, sys, hashlib
tag=sys.argv[1]
URL='http://127.0.0.1:18141/v1/chat/completions'
def send(req):
    w0=time.time()
    out=subprocess.run(['curl','-s','-m','3600',URL,'-H','Content-Type: application/json','--data',json.dumps(req)],capture_output=True,text=True).stdout
    wall=time.time()-w0
    j=json.loads(out); txt=j['choices'][0]['message']['content']; u=j['usage']
    return txt,u,wall
base=json.load(open('agentic_req.json'))
t1,u1,w1=send(base)
print(f"{tag} T1 prompt={u1['prompt_tokens']} completion={u1['completion_tokens']} wall={w1:.1f}s tok/s={u1['completion_tokens']/w1:.2f} sha={hashlib.sha256(t1.encode()).hexdigest()[:12]}",flush=True)
req2=dict(base); req2['messages']=base['messages']+[{'role':'assistant','content':t1},{'role':'user','content':'Good. Now list the three most likely root causes, one line each, most likely first.'}]
req2['max_tokens']=120
t2,u2,w2=send(req2)
print(f"{tag} T2 prompt={u2['prompt_tokens']} completion={u2['completion_tokens']} wall={w2:.1f}s tok/s={u2['completion_tokens']/w2:.2f} sha={hashlib.sha256(t2.encode()).hexdigest()[:12]}",flush=True)
json.dump({'t1':t1,'t2':t2,'req2':req2},open(f'turns_{tag}.json','w'),indent=1)
print("T2 reply:\n"+t2[:600])
