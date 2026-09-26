"""Explicit, resumable scoring. Paid stages require --allow-paid."""
import argparse
import fcntl
import subprocess
from full_common import *

def score_summary(m,t):
    return ROOT/'results'/m/'amber_scores/summary.json' if t=='amber' else output(m,t)/('judge/summary.json' if t=='mmhal' else 'summary.json')
def scored(m,t):
    p=ROOT/'scores'/f'{m}--{t}.json'
    if not p.exists(): return False
    r=read_json(p)
    if r['input_lock_sha256']!=sha256(ROOT/'input_lock.json') or r['model_identity']!=identity(m): raise RuntimeError('Score provenance mismatch')
    for path,digest in r['artifacts'].items():
        if sha256(path)!=digest: raise RuntimeError('Score artifact changed')
    return True
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--task',choices=['mmhal','object_halbench','amber'],required=True)
    p.add_argument('--allow-paid',action='store_true');a=p.parse_args()
    verify(full_models=True)
    if scored(a.model,a.task): print('Already scored and verified');return
    if not complete(a.model,a.task): raise RuntimeError('Inference is incomplete')
    if a.task=='amber' and not complete(a.model,'amber_disc'): raise RuntimeError('AMBER discriminative inference is incomplete')
    commands=[];summary=score_summary(a.model,a.task)
    if not summary.exists():
        if a.task=='amber': commands=[[SCORE_PY,CODE/'score_amber_full.py','--model',a.model]]
        else:
            if not a.allow_paid: p.error('Paid judging requires --allow-paid and OPENAI_API_KEY')
            if not os.environ.get('OPENAI_API_KEY'): p.error('Missing OPENAI_API_KEY')
            commands=[[SCORE_PY,CODE/'budget_judge.py','--task',a.task,'--model',a.model]]
            if a.task=='object_halbench': commands.append([SCORE_PY,CODE/'score_object_full.py','--model',a.model])
    for cmd in commands:
        environment=env('')
        if 'budget_judge.py' in str(cmd[1]): environment['OPENAI_API_KEY']=os.environ['OPENAI_API_KEY']
        subprocess.run(list(map(str,cmd)),env=environment,cwd=CODE,check=True)
    data=read_json(summary)
    assert data['samples']==(15220 if a.task=='amber' else COUNTS[a.task])
    if a.task=='mmhal':
        rows=read_jsonl(summary.parent/'judge_responses.jsonl')
        assert len(rows)==len({r['id'] for r in rows})==96
        assert all(r['judge_returned']==PLAN['judge']['model'] for r in rows)
    files=[x for x in summary.parent.iterdir() if x.is_file() and not x.name.endswith('.lock')]
    atomic_json(ROOT/'scores'/f'{a.model}--{a.task}.json',{'model_identity':identity(a.model),
        'input_lock_sha256':sha256(ROOT/'input_lock.json'),'artifacts':{str(x):sha256(x) for x in files}})
    print(json.dumps(data,indent=2))

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
