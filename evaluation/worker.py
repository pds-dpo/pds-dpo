"""One model, one GPU; durable task completion and safe partial-output recovery."""
import argparse
import fcntl
import subprocess
import time
from full_common import *

def run(command,log,gpu):
    verify();log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        f.write(json.dumps({'at':now(),'command':list(map(str,command))})+'\n');f.flush()
        subprocess.run(list(map(str,command)),cwd=ROOT,env=env(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)

def validate_lmms(m,t,out,n=None):
    n=COUNTS[t] if n is None else n
    results=list((out/'lmms').rglob('*_results.json'));samples=list((out/'lmms').rglob('*_samples_'+task_name(t)+'.jsonl'))
    assert len(results)==len(samples)==1,(m,t,results,samples)
    rows=read_jsonl(samples[0]);assert len(rows)==n and {int(r['doc_id']) for r in rows}==set(range(n))
    metrics=read_json(results[0])['results'][task_name(t)]
    assert metrics and all(r.get('filtered_resps') is not None for r in rows)
    return {'samples':n,'metrics':metrics,'artifacts':{str(p):sha256(p) for p in results+samples}}

def lmms(m,t,gpu,smoke=False):
    out=ROOT/'smoke'/m/t if smoke else output(m,t)
    if (out/'lmms').exists():
        try:return validate_lmms(m,t,out,2 if smoke else None)
        except Exception:(out/'lmms').rename(out/('lmms.incomplete-'+str(time.time_ns())))
    args=f'pretrained={model_path(m)},model_name=llava-v1.5-7b-{m},device=cuda,device_map=cuda,conv_template=vicuna_v1,attn_implementation=sdpa,use_cache=True'
    cmd=[EVAL_PY,'-m','lmms_eval','--model','llava','--model_args',args,'--tasks',task_name(t),'--include_path',ROOT/'tasks','--batch_size','1','--num_fewshot','0','--seed','0,0,0,0','--log_samples','--show_config','--output_path',out/'lmms','--verbosity','INFO']
    if smoke:cmd+=['--limit','2']
    run(cmd,out/'run.log',gpu)
    return validate_lmms(m,t,out,2 if smoke else None)

def validate_hall(m,t,p):
    rows=read_jsonl(p);manifest=read_jsonl(ROOT/'manifests'/(t+'.jsonl'))
    assert len(rows)==len({r['id'] for r in rows})==COUNTS[t]
    assert {r['id'] for r in rows}=={r['id'] for r in manifest}
    assert all(r['model_id']==EXPERIMENT+':'+m and isinstance(r['response'],str) and r['response'].strip() for r in rows)
    return {'samples':len(rows),'mean_response_words':sum(len(r['response'].split()) for r in rows)/len(rows),'artifacts':{str(p):sha256(p),str(p.with_suffix('.jsonl.run.json')):sha256(p.with_suffix('.jsonl.run.json'))},'judge_status':'not_required' if t.startswith('amber') else 'pending'}

def hall(m,t,gpu):
    out=output(m,t);p=out/'predictions.jsonl'
    run([EVAL_PY,'-m','hallucination_eval.generate','--backend','llava_1_5','--benchmark','amber' if t=='amber_disc' else t,'--manifest',ROOT/'manifests'/(t+'.jsonl'),'--protocol',ROOT/('protocol_disc.json' if t=='amber_disc' else 'protocol.json'),'--model',model_path(m),'--llava-model-name','llava-v1.5-7b-'+m,'--model-id',EXPERIMENT+':'+m,'--output',p,'--device','cuda:0'],out/'generation.log',gpu)
    return validate_hall(m,t,p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--gpu',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--tasks',nargs='+',choices=TASKS);a=p.parse_args()
    verify(full_models=True);model_identity=identity(a.model)
    folder=ROOT/'results'/a.model;folder.mkdir(exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.smoke:
            if not LMMS: raise RuntimeError('Smoke mode requires at least one lmms task')
            lmms(a.model,LMMS[0],a.gpu,True)
            atomic_json(ROOT/'smoke'/(a.model+'.json'),{'status':'passed','at':now(),'model_identity':model_identity});return
        try:
            for t in (a.tasks or TASKS):
                if complete(a.model,t):continue
                started=now();atomic_json(folder/'status.json',{'status':'running','task':t,'gpu':a.gpu,'at':started,'expected_samples':COUNTS[t],'pid':os.getpid()})
                r=lmms(a.model,t,a.gpu) if t in LMMS else hall(a.model,t,a.gpu)
                atomic_json(marker(a.model,t),{'status':'completed','task':t,'model':a.model,'model_identity':model_identity,'input_lock_sha256':sha256(ROOT/'input_lock.json'),'started_at':started,'finished_at':now(),**r})
            atomic_json(folder/'status.json',{'status':'generation_and_lmms_completed','at':now(),'gpu':a.gpu,'local_amber_scoring':'pending','paid_judging':'pending'})
        except BaseException as e:
            atomic_json(folder/'status.json',{'status':'failed','at':now(),'task':t,'error_type':type(e).__name__,'error':str(e),'gpu':a.gpu});raise

if __name__=='__main__':main()
