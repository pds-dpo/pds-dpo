"""Environment-only credentials and a conservative, durable $10 API budget."""
import argparse
import fcntl
import runpy
import sys
import uuid
from types import SimpleNamespace
from full_common import *

def reserve(messages,max_tokens):
    # UTF-8 byte count upper-bounds text token count; allowance covers chat framing.
    input_upper=sum(len(str(m.get('content','')).encode('utf-8'))+128 for m in messages)+256
    upper=(input_upper*PLAN['judge']['input_usd_per_million']+max_tokens*PLAN['judge']['output_usd_per_million'])/1e6
    p=ROOT/'judge/budget.json'
    ledger=read_json(p) if p.exists() else {'cap_usd':PLAN['judge']['budget_usd'],'calls':{}}
    if ledger['cap_usd']!=PLAN['judge']['budget_usd']: raise RuntimeError('Budget cap changed')
    used=sum(r['charged_usd'] for r in ledger['calls'].values())
    if used+upper>ledger['cap_usd']:raise RuntimeError('Authorized API budget exhausted; no request sent')
    key=str(uuid.uuid4());ledger['calls'][key]={'at':now(),'status':'reserved_or_uncertain','charged_usd':upper,'upper_bound_usd':upper}
    atomic_json(p,ledger);return key

def settle(key,response):
    p=ROOT/'judge/budget.json';ledger=read_json(p);usage=response.usage
    if usage:
        cost=(usage.prompt_tokens*PLAN['judge']['input_usd_per_million']+usage.completion_tokens*PLAN['judge']['output_usd_per_million'])/1e6
        assert cost<=ledger['calls'][key]['upper_bound_usd']+1e-9,'Budget estimate was not conservative'
        ledger['calls'][key].update(status='accounted',charged_usd=cost,usage=usage.model_dump(),response_id=response.id)
    else:ledger['calls'][key]['status']='usage_missing_upper_bound_retained'
    atomic_json(p,ledger)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['mmhal','object_halbench'],required=True);p.add_argument('--model',choices=MODELS,required=True);a=p.parse_args();verify()
    assert complete(a.model,a.task)
    assert PLAN['judge']['model']=='gpt-4.1-mini-2025-04-14', 'This release pins the judge; changing it is a new protocol'
    assert os.environ.get('OPENAI_API_KEY'),'Configured credential is missing'
    import openai
    RealClient=openai.OpenAI
    def Factory(*args,**kwargs):
        kwargs.update(max_retries=0,timeout=60.)
        client=RealClient(*args,**kwargs)
        def create(**request):
            assert request['model']=='gpt-4.1-mini-2025-04-14'
            request['service_tier']='default'
            request['max_tokens']=request.get('max_tokens',2048)
            with (ROOT/'judge/budget.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                token=reserve(request['messages'],request['max_tokens'])
                try:response=client.chat.completions.create(**request)
                except Exception as exc:raise RuntimeError('API request failed: '+type(exc).__name__) from None
                settle(token,response)
            assert response.model=='gpt-4.1-mini-2025-04-14','Returned judge model differs from pinned snapshot'
            return response
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    openai.OpenAI=Factory
    out=output(a.model,a.task)
    shared=['--manifest',str(ROOT/'manifests'/(a.task+'.jsonl')),'--predictions',str(out/'predictions.jsonl'),'--judge-model','gpt-4.1-mini-2025-04-14','--max-retries','3']
    if a.task=='mmhal':
        extra=['--official-evaluator',str(SUITE/'vendor/mmhal-bench/eval_gpt4.py'),'--output-dir',str(out/'judge')]
    else:extra=['--official-evaluator',str(SUITE/'vendor/rlhf-v/eval/eval_gpt_obj_halbench.py'),'--output',str(out/'extractions.jsonl')]
    sys.argv=['judge',*shared,*extra]
    runpy.run_module('hallucination_eval.judge_'+a.task,run_name='__main__')

if __name__=='__main__':main()
