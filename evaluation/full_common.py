"""Portable configuration and immutable evaluation artifact checks."""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get('SYNTHALIGN_EVAL_CONFIG', CODE/'config.json')).resolve()
if not CONFIG.is_file():
    raise RuntimeError('Copy evaluation/config.example.json to config.json, edit paths, and set SYNTHALIGN_EVAL_CONFIG to its absolute path.')
PLAN = json.loads(CONFIG.read_text())
def configured(value):
    p = Path(os.path.expandvars(value)).expanduser()
    return (CONFIG.parent/p).resolve() if not p.is_absolute() else p.resolve()
ROOT = configured(PLAN['run_root'])
SUITE = configured(PLAN['suite_root'])
HARNESS = configured(PLAN['lmms_eval_root'])
LLAVA = CODE/'compat'
PROJECT = CODE
EVAL_PY = configured(PLAN['inference_python'])
SCORE_PY = configured(PLAN['scoring_python'])
MODELS = list(PLAN['models'])
CONDITIONS = [m for m in MODELS if m != 'vanilla']
TASKS = PLAN['tasks']
LMMS = [t for t in TASKS if t in ('scienceqa','mme','mmmu','pope','seed_img')]
HALL = [t for t in TASKS if t not in LMMS]
ALL_COUNTS = dict(scienceqa=2017,mme=2374,mmmu=900,seed_img=14233,pope=9000,
                  mmhal=96,object_halbench=300,amber=1004,amber_disc=14216)
COUNTS = {t:ALL_COUNTS[t] for t in TASKS}
EXPERIMENT = PLAN['experiment']
sys.path.insert(0,str(CODE/'tasks'))

def now(): return datetime.now(timezone.utc).isoformat()
def read_json(p): return json.loads(Path(p).read_text())
def read_jsonl(p):
    with Path(p).open() as f: return [json.loads(line) for line in f if line.strip()]
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def text_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def atomic_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
    with os.fdopen(fd,'w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def immutable(p,value):
    if Path(p).exists():
        if read_json(p)!=value: raise RuntimeError(f'Frozen input differs: {p}')
    else: atomic_json(p,value)
def load_source(name):
    spec=importlib.util.spec_from_file_location('release_'+name.replace('.','_'),CODE/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def model_path(m): return configured(PLAN['models'][m])
def inspect_identity(m):
    p=model_path(m)
    if not (p/'config.json').is_file(): raise FileNotFoundError(p/'config.json')
    files=[p/'config.json']
    for name in ('model.safetensors.index.json','pytorch_model.bin.index.json'):
        if (p/name).exists():
            files.append(p/name);files += [p/n for n in sorted(set(read_json(p/name)['weight_map'].values()))]
    if len(files)==1:
        files += [x for x in (p/'model.safetensors',p/'pytorch_model.bin') if x.is_file()]
    if len(files)==1: raise RuntimeError(f'{m}: provide a full/merged checkpoint, not only a LoRA adapter')
    for name in ('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json',
                 'added_tokens.json','preprocessor_config.json','processor_config.json','generation_config.json'):
        if (p/name).is_file(): files.append(p/name)
    return {'path':str(p),'files':{str(x):{'sha256':sha256(x),'bytes':x.stat().st_size} for x in files}}
def identity(m):
    record=read_json(ROOT/'model_lock.json')['models'][m]
    if record['path']!=str(model_path(m)): raise RuntimeError('Model path changed')
    for path,info in record['files'].items():
        p=Path(path)
        if not p.is_file() or p.stat().st_size!=info['bytes']: raise RuntimeError(f'Checkpoint missing/changed: {p}')
    return record
def verify(full_models=False):
    for p,h in read_json(ROOT/'input_lock.json')['files'].items():
        if sha256(p)!=h: raise RuntimeError(f'Frozen evaluation input changed: {p}')
    if full_models:
        for m in MODELS:
            if inspect_identity(m)!=identity(m): raise RuntimeError(f'Checkpoint content changed: {m}')
def env(gpu):
    e={k:v for k,v in os.environ.items() if not k.startswith('OPENAI_') and k!='WANDB_API_KEY'}
    e.update(CUDA_VISIBLE_DEVICES=str(gpu),SYNTHALIGN_EVAL_CONFIG=str(CONFIG),
             TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',
             OMP_NUM_THREADS='4',WANDB_MODE='disabled',
             PYTHONPATH=f'{LLAVA}:{HARNESS}:{CODE}:{CODE/"tasks"}')
    e['NLTK_DATA']=str(SUITE/'cache/nltk_data')
    return e
def task_name(t): return 'synthalign_v10_full_'+t
def output(m,t): return ROOT/'results'/m/t
def marker(m,t): return ROOT/'steps'/(m+'--'+t+'.json')
def complete(m,t):
    p=marker(m,t)
    if not p.exists(): return False
    r=read_json(p)
    if r['samples']!=COUNTS[t] or r['model_identity']!=identity(m): raise RuntimeError('Completion identity/count mismatch')
    if r['input_lock_sha256']!=sha256(ROOT/'input_lock.json'): raise RuntimeError('Completion protocol mismatch')
    for f,h in r['artifacts'].items():
        if sha256(f)!=h: raise RuntimeError(f'Changed completed output: {f}')
    return True
