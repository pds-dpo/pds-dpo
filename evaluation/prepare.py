"""Prepare full-split manifests, verify upstream revisions, and freeze inputs."""
import argparse
import shutil
import subprocess
from collections import Counter
from full_common import *
from full_tasks import metadata
from hallucination_eval.prepare import _mmhal_rows, _object_rows, _amber_rows

SPECS={
 'scienceqa':('ScienceQA','ScienceQA-IMG','test','69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50',2017),
 'mme':('MME',None,'test','d6c9023f017b564f7b3ccccf5348166bce8fdbcd',2374),
 'mmmu':('MMMU',None,'validation','364f2e2eb107b36e07ff4c5a15f5947a759cef47',900),
 'pope':('POPE',None,'test','4db1276663dfa5eb8ad16a52d24c31a09e470896',9000),
 'seed_img':('SEED-Bench',None,'test','74c4ea0cee96786739e4ccb97d227818a05ae752',17990)}
PINS={'mmhal-bench':'f5f49a938f45ed99e235b8519ba28f76832a2add',
      'rlhf-v':'3863e39d5f541db7e3725acd8131ab99221455dc',
      'amber':'534babf6bbfcce2e735c26289dedfb21cef3c939'}

def check_repo(path,revision):
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision: raise RuntimeError(f'Wrong upstream revision: {path}')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    if 'amber' in TASKS and 'amber_disc' not in TASKS: raise ValueError('AMBER reporting requires both subsets')
    check_repo(HARNESS,'cb45ac4d4a667ea5ef89c7a148bff69b3489b981')
    adapter=HARNESS/'lmms_eval/models/simple/llava.py'
    if sha256(adapter)!=sha256(CODE/'compat/lmms_llava.py'):
        raise RuntimeError('Run install_lmms_compat.py on this isolated lmms-eval checkout first')
    for name,rev in PINS.items(): check_repo(SUITE/'vendor'/name,rev)
    required=[SUITE/'assets/coco2014/annotations'/n for n in ('captions_train2014.json','captions_val2014.json','instances_train2014.json','instances_val2014.json')]
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    for folder in ('tasks','manifests','logs','results','steps','scores','smoke','judge','audit'):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    for source in (CODE/'tasks').iterdir():
        if source.suffix not in ('.py','.yaml'): continue
        target=ROOT/'tasks'/source.name
        if target.exists() and target.read_bytes()!=source.read_bytes(): raise RuntimeError('Task source changed')
        if not target.exists(): shutil.copyfile(source,target)
    if LMMS:
        from datasets import load_dataset
    for bench in LMMS:
        name,config,split,rev,raw_count=SPECS[bench]
        ds=load_dataset('lmms-lab/'+name,config,revision=rev,split=split)
        assert len(ds)==raw_count
        rows=[]
        for i,r in enumerate(metadata(ds)):
            if bench=='seed_img' and r['data_type']!='image': continue
            cat=str(r['question_type_id']) if bench=='seed_img' else r['subject'] if bench=='scienceqa' else '_'.join(r['id'].split('_')[1:-1]) if bench=='mmmu' else r['category']
            group=str(r.get('image_source',r.get('data_id',r.get('question_id',r.get('id',i)))))
            rows.append({'raw_index':i,'metadata':r,'metadata_sha256':text_hash(r),'category':cat,'cluster_id':group})
        assert len(rows)==COUNTS[bench]
        if bench=='mme': assert set(Counter((r['category'],r['metadata']['question_id']) for r in rows).values())=={2}
        immutable(ROOT/'manifests'/f'{bench}.json',{'dataset':'lmms-lab/'+name,'config':config,'split':split,'revision':rev,
            'raw_count':raw_count,'samples':len(rows),'rows':rows,'category_counts':dict(Counter(r['category'] for r in rows))})
    builders={'mmhal':_mmhal_rows,'object_halbench':_object_rows,'amber':_amber_rows}
    from PIL import Image
    for bench in HALL:
        if bench=='amber_disc':
            anns=read_json(SUITE/'vendor/amber/data/annotations.json');rows=[]
            for i,r in enumerate(read_json(SUITE/'vendor/amber/data/query/query_discriminative.json')):
                ann=anns[r['id']-1];assert ann['id']==r['id']
                rows.append({'benchmark':'amber','id':r['id'],'ordinal':i,'image_path':str(SUITE/'assets/amber/image'/r['image']),
                    'prompt':r['query']+'\nAnswer with Yes or No only.','original_query':r['query'],'category':ann['type'],'truth':ann['truth']})
        else: rows=builders[bench](SUITE)
        assert len(rows)==len({r['id'] for r in rows})==COUNTS[bench]
        for r in rows:
            with Image.open(r['image_path']) as image: image.convert('RGB').load()
        text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows);dest=ROOT/'manifests'/f'{bench}.jsonl'
        if dest.exists() and dest.read_text()!=text: raise RuntimeError('Existing manifest differs')
        if not dest.exists(): dest.write_text(text)
    protocol=read_json(CODE/'protocol.json');protocol['suite_name']=EXPERIMENT
    for bench in ('mmhal','object_halbench'): protocol['benchmarks'][bench]['judge']['model']=PLAN['judge']['model']
    immutable(ROOT/'protocol.json',protocol)
    disc=json.loads(json.dumps(protocol));disc['benchmarks']['amber'].update(count=14216,max_new_tokens=16)
    immutable(ROOT/'protocol_disc.json',disc)
    immutable(ROOT/'model_lock.json',{'models':{m:inspect_identity(m) for m in MODELS}})
    if args.seal:
        files=list(CODE.rglob('*.py'))+list((CODE/'tasks').glob('*.yaml'))+[CONFIG,CODE/'protocol.json',ROOT/'model_lock.json',ROOT/'protocol.json',ROOT/'protocol_disc.json']
        files=[p for p in files if ROOT not in p.parents]
        files += [ROOT/'model_lock.json',ROOT/'protocol.json',ROOT/'protocol_disc.json']+list((ROOT/'manifests').iterdir())
        files += list((HARNESS/'lmms_eval').rglob('*.py'))
        for name in PINS:
            files += list((SUITE/'vendor'/name).rglob('*.py'))
        files += [SUITE/'vendor/amber/data'/n for n in ('annotations.json','relation.json','safe_words.txt','metrics.txt')]
        files += required
        immutable(ROOT/'input_lock.json',{'files':{str(p):sha256(p) for p in files if p.is_file()}})
        verify()
    print(json.dumps({'prepared':True,'sealed':args.seal,'models':MODELS,'counts':COUNTS},indent=2))

if __name__=='__main__': main()
