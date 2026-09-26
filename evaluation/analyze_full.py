"""Report all full-split metrics and paired effects, including nulls/regressions."""
import numpy as np
from full_common import *
from full_statistics import holm,metric,statistics,compare_arrays
from score import scored

BENCHES=['scienceqa','mme','mmmu','pope','seed_img','amber','mmhal','object_halbench']
def gather(m,b):
    if b in LMMS:
        spec=read_json(ROOT/'manifests'/(b+'.json'));meta=spec['rows']
        audit=ROOT/'audit'/f'{b}.json'
        if audit.exists():
            clusters=read_json(audit)['clusters']
            meta=[dict(r,cluster_id=clusters[str(r['raw_index'])]) for r in meta]
        files=list((output(m,b)/'lmms').rglob('*_samples_'+task_name(b)+'.jsonl'));assert len(files)==1
        samples=sorted(read_jsonl(files[0]),key=lambda r:int(r['doc_id']))
        assert len(samples)==len(meta)
        return [{'meta':md,'sample':r,'response':str(r['filtered_resps'])} for md,r in zip(meta,samples)]
    src=read_jsonl(ROOT/'manifests'/(b+'.jsonl'));answers={r['id']:r['response'] for r in read_jsonl(output(m,b)/'predictions.jsonl')}
    if b=='amber':
        scores={r['id']:r['metrics'] for r in read_jsonl(ROOT/'results'/m/'amber_scores/per_image.jsonl')}
        return [{'meta':{'cluster_id':r['image_path'],'category':'generative'},'metrics':scores[r['id']],'response':answers[r['id']]} for r in src]
    if b=='mmhal':
        scores={r['id']:r for r in read_jsonl(output(m,b)/'judge/judge_responses.jsonl')}
        return [{'meta':{'cluster_id':r['image_path'],'category':r['question_type']},'value':[scores[r['id']]['rating'],1],'response':answers[r['id']]} for r in src]
    per_image=read_json(output(m,b)/'official_per_image.json')['sentences'];scores={int(r['image_id']):r for r in per_image}
    return [{'meta':{'cluster_id':r['image_path'],'category':'objects'},'value':[scores[int(r['image_id'])]['metrics']['CHAIRs'],int(bool(scores[int(r['image_id'])]['mscoco_generated_words']))],'response':answers[r['id']]} for r in src]

def main():
    verify();results={m:{} for m in MODELS};tests=[];available=[]
    if MODELS != ['vanilla','A','B','C','D_small','E','F']:
        raise ValueError('Analysis requires vanilla plus A-F in the documented config order')
    if not (ROOT/'audit/summary.json').exists():
        raise RuntimeError('Run audit_images.py before image-cluster analysis')
    for b in [bench for bench in BENCHES if bench in TASKS]:
        if not all(complete(m,b) and (b in LMMS or scored(m,b)) for m in MODELS):continue
        try:all_rows={m:gather(m,b) for m in MODELS}
        except FileNotFoundError:continue
        if b not in LMMS:
            clusters_by_id=read_json(ROOT/'audit'/(b+'.json'))['clusters']
            sources=read_jsonl(ROOT/'manifests'/(b+'.jsonl'))
            for rows in all_rows.values():
                for source,row in zip(sources,rows):row['meta']['cluster_id']=clusters_by_id[str(source['id'])]
        categories=sorted({r['meta']['category'] for r in all_rows['vanilla']});clusters=sorted({r['meta']['cluster_id'] for r in all_rows['vanilla']})
        matrices={};available.append(b)
        for m,rows in all_rows.items():
            if b in ['mmhal','object_halbench']:
                idx={g:i for i,g in enumerate(clusters)};data=np.zeros((len(clusters),2))
                for r in rows:data[idx[r['meta']['cluster_id']]]+=r['value']
                detail=read_json(output(m,b)/('judge/summary.json' if b=='mmhal' else 'summary.json'))
            else:data,detail=statistics(b,rows,categories,clusters)
            score=float(metric(b,data.sum(0),len(categories)));matrices[m]=data
            if b=='amber':
                full=read_json(ROOT/'results'/m/'amber_scores/summary.json');assert abs(score-full['unrounded_generative']['Hal'])<.002;detail['full_amber']=full
            elif b=='mmhal':assert abs(score-detail['average_score'])<1e-6
            elif b=='object_halbench':assert abs(score-detail['response_hallucination'])<1e-6
            elif b!='pope':
                report=read_json(marker(m,b))['metrics']
                if b=='mme':expected=sum(v for k,v in report.items() if k.startswith(('mme_perception_score,','mme_cognition_score,')) and 'stderr' not in k)
                else:expected=100*next(v for k,v in report.items() if k.startswith({'scienceqa':'exact_match,','mmmu':'mmmu_acc,','seed_img':'seed_image,'}[b]) and 'stderr' not in k)
                assert abs(score-expected)<.002,(m,b,score,expected)
            results[m][b]={'primary_score':score,'samples':len(rows),'clusters':len(clusters),'details':detail,'mean_response_words':float(np.mean([len(r['response'].split()) for r in rows]))}
        # Batch resampling bounds memory even for the complete 14,233-row SEED split.
        rng=np.random.default_rng(20260920+BENCHES.index(b));n=len(clusters);boot={m:[] for m in CONDITIONS[1:]};perms={m:[] for m in CONDITIONS[1:]};direction=-1 if b in ['amber','object_halbench'] else 1
        a=matrices['A'];points={m:float(direction*(metric(b,a.sum(0),len(categories))-metric(b,matrices[m].sum(0),len(categories)))) for m in boot}
        for start in range(0,10000,100):
            w=rng.multinomial(n,np.full(n,1/n),size=100).astype(float);sw=rng.integers(0,2,size=(100,n)).astype(float)
            for other in boot:
                x=matrices[other];boot[other].extend((direction*(metric(b,w@a,len(categories))-metric(b,w@x,len(categories)))).tolist())
                change=sw@(x-a);perms[other].extend((direction*(metric(b,a.sum(0)+change,len(categories))-metric(b,x.sum(0)-change,len(categories)))).tolist())
        for other in boot:
            bt=np.array(boot[other]);pm=np.array(perms[other]);assert np.isfinite(bt).mean()>=.999 and np.isfinite(pm).mean()>=.999
            tests.append({'benchmark':b,'comparison':'A_vs_'+other[0],'family':'image_selection' if other[0] in 'BCD' else 'response_pairing','benefit_delta':points[other],'ci95':np.nanquantile(bt,[.025,.975]).tolist(),'raw_p':float((1+np.sum(abs(pm)>=abs(points[other])-1e-12))/(1+len(pm)))})
    for family,size in [('image_selection',24),('response_pairing',16)]:
        selected=[t for t in tests if t['family']==family]
        adjusted=holm([t['raw_p'] for t in selected]+[1.]*(size-len(selected)))
        for t,p in zip(selected,adjusted):t['holm_p']=p;t['family_size']=size
    status='completed' if len(available)==len([b for b in BENCHES if b in TASKS]) else 'partial'
    atomic_json(ROOT/'report_full.json',{'status':status,'at':now(),'scores':results,'tests':tests,'available_benchmarks':available,'training_seed':42,'confidence_intervals':'evaluation-sample uncertainty, not training-seed uncertainty','image_audit_completed':(ROOT/'audit/summary.json').exists()})
    lines=['# Full original-v1.0 LLaVA-1.5-7B ablation evaluation','',f'Status: {status}. All seven models, including nulls and regressions.','', '| Model | '+' | '.join(BENCHES)+' |','|---|'+'---:|'*len(BENCHES)]
    for m in MODELS:lines.append('| '+m+' | '+' | '.join(f"{results[m][b]['primary_score']:.2f}" if b in results[m] else 'Pending' for b in BENCHES)+' |')
    lines+=['','AMBER Hal and Object HalBench response hallucination: lower is better. Other primary metrics: higher is better. See report_full.json for every subgroup, paired 95% interval and Holm-adjusted test. One training seed; no claim of seed robustness.']
    (ROOT/'report_full.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
