"""Full hallucination tables, subsets and paired image-cluster uncertainty."""
import numpy as np
from full_common import *
from full_statistics import holm,metric,statistics
from score import scored

BENCHES=['pope','amber','mmhal','object_halbench']
def main():
    verify();gather=load_source('analyze_full.py').gather
    if MODELS != ['vanilla','A','B','C','D_small','E','F']:
        raise ValueError('The prespecified A-F comparison requires vanilla plus all six models in config order')
    if not (ROOT/'audit/summary.json').exists():
        raise RuntimeError('Run audit_images.py before image-cluster analysis')
    scores={m:{} for m in MODELS};tests=[];available=[]
    for bench in BENCHES:
        if not all(complete(m,bench) and (bench=='pope' or scored(m,bench)) for m in MODELS):continue
        all_rows={m:gather(m,bench) for m in MODELS}
        if bench!='pope':
            clusters_by_id=read_json(ROOT/'audit'/(bench+'.json'))['clusters']
            sources=read_jsonl(ROOT/'manifests'/(bench+'.jsonl'))
            for rows in all_rows.values():
                for source,row in zip(sources,rows):row['meta']['cluster_id']=clusters_by_id[str(source['id'])]
        categories=sorted({r['meta']['category'] for r in all_rows['vanilla']})
        clusters=sorted({r['meta']['cluster_id'] for r in all_rows['vanilla']});indices={g:i for i,g in enumerate(clusters)}
        matrices={};available.append(bench)
        for model,rows in all_rows.items():
            if bench in ('mmhal','object_halbench'):
                data=np.zeros((len(clusters),2))
                for row in rows:data[indices[row['meta']['cluster_id']]]+=row['value']
                detail=read_json(output(model,bench)/('judge/summary.json' if bench=='mmhal' else 'summary.json'))
            else:data,detail=statistics(bench,rows,categories,clusters)
            primary=float(metric(bench,data.sum(0),len(categories)));matrices[model]=data
            if bench=='amber':
                full=read_json(ROOT/'results'/model/'amber_scores/summary.json');detail['full_amber']=full
                assert abs(primary-full['unrounded_generative']['Hal'])<.002
            elif bench=='mmhal':assert abs(primary-detail['average_score'])<1e-6
            elif bench=='object_halbench':assert abs(primary-detail['response_hallucination'])<1e-6
            scores[model][bench]={'primary_score':primary,'samples':len(rows),'image_clusters':len(clusters),'details':detail,
                                 'mean_response_words':float(np.mean([len(r['response'].split()) for r in rows]))}
        rng=np.random.default_rng(PLAN['statistics_seed']+BENCHES.index(bench));n=len(clusters)
        pairs=[('A',m,'image_selection' if m in ['B','C','D_small'] else 'response_pairing') for m in CONDITIONS[1:]]
        pairs += [(m,'vanilla','versus_vanilla') for m in CONDITIONS]
        boots={m:[] for m in MODELS};deltas={pair:[] for pair in pairs};perms={pair:[] for pair in pairs}
        direction=-1 if bench in ['amber','object_halbench'] else 1
        for start in range(0,PLAN['bootstrap_replicates'],100):
            size=min(100,PLAN['bootstrap_replicates']-start)
            weights=rng.multinomial(n,np.full(n,1/n),size=size).astype(float)
            swaps=rng.integers(0,2,size=(size,n)).astype(float)
            resampled={m:metric(bench,weights@matrices[m],len(categories)) for m in MODELS}
            for m in MODELS:boots[m].extend(resampled[m].tolist())
            for pair in pairs:
                left,right,_=pair;a,b=matrices[left],matrices[right]
                deltas[pair].extend((direction*(resampled[left]-resampled[right])).tolist())
                change=swaps@(b-a)
                perms[pair].extend((direction*(metric(bench,a.sum(0)+change,len(categories))-metric(bench,b.sum(0)-change,len(categories)))).tolist())
        for m,values in boots.items():
            assert np.isfinite(values).mean()>=.999
            scores[m][bench]['ci95']=np.nanquantile(values,[.025,.975]).tolist()
        for pair in pairs:
            left,right,family=pair;point=direction*(scores[left][bench]['primary_score']-scores[right][bench]['primary_score'])
            assert np.isfinite(deltas[pair]).mean()>=.999 and np.isfinite(perms[pair]).mean()>=.999
            tests.append({'benchmark':bench,'left':left,'right':right,'family':family,'benefit_delta':point,
                'ci95':np.nanquantile(deltas[pair],[.025,.975]).tolist(),
                'raw_p':float((1+np.sum(np.abs(perms[pair])>=abs(point)-1e-12))/(1+len(perms[pair]))),
                'positive_means':'left model better; hallucination-rate differences sign-reversed'})
    for family,size in [('image_selection',12),('response_pairing',8),('versus_vanilla',24)]:
        selected=[r for r in tests if r['family']==family]
        for row,p in zip(selected,holm([r['raw_p'] for r in selected]+[1.]*(size-len(selected)))):
            row.update(holm_p=p,family_size=size)
    report={'status':'completed' if len(available)==4 else 'partial','at':now(),'scores':scores,'tests':tests,
        'available_benchmarks':available,'training_seed':42,'bootstrap_replicates':PLAN['bootstrap_replicates'],
        'uncertainty':'Paired image-cluster bootstrap/permutation; not training-seed or judge-repetition uncertainty',
        'audit':read_json(ROOT/'audit/summary.json'),'judge':PLAN['judge']['model'],'model_lock_sha256':sha256(ROOT/'model_lock.json')}
    atomic_json(ROOT/'report.json',report)
    lines=['# Full9K LLaVA-1.5-7B hallucination evaluation','',f"Status: {report['status']}. All models and all prespecified outcomes are retained.",'',
      '| Model | POPE macro F1 ↑ | AMBER Hal ↓ | MMHal score ↑ | Object response hallucination ↓ |',
      '|---|---:|---:|---:|---:|']
    for m in MODELS:
        cells=[]
        for b in BENCHES:
            if b not in scores[m]:cells.append('Pending');continue
            r=scores[m][b];cells.append(f"{r['primary_score']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]")
        lines.append('| '+m+' | '+' | '.join(cells)+' |')
    lines += ['', 'Intervals are 95% evaluation-image-cluster bootstrap intervals, not multiple-training-seed estimates.',
              'The JSON report contains every POPE subset, AMBER generative/discriminative section, MMHal question type, object metrics, and paired effects.',
              'D-small differs in prompt coverage, response-pool size and reward precision. Judge-based scores are not directly comparable to papers using a different judge.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'],'benchmarks':available,'report':str(ROOT/'report.md')},indent=2))

if __name__=='__main__':main()
