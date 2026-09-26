from collections import defaultdict
import numpy as np
def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);out=[0.]*len(order);previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,(len(order)-rank)*pvalues[i]));out[i]=previous
    return out

def metric(bench,totals,ncat):
    values=np.asarray(totals,dtype=float)
    with np.errstate(invalid="ignore",divide="ignore"):
        if bench=="mmhal":return values[...,0]/values[...,1]
        if bench=="object_halbench":return 100*values[...,0]/values[...,1]
        if bench=="amber":return 100*(1-values[...,0]/values[...,1])
        if bench=="pope":
            x=values.reshape(*values.shape[:-1],ncat,7)
            den=2*x[...,0]+x[...,1]+x[...,2]
            return 100*np.mean(np.divide(2*x[...,0],den,out=np.zeros_like(den),where=den!=0),axis=-1)
        if bench=="mme":
            x=values.reshape(*values.shape[:-1],ncat,2)
            return np.sum(x[...,0]/x[...,1],axis=-1)
        return 100*values[...,0]/values[...,1]

def statistics(bench,rows,categories,clusters):
    nc=len(categories);width=7*nc if bench=="pope" else 2*nc if bench=="mme" else 2
    data=np.zeros((len(clusters),width));ci={k:i for i,k in enumerate(clusters)};ct={k:i for i,k in enumerate(categories)}
    subgroup=defaultdict(list)
    if bench=="mme":
        pairs=defaultdict(list)
        for r in rows:
            s=r["sample"];m=r["meta"]
            item=s.get("mme_perception_score",s.get("mme_cognition_score"));assert item is not None
            assert item["category"]==m["category"] and item["question_id"]==m["metadata"]["question_id"]
            pairs[(m["category"],item["question_id"])].append((float(item["score"]),m["cluster_id"]))
        for (cat,qid),pair in pairs.items():
            assert len(pair)==2 and pair[0][1]==pair[1][1]
            value=100*(sum(s for s,g in pair)/2+int(all(s==1 for s,g in pair)))
            i=ci[pair[0][1]];j=2*ct[cat];data[i,j:j+2]+=[value,1];subgroup[cat].append(value)
        return data,{k:{"image_pairs":len(v),"score":sum(v)/len(v)} for k,v in subgroup.items()}
    for r in rows:
        m=r["meta"];i=ci[m["cluster_id"]];cat=m.get("category","generative")
        if bench=="amber":
            v=r["metrics"];data[i]+=[v["non_hallu_score"],v["non_hallu_num"]]
        elif bench=="pope":
            s=r["sample"]["pope_accuracy"];truth=s["ground_truth"];pred=s["prediction"]
            assert truth==m["metadata"]["answer"].lower() and s["category"]==cat
            v=[int(pred=="yes" and truth=="yes"),int(pred=="yes" and truth=="no"),int(pred!="yes" and truth=="yes"),int(pred==truth),1,int(pred=="yes"),int(pred not in ["yes","no"])]
            j=7*ct[cat];data[i,j:j+7]+=v;subgroup[cat].append(v)
        else:
            s=r["sample"]
            if bench=="mmmu":
                from mmmu_utils import evaluate_mmmu
                item=s["mmmu_acc"];assert item["id"]==m["metadata"]["id"]
                _,v=evaluate_mmmu([item]);correct=v["acc"]
            else:correct=float(s["exact_match" if bench=="scienceqa" else "seed_image"])
            data[i]+=[correct,1];subgroup[cat].append(correct)
    if bench=="pope":
        output={}
        for cat,items in subgroup.items():
            tp,fp,fn,correct,n,yes,invalid=np.sum(items,axis=0)
            output[cat]={"samples":int(n),"accuracy":100*correct/n,"precision":100*tp/(tp+fp) if tp+fp else 0.,"recall":100*tp/(tp+fn) if tp+fn else 0.,"f1":100*2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,"yes_ratio":100*yes/n,"invalid_rate":100*invalid/n}
        return data,output
    if bench=="amber":
        totals={k:sum(r["metrics"][k] for r in rows) for k in rows[0]["metrics"]}
        ratios={"CHAIR":("chair_score","chair_num"),"Cover":("safe_cover_score","safe_cover_num"),"Cog":("hallu_cover_score","hallu_cover_num")}
        extra={k:100*totals[a]/totals[b] if totals[b] else None for k,(a,b) in ratios.items()}
        extra["Hal"]=float(metric(bench,data.sum(0),nc));return data,extra
    return data,{k:{"samples":len(v),"accuracy":100*sum(v)/len(v)} for k,v in subgroup.items()}

def compare_arrays(bench,a,b,ncat,weights,swaps):
    direction=-1 if bench in ["amber","object_halbench"] else 1
    point=float(direction*(metric(bench,a.sum(0),ncat)-metric(bench,b.sum(0),ncat)))
    boot=direction*(metric(bench,weights@a,ncat)-metric(bench,weights@b,ncat));boot=boot[np.isfinite(boot)]
    assert len(boot)>=.999*len(weights)
    change=swaps@(b-a)
    perm=direction*(metric(bench,a.sum(0)+change,ncat)-metric(bench,b.sum(0)-change,ncat));perm=perm[np.isfinite(perm)]
    p=float((1+np.sum(np.abs(perm)>=abs(point)-1e-12))/(len(perm)+1))
    low,high=np.quantile(boot,[.025,.975])
    return {"benefit_delta_A_minus_comparator":point,"ci95":[float(low),float(high)],"paired_cluster_permutation_p":p,"bootstrap_replicates":len(boot),"permutation_replicates":len(perm),"positive_means":"A better (sign reversed for AMBER Hal)","units":"MME points" if bench=="mme" else "MMHal rating points" if bench=="mmhal" else "percentage points"}
