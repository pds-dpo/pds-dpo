"""Bounded-memory decoded-image audit; does not alter full evaluation splits."""
from collections import defaultdict
from full_common import *
from full_tasks import contains_image

def digest(image):
    rgb=image.convert('RGB')
    return str(rgb.size)+':'+hashlib.sha256(rgb.tobytes()).hexdigest()
def flatten(value):
    from PIL import Image
    if isinstance(value,Image.Image):yield value
    elif isinstance(value,(list,tuple)):
        for v in value:yield from flatten(v)
    elif isinstance(value,dict):
        for v in value.values():yield from flatten(v)
def group(records):
    parent={str(i):str(i) for i in records};seen={}
    def get(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,hashes in records.items():
        for h in hashes:
            if h in seen:parent[get(str(i))]=get(seen[h])
            else:seen[h]=str(i)
    return {str(i):get(str(i)) for i in records}
def main():
    from PIL import Image
    from datasets import load_dataset
    verify();out=ROOT/'audit';out.mkdir(exist_ok=True)
    training_list=PLAN.get('training_images_json')
    paths=set(read_json(configured(training_list))) if training_list else set()
    hashes=set()
    for name in paths:
        p=configured(name)
        with Image.open(p) as image:hashes.add(digest(image))
    summary={'at':now(),'training_overlap_checked':bool(training_list),'training_unique_image_paths':len(paths),'training_unique_pixel_hashes':len(hashes),'benchmarks':{},'policy':'Report exact-pixel overlaps if training images supplied; absence of a training list is NOT a clean-overlap claim. Never filter official splits.'}
    for b in LMMS:
        spec=read_json(ROOT/'manifests'/(b+'.json'));p=out/(b+'.json')
        if p.exists():
            r=read_json(p)
            assert r['manifest_sha256']==sha256(ROOT/'manifests'/(b+'.json'))
        else:
            ds=load_dataset(spec['dataset'],spec['config'],revision=spec['revision'],split=spec['split'])
            fields=[k for k,v in ds.features.items() if contains_image(v)];records={};overlap=[]
            # Iterate selected records lazily; never materialize PIL images for the full split.
            for count,meta in enumerate(spec['rows'],1):
                i=meta['raw_index'];row=ds[i];h=[digest(im) for k in fields for im in flatten(row[k])]
                assert h,(b,i);records[str(i)]=h
                if set(h)&hashes:overlap.append(i)
                if count%500==0:print(b,count,'/',len(spec['rows']),flush=True)
            r={'samples':len(records),'pixel_hashes':records,'clusters':group(records),'training_overlap_indices':overlap,'manifest_sha256':sha256(ROOT/'manifests'/(b+'.json'))}
            atomic_json(p,r)
        summary['benchmarks'][b]={'samples':r['samples'],'image_clusters':len(set(r['clusters'].values())),'overlap_samples':len(r['training_overlap_indices'])}
    for b in HALL:
        rows=read_jsonl(ROOT/'manifests'/(b+'.jsonl'));overlap=[];records={}
        for r in rows:
            with Image.open(r['image_path']) as image:
                h=digest(image);records[str(r['id'])]=[h]
                if h in hashes:overlap.append(r['id'])
        atomic_json(out/(b+'.json'),{'samples':len(rows),'clusters':group(records),'pixel_hashes':records,'training_overlap_ids':overlap,'manifest_sha256':sha256(ROOT/'manifests'/(b+'.jsonl'))})
        summary['benchmarks'][b]={'samples':len(rows),'image_clusters':len(set(group(records).values())),'training_overlap_ids':overlap,'overlap_samples':len(overlap)}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
