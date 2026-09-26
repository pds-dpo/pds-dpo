"""Full-split task adapters and image-safe metadata access."""
from full_common import ROOT,read_json,text_hash

def contains_image(feature):
    from datasets import Image
    if isinstance(feature,Image):return True
    if isinstance(feature,dict):return any(contains_image(v) for v in feature.values())
    if isinstance(feature,(list,tuple)):return any(contains_image(v) for v in feature)
    return hasattr(feature,'feature') and contains_image(feature.feature)

def metadata(ds): return ds.remove_columns([k for k,v in ds.features.items() if contains_image(v)])
def process(ds,bench):
    spec=read_json(ROOT/'manifests'/f'{bench}.json')
    rows=list(metadata(ds));assert len(rows)==spec['raw_count']
    indices=[r['raw_index'] for r in spec['rows']]
    assert all(text_hash(rows[r['raw_index']])==r['metadata_sha256'] for r in spec['rows'])
    return ds.select(indices)
def scienceqa(ds):return process(ds,'scienceqa')
def mme(ds):return process(ds,'mme')
def mmmu(ds):return process(ds,'mmmu')
def pope(ds):return process(ds,'pope')
def seed_img(ds):return process(ds,'seed_img')

def yes_no(text):
    # Freeze this parser before outcomes: accept only a leading yes/no token.
    import re
    m=re.match(r'^\s*(yes|no)\b',text,re.I)
    return m.group(1).capitalize() if m else 'Invalid'
