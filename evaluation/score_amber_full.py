"""Run pinned AMBER unchanged, instrumenting counters for paired analyses."""
import argparse
import ast
import contextlib
import io
import sys
from full_common import *
from full_tasks import yes_no

def main():
    a=argparse.ArgumentParser();a.add_argument('--model',choices=MODELS,required=True);args=a.parse_args();verify()
    m=args.model;assert complete(m,'amber') and complete(m,'amber_disc')
    gen=read_jsonl(output(m,'amber')/'predictions.jsonl');disc=read_jsonl(output(m,'amber_disc')/'predictions.jsonl')
    out=ROOT/'results'/m/'amber_scores';out.mkdir(exist_ok=True)
    official=SUITE/'vendor/amber/inference.py'
    combined=[{'id':int(r['id']),'response':r['response']} for r in gen]+[{'id':int(r['id']),'response':yes_no(r['response'])} for r in disc]
    assert len(combined)==15220 and {r['id'] for r in combined}==set(range(1,15221))
    atomic_json(out/'official_input.json',sorted(combined,key=lambda r:r['id']))
    previous={};samples=[];totals={}
    def record(i,metrics):
        numeric={k:float(v) for k,v in metrics.items() if isinstance(v,(int,float))}
        samples.append({'id':int(i),'metrics':{k:v-previous.get(k,0.) for k,v in numeric.items()}});previous.update(numeric)
    tree=ast.parse(official.read_text(),str(official))
    class Instrument(ast.NodeTransformer):
        count=0
        def visit_AugAssign(self,node):
            v=node.target
            if isinstance(v,ast.Subscript) and isinstance(v.value,ast.Name) and v.value.id=='metrics' and isinstance(v.slice,ast.Constant) and v.slice.value=='non_hallu_num':
                self.count+=1;return [node,ast.copy_location(ast.parse('_record(id,metrics)').body[0],node)]
            return node
        def visit_FunctionDef(self,node):
            node=self.generic_visit(node)
            if node.name=='main':node.body.append(ast.parse('_totals.update(metrics)').body[0])
            return node
    inst=Instrument();tree=inst.visit(tree);assert inst.count==1;ast.fix_missing_locations(tree)
    sys.argv=[str(official),'--word_association',str(SUITE/'vendor/amber/data/relation.json'),'--safe_words',str(SUITE/'vendor/amber/data/safe_words.txt'),'--annotation',str(SUITE/'vendor/amber/data/annotations.json'),'--metrics',str(SUITE/'vendor/amber/data/metrics.txt'),'--inference_data',str(out/'official_input.json'),'--evaluation_type','a']
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(compile(tree,str(official),'exec'),{'__name__':'__main__','__file__':str(official),'_record':record,'_totals':totals})
    text=stream.getvalue();print(text);(out/'official_stdout.txt').write_text(text)
    assert len(samples)==len({r['id'] for r in samples})==1004
    from hallucination_eval.score_amber import _metric
    metrics={k:_metric(k,text) for k in ['CHAIR','Cover','Hal','Cog']}
    exact={'CHAIR':100*totals['chair_score']/totals['chair_num'],'Cover':100*totals['safe_cover_score']/totals['safe_cover_num'],'Hal':100*(1-totals['non_hallu_score']/totals['non_hallu_num']),'Cog':100*totals['hallu_cover_score']/totals['hallu_cover_num']}
    assert all(abs(exact[k]-metrics[k])<.051 for k in exact)
    (out/'per_image.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in samples))
    sections={};section=None
    for line in text.splitlines():
        parts=line.split(':',1)
        if len(parts)!=2:continue
        label,value=parts[0].strip(),parts[1].strip()
        if not value:section=label;sections.setdefault(section,{})
        elif section:
            try:sections[section][label]=float(value)
            except ValueError:pass
    atomic_json(out/'summary.json',{'status':'completed','samples':15220,'generative_samples':1004,'discriminative_samples':14216,'generative':metrics,'unrounded_generative':exact,'official_sections':sections,'official_counters':totals,'invalid_discriminative_answers':sum(yes_no(r['response'])=='Invalid' for r in disc),'parser':'leading yes/no token; invalid remains incorrect; official F1 treats No as positive','input_lock_sha256':sha256(ROOT/'input_lock.json')})

if __name__=='__main__':main()
