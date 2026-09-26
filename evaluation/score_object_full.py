"""Existing Object HalBench scorer plus preservation of official per-image data."""
import argparse
import ast
import sys
from full_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);a=p.parse_args();verify()
    out=output(a.model,'object_halbench');source=CODE/'hallucination_eval/score_object_halbench.py'
    tree=ast.parse(source.read_text(),str(source))
    class Instrument(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='metrics' for t in node.targets):
                self.count+=1;return [node,ast.copy_location(ast.parse('_save(metrics)').body[0],node)]
            return node
    inst=Instrument();tree=inst.visit(tree);assert inst.count==1;ast.fix_missing_locations(tree)
    sys.argv=[str(source),'--manifest',str(ROOT/'manifests/object_halbench.jsonl'),'--predictions',str(out/'predictions.jsonl'),'--extractions',str(out/'extractions.jsonl'),'--official-evaluator',str(SUITE/'vendor/rlhf-v/eval/eval_gpt_obj_halbench.py'),'--coco-annotations',str(SUITE/'assets/coco2014/annotations'),'--output',str(out/'summary.json')]
    exec(compile(tree,str(source),'exec'),{'__name__':'__main__','__file__':str(source),'__package__':'hallucination_eval','_save':lambda data:atomic_json(out/'official_per_image.json',data)})

if __name__=='__main__':main()
