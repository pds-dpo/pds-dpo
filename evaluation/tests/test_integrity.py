import json
import tempfile
import unittest
from pathlib import Path
from hallucination_eval.common import join_predictions

class IntegrityTests(unittest.TestCase):
    def exercise(self,source,answers):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'source.jsonl';b=Path(folder)/'answers.jsonl'
            a.write_text(''.join(json.dumps(x)+'\n' for x in source))
            b.write_text(''.join(json.dumps(x)+'\n' for x in answers))
            return join_predictions(a,[b])
    def test_join_by_id_not_order(self):
        rows=self.exercise([{'id':1},{'id':2}],[{'id':2,'model_id':'A'},{'id':1,'model_id':'A'}])
        self.assertEqual([r[1]['id'] for r in rows],[1,2])
    def test_duplicate_source(self):
        with self.assertRaises(ValueError):self.exercise([{'id':1},{'id':1}],[{'id':1,'model_id':'A'}])
    def test_missing_prediction(self):
        with self.assertRaises(ValueError):self.exercise([{'id':1},{'id':2}],[{'id':1,'model_id':'A'}])
    def test_duplicate_prediction(self):
        with self.assertRaises(ValueError):self.exercise([{'id':1}],[{'id':1,'model_id':'A'},{'id':1,'model_id':'A'}])
    def test_mixed_models(self):
        with self.assertRaises(ValueError):self.exercise([{'id':1},{'id':2}],[{'id':1,'model_id':'A'},{'id':2,'model_id':'B'}])

if __name__=='__main__': unittest.main()
