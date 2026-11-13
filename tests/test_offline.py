"""Synthetic contract checks. No source articles, models, network or training."""
from pathlib import Path
import json,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'llm_study/implementation'))
from score import evaluate
from common import LABELS
from analysis import assignment_objective
from study import compile_response

class OfflineContracts(unittest.TestCase):
    def test_exact_occurrence(self):
        r=compile_response('ab ab',json.dumps([['ab',1,'LOCATION']]))
        self.assertEqual(r['label'],[[3,5,'LOCATION']])
    def test_no_json_repair(self):
        r=compile_response('ab','```json\n[]\n```')
        self.assertTrue(r['whole_response_invalid'])
    def test_reject_bad_occurrence(self):
        r=compile_response('ab',json.dumps([['ab',2,'LOCATION']]))
        self.assertEqual(r['label'],[])
        self.assertTrue(r['structural_rejection'])
    def test_full_denominator_and_reordered_ids(self):
        ref=[{'id':'a','sentence_length':2,'entities':[dict(start=0,end=2,label='LOCATION')]},{'id':'b','sentence_length':2,'entities':[dict(start=0,end=2,label='TIME')]}]
        pred=[{'id':'b','label':[]},{'id':'a','label':[[0,2,'LOCATION']]}]
        self.assertAlmostEqual(evaluate(ref,pred)['strict']['micro']['f1'],2/3)
    def test_missing_or_duplicate_prediction_ids_fail(self):
        ref=[{'id':'a','label':[]}]
        for pred in [[],[{'id':'a','label':[]},{'id':'a','label':[]}]]:
            with self.assertRaises(ValueError):evaluate(ref,pred)
    def test_maximum_intersection_not_maximum_cardinality(self):
        self.assertEqual(assignment_objective([[0,1,'LOCATION'],[1,102,'LOCATION']],[[0,100,'LOCATION'],[101,102,'LOCATION']]),(99,1))
    def test_overlap_tie_prefers_more_pairs(self):
        self.assertEqual(assignment_objective([[0,1,'LOCATION'],[1,5,'LOCATION']],[[0,3,'LOCATION'],[4,5,'LOCATION']]),(2,2))
    def test_hash_mismatch_fails(self):
        ref=[{'id':'a','sentence_sha256':'a'*64,'label':[]}]
        with self.assertRaises(ValueError):evaluate(ref,[{'id':'a','text_sha256':'b'*64,'label':[]}])
    def test_live_study_disabled(self):
        c=json.loads((ROOT/'llm_study/supplied/config/study.json').read_text(encoding='utf-8'))
        self.assertIs(c['live_execution_enabled'],False)
        self.assertEqual(len(LABELS),10)

if __name__=='__main__':unittest.main()
