"""Synthetic offline tests; no model generation, references or provider calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import full_cpu_driver as driver


def source(i=0):
    text='😀 lũ ở Huế'
    return {'id':f'SYNTHETIC_{i:04d}','text':text,'text_sha256':driver.hashlib.sha256(text.encode()).hexdigest()}


class Fake:
    def __init__(self,response='[["lũ",0,"DISASTER_TYPE"]]',finish='eos',failure=None):
        self.calls=0;self.response=response;self.finish=finish;self.failure=failure
    def prepare(self,row,prompt):
        return {'messages':driver.core.messages(row,prompt),'rendered_prompt':'SYNTHETIC_TEMPLATE',
                'input_token_ids':[1,2],'input_tokens':2,'official_tokenizer_match':True}
    def generate(self,req):
        self.calls+=1
        # A permanent claim must already be fsynced/visible before generation.
        assert len(list((driver.OUTPUT/'ledger').glob('*/claim_*.json')))>=1
        if self.failure:raise self.failure
        return dict(response=self.response,input_tokens=2,output_tokens=2,output_token_ids=[3,4],
                    stopped_on_eos=self.finish=='eos',finish_reason=self.finish,
                    elapsed_seconds=1.25,cpu_process_seconds=3.5,peak_process_rss_kib=100)


class FullTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name);(self.out/'ledger').mkdir();(self.out/'predictions').mkdir()
        p=patch.object(driver,'OUTPUT',self.out);p.start();self.addCleanup(p.stop)
        p=patch.object(driver,'prompt',return_value='UNCHANGED_FIXED_PROMPT');p.start();self.addCleanup(p.stop)
    def attempt(self,fake=None,lock='LOCK',run='M1-P1',i=0):
        return driver.execute_one(i,source(i),'P1',run,fake or Fake(),lock,'SYNTHETIC_SESSION')

    def test_payload_contains_only_exact_sentence_and_fixed_prompt(self):
        row=dict(source(),label=['MUST_NOT_ENTER'],document_id='HIDDEN_META')
        req=Fake().prepare(row,'FIXED')
        self.assertEqual(req['messages'][0],{'role':'system','content':'FIXED'})
        self.assertEqual(json.loads(req['messages'][1]['content']),{'sentence':source()['text']})
        self.assertEqual(len(req['messages']),2)

    def test_source_unicode_offsets_and_request_hash(self):
        value,_=self.attempt()
        self.assertEqual(value['label'],[[2,4,'DISASTER_TYPE']])
        self.assertEqual(value['text'][2:4],'lũ')
        self.assertEqual(value['request_record_sha256'],driver.core.file_hash(self.out/'ledger/M1-P1/claim_0000.json'))

    def test_occurrence_string_is_rejected_without_repair(self):
        v,_=self.attempt(Fake('[["lũ","0","DISASTER_TYPE"]]'))
        self.assertEqual(v['label'],[]);self.assertEqual(v['outcome'],'partially_rejected')
        self.assertEqual(v['rejections'][0]['reason'],'invalid_occurrence')

    def test_malformed_is_observation_and_not_retried(self):
        fake=Fake('not JSON');v,resumed=self.attempt(fake)
        self.assertEqual(v['outcome'],'malformed');self.assertFalse(resumed)
        _,resumed=self.attempt(fake);self.assertTrue(resumed);self.assertEqual(fake.calls,1)

    def test_capped_output_accepts_no_spans(self):
        v,_=self.attempt(Fake(finish='length'))
        self.assertEqual(v['outcome'],'truncated');self.assertEqual(v['label'],[])
        self.assertTrue(v['structural_rejection'])

    def test_completed_resume_does_not_generate(self):
        fake=Fake();first,_=self.attempt(fake);second,resumed=self.attempt(fake)
        self.assertEqual(first,second);self.assertTrue(resumed);self.assertEqual(fake.calls,1)

    def test_changed_runtime_identity_blocks_before_generation(self):
        fake=Fake();self.attempt(fake)
        with self.assertRaisesRegex(ValueError,'identity differs'):self.attempt(fake,lock='CHANGED')
        self.assertEqual(fake.calls,1)

    def test_corrupt_claim_blocks_resume(self):
        fake=Fake();self.attempt(fake);p=self.out/'ledger/M1-P1/claim_0000.json'
        c=driver.core.read(p);c['request']['input_token_ids']=[99];p.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'identity differs'):self.attempt(fake)
        self.assertEqual(fake.calls,1)

    def test_corrupt_generation_blocks_resume(self):
        fake=Fake();self.attempt(fake);p=self.out/'ledger/M1-P1/result_0000.json'
        c=driver.core.read(p);c['generation']['response']='CORRUPT';p.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'integrity differs'):self.attempt(fake)
        self.assertEqual(fake.calls,1)

    def test_unresolved_claim_blocks_and_is_not_removed(self):
        directory=self.out/'ledger/M1-P1';directory.mkdir();p=directory/'claim_0000.json';driver.core.save(p,{'crashed':True})
        fake=Fake()
        with self.assertRaisesRegex(ValueError,'Unresolved claim'):self.attempt(fake)
        self.assertEqual(fake.calls,0);self.assertTrue(p.exists())
        with self.assertRaisesRegex(ValueError,'Unresolved claim'):driver.preexisting_failures()

    def test_runtime_failure_is_permanent_and_cost_unknown(self):
        fake=Fake(failure=RuntimeError('SYNTHETIC FAILURE'))
        with self.assertRaisesRegex(RuntimeError,'SYNTHETIC'):self.attempt(fake)
        v=driver.core.read(self.out/'ledger/M1-P1/result_0000.json')
        self.assertEqual(v['outcome'],'runtime_failed');self.assertIsNone(v['local_inference_cost_usd'])
        with self.assertRaisesRegex(ValueError,'blocks generation'):self.attempt(fake)
        self.assertEqual(fake.calls,1)

    def test_crash_interrupt_keeps_claim_and_failure(self):
        fake=Fake(failure=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):self.attempt(fake)
        with self.assertRaisesRegex(ValueError,'blocks generation'):self.attempt(fake)
        self.assertEqual(fake.calls,1)

    def test_orphan_result_without_claim_blocks_resume(self):
        fake=Fake();self.attempt(fake);(self.out/'ledger/M1-P1/claim_0000.json').unlink()
        with self.assertRaisesRegex(ValueError,'identity differs'):self.attempt(fake)
        self.assertEqual(fake.calls,1)

    def test_run_names_have_distinct_claims(self):
        fake=Fake();a,_=self.attempt(fake,run='M1-P1');b,_=self.attempt(fake,run='M1-P2')
        self.assertNotEqual(a['request_sha256'],b['request_sha256']);self.assertEqual(fake.calls,2)

    def test_full_denominator_and_partial_flags(self):
        data=[source(i) for i in range(1000)]
        self.attempt();directory=self.out/'ledger/M1-P1';driver.core.save(directory/'claim_0001.json',{'pending':True})
        with contextlib.redirect_stdout(io.StringIO()):summary=driver.export_run(data,'M1-P1',self.out/'export')
        values=[json.loads(s)for s in (self.out/'export/M1-P1.jsonl').read_text().splitlines()]
        self.assertEqual(len(values),1000);self.assertEqual([r['id']for r in values],[r['id']for r in data])
        self.assertFalse(summary['complete']);self.assertEqual(summary['completed_generations'],1)
        self.assertEqual(values[1]['outcome'],'unresolved_claim');self.assertEqual(values[2]['outcome'],'not_run')
        self.assertIsNone(summary['local_inference_cost_usd']);self.assertIsNone(summary['cached_input_tokens'])
        self.assertIsNotNone(values[0]['request_record_sha256']);self.assertIsNotNone(values[0]['response_record_sha256'])

    def test_lock_change_cannot_overwrite_seal(self):
        p=self.out/'lock.json';driver.lock_once(p,{'version':1});before=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'Sealed lock differs'):driver.lock_once(p,{'version':2})
        self.assertEqual(p.read_bytes(),before)

    def test_exact_blind_hash_and_fields_required(self):
        p=self.out/'blind.jsonl';value=source();value['label']=[];p.write_text(json.dumps(value)+'\n')
        with self.assertRaisesRegex(ValueError,'Blind fields'):driver.rows(p,1)
        value=source();value['text']='CHANGED';p.write_text(json.dumps(value)+'\n')
        with self.assertRaisesRegex(ValueError,'hash changed'):driver.rows(p,1)

    def test_output_path_is_fixed_and_gold_false_lock_is_preserved(self):
        old=driver.core.read(driver.HERE/'pilot_cpu_20_P2/RUNTIME_LOCK.json')
        self.assertIs(old['gold_generation_allowed'],False)
        self.assertNotIn('--output',Path(driver.__file__).read_text())


if __name__=='__main__':unittest.main(verbosity=2)
