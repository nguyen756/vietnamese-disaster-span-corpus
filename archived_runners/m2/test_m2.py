import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import run_m2 as runner

CONFIG={'model':'gpt-6-luna','max_output_tokens':1024,'temperature':0,'top_p':1,'prior_completed_pilot_reservations_usd':'.184320','reservation_per_request_usd':'.004608','reservation_ceiling_usd':'10'}
JOB={'id':'dev_synthetic','text':'Huế','text_sha256':runner.sha('Huế'.encode()),'run_id':'M2-DEV-CANARY','prompt_id':'P2','phase':'development'}
def response():return {'http_status':200,'elapsed_seconds':.1,'error':None,'request_id':'synthetic','retry_after':None,'data':{'status':'completed','model':'gpt-6-luna','temperature':0,'top_p':1,'usage':{'input_tokens':10,'output_tokens':5,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}},'output':[{'type':'message','content':[{'type':'output_text','text':'[["Huế",0,"LOCATION"]]'}]}]}}
class Checks(unittest.TestCase):
    def test_payload_is_blind_and_settings_explicit(self):
        body=runner.body(JOB,{'P2':'fixed examples'},CONFIG)
        self.assertEqual(json.loads(body['input']),{'sentence':'Huế'})
        self.assertEqual(body['temperature'],0);self.assertEqual(body['top_p'],1)
        self.assertNotIn('seed',body);self.assertFalse(body['store'])
    def test_raw_compile_and_usage(self):
        result=runner.decode(JOB,response(),CONFIG)
        self.assertEqual(result['label'],[[0,3,'LOCATION']]);self.assertEqual(result['outcome'],'valid')
        self.assertEqual(runner.Decimal(result['calculated_cost_usd']),runner.Decimal('.0000035'))
    def test_sampling_canary_must_be_verified(self):
        value=response();del value['data']['temperature']
        self.assertTrue(runner.fatal(runner.decode(JOB,value,CONFIG)))
    def test_failed_canary_prevents_Gold_and_repeat(self):
        with tempfile.TemporaryDirectory()as directory,patch.object(runner,'ROOT',Path(directory)),contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);(root/'MANIFEST.json').write_text('{}');out=root/'out'
            transport=Mock(return_value={'http_status':429,'elapsed_seconds':.1,'data':None,'error':{'code':'synthetic'}})
            jobs=[JOB,{**JOB,'run_id':'M2-P1','phase':'gold'}]
            runner.execute(jobs,{'P2':'P2'},CONFIG,'synthetic',transport,out)
            self.assertEqual(transport.call_count,1)
            with self.assertRaises(ValueError):runner.execute(jobs,{'P2':'P2'},CONFIG,'synthetic',transport,out)
            self.assertEqual(transport.call_count,1)
            self.assertIsNone(runner.read(out/'result_0000.json')['calculated_cost_usd'])
    def test_resume_reuses_completed_requests(self):
        with tempfile.TemporaryDirectory()as directory,patch.object(runner,'ROOT',Path(directory)),contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);(root/'MANIFEST.json').write_text('{}');out=root/'out';transport=Mock(return_value=response())
            for _ in range(2):runner.execute([JOB],{'P2':'P2'},CONFIG,'synthetic',transport,out)
            self.assertEqual(transport.call_count,1)
    def test_crash_claim_blocks_duplicate(self):
        with tempfile.TemporaryDirectory()as directory,patch.object(runner,'ROOT',Path(directory)),contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);(root/'MANIFEST.json').write_text('{}');out=root/'out'
            def transport(*args):
                self.assertTrue((out/'claim_0000.json').exists());raise RuntimeError('simulated interruption')
            with self.assertRaises(RuntimeError):runner.execute([JOB],{'P2':'P2'},CONFIG,'synthetic',transport,out)
            with self.assertRaises(ValueError):runner.execute([JOB],{'P2':'P2'},CONFIG,'synthetic',Mock(),out)
    def test_budget_carries_prior_pilots(self):
        with tempfile.TemporaryDirectory()as directory,patch.object(runner,'ROOT',Path(directory)),contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);(root/'MANIFEST.json').write_text('{}');transport=Mock()
            with self.assertRaises(ValueError):runner.execute([JOB],{'P2':'P2'},{**CONFIG,'prior_completed_pilot_reservations_usd':'9.999'},'synthetic',transport,root/'out')
            transport.assert_not_called()
    def test_partial_export_keeps_full_denominator(self):
        with tempfile.TemporaryDirectory()as directory,patch.object(runner,'ROOT',Path(directory)),contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);(root/'MANIFEST.json').write_text('{"files":{}}');(root/'m2_luna').mkdir()
            jobs=[JOB,{**JOB,'run_id':'M2-P1','phase':'gold'},{**JOB,'run_id':'M2-P2','phase':'gold'}]
            runner.export(jobs)
            folder=next((root/'m2_luna/results').glob('export_*'));status=runner.read(folder/'STATUS.json')
            self.assertFalse(status['runs']['M2-P1']['complete']);self.assertEqual(status['runs']['M2-P1']['outcomes'],{'not_run':1})
    def test_canary_two_prompt_execute_export_end_to_end(self):
        with tempfile.TemporaryDirectory()as directory,patch.object(runner,'ROOT',Path(directory)),contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);(root/'MANIFEST.json').write_text('{"files":{}}');(root/'m2_luna').mkdir()
            jobs=[JOB,{**JOB,'run_id':'M2-P1','prompt_id':'P1','phase':'gold'},{**JOB,'run_id':'M2-P2','phase':'gold'}]
            transport=Mock(side_effect=[response()for _ in jobs])
            runner.execute(jobs,{'P1':'fixed P1','P2':'fixed P2'},CONFIG,'synthetic',transport)
            runner.export(jobs)
            self.assertEqual(transport.call_count,3)
            self.assertEqual([c.args[3]['instructions']for c in transport.call_args_list],['fixed P2','fixed P1','fixed P2'])
            folder=next((root/'m2_luna/results').glob('export_*'));status=runner.read(folder/'STATUS.json')
            self.assertTrue(all(r['complete']for r in status['runs'].values()))
            self.assertFalse(status['full_E1_grid_complete']);self.assertFalse(status['reference_labels_read'])
            with runner.zipfile.ZipFile(next(root.glob('E1_M2_return_*.zip')))as z:
                self.assertIsNone(z.testzip())
                for name,expected in json.loads(z.read('RETURN_MANIFEST.json')).items():self.assertEqual(runner.sha(z.read(name)),expected)
if __name__=='__main__':unittest.main(verbosity=2)
