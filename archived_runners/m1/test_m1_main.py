"""Synthetic, offline tests for main reporting, accounting and durable attempts."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import run_m1_cpu as main
import m1_reporting as report
import pilot_cpu as core


def row(index=0):
    text = 'lũ ở Huế'
    return dict(id=f'SYNTHETIC_{index:04d}', text=text,
                text_sha256=core.hashlib.sha256(text.encode()).hexdigest(),
                split='gold_v1', input_row_index_1based=index+1,
                document_id=None, original_row_index=None, source_index_base=None)


class Fake:
    def __init__(self, response='[["lũ",0,"DISASTER_TYPE"]]', finish='eos', failure=False):
        self.calls = 0; self.response = response; self.finish = finish; self.failure = failure

    def prepare(self, source, prompt):
        return dict(messages=core.messages(source, prompt), input_token_ids=[1,2,3],
                    input_tokens=3, official_tokenizer_match=True)

    def generate(self, request):
        self.calls += 1
        if self.failure:
            raise RuntimeError('SYNTHETIC model failure')
        return dict(response=self.response, input_tokens=3, output_tokens=2, output_token_ids=[4,5],
                    finish_reason=self.finish, stopped_on_eos=self.finish == 'eos',
                    elapsed_seconds=2.0, cpu_process_seconds=10.0, peak_process_rss_kib=1024)


class ReportingTests(unittest.TestCase):
    def accounting(self, text, raw, truncated=False):
        c = core.classify(text, raw, truncated)
        return report.item_accounting(text, raw, c, truncated)

    def test_sentence_and_item_denominators_differ(self):
        value = self.accounting('lũ ở Huế', '[["lũ",0,"DISASTER_TYPE"],["Huế","0","LOCATION"]]')
        self.assertEqual(value['parsed_items'], 2)
        self.assertEqual(value['accepted_spans'], 1)
        self.assertEqual(value['rejected_items'], 1)
        self.assertEqual(value['rejected_item_rate_pct'], 50)
        self.assertEqual(value['rejected_by_reason'], {'invalid_occurrence':1})

    def test_empty_model_and_all_rejected_are_distinct(self):
        empty = self.accounting('lũ', '[]')
        rejected = self.accounting('lũ', '[["lũ",99,"DISASTER_TYPE"]]')
        self.assertTrue(empty['model_returned_empty_array'])
        self.assertFalse(empty['all_items_rejected'])
        self.assertIsNone(empty['rejected_item_rate_pct'])
        self.assertFalse(rejected['model_returned_empty_array'])
        self.assertTrue(rejected['all_items_rejected'])
        self.assertEqual(rejected['rejected_item_rate_pct'], 100)

    def test_malformed_has_no_parsed_item_denominator(self):
        for raw in ['bad JSON', '{"not":"array"}', '']:
            value = self.accounting('lũ', raw)
            self.assertFalse(value['json_array_valid'])
            self.assertEqual(value['parsed_items'], 0)
            self.assertEqual(value['rejected_items'], 0)
            self.assertIsNone(value['rejected_item_rate_pct'])

    def test_truncation_discard_is_separate_from_compiler_rejection(self):
        value = self.accounting('lũ ở Huế', '[["lũ",0,"DISASTER_TYPE"],["Huế","0","LOCATION"]]', True)
        self.assertEqual(value['accepted_spans'], 0)
        self.assertEqual(value['rejected_items'], 1)
        self.assertEqual(value['discarded_items_due_to_truncation'], 1)

    def test_console_contains_raw_output_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            value, _ = main.execute_one(0, row(), 'P', Fake(), Path(d), 'M1-P2', 'LOCK', 'S')
            summary = report.summarize([row()], {row()['id']:value}, 'M1-P2')
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                report.print_record(value, 1, 1, summary)
            text = capture.getvalue()
            for expected in ['TOKENS', 'TIME', 'SPANS', 'REJECTION_REASONS', 'PROGRESS',
                             'PROVENANCE', 'RAW_RESPONSE', 'ACCEPTED', 'REJECTED', 'lũ',
                             'document_id', 'request_sha256', 'runtime_lock_sha256', 'cached=NA',
                             'TIMESTAMPS', 'structural_rejection', 'cost_USD=NA']:
                self.assertIn(expected, text)


class AttemptTests(unittest.TestCase):
    def test_resume_does_not_infer_and_changed_lock_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Fake(); args = (0,row(),'P',fake,Path(d),'M1-P2','LOCK','S')
            first, resumed = main.execute_one(*args)
            self.assertFalse(resumed)
            second, resumed = main.execute_one(*args)
            self.assertTrue(resumed); self.assertEqual(first, second); self.assertEqual(fake.calls, 1)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                main.execute_one(0,row(),'P',fake,Path(d),'M1-P2','CHANGED','S')
            self.assertEqual(fake.calls, 1)

    def test_unresolved_claim_blocks_inference(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Fake(); out = Path(d)
            core.save(out/'claim_0000.json', {'interrupted':True})
            with self.assertRaisesRegex(ValueError, 'Unresolved claim'):
                main.execute_one(0,row(),'P',fake,out,'M1-P2','LOCK','S')
            self.assertEqual(fake.calls, 0)
            self.assertEqual(main.collect([row()],out)[row()['id']]['outcome'], 'unresolved_claim')

    def test_runtime_failure_is_saved_and_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Fake(failure=True); out = Path(d)
            with self.assertRaisesRegex(RuntimeError, 'SYNTHETIC'):
                main.execute_one(0,row(),'P',fake,out,'M1-P2','LOCK','S')
            saved = core.read(out/'result_0000.json')
            self.assertEqual(saved['outcome'], 'runtime_failed')
            self.assertTrue(saved['record_complete'])
            with self.assertRaisesRegex(ValueError, 'blocks retry'):
                main.execute_one(0,row(),'P',fake,out,'M1-P2','LOCK','S')
            self.assertEqual(fake.calls, 1)

    def test_cap_never_accepts_spans(self):
        with tempfile.TemporaryDirectory() as d:
            value, _ = main.execute_one(0,row(),'P',Fake(finish='length'),Path(d),'M1-P2','LOCK','S')
            self.assertEqual(value['outcome'], 'truncated')
            self.assertEqual(value['label'], [])
            self.assertEqual(value['discarded_items_due_to_truncation'], 1)

    def test_corrupt_claim_blocks_resume(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Fake(); out = Path(d)
            main.execute_one(0,row(),'P',fake,out,'M1-P2','LOCK','S')
            claim = core.read(out/'claim_0000.json');claim['request']['input_token_ids'] = [99]
            (out/'claim_0000.json').write_text(json.dumps(claim))
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                main.execute_one(0,row(),'P',fake,out,'M1-P2','LOCK','S')
            self.assertEqual(fake.calls, 1)

    def test_full_1000_denominator_and_unknown_values(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); rows = [row(i) for i in range(1000)]
            main.execute_one(0,rows[0],'P',Fake(),out,'M1-P2','LOCK','S')
            main.execute_one(1,rows[1],'P',Fake(response='[]'),out,'M1-P2','LOCK','S')
            core.save(out/'claim_0002.json', {'interrupted':True})
            records = main.collect(rows,out)
            summary = report.summarize(rows,records,'M1-P2')
            self.assertEqual(summary['outcomes'], {'valid':2,'unresolved_claim':1,'not_run':997})
            self.assertEqual(summary['inference_coverage_pct'], .2)
            self.assertEqual(summary['model_empty_array_sentences'], 1)
            self.assertEqual(summary['cpu_process_seconds'], 20)
            self.assertEqual(summary['inference_seconds'], 4)
            for field in ['local_inference_cost_usd','cached_input_tokens','strict_f1','overlap_f1','gpu_minutes']:
                self.assertIsNone(summary[field])
            self.assertFalse(summary['document_bootstrap_metadata_ready'])
            with contextlib.redirect_stdout(io.StringIO()):
                target = main.export(rows,records,out,'M1-P2',summary)
            exported = [json.loads(s) for s in (target/'M1-P2.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in exported], [r['id'] for r in rows])
            self.assertEqual(exported[2]['outcome'], 'unresolved_claim')
            with (target/'requests_summary.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1000)

    def test_source_metadata_never_enters_messages(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [row()]; value = dict(id=rows[0]['id'],text_sha256=rows[0]['text_sha256'],
                                       document_id='DOC',original_row_index=11,source_index_base=1)
            path = Path(d)/'metadata.jsonl'; path.write_text(json.dumps(value)+'\n')
            main.attach_metadata(rows,path)
            self.assertEqual(rows[0]['document_id'], 'DOC')
            self.assertEqual(json.loads(core.messages(rows[0],'P')[1]['content']), {'sentence':'lũ ở Huế'})
            summary = report.summarize(rows,{},'M1-P2')
            self.assertTrue(summary['document_bootstrap_metadata_ready'])
            value['label'] = [];path.write_text(json.dumps(value)+'\n')
            with self.assertRaisesRegex(ValueError, 'annotations are forbidden'):
                main.attach_metadata(rows,path)


class MainLoopTests(unittest.TestCase):
    def lock(self):
        return dict(model_source={'synthetic':True},settings={'synthetic':True},cpu='SYNTHETIC',
                    logical_cpus=1,versions={},prompt_sha256={},input_sha256='SYNTHETIC',
                    metadata={},reporting_protocol={'synthetic':True})

    def test_end_to_end_loop_exports_every_synthetic_row(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [row(i) for i in range(3)]; backend = Fake(); output = Path(d)
            with patch.object(main,'OUTPUT',output), patch.object(main,'make_lock',return_value=self.lock()), \
                 patch.object(core,'CPUModel',return_value=backend), contextlib.redirect_stdout(io.StringIO()):
                main.run(rows,{'P2':'P'},'M1-P2',None)
            self.assertEqual(backend.calls,3)
            out = output/'M1-P2'
            status = core.read(next((out/'exports').glob('*/STATUS.json')))
            self.assertTrue(status['complete'])
            self.assertEqual(status['session']['new_requests'],3)
            self.assertEqual(status['session']['skipped_completed_requests'],0)

    def test_failed_generation_counted_in_session_and_full_export(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [row(i) for i in range(3)]; backend = Fake(failure=True); output = Path(d)
            with patch.object(main,'OUTPUT',output), patch.object(main,'make_lock',return_value=self.lock()), \
                 patch.object(core,'CPUModel',return_value=backend), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'SYNTHETIC'):
                    main.run(rows,{'P2':'P'},'M1-P2',None)
            self.assertEqual(backend.calls,1)
            status = core.read(next((output/'M1-P2/exports').glob('*/STATUS.json')))
            self.assertEqual(status['session']['new_requests'],1)
            self.assertEqual(status['outcomes'],{'runtime_failed':1,'not_run':2})
            self.assertFalse(status['complete'])

    def test_context_failure_stops_all_generation_and_identifies_row(self):
        class Overflow(Fake):
            def prepare(self, source, prompt):
                if source['id']==row(1)['id']:
                    raise ValueError('SYNTHETIC context overflow')
                return super().prepare(source,prompt)
        with tempfile.TemporaryDirectory() as d:
            rows = [row(i) for i in range(3)]; backend = Overflow(); output = Path(d)
            with patch.object(main,'OUTPUT',output), patch.object(main,'make_lock',return_value=self.lock()), \
                 patch.object(core,'CPUModel',return_value=backend), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, row(1)['id']):
                    main.run(rows,{'P2':'P'},'M1-P2',None)
            self.assertEqual(backend.calls,0)
            status = core.read(next((output/'M1-P2/exports').glob('*/STATUS.json')))
            self.assertEqual(status['outcomes'],{'not_run':3})
            self.assertEqual(status['session']['new_requests'],0)


if __name__ == '__main__':
    unittest.main()
