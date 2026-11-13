"""Offline checks with synthetic examples, never reported as model results."""
import hashlib
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import pilot_cpu as p


class Fake:
    def __init__(self):
        self.calls = 0

    def prepare(self, row, prompt):
        return dict(messages=p.messages(row, prompt), input_token_ids=[1, 2], input_tokens=2)

    def generate(self, request):
        self.calls += 1
        return dict(response='[["lũ", 0, "DISASTER_TYPE"]]', output_token_ids=[3, 4],
                    input_tokens=2, output_tokens=2, finish_reason='eos', elapsed_seconds=0.01)


class PilotTests(unittest.TestCase):
    def row(self):
        text = 'lũ ở Huế; lũ'
        return dict(id='SYNTHETIC', text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def test_payload_exact_unicode_no_labels(self):
        msg = p.messages(self.row(), 'UNCHANGED')
        self.assertEqual(msg[0], dict(role='system', content='UNCHANGED'))
        self.assertEqual(json.loads(msg[1]['content']), {'sentence': self.row()['text']})

    def test_numeric_string_strictly_rejected(self):
        r = p.classify('lũ ở Huế', '[["lũ","0","DISASTER_TYPE"],["Huế",0,"LOCATION"]]')
        self.assertEqual(r['outcome'], 'partially_rejected')
        self.assertEqual(r['label'], [[5, 8, 'LOCATION']])
        self.assertEqual(r['rejections'][0]['reason'], 'invalid_occurrence')

    def test_overlap_first_item_wins(self):
        r = p.classify('tránh lũ', '[["tránh lũ",0,"RESPONSE_ACTION"],["lũ",0,"DISASTER_TYPE"]]')
        self.assertEqual(r['label'], [[0, 8, 'RESPONSE_ACTION']])
        self.assertEqual(r['rejections'][0]['reason'], 'overlap_or_duplicate')

    def test_capped_valid_json_never_accepted(self):
        r = p.classify('lũ', '[["lũ",0,"DISASTER_TYPE"]]', truncated=True)
        self.assertEqual(r['outcome'], 'truncated')
        self.assertEqual(r['label'], [])

    def test_malformed_and_empty(self):
        self.assertEqual(p.classify('lũ', 'not JSON')['outcome'], 'malformed')
        self.assertEqual(p.classify('lũ', '[]')['outcome'], 'valid')

    def test_completed_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); fake = Fake(); row = self.row()
            first = p.execute_one(0, row, 'PROMPT', fake, out)
            self.assertEqual(first, p.execute_one(0, row, 'PROMPT', fake, out))
            self.assertEqual(fake.calls, 1)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                p.execute_one(0, row, 'CHANGED', fake, out)
            self.assertEqual(fake.calls, 1)

    def test_unresolved_claim_blocks_generation(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); fake = Fake()
            p.save(out / 'claim_0000.json', {'interrupted': True})
            with self.assertRaisesRegex(ValueError, 'Unresolved prior claim'):
                p.execute_one(0, self.row(), 'PROMPT', fake, out)
            self.assertEqual(fake.calls, 0)

    def test_missing_jobs_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [dict(self.row(), id=f'SYNTHETIC_{i}') for i in range(20)]
            with contextlib.redirect_stdout(io.StringIO()):
                report = p.summarize(Path(d), rows)
            self.assertEqual(report['configured_requests'], 20)
            self.assertEqual(report['outcomes'], {'not_run': 20})
            exported = list(Path(d).glob('predictions_*.jsonl'))[0]
            self.assertEqual(len(exported.read_text().splitlines()), 20)


if __name__ == '__main__':
    unittest.main()
