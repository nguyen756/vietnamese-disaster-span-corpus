"""Reporting for local M1 inference; never reads reference annotations."""
import datetime
import json
import math
import statistics
from collections import Counter


def percentage(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def ratio_text(numerator, denominator):
    rate = percentage(numerator, denominator)
    return f'{numerator}/{denominator} ({rate:.2f}%)' if rate is not None else f'{numerator}/{denominator} (NA)'


def number_text(value, scale=1):
    return f'{value/scale:.3f}' if value is not None else 'NA'


def item_accounting(text, raw, compiled, truncated=False):
    try:
        items = json.loads(raw)
    except (TypeError, ValueError):
        items = None
    array = isinstance(items, list)
    rejected = sum(r.get('item_index') is not None for r in compiled['rejections'])
    count = len(items) if array else 0
    kept = len(compiled['label'])
    return dict(json_array_valid=array, raw_response_empty=not raw.strip(),
                model_returned_empty_array=array and not items,
                parsed_items=count, accepted_spans=kept, rejected_items=rejected,
                rejected_item_rate_pct=percentage(rejected, count),
                discarded_items_due_to_truncation=count-rejected if truncated else 0,
                accepted_output_empty=kept == 0,
                all_items_rejected=array and count > 0 and rejected == count,
                rejected_by_reason=dict(Counter(r['reason'] for r in compiled['rejections'])),
                accepted_spans_detail=[dict(start=a, end=b, category=c, text=text[a:b])
                                       for a, b, c in compiled['label']])


def summarize(rows, records, run_id, session=None):
    """Fixed full denominator, including interrupted and failed jobs."""
    values = list(records.values())
    outcomes = Counter(v.get('outcome', 'record_invalid') for v in values)
    outcomes['not_run'] += len(rows)-len(values)
    outcomes = {k: v for k, v in outcomes.items() if v}
    terminal = [v for v in values if v.get('record_complete') is True]
    responses = [v for v in terminal if 'response' in v]
    generated = [v for v in responses if v.get('finish_reason') == 'eos']
    latencies = [v['inference_seconds'] for v in responses if v.get('inference_seconds') is not None]
    parsed = sum(v.get('parsed_items', 0) for v in responses)
    rejected = sum(v.get('rejected_items', 0) for v in responses)
    reasons = Counter()
    for value in responses:
        reasons.update(value.get('rejected_by_reason', {}))
    n = len(rows)
    logical_input = sum(v.get('input_tokens', 0) for v in responses)
    output = sum(v.get('output_tokens', 0) for v in responses)
    infer_seconds = sum(latencies)
    cpu_seconds = sum(v.get('cpu_process_seconds', 0) for v in responses)
    docs = [r.get('document_id') for r in rows]
    source_rows = [r.get('original_row_index') for r in rows]
    return dict(run_id=run_id, expected_sentences=n, primary_run='M1-P2',
                recorded_attempts=len(values), finished_attempts=len(terminal),
                complete=len(terminal) == n,
                inference_coverage_pct=percentage(len(responses), n),
                eos_coverage_pct=percentage(len(generated), n),
                outcomes=outcomes, json_array_valid_sentences=sum(v.get('json_array_valid', False) for v in responses),
                malformed_sentences=sum(v.get('outcome') == 'malformed' for v in values),
                raw_empty_response_sentences=sum(v.get('raw_response_empty', False) for v in responses),
                model_empty_array_sentences=sum(v.get('model_returned_empty_array', False) for v in responses),
                accepted_output_empty_sentences=sum(v.get('accepted_output_empty', False) for v in responses),
                all_items_rejected_sentences=sum(v.get('all_items_rejected', False) for v in responses),
                parsed_items=parsed, accepted_spans=sum(len(v.get('label', [])) for v in terminal),
                rejected_items=rejected, rejected_item_rate_pct=percentage(rejected, parsed),
                rejected_by_reason=dict(reasons),
                discarded_items_due_to_truncation=sum(v.get('discarded_items_due_to_truncation', 0) for v in responses),
                input_tokens=logical_input, output_tokens=output, total_tokens=logical_input+output,
                cached_input_tokens=None, input_tokens_are_full_logical_prompt=True,
                prefix_cache_policy='llama.cpp default in-process reuse; cached token count not instrumented',
                inference_seconds=infer_seconds, inference_wall_minutes=infer_seconds/60,
                cpu_process_seconds=cpu_seconds, cpu_process_minutes=cpu_seconds/60,
                latency_seconds=dict(mean=statistics.mean(latencies) if latencies else None,
                                     median=statistics.median(latencies) if latencies else None,
                                     p95=sorted(latencies)[math.ceil(.95*len(latencies))-1] if latencies else None,
                                     minimum=min(latencies) if latencies else None,
                                     maximum=max(latencies) if latencies else None),
                output_tokens_per_inference_second=output/infer_seconds if infer_seconds else None,
                rate_includes_prompt_processing=True,
                peak_process_rss_kib=max((v['peak_process_rss_kib'] for v in terminal
                                         if v.get('peak_process_rss_kib') is not None), default=None),
                automatic_retries=0, local_inference_cost_usd=None, gpu_minutes=None,
                document_id_coverage_pct=percentage(sum(d is not None for d in docs), n),
                original_row_index_coverage_pct=percentage(sum(i is not None for i in source_rows), n),
                document_bootstrap_metadata_ready=all(d is not None for d in docs),
                evaluation_status='not_evaluated; reference annotations are not loaded by inference',
                strict_precision=None, strict_recall=None, strict_f1=None,
                overlap_precision=None, overlap_recall=None, overlap_f1=None,
                reference_labels_read=False, session=session)


def print_header(lock, lock_sha, run_id, output_dir):
    print('\n=== M1 MAIN RUN: CONFIGURATION ===', flush=True)
    print(json.dumps(dict(run_id=run_id, expected_sentences=1000,
                          runtime_lock_sha256=lock_sha, output_directory=str(output_dir),
                          model=lock['model_source'], settings=lock['settings'],
                          hardware=dict(cpu=lock['cpu'], logical_cpus=lock['logical_cpus']),
                          software=lock['versions'], prompt_sha256=lock['prompt_sha256'],
                          input_sha256=lock['input_sha256'],
                          metadata=lock['metadata'], protocol=lock['reporting_protocol'],
                          reference_labels_read=False, evaluation_metrics='NA until separate scoring',
                          local_inference_cost_usd=None), ensure_ascii=False, indent=2), flush=True)


def print_record(record, progress, expected, cumulative, *, resumed=False, detailed=True):
    stamp = record.get('finished_at_utc') or record.get('started_at_utc') or 'NA'
    if stamp != 'NA':
        stamp = datetime.datetime.fromisoformat(stamp).astimezone(
            datetime.timezone(datetime.timedelta(hours=7))).isoformat(timespec='seconds')
    print(f"\n[{stamp}] {record['run_id']} {progress:04d}/{expected} "
          f"id={record['id']} outcome={record['outcome']} "
          f"attempt={record.get('attempt', 1)} resumed={str(resumed).lower()}", flush=True)
    print(f"  TIMESTAMPS start_utc={record.get('started_at_utc', 'NA')} "
          f"end_utc={record.get('finished_at_utc', 'NA')}", flush=True)
    print('  TOKENS input={input_tokens} output={output_tokens} total={total_tokens} cached=NA '
          'finish={finish_reason}'.format(**{k:record.get(k, 'NA') for k in
             ['input_tokens','output_tokens','total_tokens','finish_reason']}), flush=True)
    seconds = record.get('inference_seconds')
    cpu = record.get('cpu_process_seconds')
    rate = record.get('output_tokens_per_inference_second')
    print(f"  TIME inference_seconds={number_text(seconds)} "
          f"end_to_end_seconds={number_text(record.get('end_to_end_seconds'))} "
          f"cpu_process_seconds={number_text(cpu)} "
          f"output_tokens/inference_second={number_text(rate)} "
          f"current_RSS_GiB={number_text(record.get('current_process_rss_kib'), 1024**2)} "
          f"peak_RSS_GiB={number_text(record.get('peak_process_rss_kib'), 1024**2)} cost_USD=NA", flush=True)
    print(f"  SPANS parsed={record.get('parsed_items', 0)} kept={len(record.get('label', []))} "
          f"rejected={ratio_text(record.get('rejected_items', 0), record.get('parsed_items', 0))} "
          f"json_array_valid={record.get('json_array_valid')} "
          f"structural_rejection={record.get('structural_rejection')} "
          f"raw_empty={record.get('raw_response_empty')} "
          f"model_empty_array={record.get('model_returned_empty_array')} "
          f"accepted_empty={record.get('accepted_output_empty')} "
          f"all_items_rejected={record.get('all_items_rejected')} "
          f"truncation_discarded={record.get('discarded_items_due_to_truncation', 0)}", flush=True)
    print('  REJECTION_REASONS '+json.dumps(record.get('rejected_by_reason', {}), ensure_ascii=False), flush=True)
    print(f"  PROGRESS coverage={ratio_text(cumulative['finished_attempts'], expected)} "
          f"inference_coverage_pct={cumulative['inference_coverage_pct']} "
          f"cumulative_rejected={ratio_text(cumulative['rejected_items'], cumulative['parsed_items'])} "
          f"run_wall_seconds={record.get('session_elapsed_seconds', 'NA')} "
          f"ETA_seconds={record.get('estimated_remaining_seconds', 'NA')}", flush=True)
    print('  PROVENANCE '+json.dumps({k: record.get(k) for k in
          ['document_id','split','input_row_index_1based','original_row_index','source_index_base',
           'text_sha256','prompt_sha256','runtime_lock_sha256','request_sha256','response_record_sha256',
           'claim_path','result_path']}, ensure_ascii=False), flush=True)
    if detailed:
        print('  TEXT '+json.dumps(record.get('text'), ensure_ascii=False), flush=True)
        print('  RAW_RESPONSE '+json.dumps(record.get('response'), ensure_ascii=False), flush=True)
        print('  ACCEPTED '+json.dumps(record.get('accepted_spans_detail', []), ensure_ascii=False), flush=True)
        print('  REJECTED '+json.dumps(record.get('rejections', []), ensure_ascii=False), flush=True)
    if record.get('error'):
        print('  ERROR '+json.dumps(record['error'], ensure_ascii=False), flush=True)


def print_summary(summary):
    print('\n=== M1 RUN SUMMARY (not NER accuracy) ===', flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
