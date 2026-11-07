"""Compare greedy native and llama.cpp Manta traces on identical token IDs.

Requires Python tokenizers and the compare_manta_runtime Rust example.
Build scripts/reference/manta_runtime.cpp against a pinned llama.cpp checkout.
Run with --help for paths. Reports retain raw responses and top-10 logits.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=pathlib.Path)
    parser.add_argument('--tokenizer', required=True, type=pathlib.Path)
    parser.add_argument('--suite', required=True, type=pathlib.Path)
    parser.add_argument('--reference', required=True, type=pathlib.Path)
    parser.add_argument('--reference-revision', required=True)
    parser.add_argument('--reference-microbatch', type=int, default=512)
    parser.add_argument('--reference-no-repack', action='store_true')
    parser.add_argument('--native', type=pathlib.Path, default=pathlib.Path('target/release/examples/compare_manta_runtime'))
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--reuse-native', type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        path = args.output / name
        path.write_text(json.dumps(value, indent=2) + '\n')
        return path
    def native(forced=None):
        cmd = [str(args.native.resolve()), str(args.model), str(args.tokenizer), str(args.suite)]
        if forced is not None:
            cmd.append(str(forced))
        return json.loads(subprocess.check_output(cmd, text=True, timeout=600))
    rows = json.loads(args.reuse_native.read_text()) if args.reuse_native else native()
    write('native_greedy.json', rows)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding()
    reference = []
    for i, row in enumerate(rows):
        prefix = args.output / f'case_{i}'
        prompt_path = prefix.with_suffix('.prompt.txt')
        input_path = prefix.with_suffix('.input.txt')
        stop_path = prefix.with_suffix('.stop.txt')
        prompt_path.write_text(row['formatted_prompt'])
        input_path.write_text(' '.join(map(str, row['prompt_tokens'])))
        stop_path.write_text(' '.join(map(str, row['stop_ids'])))
        cmd = [str(args.reference.resolve()), str(args.model), str(prompt_path), str(input_path), str(stop_path), '128', '-',
               str(args.reference_microbatch), '0' if args.reference_no_repack else '1']
        with prefix.with_suffix('.stderr.txt').open('w') as log:
            result = json.loads(subprocess.check_output(cmd, stderr=log, text=True, timeout=600))
        result['name'] = row['name']
        result['response'] = tokenizer.decode(result['generated_tokens'], skip_special_tokens=False)
        reference.append(result)
        write('reference_greedy.json', reference)
        print('Reference completed:', row['name'], flush=True)
    # Compare next-token choices on the same first 32 reference-generated tokens.
    forced_path = write('reference_forcing.json', reference)
    forced_native = native(forced_path)
    write('native_forced.json', forced_native)
    forced_reference = []
    for i, row in enumerate(rows):
        prefix = args.output / f'case_{i}'
        force_path = prefix.with_suffix('.force.txt')
        force_path.write_text(' '.join(map(str, reference[i]['generated_tokens'][:32])))
        cmd = [str(args.reference.resolve()), str(args.model), str(prefix.with_suffix('.prompt.txt')),
               str(prefix.with_suffix('.input.txt')), str(prefix.with_suffix('.stop.txt')), '32', str(force_path),
               str(args.reference_microbatch), '0' if args.reference_no_repack else '1']
        with prefix.with_suffix('.forced.stderr.txt').open('w') as log:
            forced_reference.append(json.loads(subprocess.check_output(cmd, stderr=log, text=True, timeout=600)))
    write('reference_forced.json', forced_reference)
    cases = []
    for a, b, fa, fb in zip(rows, reference, forced_native, forced_reference):
        first_difference = next((i for i, (x, y) in enumerate(zip(a['generated_tokens'], b['generated_tokens'])) if x != y), None)
        if first_difference is None and len(a['generated_tokens']) != len(b['generated_tokens']):
            first_difference = min(len(a['generated_tokens']), len(b['generated_tokens']))
        pairs = list(zip(fa['trace'], fb['trace']))
        agree = sum(x['greedy'] == y['greedy'] for x, y in pairs)
        overlaps = [len({t['id'] for t in x['top']} & {t['id'] for t in y['top']}) for x, y in pairs]
        differences = []
        logit_errors = []
        for x, y in pairs:
            lx = {t['id']: t['logit'] for t in x['top']}
            ly = {t['id']: t['logit'] for t in y['top']}
            logit_errors.extend(abs(lx[t] - ly[t]) for t in lx.keys() & ly.keys())
            if x['greedy'] != y['greedy']:
                differences.append({'step': x['step'], 'native_token': x['greedy'],
                                    'reference_token': y['greedy'],
                                    'native_piece': tokenizer.decode([x['greedy']], skip_special_tokens=False),
                                    'reference_piece': tokenizer.decode([y['greedy']], skip_special_tokens=False),
                                    'reference_top2_margin': y['top'][0]['logit'] - y['top'][1]['logit']})
        cases.append({
            'name': a['name'], 'prompt_token_count': len(a['prompt_tokens']),
            'identical_prompt_tokens': a['prompt_tokens'] == b['prompt_tokens'],
            'native_output_tokens': len(a['generated_tokens']), 'reference_output_tokens': len(b['generated_tokens']),
            'identical_greedy_tokens': a['generated_tokens'] == b['generated_tokens'],
            'first_different_output_token_index': first_difference,
            'forced_steps': len(pairs), 'forced_top1_agreements': agree,
            'mean_forced_top10_overlap': sum(overlaps) / len(overlaps) if overlaps else None,
            'mean_common_top10_logit_absolute_difference': sum(logit_errors) / len(logit_errors) if logit_errors else None,
            'forced_disagreements': differences,
            'native_response': a['response'], 'reference_response': b['response'],
        })
    with args.model.open('rb') as source:
        model_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    report = {
        'model_sha256': model_hash,
        'native_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'reference_revision': args.reference_revision,
        'configuration': {'device': 'cpu', 'context': 8192, 'max_output_tokens': 128,
                          'greedy': True, 'repetition_penalty': 1.0, 'suppressed_tokens': [],
                          'speculative_decoding': False, 'output_filter': False,
                          'kv_cache': 'f32', 'reference_flash_attention': False,
                          'reference_microbatch': args.reference_microbatch,
                          'reference_weight_repacking': not args.reference_no_repack},
        'limitations': 'Same quantized GGUF in both runtimes. Does not compare original full-precision weights, GPU behavior, 8k inputs, or the production speculative/filtered path. Small logit differences can change greedy output near ties.',
        'cases': cases,
    }
    write('comparison.json', report)
    print(json.dumps(cases, indent=2), flush=True)


if __name__ == '__main__':
    main()
