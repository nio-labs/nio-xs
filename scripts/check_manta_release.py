"""Run small 0.5B-scope CLI checks; record responses without declaring factual accuracy."""
import json
import pathlib
import subprocess
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = [
    ('http_429', [], 128, 'What does HTTP status 429 mean? Answer in one short sentence.'),
    ('work_queue', [], 128, 'Explain a work queue and what a worker does, in two short sentences.'),
    ('idempotency', [], 128, 'Why use an idempotency key when retrying an order request? Answer in one short sentence.'),
    ('simple_rollout', [], 192, 'Give exactly three short bullet points for deploying a stateless API: test in staging, deploy to production, check health. No code or XML.'),
    ('xml_one_step', ['--plan'], 384, 'Return one complete XML plan with one step: id="check", depends_on="". Check the staging API at https://staging.example.test/health using curl. Include name, description, command, and rollback. The rollback should state that this read-only check needs no rollback.'),
    ('xml_two_steps', ['--plan'], 512, 'Create a complete two-step XML deployment plan: first validate staging, then deploy to production after staging succeeds. Include every required field, unique IDs, and an explicit depends_on attribute on both steps.'),
]


def main():
    output = ROOT / 'benchmarks/manta_release_checks_2026-11-06/short_tasks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'model': 'Manta 494M', 'device': 'cpu', 'scope': 'Short bounded tasks suited to a 0.5B companion. Responses require manual factual review.', 'cases': []}
    for name, flags, limit, prompt in CASES:
        peak = [0]
        done = threading.Event()
        start = time.monotonic()
        proc = subprocess.Popen([str(ROOT / 'target/release/nioxs'), 'manta', '--cpu', '-n', str(limit), *flags, prompt], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def sample():
            while not done.is_set():
                try:
                    for line in pathlib.Path(f'/proc/{proc.pid}/status').read_text().splitlines():
                        if line.startswith('RssAnon:'):
                            peak[0] = max(peak[0], int(line.split()[1]))
                except (FileNotFoundError, ProcessLookupError):
                    pass
                done.wait(0.01)
        monitor = threading.Thread(target=sample)
        monitor.start()
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, stderr = proc.communicate()
        finally:
            done.set()
            monitor.join()
        row = {'name': name, 'prompt': prompt, 'max_tokens': limit, 'plan_mode': bool(flags),
               'exit_code': proc.returncode, 'timed_out': timed_out,
               'seconds': round(time.monotonic()-start, 3), 'sampled_peak_anon_mib': round(peak[0]/1024, 2),
               'response': stdout.strip(), 'stderr': stderr.strip(),
               'invalid_plan_was_rejected': bool(flags) and proc.returncode != 0 and 'Manta plan rejected:' in stderr and not stdout.strip(),
               'has_code_fence': '```' in stdout, 'visible_words': len(stdout.split())}
        report['cases'].append(row)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    print('Saved '+str(output), flush=True)


if __name__ == '__main__':
    main()
