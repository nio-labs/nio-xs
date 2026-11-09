"""Measure Beluga CLI speed and sampled Linux RAM using identical prompts.

Build target/release/nioxs first. Results include startup and prefill; the
first-output timestamp observes buffered CLI output, not the first model token.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "Explain why the sky appears blue in three concise sentences.",
    "Explain how a refrigerator moves heat, in three concise sentences.",
]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def run(binary, prompt, layers, limit, timeout):
    env = os.environ.copy()
    env.pop("NIOXS_LOW_RAM", None)
    env["NIOXS_BELUGA_RESIDENT_LAYERS"] = str(layers)
    env["NIOXS_BELUGA_ROLLING_WINDOW"] = "1"
    peak_anon = peak_rss = 0
    first = None
    timed_out = False
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        start = time.monotonic()
        proc = subprocess.Popen(
            [str(binary), "beluga", "--cpu", "-n", str(limit), prompt],
            cwd=ROOT, env=env, stdout=out, stderr=err,
        )
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if first is None and os.fstat(out.fileno()).st_size:
                    first = elapsed
                try:
                    fields = dict(
                        line.split(":", 1)
                        for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines()
                        if ":" in line
                    )
                    peak_anon = max(peak_anon, int(fields.get("RssAnon", "0").split()[0]))
                    peak_rss = max(peak_rss, int(fields.get("VmRSS", "0").split()[0]))
                except (FileNotFoundError, ProcessLookupError):
                    pass
                if elapsed > timeout:
                    timed_out = True
                    proc.kill()
                    proc.wait()
                    break
                time.sleep(0.01)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        seconds = time.monotonic() - start
        out.seek(0)
        response = out.read().decode(errors="replace")
        err.seek(0)
        stderr = err.read().decode(errors="replace")
    plain = re.sub(r"</?think>", "", ANSI.sub("", response))
    words = sum(any(c.isalnum() for c in word) for word in plain.split())
    return {
        "prompt": prompt, "resident_layers": layers, "rolling_window": 1,
        "max_tokens": limit, "seconds": round(seconds, 4),
        "first_output_seconds": round(first, 4) if first is not None else None,
        "words": words, "words_per_second": round(words / seconds, 3),
        "sampled_peak_anon_mib": round(peak_anon / 1024, 2),
        "sampled_peak_rss_mib": round(peak_rss / 1024, 2),
        "exit_code": proc.returncode, "timed_out": timed_out,
        "response": response, "stderr": stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/nioxs")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 14])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or not args.layers or min(args.layers) < 0 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("repeat, token limit and timeout must be positive; layer counts nonnegative")
    binary = args.binary.resolve()
    report = {
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "version": subprocess.check_output([str(binary), "--version"], text=True).strip(),
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown"),
        "device": "cpu", "method": "Visible words / wall time, including startup and prefill. Sampling interval 10 ms; brief RAM peaks may be missed. Thought text, if displayed, counts as output. Hardware and response quality require separate review.",
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    originals = {}
    for repeat in range(args.repeat):
        for prompt in PROMPTS:
            order = args.layers if repeat % 2 == 0 else list(reversed(args.layers))
            for layers in order:
                row = run(binary, prompt, layers, args.max_tokens, args.timeout)
                row["repeat"] = repeat + 1
                originals.setdefault(prompt, row["response"])
                row["matches_first_response"] = row["response"] == originals[prompt]
                row["target_10_words_s_met"] = row["words_per_second"] >= 10
                report["runs"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k not in ("response", "stderr")}), flush=True)


if __name__ == "__main__":
    main()
