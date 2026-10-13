import { spawnSync } from 'node:child_process';
import { appendFileSync, readFileSync } from 'node:fs';

const { name, version } = JSON.parse(readFileSync('package.json', 'utf8'));
const spec = `${name}@${version}`;
const registry = 'https://registry.npmjs.org/';

function npm(args) {
  const result = spawnSync('npm', args, { encoding: 'utf8' });
  if (result.error) throw result.error;
  if (result.signal) throw new Error(`npm terminated by ${result.signal}`);
  return result;
}

function summary(message) {
  console.log(message);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${message}\n`);
  }
}

function errorDetails(result) {
  // npm emits JSON errors on stdout; stderr also includes its normal log.
  for (const output of [result.stdout, result.stderr]) {
    try {
      const parsed = JSON.parse(output);
      if (parsed.error) return parsed.error;
    } catch {
      // Keep the original output when npm did not return JSON.
    }
  }
  return {};
}

function fail(result) {
  process.stdout.write(result.stdout);
  process.stderr.write(result.stderr);
  process.exit(result.status || 1);
}

const existing = npm(['view', spec, 'version', '--json', '--registry', registry]);
if (existing.status === 0) {
  if (JSON.parse(existing.stdout) !== version) {
    throw new Error(`Unexpected registry version for ${spec}: ${existing.stdout}`);
  }
  summary(`${spec} is already published; skipping npm publish.`);
} else {
  // A staged version may be absent from public metadata. Only E404 means
  // publishing should proceed; authentication and network errors remain fatal.
  if (errorDetails(existing).code !== 'E404') fail(existing);

  const result = npm(['publish', '--access', 'public', '--json', '--registry', registry]);
  if (result.status === 0) {
    process.stdout.write(result.stdout);
    process.stderr.write(result.stderr);
    summary(`npm publish completed for ${spec}.`);
  } else {
    const error = errorDetails(result);
    const message = `${error.summary || ''}\n${error.detail || ''}\n${result.stderr}`;
    const stagedConflict = `Cannot publish over previously staged version "${version}"`;
    if (error.code !== 'E409' || !message.includes(stagedConflict)) fail(result);

    const guidance = `${spec} is already staged in npm and still needs maintainer approval. Review it in npm's Staged Packages tab and approve it to make it installable. If the staged contents are wrong, reject the staged upload before retrying this tag.`;
    console.warn(`::warning::${guidance}`);
    summary(guidance);
  }
}
