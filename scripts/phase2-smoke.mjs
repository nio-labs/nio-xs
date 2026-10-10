// Local fixtures only. Generates temporary credentials without printing them.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import net from 'node:net';
import http from 'node:http';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, i, all) => {
  if (i % 2 === 0) pairs.push([value.replace(/^--/, ''), all[i + 1]]);
  return pairs;
}, []));
for (const name of ['niodb', 'nio', 'provider', 'report']) if (!args[name]) throw new Error(`--${name} is required`);
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'nioxs-smoke-'));
fs.chmodSync(temp, 0o700);
const children = new Set();
const checks = [];
const report = { phase: 2, version: '0.1.0-alpha.2', checks, completed: false };
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const record = name => { checks.push({ name, passed: true }); console.log(`PASS ${name}`); };
const cleanEnv = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/^(NIO|NIODB|NIOXS|OPENAI|GEMINI|ANTHROPIC|HTTP_PROXY|HTTPS_PROXY|ALL_PROXY)/i.test(key)));
cleanEnv.NO_PROXY = '*';
async function freePort() {
  const server = net.createServer(); server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const port = server.address().port; await new Promise(resolve => server.close(resolve)); return port;
}
function start(binary, argv, env, name) {
  const log = fs.openSync(path.join(temp, `${name}.log`), 'a', 0o600);
  const child = spawn(binary, argv, { env: { ...cleanEnv, ...env }, stdio: ['ignore', log, log] });
  fs.closeSync(log); children.add(child); return child;
}
async function stop(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) { children.delete(child); return; }
  const done = once(child, 'exit'); child.kill('SIGTERM');
  const timer = setTimeout(() => child.kill('SIGKILL'), 2000);
  await done; clearTimeout(timer); children.delete(child);
}
async function waitReady(url) {
  const end = Date.now() + 10000;
  while (Date.now() < end) {
    try { const response = await fetch(url, { signal: AbortSignal.timeout(1000) }); if (response.ok) return; } catch {}
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error(`service not ready: ${url}`);
}
async function request(url, body, extra = {}) {
  const response = await fetch(url, { method: body === undefined ? 'GET' : 'POST', headers: { 'content-type': 'application/json', ...extra }, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(10000) });
  const value = await response.json(); return { status: response.status, value };
}
let slowServer;
try {
  const data = path.join(temp, 'db'); fs.mkdirSync(data, { mode: 0o700 });
  const credentials = JSON.parse(execFileSync(args.niodb, ['init-auth', '--dir', data, '--name', 'nioxs-smoke', '--workspace', 'nioxs-smoke'], { encoding: 'utf8', env: cleanEnv }));
  const tokenFile = path.join(temp, 'db.token'); fs.writeFileSync(tokenFile, credentials.client_token, { mode: 0o600 });
  const auth = { authorization: `Bearer ${credentials.client_token}` };
  const dbPort = await freePort(), providerPort = await freePort();
  const dbUrl = `http://127.0.0.1:${dbPort}`, providerUrl = `http://127.0.0.1:${providerPort}`;
  const dbArgs = ['serve', '--dir', data, '--listen', `127.0.0.1:${dbPort}`, '--no-pg', '--no-demo', '--nio-bin', 'nioxs-assistance-disabled', '--node-bin', 'nioxs-sql-disabled'];
  let db = start(args.niodb, dbArgs, {}, 'niodb'); await waitReady(`${dbUrl}/health`);
  const status = await request(`${dbUrl}/api/v1/status`, undefined, auth);
  assert(status.status === 200 && status.value.nio.status !== 'ready' && status.value.alasql.status === 'unavailable', 'assistance/SQL unexpectedly active');
  report.niodb = { version: execFileSync(args.niodb, ['--version'], { encoding: 'utf8' }).trim(), assistance: status.value.nio.status, sql: status.value.alasql.status };
  record('NioDB core works with assistance and SQL disabled');
  const payload = { collection: 'nano_probe', data: { schema_version: 1, deployment_id: 'smoke', value: 'durable' } };
  const created = await request(`${dbUrl}/api/v1/records`, payload, { ...auth, 'idempotency-key': 'phase1-probe' });
  const repeated = await request(`${dbUrl}/api/v1/records`, payload, { ...auth, 'idempotency-key': 'phase1-probe' });
  assert(created.status === 201 && created.value.id === repeated.value.id, 'idempotent create failed');
  const conflict = await request(`${dbUrl}/api/v1/records`, { ...payload, data: { ...payload.data, value: 'changed' } }, { ...auth, 'idempotency-key': 'phase1-probe' });
  assert(conflict.status === 409, 'idempotency conflict not rejected'); record('durable record create/read and idempotency');
  const got = await request(`${dbUrl}/api/v1/records/${created.value.id}`, undefined, auth);
  assert(got.status === 200 && got.value.data.value === 'durable', 'record read failed');
  const patchedResponse = await fetch(`${dbUrl}/api/v1/records/${created.value.id}`, { method:'PATCH', headers:{...auth,'content-type':'application/json'}, body:JSON.stringify({data:{value:'updated'}}) });
  const patched=await patchedResponse.json();
  assert(patchedResponse.ok && patched.revision === 2 && patched.data.value === 'updated', 'record update failed'); record('core record update increments revision');
  const badAuth = await request(`${dbUrl}/api/v1/records`, undefined, { authorization: 'Bearer invalid' });
  assert(badAuth.status === 401, 'database authentication failed'); record('database rejects invalid credentials');
  await stop(db); db=start(args.niodb,dbArgs,{},'niodb'); await waitReady(`${dbUrl}/health`);
  const recovered=await request(`${dbUrl}/api/v1/records/${created.value.id}`,undefined,auth);
  assert(recovered.value.data.value==='updated'&&recovered.value.revision===2,'restart lost records'); record('NioDB record survives restart');

  const providerEnv = { NIOXS_DB_URL:dbUrl, NIOXS_DB_TOKEN_FILE:tokenFile, NIOXS_LISTEN:`127.0.0.1:${providerPort}`, NIOXS_DEPLOYMENT_ID:'smoke' };
  let provider=start(args.provider,['serve'],providerEnv,'provider'); await waitReady(`${providerUrl}/ready`);
  const catalog=await request(`${providerUrl}/v1/models`);
  assert(catalog.value.data[0].id==='nio-nano','model missing'); record('NioXS provider catalog and authenticated storage readiness');
  const firstBody={model:'nio-nano',messages:[{role:'user',content:'read README.md'}],tools:[{type:'function',function:{name:'read_file',parameters:{type:'object',properties:{path:{type:'string'},line_count:{type:'integer'}},required:['path']}}}]};
  const first=await request(`${providerUrl}/v1/chat/completions`,firstBody);
  assert(first.status===200&&first.value.choices[0].finish_reason==='tool_calls','tool call missing');
  const call=first.value.choices[0].message.tool_calls[0];
  const resultBody={...firstBody,messages:[...firstBody.messages,first.value.choices[0].message,{role:'tool',tool_call_id:call.id,content:'# Actual client README\nFixture text'}]};
  const done=await request(`${providerUrl}/v1/chat/completions`,resultBody);
  assert(done.value.choices[0].message.content.includes('Fixture text')&&done.value.nioxs.outcome==='completed','result correlation failed'); record('provider client-tool round trip');
  const disabled=await request(`${providerUrl}/v1/chat/completions`,{...firstBody,tools:[]});
  assert(!disabled.value.choices[0].message.tool_calls&&!disabled.value.nioxs.handoff.continuation_required,'disabled tool policy bypassed'); record('no-tools request cannot dispatch read_file');
  const unsupported=await request(`${providerUrl}/v1/chat/completions`,{model:'nio-nano',messages:[{role:'user',content:'invent a new compiler'}]});
  assert(unsupported.value.nioxs.outcome==='capability_limitation'&&unsupported.value.nioxs.handoff.continuation_required,'limitation metadata missing'); record('unsupported request returns explicit handoff metadata');

  const project=path.join(temp,'project'); fs.mkdirSync(project); fs.writeFileSync(path.join(project,'README.md'),'# Real CLI fixture\nProof from client filesystem');
  const config=path.join(temp,'nio.json');fs.writeFileSync(config,JSON.stringify({request_interval_seconds:0,default_model:"nioxs::nio-nano",providers:[{id:'nioxs',name:'NioXS',base_url:`${providerUrl}/v1`}]}),{mode:0o600});
  const cli=execFileSync(args.nio,['run','-m','nioxs::nio-nano','--mode','ask','--trust-project','--dir',project,'--format','json','--','read README.md'],{encoding:'utf8',env:{...cleanEnv,NIO_CONFIG:config},timeout:15000});
  const cliEvents=cli.trim().split('\n').map(line=>JSON.parse(line));
  assert(cliEvents.some(e=>e.type==='tool_use'&&e.part.tool==='read_file'&&e.part.state.status==='completed'),'CLI did not execute tool');
  assert(cliEvents.some(e=>e.type==='text'&&e.part.text.includes('Proof from client filesystem')),'CLI result missing'); record('real Nio CLI → NioXS → Nio read_file → final answer');
  const capabilities=await request(`${providerUrl}/v1/capabilities`);
  assert(capabilities.value.trained===true&&capabilities.value.model_version==='0.1.0-alpha.2','trained model not ready');record('trained model capability and version');
  fs.mkdirSync(path.join(project,'src'));
  fs.writeFileSync(path.join(project,'src/example.rs'),'fn regression_token() {}\n');
  fs.writeFileSync(path.join(project,'check.mjs'),"import fs from 'node:fs';fs.appendFileSync('check-runs.log',process.argv[2]+'\\n');console.log('fixture check passed');\n");
  fs.writeFileSync(path.join(project,'package.json'),JSON.stringify({name:'phase2-fixture',version:'1.0.0',scripts:{test:'node check.mjs test',build:'node check.mjs build',lint:'node check.mjs lint'},dependencies:{'fixture-dependency':'1.0.0'}}));
  function nio(prompt,mode='ask',flags=[],trust=true){
    const argv=['run','--mode',mode,'--dir',project,'--format','json',...flags];if(trust)argv.push('--trust-project');argv.push('--',prompt);
    const output=execFileSync(args.nio,argv,{encoding:'utf8',env:{...cleanEnv,NIO_CONFIG:config},timeout:20000,stdio:['ignore','pipe','pipe']});
    return output.trim().split('\n').map(line=>JSON.parse(line));
  }
  const called=(events,tool)=>events.some(e=>e.type==='tool_use'&&e.part?.tool===tool&&e.part?.state?.status==='completed');
  const answer=events=>events.filter(e=>e.type==='text').map(e=>e.part.text).join('\n');
  for(const prompt of ['now read readme.md','please open README.md','show me README.md']){
    const events=nio(prompt);assert(called(events,'read_file')&&answer(events).includes('Proof from client filesystem'),'natural read failed');
  }record('natural read phrasing through installed Nio');
  const inspection=nio('inspect this project');assert(called(inspection,'find_files')&&called(inspection,'read_file')&&answer(inspection).includes('phase2-fixture')&&answer(inspection).includes('fixture-dependency'),'project evidence missing');record('project manifests and dependency evidence');
  const filenames=nio('find files matching *.md');assert(called(filenames,'find_files')&&answer(filenames).includes('README.md'),'file search failed');record('file glob search');
  const code=nio('search code for regression_token');assert(called(code,'search_code')&&answer(code).includes('src/example.rs:1')&&answer(code).includes('fn regression_token'),'code evidence missing');record('ranked literal code search with exact source lines');
  for(const kind of ['test','build','lint']){const events=nio(`run ${kind}`, 'build',['--auto']);assert(called(events,'run_command')&&answer(events).includes('exit code 0'),'configured check failed');}record('approved test build lint checks in Build mode');
  for(const mode of ['ask','plan']){const events=nio('run tests',mode);assert(!called(events,'run_command'),'read-only mode executed command');}record('Ask and Plan do not dispatch commands');
  const beforeDenied=fs.readFileSync(path.join(project,'check-runs.log'),'utf8');
  const denied=nio('run tests','build');assert(!called(denied,'run_command')&&fs.readFileSync(path.join(project,'check-runs.log'),'utf8')===beforeDenied,'denied command executed');record('command approval denial prevents execution');
  for(const flags of [['--no-tools'],['--no-project-tools']]){const events=nio('now read readme.md','ask',flags);assert(!called(events,'read_file'),'disabled project policy bypassed');}record('no-tools and no-project-tools stay enforced');
  const untrusted=nio('read README.md','ask',[],false);assert(!called(untrusted,'read_file'),'untrusted project read');record('untrusted folder cannot dispatch project tools');
  const missing=await request(`${providerUrl}/v1/chat/completions`,{model:'nio-nano',messages:[{role:'user',content:'read it'}]});assert(missing.value.nioxs.outcome==='clarification'&&!missing.value.choices[0].message.tool_calls,'missing input dispatched');record('missing arguments clarify without actions');
  fs.writeFileSync(path.join(project,'.nioxs.json'),JSON.stringify({schema_version:1,commands:{test:'npm run test'}}));
  const configured=nio('run tests','build',['--auto']);assert(called(configured,'run_command')&&answer(configured).includes('.nioxs.json'),'explicit config ignored');record('explicit supported command configuration');
  fs.writeFileSync(path.join(project,'.nioxs.json'),JSON.stringify({schema_version:1,commands:{test:'printf unsupported'}}));
  const unsupportedCommand=nio('run tests','build',['--auto']);assert(!called(unsupportedCommand,'run_command'),'unsupported configured command dispatched');record('unsupported configured commands abstain');
  fs.unlinkSync(path.join(project,'.nioxs.json'));
  const big=await request(`${providerUrl}/v1/chat/completions`,{...resultBody,messages:[...firstBody.messages,first.value.choices[0].message,{role:'tool',tool_call_id:call.id,content:'x'.repeat(33000)}]});assert(big.status===400,'output limit missing');record('oversized client result rejected');
  const allTools=['find_files','read_file','run_command'].map(name=>({type:'function',function:{name}}));
  const sideEffect={model:'nio-nano',messages:[{role:'user',content:'run tests'}],tools:allTools};
  const planned=await request(`${providerUrl}/v1/chat/completions`,sideEffect,{'x-nioxs-turn-id':'turn_recovery_fixture'});
  assert(planned.value.choices[0].message.tool_calls[0].function.name==='find_files','check discovery missing');
  const discoverCall=planned.value.choices[0].message.tool_calls[0];
  const commandBody={...sideEffect,messages:[...sideEffect.messages,planned.value.choices[0].message,{role:'tool',tool_call_id:discoverCall.id,content:JSON.stringify({files:['Cargo.toml'],next_offset:null,scan_limited:false})}]};
  const parallel=await Promise.all([request(`${providerUrl}/v1/chat/completions`,commandBody,{'x-nioxs-turn-id':'turn_recovery_fixture'}),request(`${providerUrl}/v1/chat/completions`,commandBody,{'x-nioxs-turn-id':'turn_recovery_fixture'})]);
  const dispatches=parallel.filter(r=>r.value.choices?.[0]?.message?.tool_calls?.[0]?.function?.name==='run_command');
  assert(dispatches.length===1,'concurrent command dispatch repeated');record('concurrent continuation dispatches one command');
  const commandReply=dispatches[0].value.choices[0].message;
  await stop(provider);provider=start(args.provider,['serve'],providerEnv,'provider-restarted');await waitReady(`${providerUrl}/ready`);
  const unknown=await request(`${providerUrl}/v1/chat/completions`,commandBody,{'x-nioxs-turn-id':'turn_recovery_fixture'});
  assert(unknown.value.nioxs.outcome==='unknown_action'&&!unknown.value.choices[0].message.tool_calls,'restart replayed unknown command');record('provider restart marks uncertain command unknown without replay');
  const completeBody={...commandBody,messages:[...commandBody.messages,commandReply,{role:'tool',tool_call_id:commandReply.tool_calls[0].id,content:'exit code: 0\nstdout:\nfixture passed\nstderr:\n'}]};
  const recordedResult=await request(`${providerUrl}/v1/chat/completions`,completeBody,{'x-nioxs-turn-id':'turn_recovery_fixture'});assert(recordedResult.value.nioxs.outcome==='completed','command result not recorded');
  await stop(provider);provider=start(args.provider,['serve'],providerEnv,'provider-completed-restart');await waitReady(`${providerUrl}/ready`);
  const recoveredResult=await request(`${providerUrl}/v1/chat/completions`,commandBody,{'x-nioxs-turn-id':'turn_recovery_fixture'});assert(recoveredResult.value.nioxs.outcome==='completed'&&!recoveredResult.value.choices[0].message.tool_calls,'completed command replayed');record('completed command survives restart without replay');
  const changedResult={...completeBody,messages:[...completeBody.messages.slice(0,-1),{role:'tool',tool_call_id:commandReply.tool_calls[0].id,content:'exit code: 1\nstdout:\nchanged result\nstderr:\n'}]};
  const conflictResult=await request(`${providerUrl}/v1/chat/completions`,changedResult,{'x-nioxs-turn-id':'turn_recovery_fixture'});assert(conflictResult.status===409,'conflicting result accepted');record('conflicting continuation result rejected');
  const actionRecords=await request(`${dbUrl}/api/v1/records?collection=nano_actions&limit=1000`,undefined,auth);
  const claimRecord=actionRecords.value.items.find(r=>r.data.action_id===commandReply.tool_calls[0].id&&r.data.status==='dispatch_intent');
  assert(claimRecord,'dispatch journal absent');
  const revisionPatch=await fetch(`${dbUrl}/api/v1/records/${claimRecord.id}`,{method:'PATCH',headers:{...auth,'content-type':'application/json'},body:JSON.stringify({data:{external_edit:true}})});assert(revisionPatch.ok,'revision fixture failed');
  const revisionConflict=await request(`${providerUrl}/v1/chat/completions`,commandBody,{'x-nioxs-turn-id':'turn_recovery_fixture'});assert(revisionConflict.status===409&&!revisionConflict.value.choices,'changed record revision not detected');record('unexpected durable record revision blocks dispatch');

  const before=await request(`${dbUrl}/api/v1/records?collection=nano_events`,undefined,auth);
  assert(before.value.items?.length>0,'no durable provider events');
  assert(!JSON.stringify(before.value).includes('Proof from client filesystem'),'raw project text persisted'); record('provider persists digests rather than file contents');

  // Cancel during a deliberately delayed storage acknowledgement, without browser/network traffic.
  let notifyStorage;const entered=new Promise(resolve=>{notifyStorage=resolve;});
  slowServer=http.createServer((req,res)=>{notifyStorage(); const timer=setTimeout(()=>{res.writeHead(201,{'content-type':'application/json'});res.end('{}');},5000);res.on('close',()=>clearTimeout(timer));req.resume();});
  slowServer.listen(0,'127.0.0.1');await once(slowServer,'listening');
  const slowPort=await freePort();const slowUrl=`http://127.0.0.1:${slowPort}`;
  const slowProvider=start(args.provider,['serve'],{...providerEnv,NIOXS_LISTEN:`127.0.0.1:${slowPort}`,NIOXS_DB_URL:`http://127.0.0.1:${slowServer.address().port}`},'slow-provider'); await waitReady(`${slowUrl}/health`);
  const pending=request(`${slowUrl}/v1/chat/completions`,firstBody,{'x-nioxs-turn-id':'turn_cancel_probe'});
  await entered;
  const cancelled=await request(`${slowUrl}/v1/turns/turn_cancel_probe/cancel`,{});
  const stopped=await pending;
  assert(cancelled.status===200&&stopped.status===409&&stopped.value.error.code==='turn_cancelled','cancellation failed'); record('explicit cancellation stops pending client-action dispatch');
  await stop(slowProvider);slowServer.closeAllConnections();await new Promise(resolve=>slowServer.close(resolve));slowServer=null;

  await stop(db);
  const outage=await request(`${providerUrl}/v1/chat/completions`,firstBody);
  assert(outage.status===503&&!outage.value.choices,'storage outage dispatched action');record('storage outage fails closed before dispatch');
  await stop(provider);
  report.completed=true;
} finally {
  if(slowServer){slowServer.closeAllConnections();slowServer.close();}
  for(const child of [...children]) await stop(child);
  fs.mkdirSync(path.dirname(path.resolve(args.report)),{recursive:true});
  fs.writeFileSync(args.report,JSON.stringify(report,null,2)+'\n');
  // Temporary credentials remain private and can be removed by the operator.
  console.log(`Report: ${args.report}`);
}
