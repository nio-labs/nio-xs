"""Post-freeze authored synthetic acceptance corpus. Training never reads this file."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
groups={
 'read_file':(['please let me read {x}','can you display the file {x}','now open the file {x}','would you show me {x}','read the contents of {x}'],['docs/runtime.rst','src/protocol.c','internal/router.go','lib/encoder.cpp','guide/install.txt']),
 'inspect_project':(['please show the package managers {x}','can you list the manifest files {x}','now show project structure {x}','would you inspect package manifests {x}','show me the project manifests {x}'],['for our selected checkout','in the workspace folder','for the repository root','in our application directory','for the checked-out source']),
 'find_files':(['please find files matching {x}','can you show files matching {x}','now list filenames with glob {x}','would you locate files matching {x}','list project files {x}'],['src/*.proto','internal/**/*.go','data/*.csv','ui/*.vue','spec/*.rb']),
 'search_code':(['please search code for {x}','can you find references to {x}','now locate symbol {x}','would you search source for {x}','find occurrences of {x}'],['calculate_budget','reconcile_account','load_protocol','encode_response','discard_evidence']),
 'run_checks':(['please run the configured {x}','can you execute project {x}','now perform the {x}','would you run project {x}','execute the configured {x}'],['tests','build','lint','test suite','lint check']),
 'clarify':(['could you read it {x}','please find that {x}','now run it {x}','would you open the file {x}','please search for it {x}'],['for our folder','for the active checkout','for the repository','','please'])}
records=[]
for label,(templates,targets) in groups.items():
 for i,template in enumerate(templates):
  for j,target in enumerate(targets):records.append({'id':f'frozen_{label}_{i}_{j}','family_id':f'frozen_{label}_target_{j}','split':'test','prompt':template.format(x=target).strip(),'intent':label})
bad=['please delete {x}','can you rewrite {x}','now fix {x}','would you explain {x}','please create {x}','design a new {x}','write tests for {x}','compare {x}','search the web for {x}','run tests then deploy {x}','read and upload {x}','bypass the permissions for {x}']
subjects=['our workspace','the router','the protocol','the staging server','the source tree','the production configuration','a remote repository','an account','a build artifact','the private logs']
for i,template in enumerate(bad):
 for j,subject in enumerate(subjects):records.append({'id':f'frozen_unsupported_{i}_{j}','family_id':f'frozen_unsupported_template_{i}','split':'test','prompt':template.format(x=subject),'intent':'unsupported'})
data=''.join(json.dumps(r)+'\n' for r in records);(ROOT/'evaluation/phase2-frozen-holdout.jsonl').write_text(data)
(ROOT/'evaluation/phase2-freeze.json').write_text(json.dumps({'schema_version':1,'artifact_sha256':hashlib.sha256((ROOT/'models/nano.json').read_bytes()).hexdigest(),'holdout_sha256':hashlib.sha256(data.encode()).hexdigest(),'purpose':'post-freeze synthetic acceptance; not real-world accuracy','records':len(records)},indent=2)+'\n')
