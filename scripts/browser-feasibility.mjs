import fs from 'node:fs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const exec=promisify(execFile);
const [binary, output]=process.argv.slice(2);
if(!binary||!output)throw new Error('usage: node browser-feasibility.mjs OBSCURA_BINARY REPORT.json');
const probes=[
 ['destination','https://example.com'],
 ['destination','https://en.wikipedia.org/wiki/Rust_(programming_language)'],
 ['google','https://www.google.com/search?q=Rust+programming+language+official+documentation'],
 ['duckduckgo','https://duckduckgo.com/html/?q=Rust+programming+language+official+documentation'],
 ['bing','https://www.bing.com/search?q=Rust+programming+language+official+documentation'],
];
const results=[];
for(const [source,url] of probes){
 const start=Date.now();
 try{
  const {stdout}=await exec(binary,['fetch',url,'--timeout','12','--quiet','--eval',"JSON.stringify({title:document.title,text:document.body.innerText.slice(0,10000),links:Array.from(document.querySelectorAll('a[href]')).map(a=>({title:a.textContent.trim().slice(0,160),url:a.href})).filter(a=>a.title&&a.url.startsWith('http')).slice(0,80)})"],{timeout:18000,maxBuffer:512*1024});
  let data;try{data=JSON.parse(stdout.trim());if(typeof data==='string')data=JSON.parse(data);}catch{throw new Error('browser returned non-JSON evaluation output');}
  const text=data.text.toLowerCase();
  const challenge=['verify you are human','unusual traffic','captcha'].some(x=>text.includes(x));
  const destinations=data.links.map(l=>{
   try{const u=new URL(l.url);const target=u.searchParams.get('uddg')||(u.pathname==='/url'?(u.searchParams.get('q')||u.searchParams.get('url')):null);return target?{...l,url:new URL(target).href}:l;}catch{return l;}
  }).filter(l=>{try{return !['google.com','www.google.com','bing.com','www.bing.com','duckduckgo.com'].includes(new URL(l.url).hostname);}catch{return false;}});
  const result={source,url,status:challenge?'challenge':data.text.trim()?'readable':'empty',title:data.title,text_bytes:data.text.length,link_count:data.links.length,destination_links:destinations.slice(0,6),official_destination:destinations.find(l=>{try{return new URL(l.url).hostname.endsWith('rust-lang.org');}catch{return false;}})?.url,elapsed_ms:Date.now()-start};
  results.push(result);console.log(JSON.stringify(result));
 }catch(error){const result={source,url,status:'error',error:error.message.includes('timed out')?'timeout':error.message.split('\n')[0],elapsed_ms:Date.now()-start};results.push(result);console.log(JSON.stringify(result));}
}
const found=results.find(r=>r.status==='readable'&&r.official_destination);
if(found){
 try{
  const {stdout}=await exec(binary,['fetch',found.official_destination,'--dump','text','--timeout','12','--quiet'],{timeout:18000,maxBuffer:512*1024});
  const followed={source:'follow_search_result',search_engine:found.source,url:found.official_destination,status:stdout.trim()?'readable':'empty',text_bytes:stdout.length,excerpt:stdout.slice(0,250)};
  results.push(followed);console.log(JSON.stringify(followed));
 }catch(error){results.push({source:'follow_search_result',url:found.official_destination,status:'error',error:error.message.split('\n')[0]});}
}
fs.writeFileSync(output,JSON.stringify({browser:'Obscura',version:'0.2.4',chromium:false,results},null,2)+'\n');
