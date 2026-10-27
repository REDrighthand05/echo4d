import fs from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';import test from 'node:test';import path from 'node:path';import {fileURLToPath} from 'node:url';
const __dirname=path.dirname(fileURLToPath(import.meta.url));
const page=process.env.ECHO_HISTORY_PAGE||path.resolve(__dirname,'../dist/history.html');
const source=fs.readFileSync(page,'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
function setup(){
  const elements=new Map(),pending=[];
  function el(id){if(!elements.has(id))elements.set(id,{hidden:true,disabled:false,value:'0',currentTime:0,files:[],handlers:{},addEventListener(k,f){this.handlers[k]=f},load(){},pause(){},play(){return Promise.resolve()},removeAttribute(){}});return elements.get(id)}
  const ctx=vm.createContext({Blob,URL:{createObjectURL(){return 'blob:mock'},revokeObjectURL(){}},document:{getElementById:el},window:{addEventListener(){},MediaRecorder:function(){}},navigator:{mediaDevices:{getUserMedia(){return new Promise((resolve,reject)=>pending.push({resolve,reject}))}}},performance:{now(){return 0}},setInterval(){},clearInterval(){},setTimeout(){}});
  vm.runInContext(source,ctx);return {el,ctx,pending,run:s=>vm.runInContext(s,ctx)};
}
const sample={t_s:0,acceleration_including_gravity_m_s2:[1,2,3],rotation_rate_deg_s:null};
function manifest(samples=[sample]){return {version:'echo-history-stream-0.1',duration_s:1,media:{file:'old.webm'},imu:{samples}}}
test('pending permission allows only one capture request and denial restores controls',async()=>{
  const h=setup(),first=h.el('startBtn').handlers.click();h.el('startBtn').handlers.click();
  assert.equal(h.pending.length,1);assert.equal(h.el('startBtn').disabled,true);
  h.pending[0].reject(Error('denied'));await first;
  assert.equal(h.el('startBtn').disabled,false);assert.equal(h.el('videoImport').disabled,false);
});
test('changing video clears all previous IMU association even with identical filename',()=>{
  const h=setup();h.ctx.saved=manifest();h.run('manifest=saved');
  h.el('videoImport').files=[{name:'old.webm'}];h.el('videoImport').handlers.change();h.el('replay').handlers.timeupdate();
  assert.equal(h.run('manifest'),null);assert.doesNotMatch(h.el('motionAtTime').textContent,/1.00, 2.00, 3.00/);
});
test('malformed manifest is rejected before changing effective state',async()=>{
  const h=setup();h.ctx.saved=manifest();h.run('manifest=saved');
  const bad=manifest([{t_s:0,acceleration_including_gravity_m_s2:['bad',2,3]}]);
  h.el('manifestImport').files=[{size:100,text:async()=>JSON.stringify(bad)}];await h.el('manifestImport').handlers.change();
  assert.equal(h.run('manifest===saved'),true);assert.match(h.el('notice').textContent,/未导入清单/);
  assert.doesNotThrow(()=>h.el('replay').handlers.timeupdate());
});
test('mismatched media cannot activate sensor samples',async()=>{
  const h=setup();h.run('boundVideoFile="different.webm"');
  h.el('manifestImport').files=[{size:100,text:async()=>JSON.stringify(manifest())}];await h.el('manifestImport').handlers.change();
  assert.equal(h.run('manifest'),null);assert.match(h.el('notice').textContent,/文件名不匹配/);
});
test('media switch while manifest read is pending rejects stale completion',async()=>{
  const h=setup();let resolve;h.run('boundVideoFile="old.webm"');
  h.el('manifestImport').files=[{size:100,text:()=>new Promise(r=>resolve=r)}];const reading=h.el('manifestImport').handlers.change();
  h.el('videoImport').files=[{name:'new.webm'}];h.el('videoImport').handlers.change();resolve(JSON.stringify(manifest()));await reading;
  assert.equal(h.run('manifest'),null);assert.match(h.el('notice').textContent,/读取期间视频已切换/);
});

test('normalized history archive reconnects the original video timeline and IMU without reconstructing geometry',async()=>{
  const h=setup();h.run('boundVideoFile="old.webm"');
  const timeline={
    ...manifest(),
    version:'echo-history-timeline-1',
    started_utc:'2026-10-28T09:00:00Z',
    geometry_status:'not_reconstructed',
    auxiliary_streams:[{sensor_id:'depth-a',clock_status:'source_claim_same_clock_unverified'}],
  };
  h.el('manifestImport').files=[{size:1000,text:async()=>JSON.stringify(timeline)}];
  await h.el('manifestImport').handlers.change();
  assert.equal(h.run('manifest.version'),'echo-history-timeline-1');
  assert.match(h.el('metadata').textContent,/附加传感流：1/);
  assert.match(h.el('metadata').textContent,/2026-10-28T09:00:00Z/);
  assert.match(h.el('metadata').textContent,/not_reconstructed/);
});
