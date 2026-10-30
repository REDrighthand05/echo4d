// E2: execute the real quality module against a minimal DOM, not a browser claim.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { SLOT_IDS } from '../dist/quality-intake.mjs';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const defer = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; };
let serial = 0;
async function setup(t) {
  const hosts = new Map(), nodes = [], urls = new Map(), exported = [];
  class Node {
    constructor(tag='div') { this.tag=tag; this.children=[]; this.events={}; this.files=[]; this.style={}; this._value=''; nodes.push(this); }
    append(...children) {this.children.push(...children);}
    replaceChildren(...children) {this.children=children;}
    setAttribute() {}
    addEventListener(name,fn) {this.events[name]=fn;}
    set value(value) {this._value=value;if(this.type==='file' && value==='')this.files=[];}
    get value() {return this._value;}
    click() {if(this.href)exported.push(urls.get(this.href));}
    remove() {}
    async change(file) {this.files=[file];await this.events.change({target:this});}
  }
  const previous = {document:globalThis.document, window:globalThis.window, Image:globalThis.Image, create:URL.createObjectURL,revoke:URL.revokeObjectURL};
  globalThis.document={createElement:tag=>new Node(tag),getElementById:id=>{if(!hosts.has(id))hosts.set(id,new Node());return hosts.get(id);},body:new Node('body')};
  globalThis.window={addEventListener(){}};
  globalThis.Image=class {constructor(){this.naturalWidth=640;this.naturalHeight=480;}set src(value){queueMicrotask(()=>this.onload());}};
  URL.createObjectURL=value=>{const key='blob:test-'+urls.size;urls.set(key,value);return key;};
  URL.revokeObjectURL=()=>{};
  t.after(()=>{globalThis.document=previous.document;globalThis.window=previous.window;globalThis.Image=previous.Image;URL.createObjectURL=previous.create;URL.revokeObjectURL=previous.revoke;});
  await import('../dist/quality.mjs?events='+serial++);
  const imageInputs=nodes.filter(n=>n.tag==='input' && n.type==='file');
  const host=id=>hosts.get(id);
  return {host,imageInputs,nodes,export:async()=>{host('exportManifest').events.click();return JSON.parse(await exported.at(-1).text());}};
}
function manifest(scene='desk',hash='a'.repeat(64)) {return {format:'echo-visual-review-intake-1',scene_id:scene,notes:'holes',mesh_sha256:'b'.repeat(64),files:SLOT_IDS.map(slot=>({slot,sha256:hash,file_name:slot+'.png'}))};}
const jsonFile=value=>({size:100,text:async()=>JSON.stringify(value)});
function imageFile(name,bytes,read) {return {name,type:'image/png',size:bytes.length,arrayBuffer:read|| (async()=>Uint8Array.from(bytes).buffer)};}

test('resume requires actual matching bytes; missing and mismatched files stay incomplete',async t=>{
  const ui=await setup(t),bytes=Buffer.from('actual local image bytes'),expected=sha(bytes);
  await ui.host('importManifest').change(jsonFile(manifest('desk',expected)));
  let out=await ui.export();assert.equal(out.images_complete,false);assert.equal(out.files[0].sha256,null);assert.equal(out.files[0].expected_sha256,expected);
  await ui.imageInputs[0].change(imageFile('matching.png',bytes));
  out=await ui.export();assert.equal(out.files[0].sha256,expected);assert.equal(out.overall_accepted,false);
  await ui.imageInputs[0].change(imageFile('wrong.png',Buffer.from('different bytes')));
  out=await ui.export();assert.equal(out.files[0].sha256,null);assert.equal(out.files[0].expected_sha256,expected);assert.equal(out.images_complete,false);
});
test('a slow earlier manifest cannot replace a newer imported scene',async t=>{
  const ui=await setup(t),slow=defer();
  const first=ui.host('importManifest').change({size:100,text:()=>slow.promise});
  await ui.host('importManifest').change(jsonFile(manifest('newer')));
  slow.resolve(JSON.stringify(manifest('older')));await first;
  assert.equal(ui.host('sceneId').value,'newer');
});
test('a stale image failure cannot clear a newer selected image',async t=>{
  const ui=await setup(t),slow=defer(),bytes=Buffer.from('new selected image');
  const first=ui.imageInputs[0].change(imageFile('old.png',bytes,()=>slow.promise));
  const newest=imageFile('new.png',bytes);await ui.imageInputs[0].change(newest);
  slow.reject(new Error('old read failed'));await first;
  assert.equal(ui.imageInputs[0].files[0],newest);assert.equal((await ui.export()).files[0].sha256,sha(bytes));
});
test('invalid manifests leave existing scene and expected hashes intact',async t=>{
  const ui=await setup(t);await ui.host('importManifest').change(jsonFile(manifest('keep')));
  const bad=manifest('discard');bad.files[1].slot=bad.files[0].slot;
  await ui.host('importManifest').change(jsonFile(bad));
  assert.equal(ui.host('sceneId').value,'keep');assert.equal((await ui.export()).files[0].expected_sha256,'a'.repeat(64));
  const contradictory=manifest('discard');contradictory.files[0].expected_sha256='c'.repeat(64);
  await ui.host('importManifest').change(jsonFile(contradictory));
  assert.match(ui.host('importStatus').textContent,/冲突/);
  assert.equal(ui.host('sceneId').value,'keep');
  assert.equal((await ui.export()).files[0].expected_sha256,'a'.repeat(64));
});
