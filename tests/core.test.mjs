import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {unproject,cameraMatrix,validateRecording,reconstruct,sampleEntity,toPLY,validateScene,unpackScene} from '../dist/core.mjs';
import {createDemo,evaluate} from '../dist/demo.mjs';
const demo=createDemo();
test('calibrated projection gives world coordinates',()=>{assert.deepEqual(unproject(1,1,2,{fx:1,fy:1,cx:1,cy:1},cameraMatrix([2,3,4],[2,3,3])),[2,3,2]);});
test('depth values drive actual geometry',()=>{
 const d={version:'echo-1',source:'recorded',units:'meters',duration:1,intrinsics:{width:2,height:2,fx:2,fy:2,cx:.5,cy:.5},frames:[{t:0,cameraToWorld:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],depth:[1,1,1,1]}]};
 validateRecording(d);const m=reconstruct(d,{voxel:.01});assert.equal(m.faces.length,2);assert.ok(m.vertices.every(v=>v.p[2]===1));d.frames[0].depth.fill(2);assert.ok(reconstruct(d).vertices.every(v=>v.p[2]===2));
});
test('zero depth creates no geometry',()=>{const d=structuredClone(demo);d.frames=d.frames.slice(0,1);d.frames[0].depth.fill(0);assert.equal(reconstruct(d).vertices.length,0);});
test('dynamic exclusion mask is applied and validated',()=>{const d=structuredClone(demo);d.frames=d.frames.slice(0,1);d.frames[0].excluded=d.frames[0].depth.map(()=>true);validateRecording(d);assert.equal(reconstruct(d).vertices.length,0);d.frames[0].excluded[0]=1;assert.throws(()=>validateRecording(d));});
test('malformed input rejected',()=>{for(const f of [d=>d.frames[0].depth.pop(),d=>d.frames[0].depth[0]=NaN,d=>d.frames[0].cameraToWorld[0]*=-1,d=>d.frames[1].t=d.frames[0].t,d=>d.intrinsics.fx=0,d=>d.intrinsics.fx=1e-300]){const d=structuredClone(demo);f(d);assert.throws(()=>validateRecording(d));}});
test('pose drift increases independent synthetic surface error without changing inputs',()=>{const before=JSON.stringify(demo),normal=evaluate(reconstruct(demo,{stride:2})),shifted=evaluate(reconstruct(demo,{stride:2,drift:.2}));assert.ok(shifted.rmse>normal.rmse*10);assert.equal(JSON.stringify(demo),before);});
test('sparse sampling reduces heldout coverage',()=>{assert.ok(evaluate(reconstruct(demo,{stride:12})).coverage<evaluate(reconstruct(demo)).coverage);});
test('noise is deterministic for controlled experiments',()=>{assert.deepEqual(reconstruct(demo,{stride:12,noise:.02}).vertices,reconstruct(demo,{stride:12,noise:.02}).vertices);});
test('unknown motion can stay unknown and is never extrapolated',()=>{const e=demo.entities[0];assert.equal(sampleEntity(e,6.4,true).gap,true);assert.equal(sampleEntity(e,6.4,false),null);assert.equal(sampleEntity(e,-1),null);assert.equal(sampleEntity(e,18),null);assert.equal(sampleEntity(e,4).type,'sample');});
test('time cutoff never uses future depth',()=>{const m=reconstruct(demo,{until:2});assert.ok(m.usedFrames>0);assert.ok(m.vertices.every(v=>v.last<=2));});
test('PLY counts match body',()=>{const m=reconstruct(demo,{stride:12}),p=toPLY(m);assert.ok(p.includes(`element vertex ${m.vertices.length}`));assert.ok(p.includes(`element face ${m.faces.length}`));assert.equal(p.split('end_header\n')[1].trim().split('\n').length,m.vertices.length+m.faces.length);});
test('native scene topology and evidence validated',()=>{const s={version:'echo-scene-1',algorithm:'projective-tsdf-v1',recording:demo,mesh:{vertices:[[0,0,0],[1,0,0],[0,1,0]],faces:[[0,1,2]],colors:[[1,2,3],[1,2,3],[1,2,3]],observations:[1,2,3],first:[0,0,0],last:[1,1,1]}};validateScene(s);assert.equal(unpackScene(s).vertices[1].supportCount,2);s.mesh.faces[0][2]=3;assert.throws(()=>validateScene(s));});
test('preview provenance rejects contradictory dimensions and false original-capture claims',()=>{
 const scene={version:'echo-scene-1',algorithm:'projective-tsdf-v1',
   recording:structuredClone(demo),
   mesh:{vertices:[[0,0,0],[1,0,0],[0,1,0]],faces:[[0,1,2]],
     colors:[[1,2,3],[1,2,3],[1,2,3]],observations:[1,2,3],
     first:[0,0,0],last:[1,1,1]}};
 const k=scene.recording.intrinsics;
 scene.recording.preview_provenance={
   loaded_recording_resolution_px:{width:k.width*2,height:k.height*2},
   exported_preview_resolution_px:{width:k.width,height:k.height},
   pixel_stride_xy:[2,2],frame_stride:1,
   original_capture_resolution_px:null,is_original_capture:false};
 validateScene(scene);
 const mutate=(fn)=>{const bad=structuredClone(scene);fn(bad.recording.preview_provenance);
   assert.throws(()=>validateScene(bad),/预览分辨率/);};
 mutate(p=>p.exported_preview_resolution_px.width++);
 mutate(p=>p.loaded_recording_resolution_px.height++);
 mutate(p=>p.pixel_stride_xy[0]=1);
 mutate(p=>p.is_original_capture=true);
 mutate(p=>p.original_capture_resolution_px={width:640,height:480});
 delete scene.recording.preview_provenance; // legacy scene remains loadable
 validateScene(scene);
});
if(process.env.ECHO_NATIVE_SCENE)test('Python export obeys browser contract',()=>{
 const s=JSON.parse(readFileSync(process.env.ECHO_NATIVE_SCENE,'utf8'));
 validateScene(s);
 assert.ok(unpackScene(s).faces.length>0);
 const p=s.recording.preview_provenance;
 assert.equal(p?.is_original_capture,false);
 assert.equal(p?.original_capture_resolution_px,null);
 assert.deepEqual(p?.exported_preview_resolution_px,
   {width:s.recording.intrinsics.width,height:s.recording.intrinsics.height});
});
