import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const root='C:/Users/15637/PycharmProjects/ISAA';
const build=path.join(root,'.ntu60_classes_revision');
const base=path.join(root,'outputs/基于133节点全身骨架的动作识别研究_SkeleT简介补充版.pptx');
const final=path.join(root,'outputs/基于133节点全身骨架的动作识别研究_60类动作发表版.pptx');
const receipt=JSON.parse(await fs.readFile(path.join(build,'validation-v3.json'),'utf8'));
const actualHash=createHash('sha256').update(await fs.readFile(final)).digest('hex');
assert.equal(actualHash,receipt.finalSha256);
const classes=JSON.parse(await fs.readFile(path.join(build,'verified_classes.json'),'utf8')).classes;
const sourceDeck=await PresentationFile.importPptx(await FileBlob.load(base));
const p=await PresentationFile.importPptx(await FileBlob.load(final));
assert.equal(sourceDeck.slides.items.length,12);
assert.equal(p.slides.items.length,13);
const inspected=(await p.inspect({select:'$.slides[2]',include:['id','index','elements.id','elements.kind','elements.text','notes.text']})).records[0];
assert.equal(inspected.index,2);
const tables=inspected.elements.filter(e=>e.kind==='table');
assert.equal(tables.length,4);
const text=tables.map(t=>t.text).join('\n');
for(const c of classes){
 assert.equal(text.split(c.id).length-1,1,`${c.id} appears exactly once`);
 assert(text.includes(`${c.id}\t${c.chinese}`),`${c.id} maps to ${c.chinese}`);
}
for(let originalIndex=0;originalIndex<12;originalIndex++){
 const newIndex=originalIndex<2?originalIndex:originalIndex+1;
 const sourcePreview=await sourceDeck.export({slide:sourceDeck.slides.items[originalIndex],format:'png',scale:1});
 await fs.writeFile(path.join(build,`source-import-${String(originalIndex+1).padStart(2,'0')}.png`),new Uint8Array(await sourcePreview.arrayBuffer()));
 const targetPreview=await p.export({slide:p.slides.items[newIndex],format:'png',scale:1});
 await fs.writeFile(path.join(build,`final-import-${String(newIndex+1).padStart(2,'0')}.png`),new Uint8Array(await targetPreview.arrayBuffer()));
 const a=await sharp(path.join(build,`source-import-${String(originalIndex+1).padStart(2,'0')}.png`)).extract({left:0,top:0,width:1280,height:672}).raw().toBuffer();
 const b=await sharp(path.join(build,`final-import-${String(newIndex+1).padStart(2,'0')}.png`)).extract({left:0,top:0,width:1280,height:672}).raw().toBuffer();
 assert(a.equals(b),`Original slide ${originalIndex+1} content pixels remain unchanged`);
}
const preview=await p.export({slide:p.slides.items[2],format:'png',scale:1});
await fs.writeFile(path.join(build,'final-slide-03.png'),new Uint8Array(await preview.arrayBuffer()));
console.log(JSON.stringify({final,slides:p.slides.items.length,classRows:classes.length,originalSlideBodiesPixelIdentical:true,sha256:actualHash}));
