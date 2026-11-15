import fs from 'node:fs/promises';
import path from 'node:path';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const dir='C:/Users/15637/PycharmProjects/ISAA/.ppt_teacher_20261115';
const final='C:/Users/15637/PycharmProjects/ISAA/outputs/ISAA研究報告_教師向け説明版_日本語.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(final));
const out=path.join(dir,'final_render');
await fs.mkdir(out,{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(JSON.stringify({slides:p.slides.items.length,renderDir:out}));
