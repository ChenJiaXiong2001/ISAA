import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const input='C:/Users/15637/PycharmProjects/ISAA/outputs/高精度全身骨架细粒度动作识别研究汇报.pptx';
const out='C:/Users/15637/PycharmProjects/ISAA/.ppt_build/rendered';
await fs.mkdir(out,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(input));
for (const [i,s] of p.slides.items.entries()) {
  const b=await p.export({slide:s,format:'png',scale:1});
  await fs.writeFile(`${out}/slide-${String(i+1).padStart(2,'0')}.png`, new Uint8Array(await b.arrayBuffer()));
}
console.log(out);
