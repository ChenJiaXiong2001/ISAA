import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import sharp from 'sharp';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const ROOT='C:/Users/15637/PycharmProjects/ISAA';
const BUILD=path.join(ROOT,'.skelet_intro_revision');
const SOURCE=path.join(ROOT,'outputs/基于133节点全身骨架的动作识别研究_背景与单流结果修订版.pptx');
const FINAL=path.join(ROOT,'outputs/基于133节点全身骨架的动作识别研究_SkeleT简介补充版.pptx');
const SKILL='C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const FONT='Microsoft YaHei';
const C={ink:'#1F2937',navy:'#12304A',muted:'#60758A',teal:'#0B8F82',blue:'#2563EB'};
await fs.mkdir(BUILD,{recursive:true});
// Crop the original paper figure only. No redrawing or generated visual content.
const FIG=path.join(BUILD,'skelet-figure2.png');
await sharp(path.join(BUILD,'paper-page-4.png')).extract({left:243,top:164,width:915,height:438}).png().toFile(FIG);
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const query={select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','notes.text']};
const before=(await p.inspect(query)).records;
await fs.writeFile(path.join(BUILD,'before.json'),JSON.stringify(before,null,2));
const originalSlides=[...p.slides.items];
const s=p.slides.add();
s.moveTo(8);
if(p.slides.items[8].id!==s.id)throw new Error('Intro slide order is incorrect');
s.background.fill='#FFFFFF';
function tx(text,x,y,w,h,size=21,color=C.ink,bold=false,align='left'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=text;q.text.insets=0;q.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,autoFit:'shrinkText'};return q;
}
tx('不足与后续发现',82,28,1130,20,13,C.teal,true);
tx('SkeleT 骨架变换方法',82,60,1120,48,32,C.ink,true);
tx('Skeleton Transformation：可嵌入CTR-GCN等骨干的骨架加权与压缩策略',82,119,1120,34,22,C.navy);
const bytes=await fs.readFile(FIG);
s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:'SkeleT原论文Figure 2：关键点选择、骨架变换与可选的多人实例池化',fit:'contain',position:{left:72,top:174,width:770,height:380}});
tx('输入选择',872,176,336,32,23,C.teal,true);
tx('133点中去掉68个面部点。\n身体17点、脚6点、双手42点，\n共65点。',872,216,336,112,21);
tx('分组映射',872,336,336,32,23,C.teal,true);
tx('特征按通道分组，各组独立\n学习映射。普通块加权节点，\n下采样块合并相关节点。',872,376,336,112,21);
tx('65点逐步压缩至27点、11点',82,570,1120,36,27,C.blue,true);
tx('各组图卷积后拼接，再做时间卷积。SkeleT-CTRGCN即将该策略结合CTR-GCN骨干。',82,615,1120,30,20,C.navy);
tx('图源：Yang等，2024，Figure 2。图(c)为可选多人池化，当前本地NTU实验未启用。',82,651,1120,22,14,C.muted);
tx('基于133节点全身骨架的动作识别研究',82,686,900,18,11,C.muted);
tx('09',1160,682,48,22,14,C.teal,true,'right');
s.speakerNotes.text='Source: Yijie Yang, Jinlu Zhang, Jiaxu Zhang, Zhigang Tu. Expressive Keypoints for Skeleton-based Action Recognition via Skeleton Transformation. arXiv:2406.18011 (2024), Sections 3.1, 3.2 and Figure 2. https://arxiv.org/abs/2406.18011 . Original Figure 2 cropped from the locally retained paper page 4. The authors select 65 expressive keypoints by removing the 68 face landmarks from COCO-WholeBody 133. Grouped Mapping splits features by channel, learns per-group node mappings, reweights nodes in normal blocks and reduces node count in downsampling blocks, using 65, 27 and 11 nodes. Group GCN outputs are concatenated before temporal convolution. SkeleT can wrap CTR-GCN and other GCN backbones. Figure 2(c) Instance Pooling is an optional multi-person extension, not enabled in the current local NTU experiment. This slide describes the paper method, not a new accuracy result. The project itself retains 133 input points and parallel body/local branch encoding.';

// Preserve all previous content, editing only shifted page numbers.
for(let old=8;old<11;old++){
 const slide=originalSlides[old];
 const oldLabel=String(old+1).padStart(2,'0');
 const slideRecords=(await p.inspect({select:`$.slides[${old+1}].elements[*]`,include:['id','text']})).records;
 const matches=slideRecords.filter(r=>r.text===oldLabel);
 if(matches.length!==1)throw new Error(`Page label ${oldLabel}: ${matches.length}`);
 p.resolve(matches[0].id).text.replace(oldLabel,String(old+2).padStart(2,'0'));
 if(p.slides.items[old+1].id!==slide.id)throw new Error('Existing slide order changed');
}
const after=(await p.inspect(query)).records;
await fs.writeFile(path.join(BUILD,'after.json'),JSON.stringify(after,null,2));
for(let i=0;i<11;i++){
 const a=structuredClone(before[i]);
 const b=structuredClone(after[i<8?i:i+1]);
 delete a.index;delete b.index;
 if(i>=8){const label=String(i+1).padStart(2,'0');a.elements.find(e=>e.text===label).text=String(i+2).padStart(2,'0');}
 if(JSON.stringify(a)!==JSON.stringify(b))throw new Error(`Unintended content change on source slide ${i+1}`);
}
const candidate=path.join(BUILD,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const writeBlob=async(file,blob)=>fs.writeFile(file,new Uint8Array(await blob.arrayBuffer()));
for(let i=0;i<p.slides.items.length;i++)await writeBlob(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.png`),await p.export({slide:p.slides.items[i],format:'png',scale:1}));
await fs.writeFile(path.join(BUILD,'intro.layout.json'),await(await s.export({format:'layout'})).text());
if(process.argv.includes('--finalize')){
 const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
 const owners=[2,4,5,7,8,10];
 console.log(JSON.stringify(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,
  pythonExecutable:'C:/Users/15637/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:owners,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')})));
}
console.log(JSON.stringify({candidate,final:FINAL,slides:p.slides.items.length,previousContentPreserved:true}));
