import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const ROOT='C:/Users/15637/PycharmProjects/ISAA';
const BUILD=path.join(ROOT,'.background_single_stream_revision');
const SOURCE=path.join(ROOT,'outputs','基于133节点全身骨架的动作识别研究_模型结构解释版.pptx');
const FINAL=path.join(ROOT,'outputs','基于133节点全身骨架的动作识别研究_背景与单流结果修订版.pptx');
const SKILL='C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const FONT='Microsoft YaHei';
const C={navy:'#12304A',ink:'#1F2937',muted:'#60758A',teal:'#0B8F82',blue:'#2563EB',paper:'#F5F7FA',line:'#CFD8E3',white:'#FFFFFF'};
await fs.mkdir(BUILD,{recursive:true});
const E=JSON.parse(await fs.readFile(path.join(ROOT,'outputs/evaluations/pool_weighted_100_cosine_20261016/summary.json'),'utf8'));
const baseline=89.9;
const improvement=(E.top1_percent-baseline).toFixed(4);
if(E.split!=='xsub60'||E.samples!==16306||E.correct!==15432||improvement!=='4.7400')throw new Error('Unexpected project evaluation identity');

// Verify the original authors' README links to the public checkpoint directory,
// then follow NTU60_Xsub and its explicitly labelled Joint-only folder.
const sourceUrls={
 readme:'https://raw.githubusercontent.com/Uason-Chen/CTR-GCN/main/README.md',
 root:'https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd',
 xsub:'https://drive.google.com/drive/folders/17nXNGySMhQDpCQ5oEhXYc8J-Wr62gtT3',
 joint:'https://drive.google.com/drive/folders/1b_r1gGuWCxn-YBPij3R4BULkd-HASp7e',
};
const fetched=Object.fromEntries(await Promise.all(Object.entries(sourceUrls).map(async([k,url])=>{
 const response=await fetch(url);if(!response.ok)throw new Error(`${url}: ${response.status}`);
 return [k,await response.text()];
})));
if(!fetched.readme.includes('1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd'))throw new Error('Official README link missing');
if(!fetched.root.includes('NTU60_Xsub')||!fetched.root.includes('17nXNGySMhQDpCQ5oEhXYc8J-Wr62gtT3'))throw new Error('NTU60 XSub folder not confirmed');
if(!fetched.xsub.includes('CTRGCN_joint_89.9')||!fetched.xsub.includes('1b_r1gGuWCxn-YBPij3R4BULkd-HASp7e'))throw new Error('Joint 89.9 folder not confirmed');
if(!fetched.joint.includes('runs-60-37560.pt'))throw new Error('Official Joint checkpoint not confirmed');
for(const [key,content]of Object.entries(fetched))await fs.writeFile(path.join(BUILD,`official-${key}.${key==='readme'?'md':'html'}`),content);
await fs.writeFile(path.join(BUILD,'verified-comparison.json'),JSON.stringify({
 checkedAt:new Date().toISOString(),dataset:'NTU RGB+D 60',split:'X-Sub',baseline:{model:'Original CTR-GCN',stream:'Joint only',top1_percent:baseline,source_type:'Original authors public pretrained checkpoint folder label',folder:'NTU60_Xsub/CTRGCN_joint_89.9',checkpoint:'runs-60-37560.pt',urls:sourceUrls},
 project:{top1_percent:E.top1_percent,display_top1:E.top1_percent.toFixed(4),samples:E.samples,correct:E.correct,checkpoint_sha256:E.checkpoint_sha256},
 improvement_percentage_points:Number(improvement),caveat:'Different input sources, node representations and temporal preprocessing. A system-level published-result difference, not a controlled model-only ablation.',
 source_sha256:Object.fromEntries(Object.entries(fetched).map(([k,v])=>[k,createHash('sha256').update(v).digest('hex')]))
},null,2));

const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const before=(await p.inspect({select:'$.slides[*]',include:['id','index','elements.kind','elements.text','notes.text']})).ndjson;
await fs.writeFile(path.join(BUILD,'before.ndjson'),before);
const writeBlob=async(f,b)=>fs.writeFile(f,new Uint8Array(await b.arrayBuffer()));
for(const i of [1,7,10]){
 await writeBlob(path.join(BUILD,`before-slide-${i+1}.png`),await p.export({slide:p.slides.items[i],format:'png',scale:1}));
 await fs.writeFile(path.join(BUILD,`before-slide-${i+1}.layout.json`),await(await p.slides.items[i].export({format:'layout'})).text());
}
async function exact(slideIndex,text){
 const result=await p.inspect({select:`$.slides[${slideIndex}].elements[*]`,include:['id','kind','text']});
 const matches=result.ndjson.trim().split('\n').filter(Boolean).map(JSON.parse).filter(x=>x.text===text);
 if(matches.length!==1)throw new Error(`Expected one target for ${text}, got ${matches.length}`);
 return p.resolve(matches[0].id);
}
async function replaceText(i,oldText,newText,position,style){
 const q=await exact(i,oldText);q.text.replace(oldText,newText);
 if(position)q.position=position;
 if(style)q.text.style={typeface:FONT,color:C.ink,autoFit:'shrinkText',...style};
 return q;
}
function tx(s,text,x,y,w,h,size=20,color=C.ink,bold=false,alignment='left'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=text;q.text.insets=0;q.text.style={typeface:FONT,fontSize:size,color,bold,alignment,autoFit:'shrinkText'};return q;
}
function table(s,rows,x,y,w,h,widths){
 const t=s.tables.add({rows:rows.length,columns:rows[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values:rows});
 t.borders.assign({style:'solid',fill:C.line,width:1});
 for(let i=0;i<rows.length;i++)for(let j=0;j<rows[0].length;j++){
  const c=t.getCell(i,j);c.fill=i===0?C.navy:(i%2?C.white:'#EDF3F7');
  c.text.style={typeface:FONT,fontSize:i===1&&j>0?28:19,color:i===0?C.white:(i===1&&j>0?C.blue:C.ink),bold:i===0||i===1};
 }
 return t;
}
const officialNotes='Original CTR-GCN Joint-only baseline: official repository https://github.com/Uason-Chen/CTR-GCN README links to '+sourceUrls.root+'. NTU60_Xsub folder '+sourceUrls.xsub+' contains CTRGCN_joint_89.9 ('+sourceUrls.joint+'), checkpoint runs-60-37560.pt. The 89.9% value is the original authors public pretrained Joint-only result, not the multi-stream main-table number. Verified 2026-10-19. Project independent Top-1 94.64000981233902%; difference 4.74000981233902 percentage points, displayed 4.7400. Inputs and preprocessing differ, so this is not a controlled model-only ablation.';

// Slide 2: preserve the existing application image and native project table.
await replaceText(1,'研究目标','研究背景与目标');
await replaceText(1,'全身骨架中的细粒度动作信息','稀疏骨架中的手部细节不足');
await replaceText(1,'身体提供整体运动与姿态','精细动作需要更完整的手部表示',
 {left:445,top:178,width:730,height:44},{fontSize:28,bold:true,color:C.navy});
await replaceText(1,'双手补充手指形态及双手相对关系，面部与足部提供局部线索。项目以单套关节输入完成60类动作识别。',
 '传统GCN多采用较少关节，手部细节表达有限。\nNTU 25点虽含手端与拇指点，但缺少完整手指关节。\n为加强精细动作识别，本项目采用133点全身骨架，补充双手各21点，研究阅读、写字、玩手机等动作。',
 {left:445,top:240,width:730,height:165},{fontSize:22});
const oldScope=await exact(1,'项目组成\t本研究采用的方案\n输入表示\tRTMW 133点，二维坐标与置信度\n模型\t身体与局部分支并行编码和融合\n评估\tNTU60 X-Sub，16,306个验证样本\n应用展示\t人物框、对齐骨架、动作编号和中文名称');
oldScope.position={left:445,top:430,width:730,height:200};
p.slides.items[1].speakerNotes.text+='\nBackground: original NTU RGB+D skeleton has 25 joints, including hand tips and thumbs, but not full articulated finger chains. This motivates richer hand detail; no claim that all traditional GCNs use few nodes or that overall accuracy proves per-class fine-action gains. CTR-GCN original skeleton: https://github.com/Uason-Chen/CTR-GCN/blob/main/graph/ntu_rgb_d.py .';

// Slide 8: quantitative Joint-only comparison, still after the full model explanation.
const s=p.slides.items[7];
const targets=await p.inspect({select:'$.slides[7].elements[*]',include:['id']});
p.delete(targets.ndjson.trim().split('\n').filter(Boolean).map(v=>JSON.parse(v).id));
s.background.fill=C.paper;
tx(s,'原始CTR-GCN Joint单流',72,28,1136,20,13,C.teal,true);
tx(s,'NTU60 X-Sub单流准确率与提升',72,60,1136,48,32,C.ink,true);
tx(s,'原始CTR-GCN采用原作者公开Joint单流结果，本项目采用最佳权重独立评估',72,113,1136,30,18,C.muted);
table(s,[
 ['维度','原始CTR-GCN Joint单流','本项目原池化'],
 ['Top-1','89.9%',`${E.top1_percent.toFixed(4)}%`],
 ['评测划分','NTU60 X-Sub','NTU60 X-Sub'],
 ['骨架表示','NTU原始25点，三维坐标','RTMW 133点，二维坐标与score'],
 ['局部建模','原始人体关节图','双手21点、跨手距离方向、面部token'],
 ['分类方式','Joint单流','单套关节输入，身体局部分支融合'],
],72,162,1136,294,[170,420,546]);
tx(s,`本项目提高 ${improvement} 个百分点`,72,482,1136,58,39,C.blue,true,'center');
tx(s,'数据来源：CTR-GCN官方预训练模型 NTU60_Xsub / CTRGCN_joint_89.9',72,560,1136,27,17,C.muted);
tx(s,'两者均为关节单流，但输入来源、节点维度和时序处理不同。\n上述提升为方案结果差异，尚不能归因于某个结构模块。',72,605,1136,59,19,C.navy);
tx(s,'基于133节点全身骨架的动作识别研究',72,682,850,20,11,C.muted);
tx(s,'08',1160,678,48,22,14,C.teal,true,'right');
s.speakerNotes.text=officialNotes+'\n'+JSON.parse(before.trim().split('\n')[6]).notes.text;

// Slide 11: synchronize the conclusion without changing its visual hierarchy.
await replaceText(10,'身体与局部并行建模，双手共享编码并引入跨手几何关系。\n真实RGB样例完成骨架对齐与整段动作识别展示。\n相似动作区分及同协议基准评估仍需改进。',
 '原始CTR-GCN官方Joint单流89.9%，本项目高4.7400个百分点。\n身体与局部并行建模，双手共享编码并引入跨手几何关系。\n真实RGB样例展示骨架对齐与整段动作识别。\n输入设置不同，相似动作区分与同协议评估仍需改进。',
 {left:85,top:437,width:1090,height:190},{fontSize:23,color:C.white});
p.slides.items[10].speakerNotes.text+='\n'+officialNotes;

const after=(await p.inspect({select:'$.slides[*]',include:['id','index','elements.kind','elements.text','notes.text']})).ndjson;
await fs.writeFile(path.join(BUILD,'after.ndjson'),after);
const a=before.trim().split('\n').map(JSON.parse),b=after.trim().split('\n').map(JSON.parse);
for(let i=0;i<11;i++)if(![1,7,10].includes(i)&&JSON.stringify(a[i])!==JSON.stringify(b[i]))throw new Error(`Unintended change on slide ${i+1}`);
const candidate=path.join(BUILD,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<11;i++)await writeBlob(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.png`),await p.export({slide:p.slides.items[i],format:'png',scale:1}));
for(const i of [1,7,10])await fs.writeFile(path.join(BUILD,`after-slide-${i+1}.layout.json`),await(await p.slides.items[i].export({format:'layout'})).text());
await writeBlob(path.join(BUILD,'montage.webp'),await p.export({format:'webp',montage:true,scale:.4}));
if(process.argv.includes('--finalize')){
 const{finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
 console.log(JSON.stringify(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,
 pythonExecutable:'C:/Users/15637/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,4,5,7,8,9].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[2,4,5,7,8,9],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')})));
}
console.log(JSON.stringify({candidate,final:FINAL,baseline,improvement,unchangedOtherSlides:true}));
