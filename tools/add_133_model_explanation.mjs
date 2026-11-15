import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';
import sharp from 'sharp';

const ROOT='C:/Users/15637/PycharmProjects/ISAA';
const BUILD=path.join(ROOT,'.model_architecture_revision');
const SOURCE=path.join(ROOT,'outputs','基于133节点全身骨架的动作识别研究_论文原图修订版.pptx');
const FINAL=path.join(ROOT,'outputs','基于133节点全身骨架的动作识别研究_模型结构解释版.pptx');
const FIGURE=path.join(ROOT,'outputs','133节点全身骨架_模型结构总览.png');
const SKILL='C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const C={navy:'#12304A',ink:'#1F2937',muted:'#60758A',teal:'#0B8F82',blue:'#2563EB',orange:'#B96B0B',paper:'#F5F7FA',white:'#FFFFFF'};
const FONT='Microsoft YaHei';
await fs.mkdir(BUILD,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const before=(await p.inspect({select:'$.slides[*]',include:['id','index','elements.kind','elements.text','notes.text']})).ndjson;
await fs.writeFile(path.join(BUILD,'before.ndjson'),before);
const s=p.slides.items[5];
const writeBlob=async(f,b)=>fs.writeFile(f,new Uint8Array(await b.arrayBuffer()));
await writeBlob(path.join(BUILD,'before-slide-06.png'),await p.export({slide:s,format:'png',scale:1}));
await fs.writeFile(path.join(BUILD,'before-slide-06.layout.json'),await(await s.export({format:'layout'})).text());
const targets=await p.inspect({select:'$.slides[5].elements[*]',include:['id']});
p.delete(targets.ndjson.trim().split('\n').filter(Boolean).map(x=>JSON.parse(x).id));
s.background.fill=C.paper;

function tx(text,x,y,w,h,size=20,color=C.ink,bold=false,align='left'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=text;q.text.insets=0;q.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,autoFit:'shrinkText'};return q;
}
function rect(x,y,w,h,fill,line){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:1.5}});}
function line(x,y,x2,y2,col,arrow=false){
 return s.shapes.add({geometry:'connector',kind:'straight',position:{left:x,top:y,width:x2-x,height:y2-y},line:{fill:col,width:2},...(arrow?{tail:{type:'triangle',width:'med',length:'med'}}:{})});
}
function branch(x,y,w,h,title,rows,col){
 rect(x,y,w,h,C.white,col);
 tx(title,x+16,y+12,w-32,28,21,col,true);
 rows.forEach((v,i)=>tx(v,x+16,y+50+i*27,w-32,25,18.5));
}
tx('项目模型结构',72,28,1136,20,13,C.teal,true);
tx('133节点输入的身体与局部并行融合模型',72,60,1136,48,32,C.ink,true);
tx('身体保留整体姿态，双手描述细粒度操作，面部补充局部时空信息',72,113,1136,30,18,C.muted);

// One complete, native editable architecture diagram. All paths are orthogonal.
rect(48,322,172,155,C.navy,C.navy);
tx('133点序列输入',60,337,148,30,21,C.white,true,'center');
tx('每个样本',60,379,148,22,17,C.white,false,'center');
tx('3×100×133×M',55,406,158,25,18.5,C.white,false,'center');
tx('x、y、score',60,439,148,25,18.5,C.white,false,'center');
tx('M≤2，联合点/帧mask',42,486,188,44,16,C.muted,false,'center');

branch(286,176,426,118,'身体分支  22个代表点',[
 '10层CTR编码：48×4，96×3，192×3',
 '有效节点池化，每帧192维特征',
],C.blue);
branch(286,339,426,118,'双手分支  左右各21点，6通道',[
 '共享6层CTR编码：32×2，64×2，96×2',
 '投影至48维，节点池化、双手均值',
],C.teal);
branch(286,502,426,118,'面部分支  68点压缩为6个token',[
 '4层时空图卷积：24、24、48、48',
 '有效节点池化，每帧48维特征',
],C.orange);

rect(762,176,198,118,C.white,C.blue);
tx('身体聚合',775,188,172,28,21,C.blue,true,'center');
tx('时间/人物均值池化\n线性投影 + ReLU\n192维',774,226,174,66,18.5,C.ink,false,'center');
rect(762,385,198,194,C.white,C.teal);
tx('局部融合与聚合',770,399,182,28,21,C.teal,true,'center');
tx('手/脸插值对齐时间\n均值与mask门控\n时间/人物均值池化\n线性投影 + ReLU\n96维',772,439,178,130,18.5,C.ink,false,'center');

rect(1010,270,222,260,C.navy,C.navy);
tx('融合分类',1026,286,190,32,23,C.white,true,'center');
tx('拼接身体与局部特征',1022,332,198,27,19,C.white,false,'center');
tx('192 + 96 = 288',1021,370,200,32,25,C.white,true,'center');
tx('线性分类器 288至60',1021,424,200,30,19,C.white,false,'center');
tx('动作编号与类别',1024,483,194,29,21,C.white,true,'center');

line(220,399,250,399,C.navy);
line(250,235,250,561,C.navy);
for(const [y,c] of [[235,C.blue],[398,C.teal],[561,C.orange]])line(250,y,286,y,c,true);
line(712,235,762,235,C.blue,true);
line(712,398,737,398,C.teal);line(737,398,737,443,C.teal);line(737,443,762,443,C.teal,true);
line(712,561,737,561,C.orange);line(737,527,737,561,C.orange);line(737,527,762,527,C.orange,true);
line(960,235,985,235,C.blue);line(985,235,985,350,C.blue);line(985,350,1010,350,C.blue,true);
line(960,482,985,482,C.teal);line(985,449,985,482,C.teal);line(985,449,1010,449,C.teal,true);

tx('CTR编码学习关节关系与时间变化，mask节点池化处理缺失点，原池化汇总时间与人物特征',72,638,1136,29,18,C.navy,true,'center');
tx('基于133节点全身骨架的动作识别研究',72,682,850,20,11,C.muted);
tx('06',1160,678,48,22,14,C.teal,true,'right');
s.speakerNotes.text=[
 '模型结构依据：isaa/models/body_local_fusion.py，BodyLocalHandCTRWideRelativeFusion及其父类。',
 '与已核验权重的summary.json匹配的架构：rtmw_ctr22_handctr21_wide_crosshanddistdir_facest48_torso_relative_v3。',
 '身体22个代表点并非17身体点加6足点。代码索引：0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,56,91,112。',
 '手部6通道：躯干相对x/y、score、跨手对应点距离除以躯干尺度、单位方向x/y。双手共享编码器。96维输出通过1×1卷积投影至48维。',
 '面部68点按有效mask压缩为6个token，4层LocalSTBlock通道24,24,48,48。',
 '逐帧有效节点求均值后，双手取均值。手与脸以线性插值对齐身体时间长度，mask采用最近邻对齐。局部=(手+脸)/2，再乘hand_valid OR face_valid门控。',
 '身体和局部分别对人物和时间求普通均值，经过Linear+ReLU获得192维和96维，拼接288维后Linear至60类。这里的有效性门控不包含可学习注意力。',
 'CTR块由通道级拓扑细化图卷积、多尺度时序卷积与残差连接组成。身体第5和第8块下采样，手与面部编码保持时间长度。',
 '原LocalSTBlock.forward保留既有torch._dynamo.disable边界，不声称整模型fullgraph编译。',
 '讲解：输入只是一套133点骨架，各区域分支在模型内部并行，输出为同一个60类分类结果。',
].join('\n');

await fs.writeFile(path.join(BUILD,'after-slide-06.layout.json'),await(await s.export({format:'layout'})).text());
const after=(await p.inspect({select:'$.slides[*]',include:['id','index','elements.kind','elements.text','notes.text']})).ndjson;
await fs.writeFile(path.join(BUILD,'after.ndjson'),after);
const a=before.trim().split('\n').map(JSON.parse), b=after.trim().split('\n').map(JSON.parse);
for(let i=0;i<11;i++)if(i!==5 && JSON.stringify(a[i])!==JSON.stringify(b[i]))throw new Error(`Unintended content change on slide ${i+1}`);
for(let i=0;i<11;i++)await writeBlob(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.png`),await p.export({slide:p.slides.items[i],format:'png',scale:1}));
const hi=Buffer.from(await(await p.export({slide:s,format:'png',scale:3})).arrayBuffer());
await sharp(hi).extract({left:36*3,top:156*3,width:1209*3,height:514*3}).png().toFile(FIGURE);
await writeBlob(path.join(BUILD,'montage.webp'),await p.export({format:'webp',montage:true,scale:.4}));
const candidate=path.join(BUILD,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
if(process.argv.includes('--finalize')){
 const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
 console.log(JSON.stringify(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,
  pythonExecutable:'C:/Users/15637/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,4,5,7,8,9].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[2,4,5,7,8,9],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')})));
}
console.log(JSON.stringify({candidate,figure:FIGURE,final:FINAL,unchangedOtherSlides:true}));
