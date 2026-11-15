import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const ROOT='C:/Users/15637/PycharmProjects/ISAA';
const BUILD=path.join(ROOT,'.ntu60_classes_revision');
const SOURCE=path.join(ROOT,'outputs/基于133节点全身骨架的动作识别研究_SkeleT简介补充版.pptx');
const FINAL=path.join(ROOT,'outputs/基于133节点全身骨架的动作识别研究_60类动作发表版.pptx');
const SKILL='C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const FONT='Microsoft YaHei';
const C={ink:'#1F2937',navy:'#12304A',muted:'#60758A',teal:'#0B8F82',line:'#CFD8E3',band:'#EDF4F8'};
const chinese=[
 '喝水','吃饭或吃零食','刷牙','梳头','掉落物品','捡起物品','扔东西','坐下','从坐姿站起','拍手','阅读','写字','撕纸','穿外套','脱外套',
 '穿鞋','脱鞋','戴眼镜','摘眼镜','戴帽子','摘帽子','欢呼','挥手','踢东西','手伸进口袋','单脚跳','跳起','打电话或接电话','玩手机或平板','键盘打字',
 '用手指向某物','自拍','看手表时间','搓手','点头或鞠躬','摇头','擦脸','敬礼','双手合十','双手交叉示停','打喷嚏或咳嗽','踉跄','跌倒','触头（头痛）','触胸（腹痛或心痛）',
 '触背（背痛）','触颈（颈痛）','恶心或呕吐','扇风','拳击或掌掴他人','踢他人','推他人','拍他人后背','用手指向他人','拥抱他人','给他人物品','触碰他人口袋','握手','相向走近','相互走开',
];
assert.equal(chinese.length,60);
const official=await fs.readFile(path.join(BUILD,'official_readme.md'),'utf8');
const sourceRows=[...official.matchAll(/^\* A(\d+)\.\s+([^\r\n]+)$/gm)]
 .map(m=>({number:Number(m[1]),english:m[2].trim().replace(/\.$/,'')}))
 .filter(r=>r.number<=60);
assert.deepEqual(sourceRows.map(r=>r.number),Array.from({length:60},(_,i)=>i+1));
assert.equal(sourceRows[21].english,'cheer up');
assert.equal(sourceRows[22].english,'hand waving');
const classes=sourceRows.map((r,i)=>({...r,id:`A${String(r.number).padStart(3,'0')}`,chinese:chinese[i]}));
await fs.writeFile(path.join(BUILD,'verified_classes.json'),JSON.stringify({source:'https://github.com/shahroudy/NTURGB-D#action-classes',classes},null,2));

const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const query={select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','notes.text']};
const before=(await p.inspect(query)).records;
assert.equal(before.length,12);
await fs.writeFile(path.join(BUILD,'before.json'),JSON.stringify(before,null,2));
const s=p.slides.add();
s.moveTo(2);
assert.equal(p.slides.items[2].id,s.id);
s.background.fill='#FFFFFF';
function tx(text,x,y,w,h,size=20,color=C.ink,bold=false,alignment='left'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=text;
 q.text.insets=0;
 q.text.style={typeface:FONT,fontSize:size,color,bold,alignment,autoFit:'shrinkText'};
 return q;
}
tx('数据集与识别类别',82,28,1120,20,13,C.teal,true);
tx('NTU RGB+D 60类动作',82,60,1120,48,32,C.ink,true);
tx('A001至A060，按数据集官方类别顺序列出',82,110,1120,22,18,C.muted);
for(let block=0;block<4;block++){
 const values=[['编号','动作名称'],...classes.slice(block*15,(block+1)*15).map(r=>[r.id,r.chinese])];
 const t=s.tables.add({rows:16,columns:2,left:82+block*287,top:136,width:265,height:496,columnWidths:[66,199],values});
 t.styleOptions={headerRow:false,bandedRows:false};
 t.borders.assign({style:'solid',fill:C.line,width:0.6});
 for(let row=0;row<16;row++){
  t.rows[row].height=31;
  for(let col=0;col<2;col++){
   const cell=t.getCell(row,col);
   cell.fill=row===0?C.navy:(row%2===0?C.band:'#FFFFFF');
   cell.text.style={typeface:FONT,fontSize:18,color:row===0?'#FFFFFF':(col===0?C.teal:C.ink),bold:row===0||col===0,alignment:col===0?'center':'left',autoFit:'shrinkText'};
  }
 }
 t.cells.block({row:0,column:0,rowCount:16,columnCount:2}).assign({margins:{left:6,right:6,top:2,bottom:2},anchor:'center'});
}
tx('类别来源：NTU RGB+D 官方仓库。中文名称依据官方英文释义翻译。',82,655,1120,20,13,C.muted);
tx('基于133节点全身骨架的动作识别研究',82,686,900,18,11,C.muted);
tx('03',1160,682,48,22,14,C.teal,true,'right');
s.speakerNotes.text=[
 'Source: NTU RGB+D official dataset repository, Action Classes. https://github.com/shahroudy/NTURGB-D#action-classes . Raw source: https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/README.md . Accessed 2026-10-19.',
 'The 60 actions are the official A1-A60 categories, displayed as three-digit A001-A060 to match video filenames. Chinese names are translations of the official English definitions. A022 is cheer up (欢呼), and A023 is hand waving (挥手). No accuracy values are added or changed.',
 ...classes.map(r=>`${r.id}: ${r.english} / ${r.chinese}`),
].join('\n');

// Keep every original slide intact except the shifted footer page number.
for(let old=2;old<12;old++){
 const oldLabel=String(old+1).padStart(2,'0');
 const rec=(await p.inspect({select:`$.slides[${old+1}].elements[*]`,include:['id','text']})).records;
 const labels=rec.filter(r=>r.text===oldLabel);
 assert.equal(labels.length,1,`Unique footer page number on source slide ${old+1}`);
 p.resolve(labels[0].id).text.replace(oldLabel,String(old+2).padStart(2,'0'));
}
const after=(await p.inspect(query)).records;
await fs.writeFile(path.join(BUILD,'after.json'),JSON.stringify(after,null,2));
for(let old=0;old<12;old++){
 const a=structuredClone(before[old]);
 const b=structuredClone(after[old<2?old:old+1]);
 delete a.index;delete b.index;
 if(old>=2)a.elements.find(e=>e.text===String(old+1).padStart(2,'0')).text=String(old+2).padStart(2,'0');
 assert.deepEqual(b,a,`Original content on source slide ${old+1}`);
}
const added=after[2].elements.filter(e=>e.kind==='table').map(e=>e.text).join('\n');
for(const r of classes){
 assert.equal(added.split(r.id).length-1,1,`${r.id} appears once`);
 assert(added.includes(`${r.id}\t${r.chinese}`),`Correct mapping for ${r.id}`);
}
const candidate=path.join(BUILD,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
await fs.writeFile(path.join(BUILD,'classes.layout.json'),await(await s.export({format:'layout',detail:'full'})).text());
if(process.argv.includes('--finalize')){
 const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
 const owners=[2,3,5,6,8,9,11];
 console.log(JSON.stringify(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,
  pythonExecutable:'C:/Users/15637/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:13,requiredNativeTableOwnerSlides:owners,fontPolicy:{basis:'reference',families:[FONT],referencePath:SOURCE,referenceSha256:createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation-v3.json')})));
}
console.log(JSON.stringify({candidate,final:FINAL,slideCount:13,classes:60,previousContentPreserved:true}));
