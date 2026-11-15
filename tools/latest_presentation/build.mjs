import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { applyPresentationChartFont } from 'file:///C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations/container_tools/artifact_tool_utils.mjs';

const root='C:/Users/15637/PycharmProjects/ISAA';
const dir=path.join(root,'.ppt_teacher_20261115');
const out=path.join(root,'outputs');
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#F7F8FA',navy:'#17324D',blue:'#2F6B9A',teal:'#1B8A8A',ink:'#1D2733',muted:'#657381',line:'#D9E1E8',white:'#FFFFFF',orange:'#D9822B'};
const font='Arial';
const entries=[];
function text(s,v,x,y,w,h,size=20,color=C.ink,bold=false,align='left'){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=v;t.text.insets=0;t.text.style={fontFamily:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle'};return t;
}
function box(s,x,y,w,h,fill=C.white,line=C.line){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:line,width:line==='none'?0:1}});}
function arrow(s,x,y,w=36,h=28,down=false){return s.shapes.add({geometry:down?'downArrow':'rightArrow',position:{left:x,top:y,width:w,height:h},fill:C.teal,line:{fill:'none',width:0}});}
function add(key,title,sub,notes=''){
 const s=p.slides.add();s.background.fill=C.bg;
 entries.push({key,s,title});
 if(title)text(s,title,72,48,1136,60,34,C.navy,true);
 if(sub)text(s,sub,74,111,1132,36,17,C.muted);
 s.speakerNotes.text=notes;return s;
}
function table(s,values,x,y,w,h,widths,size=18,compact=false){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 t.borders.assign({style:'solid',fill:C.line,width:0.6});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:compact?4:10,right:compact?4:10,top:compact?1:5,bottom:compact?1:5}});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=h/values.length;
  for(let c=0;c<values[0].length;c++){
   const q=t.getCell(r,c);q.fill=r===0?C.navy:r%2?C.white:'#EEF3F6';
   q.text.style={fontFamily:font,fontSize:size,bold:r===0,color:r===0?C.white:C.ink};
  }
 }
 return t;
}
const notesModel='出典：isaa/models/body_local_fusion.py（BodyLocalHandCTRWideRelativeFusion と親クラス）、isaa/models/face_pair_fusion.py（FacePairFusion）。身体22点、左右手各21点、顔68点を6領域に圧縮。身体192次元、手は96チャネルから48次元へ写像、顔48次元。元の局所投影は48→96であり、従来分類器は192+96=288入力。新方式は96入力と240入力の線形分類器。原始CTR-GCNの学習済み重みを利用したモデルではなく、CTR型の構造を用いた本研究モデル。';
const notesFusion='出典：docs/face_pair_validation/best_epoch83_accuracy_weighted/report.md、summary.json、per_class_accuracy.csv、tools/evaluate_face_pair_accuracy_weighted.py。校正3,191件（被験者9・18）。正式検証は重み算出に不使用。T=1。計算はlog_softmax、logaddexp、logsumexpで安定化。全クラスに固定のクラス別重みを適用し、現在の正解ラベルは利用しない。';
const notesBaseline='出典：outputs/evaluations/pool_weighted_100_cosine_20261016/summary.json、config.json、docs/best_weighted_100_cosine.md、docs/face_pair_validation/full_config_stability.json、best_epoch83_accuracy_weighted/summary.json。従来モデル15,432/16,306、新方式15,536/16,306。値は独立評価の値で、訓練ログの94.6461%と区別する。';
const paper='https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.pdf';
const official='https://github.com/Uason-Chen/CTR-GCN';
const notesCtr=`出典：Chen et al., Channel-Wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition, ICCV 2021, Table 6。${paper}\n公式実装：${official}\n本研究の再評価：outputs/original_ctrgcn_facepair_centered_20261108/summary.json、per_class.csv、paired_predictions.csv、tools/compare_original_ctrgcn_face_pair_20261108.py。92.4%は論文4ストリームの公表値、89.9%はJoint公開重みの保存先表示。再評価88.27985465%は14,334/16,237、本研究95.28237975%は15,471/16,237。公式重みを使った再評価であり、CTR-GCNを本研究で再学習した結果ではない。`;

// 1  Cover
{
 const s=add('cover','','',notesBaseline);
 s.background.fill=C.navy;
 text(s,'全身骨格による動作認識',112,104,1080,80,46,C.white,true);
 text(s,'特徴をまとめる方式と、予測を組み合わせる方式の比較',112,206,1070,58,29,'#D9EAF4');
 text(s,'研究期間：2026年10月19日〜11月8日',112,303,1000,36,20,'#B9C9D6');
 text(s,'検証正解率  94.64% から 95.28%',112,405,1060,58,32,C.white,true);
 text(s,'本研究の2方式と、原版 CTR-GCN の評価結果',112,479,1060,36,21,'#D9EAF4');
 text(s,'研究報告',1030,622,170,36,22,'#B9C9D6',true,'right');
}
// 2  Input and task definitions
{
 const s=add('input','認識対象と入力データ','1本の動画に含まれる動作を、60種類から1つ選ぶ',notesModel+'\nNTU60はNTU RGB+Dの60動作データセット。X-Subは被験者で学習・検証を分離する評価方法。');
 const v=[['用語','この研究での意味'],['動作クラス','「水を飲む」「読書」などの動作の種類。今回は60種類'],['全身骨格・キーポイント','身体・手・顔の位置を表す点。1人につき133点を扱う'],['RTMW','RGB動画から2次元の全身キーポイントを推定する姿勢推定器'],['各点の入力値','横位置 x、縦位置 y、検出の信頼度の3値。信頼度は位置推定の確かさ'],['NTU60 / X-Sub','使用データセット / 学習に使う人物と検証に使う人物を分ける評価方法']];
 table(s,v,72,174,1136,380,[280,856],21);
 text(s,'研究課題：身体・手・顔の情報を、どの段階で組み合わせるか',80,593,1120,48,24,C.blue,true);
}
// 3  Preprocessing workflow
{
 const s=add('preprocess','データ処理の流れ','動画から骨格を作成し、全動画を同じ入力サイズにそろえる',notesModel+'\n出典：docs/face_pair_validation/best_epoch83_accuracy_weighted/summary.json、isaa/data/skeleton_dataset.py。');
 const items=[['RGB動画','カラー映像'],['姿勢推定','RTMWが各人物の\n133点を推定'],['骨格の整形','x・y・信頼度\n無効な点を処理'],['時間長の統一','100フレーム\n最大2人']];
 for(let i=0;i<4;i++){
  const x=72+i*296;box(s,x,191,248,132,C.white,C.line);
  text(s,items[i][0],x+12,204,224,35,24,C.navy,true,'center');
  text(s,items[i][1],x+12,251,224,57,19,C.ink,false,'center');
  if(i<3)arrow(s,x+258,239,28,24);
 }
 text(s,'長い動画',80,374,210,32,23,C.navy,true);
 text(s,'複数フレームをまとめ、検出信頼度の高い点を重くして100フレームに圧縮',290,370,910,52,21);
 text(s,'短い動画・少人数',80,447,250,34,23,C.navy,true);
 text(s,'不足する時間や人物の領域をゼロで埋める',350,444,850,46,21);
 text(s,'モデルへの入力',80,540,300,35,23,C.navy,true);
 text(s,'3値 × 100フレーム × 133点 × 最大2人',80,588,1110,48,29,C.blue,true,'center');
 text(s,'検証時には、座標を変えるなどのデータ拡張を行わない',80,647,1110,26,16,C.muted,false,'center');
}
// 4  Define streams before any model comparison
{
 const s=add('streams','入力ストリームと、モデル内部の分岐','ストリームは入力表現ごとの認識経路、分岐は1つのモデル内部の分類経路',notesCtr+'\n本研究では関節座標から作る距離・方向を同じモデル内の補助特徴として利用し、独立したストリームとは数えない。');
 const v=[['方式','入力と分類の構成'],['本研究の単一ストリーム','1種類の2次元全身骨格を入力。身体・手・顔を同じモデル内で処理'],['本研究の内部2分岐','同じ入力から「手＋顔」「身体＋顔」の2つの分類確率を計算'],['CTR-GCN の Joint 単流','関節座標を入力する1つの認識モデル。Joint は「関節」の意味'],['CTR-GCN の4ストリーム','関節座標、骨ベクトル、関節の時間変化、骨の時間変化を別々に分類し融合']];
 table(s,v,72,179,1136,358,[310,826],21);
 text(s,'本研究の95.28%：単一入力ストリーム、モデル内部に2つの分類分岐',80,578,1120,46,23,C.blue,true,'center');
 text(s,'本資料の CTR-GCN は Chen らの原版を指す。SkeleT の CTR-GCN は比較対象に含めない',80,637,1120,34,18,C.muted,false,'center');
}
// 5  Architecture with explicit branch connections
{
 const s=add('architecture','新方式のモデル全体','同じ骨格から特徴を抽出し、2つの分類結果を動作別の重みで組み合わせる',notesModel+notesFusion);
 const input=box(s,45,297,145,128,C.navy,'none');text(s,'全身骨格',55,315,125,30,22,C.white,true,'center');text(s,'133点\n100フレーム',55,358,125,52,18,C.white,false,'center');
 const enc=[['左右の手 各21点','左右共通のCTR型処理','96チャネルを48次元へ'],['顔 68点を6領域へ','領域の時空間グラフ処理','48次元'],['身体 22点','CTR型グラフ処理','192次元']];
 const encShapes=[];
 for(let i=0;i<3;i++){let y=177+i*129;encShapes.push(box(s,276,y,250,103));text(s,enc[i][0],289,y+6,224,29,21,C.navy,true);text(s,enc[i][1],289,y+38,224,27,17);text(s,enc[i][2],289,y+71,224,26,17,C.teal,true);}
 const hf=box(s,606,177,240,103,'#E7F0F7','none');text(s,'手＋顔の分類器',616,186,220,32,22,C.navy,true,'center');text(s,'48＋48＝96次元\n60動作の確率',616,227,220,45,18,C.ink,false,'center');
 const bf=box(s,606,435,240,103,'#E5F4F1','none');text(s,'身体＋顔の分類器',616,444,220,32,22,C.navy,true,'center');text(s,'192＋48＝240次元\n60動作の確率',616,485,220,45,18,C.ink,false,'center');
 const fusion=box(s,900,277,196,166,C.navy,'none');text(s,'動作別の加重融合',911,294,174,34,20,C.white,true,'center');text(s,'2つの確率を合成\n合計を1に調整',911,344,174,62,18,C.white,false,'center');
 const result=box(s,1140,304,105,112,'none','none');text(s,'最終予測',1135,305,115,34,21,C.navy,true,'center');text(s,'最大確率の\n動作を出力',1137,359,111,53,18,C.ink,false,'center');
 const link=(a,b,toSide='left')=>s.shapes.connect(a,b,{kind:'elbow',fromSide:'right',toSide,line:{style:'solid',fill:C.teal,width:1.6},tail:{type:'arrow',width:'sm',length:'sm'}});
 for(const e of encShapes)link(input,e);
 link(encShapes[0],hf);link(encShapes[1],hf,'bottom');link(encShapes[1],bf,'top');link(encShapes[2],bf);link(hf,fusion);link(bf,fusion);link(fusion,result);
 text(s,'手・顔は体幹中心からの相対座標を使用。手には左右対応点の距離・方向も追加',72,557,1136,30,19,C.ink,false,'center');
 text(s,'「次元・チャネル」は特徴を表す数値の個数。「点」は位置を測るキーポイントの個数',72,598,1136,31,20,C.blue,true,'center');
 text(s,'グラフ処理＝点同士の関係と時間変化を学習する処理。分類器＝特徴から60動作の確率を求める処理',72,643,1136,33,17,C.muted,false,'center');
}
// 6  Old/new readout diagrams and exact dimensions
{
 const s=add('versions','従来方式と新方式の具体的な違い','特徴を先にまとめて1回分類する方式と、2回分類した結果を組み合わせる方式',notesModel+notesBaseline);
 text(s,'従来方式：特徴を集約して1回分類',72,173,560,36,25,C.navy,true);
 text(s,'新方式：2つの分類結果を加重融合',686,173,535,36,25,C.teal,true);
 const left=[['手48次元と顔48次元を平均','同じ時刻の特徴を1つにまとめる'],['時間・人物を平均し、48次元を96次元へ','身体の192次元と並べて288次元にする'],['1つの分類器で60動作の確率を計算','最大の確率を持つ動作を出力']];
 const right=[['身体・手・顔を別々に集約','有効な時刻・人物を使って平均する'],['手＋顔96次元、身体＋顔240次元','2つの分類器が別々に60動作を予測'],['動作別の重みで2つの確率を合成','合計を1に調整して最終予測を出力']];
 for(let i=0;i<3;i++){
  let y=235+i*119;
  box(s,72,y,522,91,C.white,C.line);text(s,left[i][0],86,y+8,494,34,20,C.navy,true);text(s,left[i][1],86,y+49,494,30,18);
  box(s,686,y,522,91,C.white,C.line);text(s,right[i][0],700,y+8,494,34,20,C.teal,true);text(s,right[i][1],700,y+49,494,30,18);
  if(i<2){arrow(s,313,y+95,25,18,true);arrow(s,927,y+95,25,18,true);}
 }
 text(s,'両方式とも、両手の距離・方向と、身体・手・顔の特徴抽出器を使用',72,608,1136,32,20,C.blue,true,'center');
 text(s,'新方式は独立にランダム初期化して学習。従来の学習済み重みを引き継がない',72,647,1136,28,18,C.muted,false,'center');
}
// 7  Explain calibration by data roles
{
 const s=add('splits','学習用・重み決定用・検証用のデータ','新方式では、学習側の一部を融合重みの計算専用に分ける',notesFusion+notesBaseline);
 box(s,72,181,1136,68,C.navy,'none');text(s,'NTU60 X-Sub：人物を分けて評価',92,192,1096,44,25,C.white,true,'center');
 box(s,72,285,707,76,'#E7F0F7','none');text(s,'学習側 39,627件',88,303,675,40,27,C.navy,true,'center');
 box(s,828,285,380,207,'#FFF1E4','none');text(s,'正式な検証 16,306件',844,306,348,42,25,C.navy,true,'center');text(s,'学習に使わない人物の動画\n完成した方式の正解率を測る\n融合重みの計算には使わない',849,369,338,94,21,C.ink,false,'center');
 arrow(s,228,370,26,29,true);arrow(s,587,370,26,29,true);
 box(s,72,414,338,144,C.white,C.line);text(s,'パラメータ学習 36,436件',84,430,314,36,23,C.navy,true,'center');text(s,'特徴抽出器と2つの分類器を学習',84,483,314,52,20,C.ink,false,'center');
 box(s,441,414,338,144,'#E5F4F1','none');text(s,'重み決定用 3,191件',453,430,314,36,23,C.teal,true,'center');text(s,'各動作の正解率から重みを算出\n被験者9・18の動画',453,481,314,60,19,C.ink,false,'center');
 text(s,'「校正用データ」は、この重み決定用の3,191件を指す',72,592,1136,36,23,C.blue,true,'center');
 text(s,'従来方式は39,627件すべてをパラメータ学習に使用。新方式と学習データ量が異なる',72,643,1136,32,19,C.muted,false,'center');
}
// 8  All formula symbols, in Japanese, before formulas
{
 const s=add('symbols','融合式で使う記号の意味','過去の正解率から決める「重み」と、現在の動画に対する「確率」を区別する',notesFusion);
 const v=[['記号','意味'],['H / B','H＝手＋顔の分岐（コードのHF）、B＝身体＋顔の分岐（コードのBF）'],['c、k / j','cとkは60種類の候補動作。jはHまたはBの分岐を表す'],['a_j(c)','重み決定用データで、正解が動作cの動画のうち分岐jが判別できた割合'],['w_j(c)','動作cについて、分岐jの予測をどの割合で採用するかを表す固定重み'],['z_j(c)、p_j(c)','今の動画を分岐jで分類した点数zと、その点数から求めた動作cの確率p'],['q(c)、P(c)','2分岐の確率を合成した値qと、全60動作の合計を1に調整した最終確率P']];
 table(s,v,72,170,1136,414,[220,916],20);
 text(s,'重み w は動作ごとに事前決定。確率 p は入力する動画ごとに変わる',72,616,1136,44,23,C.blue,true,'center');
}
// 9  Formula and method
{
 const s=add('formula','融合の計算方法','確率の算術加重平均を、60種類すべての候補動作について計算する',notesFusion);
 text(s,'1  動作別正解率から重みを計算',72,173,560,35,24,C.navy,true);
 text(s,'a_j(c) = 動作cを正しく分類した数 / 動作cの数',72,221,555,43,20,C.ink);
 text(s,'w_H(c) = a_H(c) / [a_H(c) + a_B(c)]',72,280,555,40,22,C.teal,true);
 text(s,'w_B(c) = 1 − w_H(c)',72,331,555,37,22,C.teal,true);
 text(s,'両分岐の正解率の合計が0の場合、各重みを0.5とする',72,384,552,55,18,C.muted);
 text(s,'2  各分岐の点数を確率に変換',690,173,520,35,24,C.navy,true);
 text(s,'p_j(c) = exp(z_j(c)) / Σ[k=1…60] exp(z_j(k))',690,221,520,46,20,C.ink);
 text(s,'softmax：60動作の点数を、合計1の確率へ変換',690,277,520,55,19,C.muted);
 text(s,'3  2つの確率を合成し、合計を1に調整',690,352,520,35,23,C.navy,true);
 text(s,'q(c) = w_H(c) p_H(c) + w_B(c) p_B(c)',690,404,520,40,22,C.teal,true);
 text(s,'P(c) = q(c) / Σ[k=1…60] q(k)',690,457,520,40,22,C.teal,true);
 text(s,'Σは全候補の合計。クラスごとに重みが違うため再調整する',690,508,520,57,18,C.muted);
 text(s,'最終予測 = argmax_c P(c)    （最終確率が最も高い動作を選択）',72,592,1136,43,24,C.navy,true,'center');
 text(s,'推論では全候補に事前の重みを適用する。入力動画の正解ラベルは不要',72,643,1136,31,19,C.blue,true,'center');
}
// 10  Verified calibration examples, hypothetical prediction distinguished
{
 const s=add('example','実データの重みと、計算例','実際の校正結果から、動作ごとに異なる重みを得る',notesFusion+'\n下段のp_H=0.60、p_B=0.40は方法を説明する仮の値。実測の入力動画の出力値ではない。qは再正規化前の値。');
 const v=[['動作','校正件数','手＋顔の正解率','身体＋顔の正解率','手＋顔の重み','身体＋顔の重み'],['水を飲む','53','53/53 = 100.00%','51/53 = 96.23%','50.96%','49.04%'],['よろめく','53','32/53 = 60.38%','52/53 = 98.11%','38.10%','61.90%']];
 table(s,v,48,177,1184,192,[170,100,230,230,227,227],19);
 text(s,'仮の動画で「水を飲む」の確率が、手＋顔60%、身体＋顔40%の場合',72,412,1136,40,22,C.navy,true);
 text(s,'q(水を飲む) = 0.5096 × 0.60 + 0.4904 × 0.40 ≈ 0.5019',72,474,1136,50,27,C.teal,true,'center');
 text(s,'この0.5019は再調整前の値。残り59動作も計算し、全体の合計で割る',72,548,1136,40,21,C.ink,false,'center');
 text(s,'上段は実測した校正値。下段の60%と40%は計算方法を説明するための仮定',72,623,1136,39,19,C.muted,false,'center');
}
// 11  Training differences in plain language, complete technical values
{
 const s=add('training','2つのバージョンの学習・評価条件','共通の入力と検証データを使用するが、学習条件には違いがある',notesBaseline);
 const v=[['項目','従来方式：1つの分類器','新方式：2分岐の加重融合'],['共通の入力・検証','RTMW 2次元133点、100フレーム、最大2人\n同じ16,306件を検証。検証時のデータ拡張なし','左と同じ。両方式とも単一ストリーム'],['パラメータ学習件数','39,627件すべて','36,436件。別の3,191件で融合重みを計算'],['学習の初期値・量','ランダム初期化、100周、乱数設定1\n1回の更新に使う動画32件','左と同じ。従来の学習済み重みは使わない'],['更新方法の共通設定','SGD＋Nesterov、慣性係数0.9\n重み減衰0.0004','左と同じ'],['学習で小さくする誤差','最終分類の誤差を1つ計算','融合の誤差＋0.5×手顔の誤差\n＋0.5×身体顔の誤差'],['学習率・更新の制限','基本学習率0.1、勾配の大きさの制限なし','基本学習率0.025、勾配ノルムの上限5'],['学習率の変化','最初の5周で増加。その後cosineで減衰\n最小学習率0.0001','左と同じ変化の方法'],['採用した学習時点・実行','98周目、torch.compile有効\nモード：reduce-overhead','83周目、torch.compile有効\nモード：default']];
 table(s,v,48,164,1184,449,[210,487,487],16,true);
 text(s,'1周（epoch）＝学習データを1回通すこと。学習率＝更新の大きさを決める係数。誤差＝予測と正解のずれ',60,626,1160,28,16,C.muted);
 text(s,'checkpoint＝学習済み保存状態。torch.compile＝計算最適化機能。比例融合は保存済み83周目に適用',60,659,1160,25,16,C.muted);
}
// 12  Overall result
{
 const s=add('results','同じ検証データにおける2方式の結果','正解率は、最大確率の動作が正解した動画の割合（Top-1）',notesBaseline);
 const chart=s.charts.add('bar',{position:{left:98,top:196,width:652,height:346},categories:['従来：特徴を集約','新：2分岐を加重融合'],series:[{name:'Top-1正解率（%）',values:[94.6400,95.2778],fill:C.blue}],hasLegend:false,dataLabels:{showValue:true,position:'outEnd'},yAxis:{min:94,max:96,majorGridlines:{style:'solid',fill:C.line,width:1}}});
 text(s,'従来方式',829,211,363,38,24,C.muted,true);text(s,'94.6400%',829,258,363,61,42,C.navy,true);
 text(s,'新方式',829,359,363,38,24,C.muted,true);text(s,'95.2778%',829,406,363,61,42,C.teal,true);
 text(s,'正解数：15,432件から15,536件',780,512,420,35,21,C.navy,true);
 text(s,'+104件、+0.6378ポイント',780,557,420,39,25,C.orange,true);
 text(s,'共通：同じ16,306動画。グラフの縦軸は94〜96%を拡大表示',72,610,1136,30,19,C.navy,true,'center');
 text(s,'構造・学習データ量・学習率も異なるため、向上分を融合式だけの効果とは判断できない',72,651,1136,29,18,C.muted,false,'center');
}
const ja=['水を飲む','食事・間食','歯を磨く','髪をとかす','物を落とす','物を拾う','物を投げる','座る','立つ','拍手','読書','書字','紙を破る','上着を着る','上着を脱ぐ','靴を履く','靴を脱ぐ','眼鏡をかける','眼鏡を外す','帽子をかぶる','帽子を外す','手を振る','足を蹴り出す','物を蹴る','手をポケットに入れる','片足跳び','跳ぶ','電話をかける','スマホ・タブレット操作','キーボード操作','物を指さす','自撮り','時計を見る','手をこする','うなずく・お辞儀','首を横に振る','顔を拭く','敬礼','手を合わせる','腕を組む','くしゃみ・咳','よろめく','転倒','頭痛','胸の痛み','背中の痛み','首の痛み','吐き気・嘔吐','扇子を使う','他人を殴る・たたく','他人を蹴る','他人を押す','他人の背中をたたく','他人を指さす','抱きしめる','他人に物を渡す','他人のポケットに触れる','握手','一緒に近づく','一緒に離れる'];
const name=id=>ja[Number(id.slice(1))-1];
async function csv(file){return (await fs.readFile(file,'utf8')).replace(/^\uFEFF/,'').trim().split(/\r?\n/).slice(1).map(line=>line.split(/","/).map(v=>v.replace(/^"|"$/g,'')));}
const old=await csv(path.join(out,'动作类别对比_20261108.csv'));
const ctr=await csv(path.join(out,'original_ctrgcn_facepair_centered_20261108/逐类对比.csv'));
if(old.length!==60||ctr.length!==60)throw new Error('Both comparisons require all 60 classes');
// 13  Class trends with grouped examples
{
 const s=add('class_summary','動作ごとの改善・低下・同値','従来方式と新方式を、同じ16,306動画で比較',notesBaseline+'\n出典：outputs/动作类别对比_20261108.csv。各クラスの正解率＝そのクラスの正解数/そのクラスの動画数。');
 text(s,'改善 26動作',72,169,390,51,32,C.teal,true);text(s,'低下 27動作',685,169,390,51,32,C.orange,true);text(s,'同値 7動作',72,572,390,45,30,C.blue,true);
 const up=[...old].filter(r=>r[9]==='提升').sort((a,b)=>Number(b[7])-Number(a[7])).slice(0,5);
 const down=[...old].filter(r=>r[9]==='下降').sort((a,b)=>Number(a[7])-Number(b[7])).slice(0,5);
 for(let i=0;i<5;i++){
  let y=239+i*57;text(s,name(up[i][0]),77,y,390,38,22);text(s,'+'+Number(up[i][7]).toFixed(2),477,y,120,38,24,C.teal,true,'right');
  text(s,name(down[i][0]),690,y,390,38,22);text(s,Number(down[i][7]).toFixed(2),1090,y,110,38,24,C.orange,true,'right');
 }
 text(s,'変化は「新方式 − 従来方式」、単位はポイント',470,579,737,33,20,C.muted);
 text(s,'動作別正解率＝その動作を正しく分類した動画数 / その動作の全動画数',72,637,1136,34,21,C.navy,true,'center');
}
function classTable(key,title,sub,data,kindCol,baseCol,newCol,deltaCol,labels,notes){
 const ordered=[...data].sort((a,b)=>({'提升':0,'下降':1,'持平':2}[a[kindCol]]-{'提升':0,'下降':1,'持平':2}[b[kindCol]])||Number(b[deltaCol])-Number(a[deltaCol]));
 const s=add(key,title,sub,notes);
 for(let part=0;part<3;part++){
  const rows=ordered.slice(part*20,part*20+20);
  const v=[['動作',...labels,'差'],...rows.map(r=>[r[0]+' '+name(r[0]),Number(r[baseCol]).toFixed(2),Number(r[newCol]).toFixed(2),(Number(r[deltaCol])>0?'+':'')+Number(r[deltaCol]).toFixed(2)])];
  const t=table(s,v,38+part*406,159,390,486,[199,62,62,67],12,true);
  for(let row=1;row<21;row++){
   const kind=rows[row-1][kindCol];let color=kind==='提升'?C.teal:kind==='下降'?C.orange:C.blue;
   t.getCell(row,3).text.style={fontFamily:font,fontSize:12,bold:true,color};
  }
 }
 return s;
}
// 14  All 60 classes on one slide, grouped
{
 const s=classTable('old_classes','従来方式との60動作比較','正解率は%。差はポイント。上から改善、低下、同値の順にまとめる',old,9,5,6,7,['従来','新'],notesBaseline+'\n出典：outputs/动作类别对比_20261108.csv。A001等はNTU60の動作番号。');
 text(s,'同じ16,306件。従来＝特徴集約、新＝2分岐加重融合。改善26、低下27、同値7',42,652,1180,27,16,C.navy,true);
}
// 15  Published and local values separated
{
 const s=add('paper','CTR-GCN：論文結果と今回の再評価','本研究の2方式とは別の、外部モデルとの比較',notesCtr);
 const v=[['出典・評価','入力と分類の構成','正解率'],['原論文 Table 6','原版 CTR-GCN、4ストリーム融合','92.4%'],['公式Joint公開重みの表示','関節座標の単流。重みフォルダーの値','89.9%'],['今回の公式重みの再評価','原版 CTR-GCN、Joint単流、共通16,237件','88.28%'],['今回の本研究の結果','単一の全身骨格入力、内部2分岐、共通16,237件','95.28%']];
 const t=table(s,v,66,180,1148,340,[315,650,183],21);
 for(let r=1;r<5;r++)t.getCell(r,2).text.style={fontFamily:font,fontSize:25,bold:true,color:C.teal};
 text(s,'89.9%は公開重みの表示値。原論文の表に単流結果として明記した数値とは区別する',72,548,1136,44,19,C.navy,true);
 text(s,'88.28%は公開重みを使った今回の再評価。学習の再現実験を行った値ではない',72,601,1136,34,19,C.muted);
 text(s,'論文：Chen et al., ICCV 2021, Table 6。各値は入力・評価条件が異なる',72,648,1136,29,17,C.muted);
}
// 16  Like and unlike CTR comparison conditions
{
 const s=add('ctr_conditions','CTR-GCNとの比較条件：共通点と相違点','今回の実測比較は、関節座標の単流モデルと、本研究の単一ストリームモデル',notesCtr);
 const v=[['項目','原版 CTR-GCN の再評価','本研究の新方式'],['共通の評価対象','NTU60 X-Sub、60動作、共通16,237件','同じ動画番号・正解ラベルを照合して使用'],['ストリーム・内部構成','Joint（関節座標）の単流、1つの分類出力','2次元全身骨格の単流、内部に2分類分岐'],['座標とキーポイント','Kinectによる3次元座標、25点','RTMWによる2次元座標と信頼度、133点'],['時間長の処理','中央95%の区間から64フレームへ変換','信頼度で加重し100フレームへ圧縮'],['モデルの学習','公式公開済みのJoint重みを使用','ランダム初期化から100周学習'],['今回の正解数・正解率','14,334 / 16,237 = 88.2799%','15,471 / 16,237 = 95.2824%']];
 table(s,v,48,169,1184,413,[215,484,485],19);
 text(s,'16,306件のうち69件が対応せず、共通16,237件に限定。前の表とは母数が異なる',60,602,1160,36,18,C.navy,true);
 text(s,'入力・前処理・学習条件が異なるため、約7.00ポイントの差を融合方式だけの効果とは判断できない',60,650,1160,31,17,C.muted);
}
// 17  Honest runtime/training caveats, plain definitions
{
 const s=add('environment','CTR-GCNとの比較条件：学習環境と実行環境','同じ学習条件へそろえた実験ではないため、確認できる範囲を表示',notesCtr+notesFusion+'\n本研究学習環境：docs/face_pair_validation/full_config_stability.json runtime。');
 const v=[['項目','原版 CTR-GCN','本研究の新方式'],['公開重みの学習設定','公式設定例は65周、1回の更新64動画\nその重みの実際の条件との一致は未確認','100周、1回の更新32動画\n設定の詳細は本資料の学習条件表を参照'],['学習機器・ソフトウェア','公開重みを学習した環境は未確認','Linux、Python 3.10.12\nRTX A6000、PyTorch 2.14.0+cu126、CUDA 12.6'],['今回の評価処理','公式Joint重みをRTX A6000・CUDAで評価','CUDA評価済みの予測を共通サンプルで集計'],['前処理の確認範囲','既存3次元データを第1フレームの関節2で中心化\n公式の全前処理との完全一致は未確認','RTMW座標、100フレーム、最大2人\n検証時のデータ拡張なし'],['torch.compile の範囲','今回の再評価スクリプトは有効化していない','身体・手、集約、2分類器、確率融合をコンパイル\n顔の時空間処理のみ既存の通常実行を維持']];
 table(s,v,48,171,1184,410,[215,484,485],17);
 text(s,'torch.compile＝ニューラル計算を最適化して実行する機能。本研究の学習では有効化を必須とする',60,604,1160,41,18,C.navy,true);
 text(s,'このCTR-GCN比較は条件差を含む参考比較であり、同条件の優劣を確定するものではない',60,653,1160,29,18,C.muted);
}
// 18  All CTR per-class results, single page
{
 const s=classTable('ctr_classes','原版 CTR-GCNとの60動作比較','正解率は%。差＝本研究 − CTR-GCN、単位はポイント。改善・低下をまとめて表示',ctr,6,3,4,5,['原版','本研究'],notesCtr+'\n出典：outputs/original_ctrgcn_facepair_centered_20261108/逐类对比.csv。原版列は今回のJoint単流再評価。論文の4ストリーム結果のクラス別値ではない。');
 text(s,'原版＝今回のJoint単流再評価。共通16,237件で改善59、低下1、同値0。入力・学習条件は異なる',42,652,1190,27,16,C.navy,true);
}

// Stable, actual page numbers are assigned after every insertion.
for(let i=0;i<entries.length;i++){
 const {s}=entries[i];
 if(i>0)text(s,`${i+1} / ${entries.length}`,1110,690,100,18,12,C.muted,false,'right');
 for(const chart of s.charts.items??[])applyPresentationChartFont(chart,{fontFamily:font});
}
await fs.mkdir(dir,{recursive:true});
await fs.writeFile(path.join(dir,'slide_map.json'),JSON.stringify(entries.map(({key,title},i)=>({number:i+1,key,title})),null,2));
for(let i=0;i<entries.length;i++){
 const s=entries[i].s;
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(dir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
await (await PresentationFile.exportPptx(p)).save(path.join(dir,'candidate.pptx'));
console.log(JSON.stringify({slides:entries.length,candidate:path.join(dir,'candidate.pptx')}));
