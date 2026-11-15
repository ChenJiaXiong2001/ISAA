import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root='C:/Users/15637/PycharmProjects/ISAA';
const skill='C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const tableOwners=[2,4,8,10,11,14,15,16,17,18];
const result=await finalizePresentation({
 workspaceDir:root,
 candidatePath:path.join(root,'.ppt_teacher_20261115/candidate.pptx'),
 finalPath:path.join(root,'outputs/ISAA研究報告_教師向け説明版_日本語.pptx'),
 pythonExecutable:'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:18,
 requiredNativeTableOwnerSlides:tableOwners,
 requiredNativeChartOwnerSlides:[12],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:['Arial']},
 verifyArtifactToolImport:true,
 receiptPath:path.join(root,'.ppt_teacher_20261115/validation_final.json')
});
console.log(JSON.stringify(result));
