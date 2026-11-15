import path from 'node:path';
import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const { finalizePresentation } = await import(pathToFileURL('C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs').href);

const workspaceDir = 'C:/Users/15637/PycharmProjects/ISAA';
const candidatePath = 'C:/Users/15637/PycharmProjects/ISAA/.ppt_build/research_study_draft.pptx';
const finalPath = 'C:/Users/15637/PycharmProjects/ISAA/outputs/高精度全身骨架细粒度动作识别研究汇报.pptx';
const stagingDir = 'C:/Users/15637/PycharmProjects/ISAA/.ppt_build/finalizer';
await fs.mkdir(stagingDir, { recursive: true });
const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: 'C:/Users/15637/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath: 'C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/inspect_presentation_package_integrity.py',
  layoutValidatorPath: 'C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs: ['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  requiredNativeTableOwnerSlides: [],
  fontPolicy: { basis:'design', families:['Aptos'] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, 'validation.json'),
});
console.log(JSON.stringify(result));
