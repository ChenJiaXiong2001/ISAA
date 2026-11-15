import path from 'node:path';
import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const SKILL_DIR = 'C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);

const workspaceDir = 'C:/Users/15637/PycharmProjects/ISAA';
const candidatePath = 'C:/Users/15637/PycharmProjects/ISAA/.ppt_build_publication/原池化100帧与CTR-GCN发表汇报_草稿.pptx';
const finalPath = 'C:/Users/15637/PycharmProjects/ISAA/outputs/原池化100帧与CTR-GCN发表汇报.pptx';
const stagingDir = 'C:/Users/15637/PycharmProjects/ISAA/.ppt_build_publication/finalizer';
await fs.mkdir(stagingDir, { recursive: true });

const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: 'C:/Users/15637/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit'],
  requiredNativeTableOwnerSlides: [],
  fontPolicy: { basis: 'design', families: ['Aptos'] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, 'validation.json'),
});
console.log(JSON.stringify(result));
