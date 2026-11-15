$taskRoot = 'C:\Users\15637\PycharmProjects\ISAA'
$taskRender = Join-Path $taskRoot '.model_architecture_revision\report_render_v2'
New-Item -ItemType Directory -Path $taskRender -Force | Out-Null
$taskDoc = Join-Path $taskRoot 'docs\基于133节点全身骨架的动作识别研究报告_模型结构解释版.docx'
$taskPdf = Join-Path $taskRender 'report.pdf'
$taskWord = New-Object -ComObject Word.Application
$taskWord.Visible = $false
$taskWord.DisplayAlerts = 0
try {
    $taskDocument = $taskWord.Documents.Open($taskDoc, $false, $true)
    $taskDocument.Repaginate()
    $taskDocument.ExportAsFixedFormat($taskPdf, 17)
    $taskDocument.Close(0)
} finally {
    $taskWord.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord) | Out-Null
}
& 'C:\Users\15637\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe' -png -scale-to 1600 $taskPdf (Join-Path $taskRender 'page')
Get-ChildItem -LiteralPath $taskRender
