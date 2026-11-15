$taskRoot = 'C:\Users\15637\PycharmProjects\ISAA'
$taskRender = Join-Path $taskRoot '.background_single_stream_revision\report_render_v2'
New-Item -ItemType Directory -Path $taskRender -Force | Out-Null
$taskDoc = Join-Path $taskRoot 'docs\基于133节点全身骨架的动作识别研究报告_背景与单流结果修订版.docx'
$taskPdf = Join-Path $taskRender 'report.pdf'
$ErrorActionPreference = 'Stop'
# Use IDispatch reflection to avoid the broken registered Office interop type library.
# This is a non-interactive read-only export of the task document, not a UI edit.
$taskWord = [Activator]::CreateInstance([type]::GetTypeFromProgID('Word.Application'))
$taskWord.GetType().InvokeMember('Visible', [Reflection.BindingFlags]::SetProperty, $null, $taskWord, @($false))
$taskWord.GetType().InvokeMember('DisplayAlerts', [Reflection.BindingFlags]::SetProperty, $null, $taskWord, @(0))
$taskDocument = $null
$taskDocuments = $null
try {
    $taskDocuments = $taskWord.GetType().InvokeMember('Documents', [Reflection.BindingFlags]::GetProperty, $null, $taskWord, $null)
    $taskDocument = $taskDocuments.GetType().InvokeMember('Open', [Reflection.BindingFlags]::InvokeMethod, $null, $taskDocuments, @($taskDoc.ToString()))
    $taskDocument.GetType().InvokeMember('Repaginate', [Reflection.BindingFlags]::InvokeMethod, $null, $taskDocument, $null)
    $taskDocument.GetType().InvokeMember('ExportAsFixedFormat', [Reflection.BindingFlags]::InvokeMethod, $null, $taskDocument, @($taskPdf.ToString(), 17))
} finally {
    if ($null -ne $taskDocument) {
        $taskDocument.GetType().InvokeMember('Close', [Reflection.BindingFlags]::InvokeMethod, $null, $taskDocument, @(0))
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskDocument) | Out-Null
    }
    if ($null -ne $taskDocuments) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskDocuments) | Out-Null }
    $taskWord.GetType().InvokeMember('Quit', [Reflection.BindingFlags]::InvokeMethod, $null, $taskWord, @(0))
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord) | Out-Null
}
& 'C:\Users\15637\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe' -png -scale-to 1600 $taskPdf (Join-Path $taskRender 'page')
Get-ChildItem -LiteralPath $taskRender
