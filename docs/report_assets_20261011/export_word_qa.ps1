$ErrorActionPreference = 'Stop'
$reportRoot = 'C:\Users\15637\PycharmProjects\ISAA\docs'
$qaRoot = Join-Path $reportRoot 'report_assets_20261011\qa'
New-Item -ItemType Directory -Path $qaRoot -Force | Out-Null
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open((Join-Path $reportRoot 'ISAA完整研究报告_20261011.docx'), $false, $true)
    $document.Fields.Update() | Out-Null
    $document.Repaginate()
    $pageCount = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat((Join-Path $qaRoot 'report.pdf'), 17)
    Write-Output "Exported PDF with $pageCount pages"
} finally {
    if ($null -ne $document) { $document.Close(0); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null }
    if ($null -ne $word) { $word.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
