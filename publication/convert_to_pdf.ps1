$docPath = (Resolve-Path (Join-Path $PSScriptRoot "MANUSCRIPT.docx")).Path
$pdfPath = [System.IO.Path]::Combine($PSScriptRoot, "MANUSCRIPT.pdf")

Write-Host "Converting DOCX to PDF..."
Write-Host "Source: $docPath"
Write-Host "Target: $pdfPath"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($docPath)
    # wdExportFormatPDF = 17
    $doc.ExportAsFixedFormat($pdfPath, 17)
    $doc.Close([ref]0) # wdDoNotSaveChanges
    Write-Host "[SUCCESS] PDF exported successfully via Microsoft Word COM ExportAsFixedFormat."
}
catch {
    Write-Error "Failed to export PDF: $_"
}
finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
