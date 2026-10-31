# Exports every slide of a deck to PNG through PowerPoint (for visual checks).
# Usage: powershell -File docs\export_slides.ps1 <deck.pptx> <out-dir>
param([string]$Deck, [string]$Out)
New-Item -ItemType Directory -Force $Out | Out-Null
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open((Resolve-Path $Deck).Path, $true, $false, $false)
    $i = 1
    foreach ($slide in $pres.Slides) {
        $slide.Export((Join-Path (Resolve-Path $Out).Path ("slide-{0:D2}.png" -f $i)), "PNG", 1600, 900)
        $i++
    }
    $pres.Close()
} finally {
    $app.Quit()
}
