# Speaks each scene of narration.json with the Windows "Microsoft Irina Desktop" voice into
# build\voice-<id>.wav (48 kHz, 16-bit, mono). Lines are passed in by make_lines.mjs as
# build\lines.json: [{ "id": "...", "say": "..." }].
param([string]$Dir)
Add-Type -AssemblyName System.Speech
$lines = Get-Content -Raw -Encoding UTF8 (Join-Path $Dir "lines.json") | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft Irina Desktop")
$synth.Rate = 1
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($line in $lines) {
    $path = Join-Path $Dir ("voice-" + $line.id + ".wav")
    $synth.SetOutputToWaveFile($path, $format)
    $synth.Speak($line.say)
    $synth.SetOutputToNull()
    Write-Output ("spoke " + $line.id)
}
$synth.Dispose()
