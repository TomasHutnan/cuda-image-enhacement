param(
	[Parameter(Mandatory = $true, Position = 0)]
	[string]$InputImage,

	[string]$TempRoot = ".\\temp\\stage-compare-isolated",

	[string]$PythonExe = "python",

	[string]$TgpuPyPath = ".\\scripts\\tgpu.py",

	[string]$CppExePath = ".\\build\\Debug\\tgpu_cli.exe",

	[switch]$SkipView
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ExistingPath {
	param(
		[Parameter(Mandatory = $true)]
		[string]$Path
	)

	if (-not (Test-Path -LiteralPath $Path)) {
		throw "Path does not exist: $Path"
	}

	return (Resolve-Path -LiteralPath $Path).Path
}

function Invoke-Checked {
	param(
		[Parameter(Mandatory = $true)]
		[string]$FilePath,

		[Parameter(Mandatory = $true)]
		[string[]]$Arguments,

		[Parameter(Mandatory = $true)]
		[string]$Label
	)

	Write-Host ""
	Write-Host ">>> $Label"
	Write-Host "$FilePath $($Arguments -join ' ')"

	& $FilePath @Arguments
	if ($LASTEXITCODE -ne 0) {
		throw "Command failed ($LASTEXITCODE): $Label"
	}
}

$inputImagePath = Resolve-ExistingPath -Path $InputImage
$tgpuPyFullPath = Resolve-ExistingPath -Path $TgpuPyPath
$cppExeFullPath = Resolve-ExistingPath -Path $CppExePath

if (-not (Test-Path -LiteralPath $TempRoot)) {
	New-Item -ItemType Directory -Path $TempRoot | Out-Null
}
$tempRootPath = (Resolve-Path -LiteralPath $TempRoot).Path

$stages = @(
	"non_local_means",
	"unsharp_mask",
	"richardson_lucy",
	"histogram_stretch"
)

$stageFiles = @{
	"non_local_means" = "10_non_local_means.png"
	"unsharp_mask" = "20_unsharp_mask.png"
	"richardson_lucy" = "30_richardson_lucy.png"
	"histogram_stretch" = "40_histogram_stretch.png"
}

$imageBaseName = [System.IO.Path]::GetFileNameWithoutExtension($inputImagePath)
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = Join-Path $tempRootPath ("$imageBaseName-$runId")

New-Item -ItemType Directory -Path $runRoot | Out-Null

Write-Host "Input image: $inputImagePath"
Write-Host "Run directory: $runRoot"

$pythonDir = Join-Path $runRoot "py"
$cppDir = Join-Path $runRoot "cpp"

New-Item -ItemType Directory -Path $pythonDir -Force | Out-Null
New-Item -ItemType Directory -Path $cppDir -Force | Out-Null

foreach ($stage in $stages) {
	$pythonTempDir = Join-Path $runRoot "py_temp_$stage"
	$cppTempDir = Join-Path $runRoot "cpp_temp_$stage"

	New-Item -ItemType Directory -Path $pythonTempDir -Force | Out-Null
	New-Item -ItemType Directory -Path $cppTempDir -Force | Out-Null

	$cppOutputImage = Join-Path $cppTempDir "out.png"

	Invoke-Checked -FilePath $PythonExe -Arguments @(
		$tgpuPyFullPath,
		"reference",
		"capture-stages",
		$inputImagePath,
		$pythonTempDir,
		"--only-stage",
		$stage
	) -Label "Python capture-stages ($stage)"

	Invoke-Checked -FilePath $cppExeFullPath -Arguments @(
		$inputImagePath,
		$cppOutputImage,
		"--dump-stages",
		$cppTempDir,
		"--only-stage",
		$stage
	) -Label "C++ capture-stages ($stage)"

	# Copy the correctly isolated stage file to the final directories
	$stageFileName = $stageFiles[$stage]
	Copy-Item -Path (Join-Path $pythonTempDir $stageFileName) -Destination (Join-Path $pythonDir $stageFileName) -Force
	Copy-Item -Path (Join-Path $cppTempDir $stageFileName) -Destination (Join-Path $cppDir $stageFileName) -Force

	# Copy the initial input from the first stage
	if ($stage -eq "non_local_means") {
		Copy-Item -Path (Join-Path $pythonTempDir "00_input_normalized.png") -Destination (Join-Path $pythonDir "00_input_normalized.png") -Force
		Copy-Item -Path (Join-Path $cppTempDir "00_input_normalized.png") -Destination (Join-Path $cppDir "00_input_normalized.png") -Force
	}

	# Copy the final output from the last stage
	if ($stage -eq "histogram_stretch") {
		Copy-Item -Path (Join-Path $pythonTempDir "90_output.png") -Destination (Join-Path $pythonDir "90_output.png") -Force
		Copy-Item -Path (Join-Path $cppTempDir "90_output.png") -Destination (Join-Path $cppDir "90_output.png") -Force
	}

	# Clean up temporary folders
	Remove-Item -Path $pythonTempDir -Recurse -Force
	Remove-Item -Path $cppTempDir -Recurse -Force
}

Invoke-Checked -FilePath $PythonExe -Arguments @(
	$tgpuPyFullPath,
	"reference",
	"compare-stages",
	$pythonDir,
	$cppDir
) -Label "Compare stages"

if (-not $SkipView) {
	Invoke-Checked -FilePath $PythonExe -Arguments @(
		$tgpuPyFullPath,
		"reference",
		"view-stages",
		$pythonDir,
		$cppDir
	) -Label "View stages"
}

Write-Host ""
Write-Host "Done. Stage dumps are in: $runRoot"
