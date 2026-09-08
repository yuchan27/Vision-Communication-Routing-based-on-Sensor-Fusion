param(
    [int]$Port = 8000,
    [switch]$Install,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvRoot = Join-Path $backendRoot ".venv313"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$requirementsPath = Join-Path $backendRoot "requirements.txt"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The Windows Python launcher (py.exe) is required. Install Python 3.13 and enable the launcher."
}

$launcherVersion = (& py -3.13 --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $launcherVersion -notmatch "Python 3\.13\.") {
    throw "Python 3.13 was not found. Install Python 3.13, then run this script again."
}

$createdEnvironment = $false
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating the project virtual environment with $launcherVersion..."
    & py -3.13 -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create $venvRoot."
    }
    $createdEnvironment = $true
}

$environmentVersion = (& $venvPython --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $environmentVersion -notmatch "Python 3\.13\.") {
    throw "The project virtual environment is not Python 3.13: $environmentVersion"
}

if ($createdEnvironment -or $Install) {
    Write-Host "Installing backend dependencies into $venvRoot..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed."
    }
    & $venvPython -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
}

Write-Host "Starting Fire Vision Command Center with $environmentVersion on http://127.0.0.1:$Port"

$uvicornArguments = @(
    "app:app",
    "--host", "127.0.0.1",
    "--port", $Port.ToString()
)
if ($Reload) {
    $uvicornArguments += "--reload"
}

Push-Location $backendRoot
try {
    & $venvPython -m uvicorn @uvicornArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Uvicorn stopped with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
