# Script PowerShell simple - pas de caracteres speciaux

Write-Host 'Building reranking Podman image...' -ForegroundColor Green

# Verifier Podman
try {
    podman --version | Out-Null
    Write-Host 'Podman trouve' -ForegroundColor Green
} catch {
    Write-Error 'Podman non disponible'
    exit 1
}

# Lire la version depuis version.py
function Get-Version {
    $VersionFile = "version.py"
    if (Test-Path $VersionFile) {
        $Content = Get-Content $VersionFile
        $VersionLine = $Content | Where-Object { $_ -match '__version__\s*=\s*"([^"]+)"' }
        if ($VersionLine) {
            return $Matches[1]
        }
    }
    Write-Warning "Impossible de lire la version depuis $VersionFile, utilisation de 'latest'"
    return "latest"
}

$Version = Get-Version
$ImageName = "cyberbobjr/reranking:$Version"

Write-Host "Construction de l'image $ImageName (version $Version)..." -ForegroundColor Blue

$env:BUILDAH_FORMAT = 'docker'

podman build -t $ImageName --platform linux/amd64 .

if ($LASTEXITCODE -eq 0) {
    Write-Host 'Image buildee avec succes!' -ForegroundColor Green
    Write-Host ''
    Write-Host 'Pour lancer le container:' -ForegroundColor Yellow
    Write-Host "podman run --rm -p 8000:8000 --device nvidia.com/gpu=all $ImageName" -ForegroundColor Cyan
    Write-Host ''
    Write-Host 'Pour tester l API:' -ForegroundColor Yellow
    Write-Host 'curl http://localhost:8000/healthz' -ForegroundColor Cyan
} else {
    Write-Error 'Echec du build'
    exit 1
}