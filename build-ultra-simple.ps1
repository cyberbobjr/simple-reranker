# Script PowerShell simple - pas de caracteres speciaux

Write-Host 'Building nwai-reranker Podman image...' -ForegroundColor Green

# Verifier Podman
try {
    podman --version | Out-Null
    Write-Host 'Podman trouve' -ForegroundColor Green
} catch {
    Write-Error 'Podman non disponible'
    exit 1
}

# Build de l'image
$ImageName = 'nwai-reranker:latest'

Write-Host "Construction de l'image $ImageName..." -ForegroundColor Blue

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