$VERSION = $(python -m iris --version)

Write-Host "Building setup version: " $VERSION

iscc /dAppVersion=$VERSION ".\iris-setup.iss"