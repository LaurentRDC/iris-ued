Write-Host "Building .exe pyinstaller"
python -O -m PyInstaller --clean -y --distpath=dist\executable --onedir iris-onedir.spec