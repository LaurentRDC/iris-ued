Write-Host "Create clean environment"
conda activate ./env

Write-Host "Building .exe pyinstaller"
python -O -m PyInstaller --clean -y --distpath=dist\executable --onedir iris-onedir.spec

Write-Host "Building setup using ISCC"
if ($ENV:PROCESSOR_ARCHITECTURE -eq "AMD64"){
    $iscc = get-item "C:\Program Files (x86)\Inno Setup 5\ISCC.exe"
}
else {
    $iscc = get-item "C:\Program Files\Inno Setup 5\ISCC.exe"
}
& $iscc "iris-setup.iss"

Write-Host "Cleanup"
conda deactivate