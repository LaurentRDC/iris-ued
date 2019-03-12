Write-Host "Building .exe pyinstaller"
$INSTALLATION_RESULT = python -O -m PyInstaller --clean --distpath=dist iris.exe.spec
if($INSTALLATION_RESULT -ne 0){
    Write-Host "PyInstaller run failed"
}

Write-Host "Building setup using ISCC"
if ($ENV:PROCESSOR_ARCHITECTURE -eq "AMD64"){
    $iscc = get-item "C:\Program Files (x86)\Inno Setup 5\ISCC.exe"
}
else {
    $iscc = get-item "C:\Program Files\Inno Setup 5\ISCC.exe"
}
& $iscc "iris-setup.iss" #*> iscc.out