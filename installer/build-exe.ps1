"Building .exe pyinstaller"
python -OO -m PyInstaller --clean --distpath=dist iris.exe.spec

"Building setup using ISCC"
if ($ENV:PROCESSOR_ARCHITECTURE -eq "AMD64"){
    $iscc = get-item "C:\Program Files (x86)\Inno Setup 5\ISCC.exe"
}
else {
    $iscc = get-item "C:\Program Files\Inno Setup 5\ISCC.exe"
}
& $iscc "iris-setup.iss" #*> iscc.out