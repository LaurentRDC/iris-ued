
# Pyinstaller spec file is used to build iris-ued binaries on Windows
#
# Inspired by the magic-wormhole spec file available here:
#   https://github.com/warner/magic-wormhole/blob/master/pyi/wormhole.exe.spec

import os, sys

# your cwd should be in the same dir as this file, so .. is the project directory:
basepath = os.path.realpath('..')
builddir = os.path.realpath('.')

a = Analysis([os.path.join(basepath, 'iris/__main__.py'), ],
             pathex=[basepath, ],
             binaries=[],
             datas=[],
             hiddenimports=["pywt", # force hook-pywt.py
                            "tifffile._tifffile"],
             hookspath=[builddir],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=None)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='iris',
          debug=False,
          strip=False,
          upx=True,
          console=False,
          icon="iris.ico")

# We prevent the creation of a one-file executable
# because one-file executables are slow
#   https://stackoverflow.com/questions/5971038/pyinstaller-creates-slow-executable
if False:
    coll = COLLECT(exe,
                   a.binaries,
                   a.zipfiles,
                   a.datas,
                   strip=False,
                   upx=True,
                   name='iris')