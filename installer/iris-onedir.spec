
# Pyinstaller spec file is used to build iris-ued binaries on Windows
#
# Inspired by the magic-wormhole spec file available here:
#   https://github.com/warner/magic-wormhole/blob/master/pyi/wormhole.exe.spec

import os, sys

# your cwd should be in the same dir as this file, so .. is the project directory:
basepath = os.path.realpath('..')
builddir = os.path.realpath('.')

images = [os.path.join(basepath, 'iris/gui/images/*')]

a = Analysis([os.path.join(basepath, 'iris/__main__.py'), ],
             pathex=[basepath, ],
             binaries=[],
             datas=[],
             hiddenimports=["pywt", # force hook-pywt.py
                            "dask", # force hook-dask.py
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
          debug=True,
          strip=False,
          upx=False,
          console=True,
          icon="iris.ico")

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='iris')