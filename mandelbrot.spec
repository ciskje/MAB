# -*- mode: python ; coding: utf-8 -*-
# Ricetta PyInstaller per Mandelbrot.app (one-dir, arm64).
# Dipendenze reali di mandel.py: numpy, Pillow(ImageTk), pyobjc(Metal, GPU),
# numba+llvmlite (fast path CPU; l'app degrada su numpy se assente).
# cupy/CUDA NON e' incluso (assente su macOS; rilevato a runtime, gia' guardato).

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# --- Numba + LLVM (dylib ~116 MB di llvmlite, dati JIT) ---
for mod in ("numba", "llvmlite"):
    d, b, h = collect_all(mod)
    datas += d
    binaries += b
    hiddenimports += h

# --- pyobjc / Metal (GPU Apple): moduli C (objc/_objc, Foundation/_nsobject,
#     Metal/_metal) + submoduli dinamici ---
for mod in ("objc", "Foundation", "Metal"):
    try:
        d, b, h = collect_all(mod)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        hiddenimports += [mod]
hiddenimports += collect_submodules("objc")

# --- Numba OpenMP pool: omppool.so richiede @rpath/libomp.dylib (il suo rpath
#     e' hardcoded a una dir CI inesistente). Unica copia disponibile:
#     torch/lib/libomp.dylib. Nel bundle con basename 'libomp.dylib' ->
#     PyInstaller fa match sul basename e riscrive il riferimento. ---
try:
    import torch
    _libomp = os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")
    if os.path.exists(_libomp):
        binaries.append((_libomp, "libomp.dylib"))
except Exception:
    pass

# --- Pillow + Tkinter (ImageTk importa _tkinter_finder a runtime) ---
hiddenimports += ["PIL._tkinter_finder"]

# --- import laziali nel sorgente (giu' in funzioni/try) da garantire ---
hiddenimports += ["Metal", "objc", "Foundation"]

block_cipher = None

a = Analysis(
    ["mandel.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["cupy", "torch", "matplotlib", "IPython", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Mandelbrot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

app = BUNDLE(
    exe, a.binaries, a.datas,
    name="Mandelbrot.app",
    icon="mandelbrot.icns",
    bundle_id="it.mandelbrot.app",
    info_plist={
        "CFBundleName": "Mandelbrot",
        "CFBundleDisplayName": "Mandelbrot",
        "CFBundleShortVersionString": "5.4.0",
        "CFBundleVersion": "5.4.0",
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
        "LSMinimumSystemVersion": "11.0",
    },
)
