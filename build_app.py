#!/usr/bin/env python
# Build self-contained (one-dir) multipiattaforma per l'app Mandelbrot.
# Ordine: icona -> PyInstaller (mandelbrot.spec) -> post (verifica + zip).
#   Windows: python build_app.py   -> dist/Mandelbrot/Mandelbrot.exe + zip
#   macOS  : ./build_app.sh        -> dist/Mandelbrot.app (firma ad-hoc)
# Nota: il runtime CUDA NON e' incluso; la GPU CUDA funziona solo se l'utente ha
# installato driver NVIDIA + runtime CUDA (trovati da cupy via cuda-pathfinder).
# La GPU Vulkan (wgpu/wgpu-native, AMD/NVIDIA/Intel) E' invece bundled e funziona
# out-of-the-box.
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IS_WIN32 = (sys.platform == "win32")
IS_DARWIN = (sys.platform == "darwin")

with open(os.path.join(HERE, "mandel.py"), encoding="utf-8") as _f:
    _head = _f.read(4096)
VERSION = re.search(r"VERSIONE:\s*([\d.]+)", _head).group(1)


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=HERE, **kw)


def fail(msg):
    print("ERRORE:", msg, file=sys.stderr)
    sys.exit(1)


# 1) Icona (icon_src.png + mandelbrot.ico) via motore CPU di mandel.py
r = run([sys.executable, os.path.join(HERE, "make_icon.py")])
if r.returncode != 0:
    fail("make_icon.py non riuscita")
if not os.path.isfile(os.path.join(HERE, "mandelbrot.ico")):
    fail("mandelbrot.ico non generata")

# 2) PyInstaller (ricetta mandelbrot.spec)
r = run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         os.path.join(HERE, "mandelbrot.spec")])
if r.returncode != 0:
    fail("PyInstaller non riuscito")

# 3) Post-build
if IS_WIN32:
    app_dir = os.path.join(HERE, "dist", "Mandelbrot")
    exe = os.path.join(app_dir, "Mandelbrot.exe")
    if not os.path.isfile(exe):
        fail("dist/Mandelbrot/Mandelbrot.exe non trovata dopo la build")
    base = os.path.join(HERE, "dist", "Mandelbrot-v%s-win64" % VERSION)
    if os.path.exists(base + ".zip"):
        os.remove(base + ".zip")
    zip_path = shutil.make_archive(base, "zip",
                                   root_dir=os.path.join(HERE, "dist"),
                                   base_dir="Mandelbrot")
    print("OK: exe =", exe)
    print("OK: zip =", zip_path)
elif IS_DARWIN:
    app = os.path.join(HERE, "dist", "Mandelbrot.app")
    if not os.path.isdir(app):
        fail("dist/Mandelbrot.app non trovato dopo la build")
    run(["codesign", "--force", "--deep", "--sign", "-", app])
    print("OK: app =", app)
else:
    fail("piattaforma non supportata per la build: %s" % sys.platform)
