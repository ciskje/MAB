#!/usr/bin/env bash
# Build di Mandelbrot.app self-contained (macOS, arm64, one-dir).
# Uso: ./build_app.sh   (richiede python3 con numpy/numba/pyobjc/Pillow installati)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
APP="dist/Mandelbrot.app"

echo "==> [1/4] Icona: render 1024x1024 -> mandelbrot.icns"
"$PY" make_icon.py
ICONSET="build/icon.iconset"
rm -rf "$ICONSET" && mkdir -p "$ICONSET"
# dimensione  nome-file
declare -a SIZES=(
  "16    icon_16x16"
  "32    icon_16x16@2x"
  "32    icon_32x32"
  "64    icon_32x32@2x"
  "128   icon_128x128"
  "256   icon_128x128@2x"
  "256   icon_256x256"
  "512   icon_256x256@2x"
  "512   icon_512x512"
  "1024  icon_512x512@2x"
)
for e in "${SIZES[@]}"; do
  read -r px name <<<"$e"
  sips -z "$px" "$px" icon_src.png --out "$ICONSET/$name.png" >/dev/null
done
iconutil -c icns -o mandelbrot.icns "$ICONSET"
echo "    ok: mandelbrot.icns"

echo "==> [2/5] PyInstaller (one-dir .app)"
"$PY" -m PyInstaller --clean --noconfirm mandelbrot.spec

echo "==> [3/5] Fix libomp.dylib (PyInstaller lo mette in una sottodir)"
FW="dist/Mandelbrot.app/Contents/Frameworks"
if [ -e "$FW/libomp__dot__dylib/libomp.dylib" ]; then
  rm -f "$FW/libomp.dylib"
  mv "$FW/libomp__dot__dylib/libomp.dylib" "$FW/libomp.dylib"
  rmdir "$FW/libomp__dot__dylib" 2>/dev/null || true
  echo "    ok: libomp.dylib in $FW/"
fi
# verifica: omppool.so deve risolvere @rpath/libomp.dylib
python3 - <<'PY'
import ctypes
p="dist/Mandelbrot.app/Contents/Frameworks/numba/np/ufunc/omppool.cpython-312-darwin.so"
try:
    ctypes.CDLL(p); print("    ok: omppool.so carica (libomp risolto)")
except OSError as e:
    print("    AVVISO: omppool non caricabile (numba usera' fallback pool):", e)
PY

echo "==> [4/5] Firma ad-hoc (Gatekeeper per uso locale)"
codesign --force --deep -s - "$APP"
codesign --verify --verbose "$APP" 2>&1 | sed 's/^/    /'

echo "==> [5/5] Fatto"
echo "    App : $APP"
du -sh "$APP" | sed 's/^/    dim : /'
echo "    Per lanciare: open \"$APP\""
echo "    (se copiato/downloadato e macOS blocca:  xattr -dr com.apple.quarantine \"$APP\")"
