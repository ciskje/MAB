# Genera icon_src.png (1024x1024) renderizzando l'insieme con l'app stessa.
# CPU f64 deterministico; palette fuoco. Uso: python make_icon.py
import mandel as M

M.apply_palette("fuoco")
half = 1.35
img = M.compute_cpu(cx=-0.6, cy=0.0, half=half, w=1024, h=1024,
                    mi=M.auto_mi(half), prec="f64")
img.save("icon_src.png")
print("icona renderizzata -> icon_src.png")
