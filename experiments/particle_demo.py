"""Offline sample of the living-graffiti background: the strokes
sit formed, a stand-in cursor sweeps through the M (drawn as a
small ring), the beads scatter in depth and drift home after it
leaves. Renders exactly what the app's backdrop shows (frosted),
plus the ring so the 'mouse' is visible in a silent clip."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import ImageDraw

from echo_app.glass import ParticleField

W, H = 1020, 800
field = ParticleField(W, H)


def mouse_at(k):
    """Scripted cursor: away, sweep through the M, linger, leave."""
    if k < 10:
        return None
    if k < 32:
        t = (k - 10) / 22.0
        t = t * t * (3 - 2 * t)
        return (120 + (480 - 120) * t, 330 + (150 - 330) * t)
    if k < 40:
        return (480.0, 150.0)
    if k < 50:
        t = (k - 40) / 10.0
        return (480 + (900 - 480) * t, 150 + (420 - 150) * t)
    return None


frames = []
for k in range(58):
    m = mouse_at(k)
    field.step(0.05, m)
    img = field.render()
    if m is not None:
        d = ImageDraw.Draw(img)
        x, y = m
        d.ellipse([x - 13, y - 13, x + 13, y + 13],
                  outline=(70, 80, 95), width=3)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(70, 80, 95))
    frames.append(img.resize((620, 487)))

out = os.path.expanduser("~/workspace/particle_graffiti_demo.gif")
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=55, loop=0, optimize=True)
print("wrote", out, os.path.getsize(out) // 1024, "KB")
