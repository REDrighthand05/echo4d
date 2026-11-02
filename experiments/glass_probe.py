"""Minimal probe: does a GlassWall pane actually show inside a
CTkFrame in a real (Xvfb) window? Prints panel image state and
samples screenshot pixels where the blue M should ghost through."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import customtkinter as ctk

from echo_app.glass import GlassWall

root = ctk.CTk()
root.geometry("1020x800+0+0")
root.configure(fg_color="#E9EDF2")
wall = GlassWall(root)

frame = ctk.CTkFrame(root, corner_radius=24, width=480, height=260)
frame.place(x=30, y=120)
wall.attach(frame, name="hero", milk=100)
lbl = ctk.CTkLabel(frame, text="PROBE TEXT")
lbl.place(x=20, y=20)

root.update_idletasks()
root.update()
wall.refresh()

for p in wall.panels:
    img = p["img"]
    print("PANEL", p["name"], "photo:", None if img is None else
          (img.width(), img.height()), "label_image:", repr(p["label"].cget("image"))[:60])
print("FRAME fg:", frame.cget("fg_color"))

root.update_idletasks()
root.update()

from PIL import ImageGrab
shot = ImageGrab.grab()
shot.save("/tmp/glass_probe.png")
# M stroke region inside the frame: window coords x 60..490, y 130..260.
for xy in [(100, 200), (200, 160), (300, 220), (450, 180), (60, 140), (250, 300)]:
    print("PIX", xy, shot.getpixel(xy))
print("DONE")
root.destroy()
