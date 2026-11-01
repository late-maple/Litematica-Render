# Litematica-Render

Offline isometric rendering of Minecraft Litematica `.litematic` schematics.

## Features

- Parse `.litematic` files: streaming NBT, lazy vectorized region unpacking
- Render PNGs from a resource pack's real blockstates, models, and textures
- Free camera: any rotation/slant angle with per-pixel depth buffering
- Pack stack: layer add-on packs (e.g. Faithful 32x) over a vanilla base, MC-style first-hit lookup

## Example

```python
from litematic import load
from litematic.render import isometric_png, unresolved_colors

schem = load("schematic.litematic")

png = isometric_png(
    schem,
    textures_path="vanilla-assets.zip",   # base pack, required
    extra_packs=["Faithful 32x.zip"],     # add-ons, earlier = higher priority
    rotation=160.0,                       # camera azimuth degrees (default 45)
    slant=25.0,                           # camera elevation in [-90, 90]
    max_size=2048,
)
with open("render.png", "wb") as f:
    f.write(png)

# block names the pack stack cannot cover; those render magenta
missing = unresolved_colors(schem, textures_path="vanilla-assets.zip")
```
