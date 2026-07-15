"""Standalone local preview of the Parts dashboard tab.

Thin wrapper over dnasc.renderer.parts.render_parts_tab() — the SAME code the dashboard
tab uses — wrapped in a minimal page shell (with the #tab-parts id so the scoped CSS
applies) and written to disk. Single source of truth: edit the renderer, not this file.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from dnasc.renderer.parts import render_parts_tab

doc = ('<!DOCTYPE html>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
       '<title>Parts Inventory</title>\n'
       '<body style="margin:0">\n'
       '<div id="tab-parts" class="tab-content active">\n'
       + render_parts_tab() +
       '\n</div>\n</body>')

out = "/Users/juliehachey/www/parts_preview.html"
open(out, "w", encoding="utf-8").write(doc)
print(f"wrote {out} ({len(doc)} bytes)")
