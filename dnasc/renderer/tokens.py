"""Design tokens — the single source of truth for dashboard colors & formats.

Every renderer (dashboard.py, inflight.py, lsp_capacity.py) imports from here.
No renderer should hardcode a hex value. To restyle the dashboard, edit THIS
file only.

Design rules (approved 2026-06-10):
  * BRAND PURPLE IS LOCKED. The Asimov purples below are the anchor the rest
    of the palette is built around. Never recolor them. If a purple combo
    fails contrast, fix the *other* side (bg or text), never the purple.
  * ONE COLOR = ONE MEANING. Each hex carries a single semantic role. The two
    documented exceptions are spelled out in EXCEPTIONS below.
  * TREATMENT ENCODES ROLE so overlapping hues can't be confused:
        statuses & customers -> light TINT bg + colored text
        phases & lsp-stages  -> SOLID fill + white text
        flags                -> tint + ICON
  * COLORBLIND-SAFE. Color is never the only signal: every status / flag /
    customer badge keeps its text label AND a small icon/shape (see *_ICON
    maps). Red (FAILED/STALLED/BLOCKED) vs green (R&D / SUCCEEDED) is always
    separated by column + label + icon + tint-vs-solid treatment.
  * EASY ON THE EYES. Soft off-white surfaces (never pure #fff), dark-gray ink
    (never pure #000), no neon. All text/bg combos meet WCAG AA
    (4.5:1 normal, 3:1 large/bold). Text font floor is 9px.

EXCEPTIONS (intentional dual-use — do NOT "fix" these):
  1. #be185d  -> FAILED status (tint bg + magenta text)  AND
                 PARTS phase / Waiting-synparts stage (solid fill + white text).
                 Distinct by solid-vs-tint + different column. The PARTS pill is
                 the brand-gradient magenta endpoint, so the phase pills read as
                 a brand sweep blue->purple->magenta.
                 -> flag any spot where a PARTS pill sits adjacent to a FAILED
                    badge.
  2. green (#166534 / #15803d) is shared by SUCCEEDED status, LSP phase/stage,
     and R&D customer. Distinct by tint-vs-solid + column + icon + leading dot
     on customer badges.
"""

# ───────────────────────────────────────────────────────────────────────────
# 1. BRAND — LOCKED. Do not change these values for any reason.
# ───────────────────────────────────────────────────────────────────────────
PURPLE          = "#6d28d9"   # primary brand purple (RUNNING text)
PURPLE_BRIGHT   = "#7c3aed"   # REPICK / accents
PURPLE_DARK     = "#4c1d95"   # secondary-root tag text
PURPLE_BG       = "#f5f3ff"   # lightest purple tint (RUNNING bg)
PURPLE_BG_2     = "#ede9fe"   # light purple tint (REPICK / pAI bg)
PURPLE_BORDER   = "#ddd6fe"
PURPLE_BORDER_2 = "#c4b5fd"
BRAND_GRADIENT  = "linear-gradient(135deg,#7c3aed 0%,#be185d 100%)"  # tab nav / exp header

# ───────────────────────────────────────────────────────────────────────────
# 2. SURFACE & INK — soft, AA-compliant (no pure black/white)
# ───────────────────────────────────────────────────────────────────────────
SURFACE_PAGE   = "#f0f0f2"    # page background
SURFACE_CARD   = "#fdfdfd"    # cards (off-white, NOT pure #fff)
SURFACE_SUNKEN = "#f5f5f7"    # headers / controls
INK            = "#1d1d1f"    # primary text (not #000)
INK_SECONDARY  = "#4b5563"
INK_MUTED      = "#6b7280"
BORDER         = "#d1d1d6"
BORDER_SOFT    = "#e5e5e7"
WHITE          = "#ffffff"    # for solid-fill text only

# ───────────────────────────────────────────────────────────────────────────
# 3. STATUS badges — Requests-tab canonical · TINT bg + text + border · + icon
#    map: KEY -> (background, text, border)
# ───────────────────────────────────────────────────────────────────────────
STATUS = {
    "SUCCEEDED":   ("#f0fdf4", "#15803d", "#bbf7d0"),   # AA-darkened from #16a34a
    "FULFILLED":   ("#f0fdf4", "#15803d", "#bbf7d0"),
    "RUNNING":     (PURPLE_BG, PURPLE,    PURPLE_BORDER),    # LOCKED purple
    "LSP_RUNNING": (PURPLE_BG, PURPLE,    PURPLE_BORDER),    # LOCKED purple
    "REPICK":      (PURPLE_BG_2, PURPLE_BRIGHT, PURPLE_BORDER_2),  # LOCKED purple
    "IN_PROGRESS": ("#eff6ff", "#1d4ed8", "#bfdbfe"),
    "READY":       ("#fff7ed", "#c2410c", "#fed7aa"),
    "WAITING":     ("#fffbeb", "#b45309", "#fde68a"),   # AA-darkened from #d97706
    "BLOCKED":     ("#fef2f2", "#b91c1c", "#fca5a5"),   # split from FAILED -> true red
    "FAILED":      ("#fff1f5", "#be185d", "#fecdd3"),   # the ONLY magenta-red status
    "CANCELED":    ("#f5f5f7", "#6b7280", "#d1d5db"),
    "DRAFT":       ("#f1f5f9", "#64748b", "#cbd5e1"),
    "UNKNOWN":     ("#f5f5f7", "#6b7280", "#d1d5db"),
}
# Status icons are Lucide line-glyphs (1.5–2px stroke, currentColor), defined
# ONCE here and consumed by BOTH tabs so a status looks identical everywhere:
#   * Tracking tab (dashboard.py) bakes each path into a CSS ::before SVG
#     data-URI (stroke colored to match the status text).
#   * Requests-In-Flight tab (inflight.py) injects these maps into its inline
#     lucide() helper (stroke = currentColor).
# LUCIDE_PATHS: icon name -> inner SVG markup.  STATUS_LUCIDE: status -> icon.
LUCIDE_PATHS = {
    "check":     '<path d="M20 6 9 17l-5-5"/>',
    "x":         '<path d="M18 6 6 18M6 6l12 12"/>',
    "refresh":   '<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 4v5h-5"/>',
    "clock":     '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "hourglass": '<path d="M5 22h14M5 2h14M17 22v-4.2a2 2 0 0 0-.6-1.4L12 12l-4.4 4.4a2 2 0 0 0-.6 1.4V22M7 2v4.2a2 2 0 0 0 .6 1.4L12 12l4.4-4.4A2 2 0 0 0 17 6.2V2"/>',
    "ban":       '<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>',
    "slash":     '<circle cx="12" cy="12" r="9"/><path d="m15 9-6 6"/>',
    "pencil":    '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    "star":      '<path d="M12 2l3 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.9 21l1.2-6.8-5-4.9 6.9-1z"/>',
    "play":      '<path d="M7 4v16l13-8z"/>',
}
STATUS_LUCIDE = {
    "SUCCEEDED": "check", "FULFILLED": "star", "FAILED": "x",
    "RUNNING": "refresh", "LSP_RUNNING": "refresh", "REPICK": "refresh",
    "IN_PROGRESS": "play", "READY": "clock", "WAITING": "hourglass",
    "BLOCKED": "ban", "CANCELED": "slash", "DRAFT": "pencil", "UNKNOWN": "",
}

# ───────────────────────────────────────────────────────────────────────────
# 4. CUSTOMER badges — TINT bg + text + leading dot (shape marker), mixed-case
#    map: KEY -> (label, background, text)
# ───────────────────────────────────────────────────────────────────────────
CUSTOMER = {
    "R_D":               ("R&D",         "#f0fdf4", "#166534"),   # green (canonical)
    "INTERNAL_CLD":      ("CLD",         "#e0f2fe", "#0369a1"),   # Kernel sky-blue, explicit boundary
    "TECH_OUT":          ("Tech Out",    "#ffedd5", "#9a3412"),   # AA-darkened
    "EXTERNAL_TECH_OUT": ("Ext TechOut", "#fce7f3", "#9d174d"),   # AA; distinct from FAILED
}
CUSTOMER_FALLBACK = ("—", "#f3f4f6", "#6b7280")
CUSTOMER_DOT = ""    # leading marker before customer label (empty = none). Single
                     # source of truth — consumers render {CUSTOMER_DOT}{label} with no
                     # manual space, so set e.g. "• " (with trailing space) to re-add.

# ───────────────────────────────────────────────────────────────────────────
# 5. KERNEL FUNCTIONAL PALETTE (approved 2026-06-11) — vibrant antibody-glyph hues,
#    rendered AA-SAFE (Option A): colored/dark text on a light tint + accent border.
#    Never solid white-on-vibrant (orange ~2.4:1, teal ~1.9:1 — both fail AA).
#    Supersedes the old phase brand-sweep. The #6d28d9 brand purple value is NOT
#    recolored — it simply moves to Partner/system tags; RUNNING status still uses it.
# ───────────────────────────────────────────────────────────────────────────
TECH_BLUE   = "#2563eb"   # active states / links / structural
SKY_BLUE    = "#dbeafe"   # muted blue tint
BIO_ORANGE  = "#f97316"   # specialized status (ASM)
TEAL_CYAN   = "#06b6d4"   # specialized status / metrics / flags (PARTS)
DEEP_PURPLE = "#7c3aed"   # group variants / system tags (Partner)

# PHASE pills — Kernel palette, AA-safe tint badges.  KEY -> (bg, text, border)
PHASE = {
    "LSP":   ("#dbeafe", "#1e40af", "#93c5fd"),   # Tech Blue — richer tint
    "ASM":   ("#ffedd5", "#9a3412", "#fb923c"),   # Vibrant Orange — richer tint
    "PARTS": ("#cffafe", "#0e7490", "#22d3ee"),   # Teal/Cyan — richer tint
}
# Partner / group-variant / system tag — Deep Purple, AA-safe tint.
PARTNER = ("#f5f3ff", "#6d28d9", "#ddd6fe")

# ───────────────────────────────────────────────────────────────────────────
# 6. LSP STAGE chips — SOLID fill + white text (kept distinct from statuses by
#    treatment + position). Phase-equivalent stages follow the brand sweep;
#    non-phase stages get hues OUTSIDE the sweep so no two stages share a color.
#    map: KEY -> (fill, text)
# ───────────────────────────────────────────────────────────────────────────
STAGE = {
    "READY_LSP":        ("#1d4ed8", WHITE),   # = LSP phase (blue)
    "IN_ASSEMBLY":      (PURPLE,    WHITE),   # = ASM phase (purple)
    "WAITING_SYNPARTS": ("#be185d", WHITE),   # = PARTS phase (magenta)
    "IN_NGS":           ("#0e7490", WHITE),   # NUDGED off #1d4ed8 -> teal (was = READY_LSP)
    "AWAITING_ASSEMBLY":("#166534", WHITE),   # green (queue)
    "ASSEMBLY_FAILED":  ("#b91c1c", WHITE),   # red (matches BLOCKED family)
    "NEW":              ("#6b7280", WHITE),   # gray
}

# ───────────────────────────────────────────────────────────────────────────
# 7. FLAGS / alerts — TINT + icon, distinct family
#    map: KEY -> (background, text, border)
# ───────────────────────────────────────────────────────────────────────────
FLAG = {
    "PAST_DUE":     ("#fee2e2", "#991b1b", "#fca5a5"),
    "AT_RISK":      ("#fef9c3", "#854d0e", "#fde047"),   # one yellow (collapses ffedd5/fef9c3 dupe)
    "STALLED":      ("#fef2f2", "#dc2626", "#fca5a5"),   # split from #be185d
    "MUST_BATCH":   ("#fef2f2", "#dc2626", "#fca5a5"),   # urgent; icon differs from STALLED
    "LOW_PICKABLE": ("#fef3c7", "#92400e", "#fcd34d"),
    "SEQ_STALLED":  ("#fef3c7", "#92400e", "#fcd34d"),
}
FLAG_ICON = {
    "PAST_DUE": "⏰", "AT_RISK": "⚠", "STALLED": "⏸",
    "MUST_BATCH": "⬆", "LOW_PICKABLE": "▽", "SEQ_STALLED": "⏱",
}

# ───────────────────────────────────────────────────────────────────────────
# 8. TIMELINE DOTS — mapped to STATUS text colors (kills dot-vs-badge mismatch)
#    map: state -> dot color
# ───────────────────────────────────────────────────────────────────────────
TIMELINE_DOT = {
    "succeeded": STATUS["SUCCEEDED"][1],   # #15803d
    "failed":    STATUS["FAILED"][1],      # #be185d
    "running":   STATUS["RUNNING"][1],     # #6d28d9  (was #7c3aed)
    "repick":    STATUS["REPICK"][1],      # #7c3aed
    "ready":     STATUS["READY"][1],       # #c2410c  (was #f59e0b)
    "canceled":  STATUS["CANCELED"][1],    # #6b7280  (was #9ca3af)
    "pending":   "#e5e5e7",
    "source":    "#6366f1",
}

# ───────────────────────────────────────────────────────────────────────────
# 9. BADGE GEOMETRY — one canonical set per badge type
#    map: type -> dict(size, weight, pad, radius, upper, mono)
# ───────────────────────────────────────────────────────────────────────────
GEOM = {
    "status":   dict(size="9px",  weight="700", pad="2px 7px", radius="4px", upper=True,  mono=False),
    "customer": dict(size="10px", weight="600", pad="2px 6px", radius="3px", upper=False, mono=False),
    "phase":    dict(size="9px",  weight="700", pad="2px 8px", radius="4px", upper=True,  mono=False),
    "flag":     dict(size="9px",  weight="700", pad="1px 6px", radius="3px", upper=True,  mono=False),
    "pai":      dict(size="9px",  weight="700", pad="1px 4px", radius="2px", upper=False, mono=True),
    "stock":    dict(size="9px",  weight="700", pad="1px 4px", radius="2px", upper=False, mono=True),
    "strain":   dict(size="10px", weight="500", pad="2px 7px", radius="4px", upper=False, mono=False),
    "date":     dict(size="9px",  weight="600", pad="0 5px",   radius="3px", upper=False, mono=False),
}

# ───────────────────────────────────────────────────────────────────────────
# Helpers — build inline-style / CSS strings from the maps above.
# ───────────────────────────────────────────────────────────────────────────
_MONO = "'SF Mono',Menlo,monospace"


def tint_style(triple, geom_key, *, with_border=True):
    """Inline style for a TINT badge (status / customer / flag): bg + text [+ border]."""
    bg, text, border = triple
    g = GEOM[geom_key]
    css = (f"display:inline-block;background:{bg};color:{text};"
           f"font-size:{g['size']};font-weight:{g['weight']};"
           f"padding:{g['pad']};border-radius:{g['radius']};white-space:nowrap;")
    if with_border:
        css += f"border:1px solid {border};"
    if g["upper"]:
        css += "text-transform:uppercase;"
    if g["mono"]:
        css += f"font-family:{_MONO};"
    return css


def solid_style(pair, geom_key):
    """Inline style for a SOLID badge (phase / stage): fill + white text, no border."""
    fill, text = pair
    g = GEOM[geom_key]
    css = (f"display:inline-block;background:{fill};color:{text};"
           f"font-size:{g['size']};font-weight:{g['weight']};"
           f"padding:{g['pad']};border-radius:{g['radius']};white-space:nowrap;")
    if g["upper"]:
        css += "text-transform:uppercase;"
    return css
