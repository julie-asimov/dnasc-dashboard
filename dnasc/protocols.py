"""
dnasc/protocols.py
───────────────────
Single source of truth for OpTracker protocol name strings.
If a protocol is renamed in OpTracker, change it here.

Import as:
    from dnasc import protocols as proto
    proto.NGS, proto.SEQ_PROTOS, etc.
"""

# ── Parts / ordering ──────────────────────────────────────────────────────────
SYNTHESIS_ORDER  = "Synthesis Order"
ORDER_OLIGOS     = "Order Oligo"
RECEIVE_SYNPART  = "Receive SynPart Synthesis"
RECEIVE_PLASMID  = "Receive Plasmid Synthesis"
PCR              = "PCR"

# ── Assembly ──────────────────────────────────────────────────────────────────
GOLDEN_GATE      = "Golden Gate Assembly"
GIBSON           = "Gibson Assembly"
STAR_TRANSF      = "STAR Transformation"
TRANSFORMATION   = "Transformation"

# ── Colony picking / downstream ───────────────────────────────────────────────
MINIPREP          = "Create Minipreps and Glycerol Stocks"
REPICK            = "Repick: Miniprep/Glycerol/Media"
REARRAY           = "Rearray 96 to 384"
DNA_QUANT         = "DNA Quantification"
NGS               = "NGS Sequence Confirmation"
FRAGMENT_ANALYZER = "Fragment Analyzer"

# ── LSP ───────────────────────────────────────────────────────────────────────
LSP_ORDER         = "LSP Order"
LSP_RECEIVING     = "LSP Receiving"
LSP_REVIEWING     = "LSP Reviewing"
LSP_RELEASING     = "LSP Releasing"
GLYCEROL_STOCKING = "Glycerol Stocking Scinomix"
DIGEST            = "Digest"

# ── Grouped sets ──────────────────────────────────────────────────────────────

# Sequencing protocols: completion (SC/FA) means seq result is known
SEQ_PROTOS = frozenset({NGS, FRAGMENT_ANALYZER})

# Progress-milestone protocols: SC means downstream work is underway
PROGRESS_PROTOS = frozenset({REARRAY, DNA_QUANT} | SEQ_PROTOS)

# Transformation-class protocols (used in colony status logic)
TRANSF_PROTOS = frozenset({STAR_TRANSF, TRANSFORMATION, MINIPREP})

# Parts-ordering protocols (used in stall detection)
ORDER_PROTOS = frozenset({SYNTHESIS_ORDER, ORDER_OLIGOS})
