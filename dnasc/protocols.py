"""
dnasc/protocols.py
───────────────────
Single source of truth for OpTracker protocol name strings.
If a protocol is renamed in OpTracker, change it here.

Import as:
    from dnasc import protocols as proto
    proto.NGS, proto.SEQ_PROTOS, etc.
"""

MINIPREP          = "Create Minipreps and Glycerol Stocks"
REPICK            = "Repick: Miniprep/Glycerol/Media"
STAR_TRANSF       = "STAR Transformation"
TRANSFORMATION    = "Transformation"
REARRAY           = "Rearray 96 to 384"
DNA_QUANT         = "DNA Quantification"
NGS               = "NGS Sequence Confirmation"
FRAGMENT_ANALYZER = "Fragment Analyzer"
SANGER            = "Sanger Sequencing"

# Sequencing protocols: completion (SC/FA) means seq result is known
SEQ_PROTOS = frozenset({NGS, FRAGMENT_ANALYZER, SANGER})

# Progress-milestone protocols: SC means downstream work is underway
PROGRESS_PROTOS = frozenset({REARRAY, DNA_QUANT} | SEQ_PROTOS)

# Transformation-class protocols (used in colony status logic)
TRANSF_PROTOS = frozenset({STAR_TRANSF, TRANSFORMATION, MINIPREP})
