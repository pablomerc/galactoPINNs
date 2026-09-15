"""ATNF psrcat reference tag -> (timing program, bibtex key) for the Donlon+2025 catalog.

The tag is the reference the ATNF Pulsar Catalogue adopts for the parameter that carries the
acceleration (PBDOT for the binary channel, F1 for the spin channel).  Programs: the three
long-running pulsar timing arrays; "Other" = a dedicated timing study of one system.
"""
TAGS = {
    "aab+21a": ("NANOGrav", "alam2021-nanograv125"),        # NANOGrav 12.5-yr
    "abb+18":  ("NANOGrav", "arzoumanian2018-nanograv11"),  # NANOGrav 11-yr
    "fcp+21":  ("NANOGrav", "fonseca2021-j0740"),           # J0740+6620 (NANOGrav + CHIME)
    "abb+23":  ("EPTA",     "antoniadis2023-eptadr2"),      # EPTA DR2 (NOT NANOGrav 15-yr)
    "dcl+16":  ("EPTA",     "desvignes2016-epta"),          # EPTA 42-MSP timing
    "rsc+21":  ("PPTA",     "reardon2021-pptadr2"),         # PPTA DR2
    "cpb+23":  ("PPTA",     "curylo2023-pptauwl"),          # PPTA UWL wide-band timing
    "rbs+24":  ("PPTA",     "reardon2024-j0437"),           # J0437-4715
    "ksm+06":  ("Other",    "kramer2006-doublepulsar"),
    "wh16":    ("Other",    "weisberg2016-b1913"),
    "fwe+12":  ("Other",    "freire2012-j1738"),
    "gfg+21":  ("Other",    "guo2021-j2222"),
    "sfa+19":  ("Other",    "stovall2019-j2234"),
    "sbb+18":  ("Other",    "spiewak2018-j2322"),
    "gvf+23":  ("Other",    "geyer2023-j1933"),
    "gfw+24":  ("Other",    "gautam2024-j1012"),
    "skm+17":  ("Other",    "swiggum2017-j1400"),
    "jsk+08":  ("Other",    "janssen2008-j1518"),
    "jsb+10":  ("Other",    "janssen2010-msps"),
}
PROGRAM_ORDER = ["NANOGrav", "EPTA", "PPTA", "Other"]

def program(tag):
    return TAGS.get(tag, ("Other", None))[0]

def bibkey(tag):
    return TAGS.get(tag, (None, None))[1]
