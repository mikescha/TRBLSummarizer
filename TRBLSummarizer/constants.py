from __future__ import annotations

import operator as op
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths and related
DATA_DIR = PROJECT_ROOT / "TRBLSummarizer" / "Data"
PMJ_DIR_NAME = "PMJ Data"
PMJ_DIR = DATA_DIR / PMJ_DIR_NAME
FIGURE_DIR = PROJECT_ROOT / "TRBLSummarizer" / "Figures"

# Columns
DATE_COL = "date"
DATETIME_COL = "dt"
SITE = "site"
SITE_COLS = {
    "id": "id",
    "recording": "recording",
    SITE: "site",
    "day": "day",
    "month": "month",
    "year": "year",
    "hour": "hour",
    "minute": "minute",
    "species": "species",
    "songtype": "songtype",
    "x1": "x1",
    "x2": "x2",
    "y1": "y1",
    "y2": "y2",
    "frequency": "frequency",
    "validated": "validated",
    "url": "url",
    "score": "score",
    "site_id": "site_id",
}

TAG_WSE = "tag_edge"
TAG_WSM = "tag_wsm"
TAG_MHE = "tag_mhe"
TAG_MHM = "tag_mhm"
TAG_MHH = "tag_mhh"
TAG_WS = "tag_ws"
TAG_MH = "tag_mh"
TAG_ = "tag_"
TAG_P1N = "tag_p1n"
TAG_P2N = "tag_p2n"
TAG_P3N = "tag_p3n"
TAG_P4N = "tag_p4n"
TAG_YNC_P2 = "tag<YNC-p2>"
TAG_YNC_P3 = "tag<YNC-p3>"
TAG_YNC_P4 = "tag<YNC-p4>"

MALE_SONG = "malesong"
ALTSONG1 = "altsong1"
ALTSONG2 = "altsong2"
COURT_SONG = "courtsong"
SIMPLE_CALL2 = "simplecall2"

FILENAME = "filename"
HOUR = "hour"
CORE_HOUR_COL = "core_hour"
DATA_COL = {
    FILENAME: "filename",
    SITE: "site",
    "day": "day",
    "month": "month",
    "year": "year",
    HOUR: "hour",
    DATE_COL: "date",
    TAG_YNC_P2: "tag<YNC-p2>",  # Young nestling call pulse 2
    TAG_YNC_P3: "tag<YNC-p3>",  # Young nestling call pulse 3
    TAG_YNC_P4: "tag<YNC-p4>",  # Young nestling call pulse 4
    "tag_p1a": "tag<p1a>",
    "tag_p1f": "tag<p1f>",
    TAG_P1N: "tag<p1n>",
    "tag_p2f": "tag<p2f>",
    TAG_P2N: "tag<p2n>",
    TAG_P3N: "tag<p3n>",
    TAG_P4N: "tag<p4n>",
    "tag_mhe2": "tag<reviewed-MH-e2>",
    "tag_mhe": "tag<reviewed-MH-e>",
    TAG_MHH: "tag<reviewed-MH-h>",
    TAG_MHM: "tag<reviewed-MH-m>",
    TAG_MH: "tag<reviewed-MH>",
    TAG_WSE: "tag<reviewed-WS-e>",
    TAG_WSM: "tag<reviewed-WS-m>",
    TAG_WS: "tag<reviewed-WS>",
    TAG_: "tag<reviewed>",
    ALTSONG2: "val<Agelaius tricolor/Alternative Song 2>",
    ALTSONG1: "val<Agelaius tricolor/Alternative Song>",
    MALE_SONG: "val<Agelaius tricolor/Common Song>",
    COURT_SONG: "val<Agelaius tricolor/Courtship Song>",
    SIMPLE_CALL2: "val<Agelaius tricolor/Simple Call 2>",
    "val<sp11/Simple Call>": "val<sp11/Simple Call>",
    "val<sp22/Simple Call>": "val<sp22/Simple Call>",
}

START = "start"
END = "end"

# Graph names
GRAPH_MANUAL = "Manual Analysis"
GRAPH_MINIMAN = "MiniMan"
GRAPH_EDGE = "Edge Analysis"
GRAPH_PM = "Pattern Matching Analysis"
GRAPH_COMPOSITE = "Composite"

PM_MALE_SONG = "Male Song"
PM_MALE_CHORUS = "Male Chorus"
PM_FEMALE = "Female"
PM_HATCHLING = "Hatchling"
PM_NESTLING = "Nestling"
PM_FLEDGLING = "Fledgling"
PM_SONG_TYPES = [PM_MALE_SONG, PM_MALE_CHORUS, PM_FEMALE, PM_HATCHLING, PM_NESTLING, PM_FLEDGLING]
PM_FILE_TYPES = PM_SONG_TYPES

EDGE_N_TAGS = [
    TAG_P1N,
    TAG_P2N,
    TAG_P3N,
    TAG_P4N,
]  # nestlings, p1 = pulse 1, p2 = pulse 2
EDGE_YNC_TAGS = [TAG_YNC_P2, TAG_YNC_P3, TAG_YNC_P4]
EDGE_TAGS = EDGE_N_TAGS + EDGE_YNC_TAGS
EDGE_COLS = [DATA_COL[t] for t in EDGE_N_TAGS]

MINI_MANUAL_TAGS = [TAG_MHH, TAG_MHM, TAG_WSM]
MINI_MANUAL_COLS = [DATA_COL[t] for t in MINI_MANUAL_TAGS]

MANUAL_TAGS = [TAG_MH, TAG_WS, TAG_]
MANUAL_COLS = [DATA_COL[t] for t in MANUAL_TAGS]

TAG_MAP = {  # map of tag_pXn to ync tag
    DATA_COL[TAG_P1N]: DATA_COL[ALTSONG1],
    DATA_COL[TAG_P2N]: DATA_COL[TAG_YNC_P2],
    DATA_COL[TAG_P3N]: DATA_COL[TAG_YNC_P3],
    DATA_COL[TAG_P4N]: DATA_COL[TAG_YNC_P4],
}
ALL_TAGS = MANUAL_TAGS + MINI_MANUAL_TAGS + EDGE_TAGS

ALL_SONGS = [MALE_SONG, COURT_SONG, ALTSONG2, ALTSONG1, SIMPLE_CALL2]

SONGS = [MALE_SONG, COURT_SONG, ALTSONG2, ALTSONG1]
SONG_COLS = [DATA_COL[s] for s in SONGS]

PHASE_MALE_CHORUS = "Settlement"
PHASE_INC = "Incubation"
PHASE_BROOD = "Brooding"
PHASE_FLDG = "Fledgling"

ABANDONED = "abandon"
VALIDATED_STR = "validated"

MISSING_DATA_FLAG = -100
PRESERVE_EDGES_FLAG = -99

_OPS = {
    ">": op.gt,
    ">=": op.ge,
    "<": op.lt,
    "<=": op.le,
    "==": op.eq,
    "!=": op.ne,
}

# Core hours, these are the only ones we check for presences
CORE_START_HOUR = 7
CORE_END_HOUR_EXCLUSIVE = 20  # includes 7 through 19, excludes 20

# Fields in the adapted public chronology summary
PULSE_COUNT = "pulse_count"
PULSES = ["p1", "p2", "p3", "p4"]
SUMMARY_FIRST_REC = "First Recording"
SUMMARY_LAST_REC = "Last Recording"
PULSE_MC_START = "mcstart"
PULSE_INC_START = "incstart"
PULSE_HATCH = "hatch"
PULSE_FIRST_FLDG = "fledgestart"
PULSE_LAST_FLDG = "fledgedisp"
# Controlled values in the accepted chronology
CONTINUOUS = "continuous"
ND_STRING = "ND"
MISSED = "missed"
INF = "inf"
CONTROLLED_DATE_VALUES = frozenset({ND_STRING, INF, MISSED, CONTINUOUS})
EVENT_MARKERS = {
    PULSE_MC_START: "M", PULSE_INC_START: "I", PULSE_HATCH: "B",
    PULSE_FIRST_FLDG: "F", PULSE_LAST_FLDG: "D",
}
PM_EVENT_ROWS = {
    PULSE_MC_START: PM_MALE_CHORUS, PULSE_INC_START: PM_FEMALE, PULSE_HATCH: PM_HATCHLING,
    PULSE_FIRST_FLDG: PM_FLEDGLING, PULSE_LAST_FLDG: PM_FLEDGLING,
}

# Color maps
# default color map
CMAP = {
    DATA_COL[MALE_SONG]: "Greens",
    DATA_COL[COURT_SONG]: "Oranges",
    DATA_COL[ALTSONG2]: "Purples",
    DATA_COL[ALTSONG1]: "Blues",
    "Fledgling": "Blues",
}

CMAP_PM = {
    PM_MALE_SONG: "Greens",
    PM_MALE_CHORUS: "Oranges",
    PM_FEMALE: "Purples",
    PM_HATCHLING: "Blues",
    PM_NESTLING: "Blues",
    PM_FLEDGLING: "Blues",
}

REFERENCE_DPI = 300
DPI = REFERENCE_DPI
FIG_WIDTH_INCHES = 6.5
EMPTY_FRAME_HEIGHT_INCHES = 1.2
HEATMAP_GAMMA = 0.85
HEATMAP_MIN_POSITIVE = 0.001
HEATMAP_MAX_FRACTION = 1.0
GRAPH_FONT = "Arial"
GRAPH_FONT_TTF = "ARIAL.TTF"  # used for output to the file using PIL
TITLE_FONT_SIZE = 18
SUBTITLE_FONT_SIZE = 12
AXIS_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
POINTS_PER_INCH = 72.0
BORDER_WIDTH = 0.25  # points, for the vertical month dividers and the exterior edges
BORDER_COLOR = "#1A1A1A"
FONT_COLOR = "#1A1A1A"

# Event badge spacing is measured in pixels at REFERENCE_DPI; outlines are points.
EVENT_MARKER_PADDING_PX = 4
EVENT_MARKER_LEFT_INSET_PX = 3
EVENT_MARKER_TOP_INSET_PX = 11
EVENT_MARKER_ARROW_GAP_PX = 10
EVENT_MARKER_MIN_ARROW_WIDTH_PX = 24
EVENT_MARKER_BORDER_WIDTH = BORDER_WIDTH * 0.6
EVENT_MARKER_LINE_WIDTH = 0.35  # points
EVENT_MARKER_LINE_TOP_GAP_PX = 0
EVENT_MARKER_HALO_WIDTH_PX = 2

# Internal marker for dates before recording began; never displayed as a calendar date.
DATE_PRE_RECORDING = pd.Timestamp("1967-06-01")
