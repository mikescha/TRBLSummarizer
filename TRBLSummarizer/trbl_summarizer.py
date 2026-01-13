from __future__ import annotations


#Set appropriately before I deploy
BEING_DEPLOYED_TO_STREAMLIT = True
SHOW_MANUAL_ANALYSIS = True  # Dec 2025, we may or may not want to show the manual analysis graph
INCLUDE_INSECT_AND_FROG_DATA = False
PROFILING = False
MAKE_ALL_GRAPHS = False
ALIGN_DATES = True
STANDARD_START  = "04/01"
STANDARD_END    = "07/30"
GRAPH_LEFT_PADDING = 0.1

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns #TODO: try to get rid of this
import math

import matplotlib as mpl
mpl.use('WebAgg') #Have to select the backend before doing other imports
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.transforms import Bbox
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import NullLocator, NullFormatter
from matplotlib.backend_bases import RendererBase
from matplotlib.font_manager import FontProperties

from matplotlib import colors
from pathlib import Path
import os
import calendar
from collections import Counter
from itertools import tee
import random
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime as dt
import glob


#Crap that I needed to fix pylance errors
from collections.abc import Sequence
from typing import Any, Dict, Tuple, Protocol, cast

#to force garbage collection and reduce memory use
import gc

#For profiling
import cProfile
import pstats

import traceback
from matplotlib.backend_bases import FigureCanvasBase

from contextlib import contextmanager
import time
@contextmanager
def timed(label: str):
    if not PROFILING:
        yield
        return
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[TIMER] {label}: {dt:.3f}s")


import matplotlib.text as mtext

def count_text_artists(fig):
    texts = [a for a in fig.findobj(mtext.Text) if a.get_visible()]
    # Filter out empty strings (Matplotlib has a lot of empty placeholders)
    texts = [t for t in texts if (t.get_text() or "").strip()]
    return len(texts), texts

import io
def st_image_figure(fig, *, dpi=120):
    fig_w, fig_h = fig.get_size_inches()

    pos = Bbox.union([ax.get_position() for ax in fig.axes])   # figure fraction
    fig_w, fig_h = fig.get_size_inches()

    bbox_in = Bbox.from_bounds(
        pos.x0 * fig_w,
        pos.y0 * fig_h,
        pos.width * fig_w,
        pos.height * fig_h,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_in)  # keep None
    buf.seek(0)
    st.image(buf)

# Constants and Globals
#
#
BAD_FILES = 'bad'
FILENAME = 'filename'
SITE = 'site'
DATE = 'date'
HOUR = 'hour'
tag_wse = 'tag_edge'
tag_wsm = 'tag_wsm'
tag_mhe = 'tag_mhe'
tag_mhm = 'tag_mhm'
tag_mhh = 'tag_mhh'
tag_mhe2= 'tag_mhe2'
tag_ws  = 'tag_ws'
tag_mh  = 'tag_mh'
tag_    = 'tag_'
tag_p1c = 'tag_p1c'  
tag_p1n = 'tag_p1n'
tag_p1a = 'tag_p1a' #used to be P1NA
tag_p1f = 'tag_p1f'
tag_p2c = 'tag_p2c'
tag_p2n = 'tag_p2n'
tag_p2f = 'tag_p2f'
tag_p3n = 'tag_p3n' 
tag_p4n = 'tag_p4n' 

tag_wsmc = 'tag_wsmc'
validated = 'validated'
tag_YNC_p2 = 'tag<YNC-p2>'
tag_YNC_p3 = 'tag<YNC-p3>' 
tag_YNC_p4 = 'tag<YNC-p4>' 
MALE_SONG = 'malesong'
ALTSONG2 = 'altsong2'
ALTSONG1 = 'altsong1'
COURT_SONG = 'courtsong'
SIMPLE_CALL2 = 'simplecall2'

PRESENT = 'present'

START = 'start'
END = 'end'

# Master list of all the columns I need. If columns get added/removed then this needs to update
# The dictionary values MUST map to what's in the data file. 
data_col = {
    FILENAME : 'filename', 
    SITE     : 'site', 
    'day'        : 'day',
    'month'      : 'month',
    'year'       : 'year',
    HOUR     : 'hour', 
    DATE     : 'date',
    tag_YNC_p2   : 'tag<YNC-p2>', #Young nestling call pulse 2
    tag_YNC_p3   : 'tag<YNC-p3>', #Young nestling call pulse 3
    tag_YNC_p4   : 'tag<YNC-p4>', #Young nestling call pulse 4
    tag_p1a      : 'tag<p1a>',
    tag_p1c      : 'tag<p1c>',
    tag_p1f      : 'tag<p1f>',
    tag_p1n      : 'tag<p1n>',
    tag_p2c      : 'tag<p2c>',
    tag_p2f      : 'tag<p2f>',
    tag_p2n      : 'tag<p2n>',
    tag_p3n      : 'tag<p3n>',
    tag_p4n      : 'tag<p4n>',
    tag_mhe2     : 'tag<reviewed-MH-e2>', 
    tag_mhe      : 'tag<reviewed-MH-e>',
    tag_mhh      : 'tag<reviewed-MH-h>',
    tag_mhm      : 'tag<reviewed-MH-m>',
    tag_mh       : 'tag<reviewed-MH>',
    tag_wse      : 'tag<reviewed-WS-e>',
    tag_wsm      : 'tag<reviewed-WS-m>',
    tag_ws       : 'tag<reviewed-WS>',
    tag_         : 'tag<reviewed>',
    ALTSONG2     : 'val<Agelaius tricolor/Alternative Song 2>',
    ALTSONG1     : 'val<Agelaius tricolor/Alternative Song>',
    MALE_SONG     : 'val<Agelaius tricolor/Common Song>',
    COURT_SONG    : 'val<Agelaius tricolor/Courtship Song>',
    SIMPLE_CALL2  : 'val<Agelaius tricolor/Simple Call 2>',
    "val<sp11/Simple Call>":"val<sp11/Simple Call>",
    "val<sp22/Simple Call>":"val<sp22/Simple Call>"
}

tag_name_map = {
    'val<Agelaius tricolor/Common Song>' : "Male Song",
    'val<Agelaius tricolor/Courtship Song>' : "Male Chorus",
    'val<Agelaius tricolor/Alternative Song 1>' : "Female Chatter",
    'val<Agelaius tricolor/Alternative Song 2>' : "Begging Calls",
}

site_columns = {
    'id'        : 'id',
    'recording' : 'recording',
    SITE    : 'site', 
    'day'       : 'day',
    'month'     : 'month',
    'year'      : 'year',
    HOUR    : 'hour', 
    'minute'    : 'minute',
    'species'   : 'species',
    'songtype'  : 'songtype',
    'x1'        : 'x1',
    'x2'        : 'x2',
    'y1'        : 'y1',
    'y2'        : 'y2',
    'frequency' : 'frequency',
    validated   : 'validated',
    'url'       : 'url',
    'score'     : 'score',
    'site_id'   : 'site_id'
}

SONGS = [MALE_SONG, COURT_SONG, ALTSONG2, ALTSONG1]
SONG_COLS = [data_col[s] for s in SONGS]
ALL_SONGS = [MALE_SONG, COURT_SONG, ALTSONG2, ALTSONG1, SIMPLE_CALL2] 

MANUAL_TAGS = [tag_mh, tag_ws, tag_]
MINI_MANUAL_TAGS = [tag_mhh, tag_mhm, tag_wsm]

EDGE_N_TAGS = [tag_p1n, tag_p2n, tag_p3n, tag_p4n] #nestlings, p1 = pulse 1, p2 = pulse 2
EDGE_YNC_TAGS = [tag_YNC_p2, tag_YNC_p3, tag_YNC_p4]
EDGE_TAGS = EDGE_N_TAGS + EDGE_YNC_TAGS
ALL_TAGS = MANUAL_TAGS + MINI_MANUAL_TAGS + EDGE_TAGS

MANUAL_COLS = [data_col[t] for t in MANUAL_TAGS]
MINI_MANUAL_COLS = [data_col[t] for t in MINI_MANUAL_TAGS]
EDGE_N_COLS = [data_col[t] for t in EDGE_N_TAGS]
#all_tag_cols = manual_cols + mini_manual_cols + edge_c_cols + edge_n_cols

EDGE_COLS = EDGE_N_COLS #make list of the right length
ALL_EDGE_COLS = EDGE_COLS + [data_col[t] for t in EDGE_YNC_TAGS]

#Constants for the graphing, so they can be shared across weather and blackbird graphs
#For setting figure width and height, values in inches
FIG_W = 6.5
FIG_H = 1

#constants for the weather data files
WEATHER_PRCP = 'prcp'
WEATHER_TMAX = 'tmax'
WEATHER_TMIN = 'tmin'
WEATHER_WIND = 'wspd'
WEATHER_COLS = [WEATHER_PRCP, WEATHER_TMAX, WEATHER_TMIN, WEATHER_WIND]

GRAPH_SUMMARY = "Summary"
GRAPH_MANUAL = 'Manual Analysis'
GRAPH_MINIMAN = 'MiniMan'
GRAPH_EDGE = 'Edge Analysis'
GRAPH_PM = 'Pattern Matching Analysis'
GRAPH_WEATHER = 'Weather'
GRAPH_TYPES = [GRAPH_SUMMARY, GRAPH_PM, GRAPH_MANUAL, GRAPH_MINIMAN,  GRAPH_EDGE, GRAPH_WEATHER]
LEGEND_NAME = 'legend.png'

#default color map
CMAP = {data_col[MALE_SONG]:'Greens', 
        data_col[COURT_SONG]:'Oranges', 
        data_col[ALTSONG2]:'Purples', 
        data_col[ALTSONG1]:'Blues', 
        "Fledgling":"Blues"
}

CMAP_NAMES = {data_col[MALE_SONG]:"Male Song",
              data_col[COURT_SONG]:"Male Chorus",
              data_col[ALTSONG2]:"Female Chatter",
              data_col[ALTSONG1]:"Hatchling/Nestling/Fledgling",
} 

#color map for pattern matching
CMAP_PM = {"Male Song":         "Greens", 
           "Male Chorus":       "Oranges", 
           "Female":            "Purples", 
           "Hatchling":         "Blues",
           "Nestling" :         "Blues",
           "Fledgling":         "Blues",
           "Insect 30":         "Greys",
           "Insect 31":   	    "Greys",
           "Insect 32":         "Greys",	
           "Insect 33":         "Greys",
           "Pacific Tree Frog": "YlGn",	
           "Red-legged Frog":   "YlGn",
           "Bull Frog":         "YlGn"}

NO_DATA_COLOR = "lightgray"
HATCH_PATTERN = "////////"
HATCH_BG_COLOR = "mintcream"
HATCH_DARK_COLOR = "silver"
BORDER_WIDTH = 0.25             #for the vertical month dividers and the exterior edges
TEMP_LINES = 0.5                #for the red lines on the weather graph
LABEL_OFFSET = 0.125            #gap between bottom of the graph and the months 
EDGE_GRAPH_BORDER_WIDTH = 0.5   
EDGE_GRAPH_BORDER_INSET = 0.02
EDGE_GRAPH_COLOR_SCALE = 0.6    #Bigger is darker

#Files, paths, etc.
DATA_FOLDER = 'Data/'
PMJ_DATA_FOLDER = 'PMJ Data/'
FIG_FOLDER = 'Figures/'
DATA_DIR = Path(__file__).parents[0] / DATA_FOLDER
PMJ_DATA_DIR = Path(__file__).parents[0] / PMJ_DATA_FOLDER
FIGURE_DIR = Path(__file__).parents[0] / FIG_FOLDER
ALL_FILE = 'TRBL Analysis tracking - All.csv'
SHEET_HEADER_SIZE = 2 #number of rows to skip over
WEATHER_FILE = 'weather_history.csv'
ERROR_FILE = Path(__file__).parents[0] / 'error.txt'
DATES_FILE = 'analyzed dates.csv'

#This is everything except the data files, because those are auto-generated
FILES = {
    ALL_FILE : DATA_DIR / ALL_FILE,
    WEATHER_FILE : DATA_DIR / WEATHER_FILE,
    DATES_FILE : Path(__file__).parents[0] / DATES_FILE
}

#TODO: For clarity, rename all symbols that are constants to be all caps.

PULSE_COUNT = "pulse_count"
ABANDONED = "abandon"
PULSES = ["p1", "p2", "p3", "p4"]
SUMMARY_FIRST_REC = "First Recording"
SUMMARY_LAST_REC = "Last Recording"
SUMMARY_EDGE_DATES = [SUMMARY_FIRST_REC, SUMMARY_LAST_REC]
PULSE_MC_START = "mcstart"
PULSE_MC_END = "mcend"
PULSE_INC_START = "incstart"
PULSE_HATCH = "hatch"
PULSE_FIRST_FLDG = "fledgestart"
PULSE_LAST_FLDG = "fledgedisp"
PULSE_DATE_TYPES = [PULSE_MC_START, PULSE_MC_END, PULSE_INC_START, PULSE_HATCH, PULSE_FIRST_FLDG, PULSE_LAST_FLDG, ABANDONED]
summary_date_cols = [p + ' ' + d for p in PULSES for d in PULSE_DATE_TYPES]
summary_numeric_cols = ["Site ID", "Altitude", "Number of Recordings"]

PHASE_MALE_CHORUS = "Settlement"
PHASE_INC = "Incubation"
PHASE_BROOD = "Brooding"
PHASE_FLDG = "Fledgling"
PULSE_PHASES = {PHASE_MALE_CHORUS : [PULSE_MC_START, PULSE_INC_START], #-1
                PHASE_INC : [PULSE_INC_START, PULSE_HATCH],            #-1
                PHASE_BROOD : [PULSE_HATCH, PULSE_FIRST_FLDG],         #-1
                PHASE_FLDG : [PULSE_FIRST_FLDG, PULSE_LAST_FLDG]}


#
#Pattern Matching Files
#edit this if we add/remove file types
#Change: Color Map for Pattern Matching, Legend Text, plus File Types. Also, there are some lists
#of column names in summarize_pm() that likely need to change
PM_SONG_TYPES = ["Male Song",
                 "Male Chorus", 
                 "Female", 
                 "Hatchling", 
                 "Nestling",
                 "Fledgling"]

#NOTE Dec 2024: The file names are matching what the PM Downloader does, which is missing the "sp" from the name
#       so to prevent having to re-download everything we'll leave it this way and change it in the graph
#       rendering code or elsewhere as necessary. If this changes, need to update the cmap and the legend text
PM_INSECT_SP30 = "Insect 30"  #Making these variables because this string is referenced in the graphing code
PM_FROG_PACTF = "Pacific Tree Frog"
PM_OTHER_TYPES = {
    PM_INSECT_SP30 : "Bug 30",
    "Insect 31": "Bug 31",	
    "Insect 32": "Bug 32",	
    "Insect 33": "Bug 33",	
    PM_FROG_PACTF: "Pacific Tree Frog",	
    "Red-legged Frog": "Red-legged Frog",
    "Bull Frog": "Bull Frog",
}

#adjust the set of file types based on what data we want to show in the graphs
if INCLUDE_INSECT_AND_FROG_DATA:
    PM_FILE_TYPES = PM_SONG_TYPES + list(PM_OTHER_TYPES.keys())
else:
    PM_FILE_TYPES = PM_SONG_TYPES

#Abbreviations are used in the summary table, to reduce column width
pm_abbreviations = ["PM-MS", "PM-MC", "PM-F", "PM-H", "PM-N", "PM-FL","PM-I30", "PM-I31", "PM-I32", "PM-I33", "PM-PTF", "PM-RLF", "PM-BF"]
pm_friendly_names = dict(zip(PM_FILE_TYPES, pm_abbreviations))

FIRST = "First"
LAST = "Last"
BEFORE_FIRST = "Before First"
AFTER_LAST = "After Last"

valid_pm_date_deltas = {PM_SONG_TYPES[1]:0, #Male Chorus to Female can be 0 days
                        PM_SONG_TYPES[2]:5, #Female to Hatchling must be at least 5 days
                        PM_SONG_TYPES[3]:0, #Hatchling to Nestling can be 0 days
                        PM_SONG_TYPES[4]:3, #Nestling to Fledgling must be at least 3 days
                        PM_SONG_TYPES[5]:0, #Nestling to Nestling is zero, here to make math easy
                        }

MISSING_DATA_FLAG = -100
PRESERVE_EDGES_FLAG = -99

DPI = 300

error_list = ''

#
#
# Helper functions
#
#
def append_to_csv(df, site, csv_filename):
    # Replace <br> with \n in the DataFrame
    df = df.replace(r"<br>", "\n", regex=True)

    # Flatten the DataFrame into one row with columns prefixed by the row index
    flat_data = {f"{pulse}{category}": value for pulse, category, value in df.stack().reset_index().values}
    flat_data["Site"] = site  # Add the site as a separate column

    # Convert the flattened data to a DataFrame
    flat_df = pd.DataFrame([flat_data])

    # Reorder columns to make "Site" the first column
    columns = ["Site"] + [col for col in flat_df.columns if col != "Site"]
    flat_df = flat_df[columns]

    # Append to CSV, creating it if it doesn't exist
    with open(csv_filename, 'a',  newline='') as f:
        write_header = f.tell() == 0  # Write header only if file is empty
        flat_df.to_csv(f, index=False, header=write_header)


def format_timestamp(ts):
    if pd.notna(ts):
        if isinstance(ts, pd.Timestamp):
            return ts.strftime('%m/%d')
        elif isinstance(ts, str):
            return ts
    else:
        return "None"
    
def my_time():
    return dt.now().strftime('%d-%b-%y %H:%M:%S')

def init_logging():
    if not BEING_DEPLOYED_TO_STREAMLIT:
        remove_file(ERROR_FILE)
        with ERROR_FILE.open("a") as f:
            f.write(f"Logging started {my_time()}\n")    

def log_error(msg: str):
    global error_list
    error_list += f"{msg}\n\n"
    # if not BEING_DEPLOYED_TO_STREAMLIT:
    #     with ERROR_FILE.open("a") as f:
    #         f.write(f"{my_time()}: {msg}\n")

def show_error(msg: str):
    #Only show the error if we're doing one graph at a time, but log it
    if not MAKE_ALL_GRAPHS:
        st.error(msg)
    log_error(msg)

def pairwise(iterable):
    a, b = tee(iterable) # Note that tee is from itertools
    next(b, None)
    return zip(a, b)

def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 10  #the 'empty' files seem to have a few bytes, so just being safe by using 10 as a min length 

def count_files_in_folder(fpath):
    i = 0
    for item in os.scandir(fpath):
        if item.is_file():
            i += 1
    return i


def make_date(row: pd.Series) -> pd.Timestamp:
    y = int(row["year"])
    m = int(row["month"])
    d = int(row["day"])
    return pd.Timestamp(year=y, month=m, day=d)

#
#
#File handling and setup
#
#
@st.cache_data
def load_all_file():
    return pd.read_csv(
        FILES[ALL_FILE],
        skiprows=SHEET_HEADER_SIZE
    )


def get_target_sites() -> list:
    #Load the list of unique site names, keep just the 'Name' column, and then convert that to a list
    #all_sites = pd.read_csv(FILES[ALL_FILE], usecols = ["Name", "Skip Site", "Comment for Skip Site"], skiprows=SHEET_HEADER_SIZE)
    all_sites_and_cols = load_all_file()
    all_sites = all_sites_and_cols[["Name", "Skip Site", "Comment for Skip Site"]]
    #Clean it up. Only keep names that start with a 4-digit number and are not to be skipped. 
    filtered_sites = all_sites.loc[
        (all_sites["Skip Site"] != "Y") & (all_sites["Name"].str.startswith("20")),
        "Name"
    ].tolist()

    if len(filtered_sites):
        filtered_sites.sort()
    else:
        show_error('No site files found')

    return filtered_sites

#TODO Remove this, all the inputs are automatically generated
#Used by the two functions that follow to do file format validation
def confirm_columns(target_cols:dict, file_cols:list, file:Path) -> list:
    errors_found = []
#    if len(target_cols) != len(file_cols):
#        log_error(f"confirm_columns: File {file} has an unexpected number of columns, {len(file_cols)} instead of {len(target_cols)}")
#        show_error(f"confirm_columns: File {file} has an unexpected number of columns, {len(file_cols)} instead of {len(target_cols)}")
    for col in target_cols:        
        if  target_cols[col] not in file_cols:
            errors_found.append(target_cols[col])
#            show_error(f"confirm_columns: Column {target_cols[col]} missing from file {file}")
    
    return errors_found


def fix_bad_values(df:pd.DataFrame):
    """
    This function finds columns containing "---", prints a warning message,
    and replaces all "---" with 0 in-place within the DataFrame. Note that the way python works,
    I'm actually modifying the original!
    """
    for col in df.columns:
        if col.startswith("tag") and -100 in df[col].values:
            log_error(f'fix_bad_values: Column {col} contains "---"')
            df[col] = df[col].replace(-100, 0)


def check_edge_cols_for_errors(df:pd.DataFrame) -> bool:
    error_found = False

    #Remove any -100 (were "---" in the original file, converted to numbers in the first cleaning pass) and log it, if there are any
    fix_bad_values(df)

    # # For each day, there should be only either P1F or P1N, never both
    # tag_errors = df.loc[(df[data_col[tag_p1f]]>=1) & (df[data_col[tag_p1n]]>=1)]

    # if len(tag_errors):
    #     error_found = True
    #     show_error("check_edge_cols_for_errors: Found recordings that have both P1F and P1N tags, see log")
    #     for f in tag_errors[FILENAME]: 
    #         log_error(f"check_edge_cols_for_errors: {f}\tRecording has both P1F and P1N tags")

    return error_found 


def find_invalid_rows(df, cols_to_filter, cols_to_check, output_file="invalid_filenames.txt"):
    """
    Find rows where filter columns contain '1' but check columns contain '---'.
    
    Args:
        df: DataFrame with all string values
        cols_to_filter: List of column names to check for '1'
        cols_to_check: List of column names to check for '---'
        output_file: File to save invalid filenames to
    
    Returns:
        List of invalid filenames
    """
    
    # Find rows where at least one filter column has value "1"
    filter_mask = df[cols_to_filter].eq(1).any(axis=1)
    filtered_rows = df[filter_mask]
    
    log_error(f"Found {len(filtered_rows)} rows where at least one filter column equals '1'")
    
    # In those filtered rows, check if any of the check columns contain "---"
    invalid_mask = filtered_rows[cols_to_check].eq("---").any(axis=1)
    invalid_rows = filtered_rows[invalid_mask]
    
    log_error(f"Found {len(invalid_rows)} rows with invalid data (containing '---' in check columns)")
    
    # Get the filenames from invalid rows
    if len(invalid_rows) > 0:
        invalid_filenames = invalid_rows["filename"].tolist()
        
        # Save to file
        with open(output_file, 'w') as f:
            for filename in invalid_filenames:
                f.write(f"{filename}\n")
        
        log_error(f"Saved {len(invalid_filenames)} invalid filenames to '{output_file}'")
    else:
        log_error("No invalid rows found!")
        
    return


def check_for_tag_errors(df: pd.DataFrame):
    #Check these cols, if any of these has a 1 then...
    cols_to_filter = MANUAL_COLS + MINI_MANUAL_COLS

    #report any rows where any of these cols has the value "---"
    cols_to_check = [data_col[ALTSONG1], data_col[ALTSONG2], data_col[MALE_SONG], data_col[COURT_SONG]]

    find_invalid_rows(df, cols_to_filter, cols_to_check)

    return



# Load the main data.csv file into a dataframe, validate that the columns are what we expect
@st.cache_data
def load_data() -> pd.DataFrame:
    files_to_load = [DATA_DIR / f"data {year}.csv" for year in range(2017, 2025)]
    combined_df = pd.DataFrame()
    for file_name in files_to_load:
        #Validate the data file format
        headers = pd.read_csv(file_name, nrows=0).columns.tolist()
        missing_columns = confirm_columns(data_col, headers, file_name)

        #The set of columns we want to use are the basic info (filename, site, date), all songs, and all tags
        usecols = [data_col[FILENAME], data_col[SITE], data_col[DATE]]
        for song in ALL_SONGS:
            usecols.append(data_col[song])
        for tag in ALL_TAGS:
            usecols.append(data_col[tag])

        #remove any columns that are missing from the data file, so we don't ask for them as that will cause
        #an exception. Hopefully the rest of the code is robust enough to deal...
        usecols = [item for item in usecols if item not in missing_columns]

        # 0) Read the file         
        df = pd.read_csv(file_name, usecols=usecols)

        # 1) Convert the date column explicitly
        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

        # 2) Make it the index
        df = df.set_index('date')

        combined_df = pd.concat([combined_df, df]) #NOTE This assumes the files don't have overlapping dates

    # We've loaded all the data, let's do a quick error check
    check_for_tag_errors(combined_df)
    return combined_df

def load_data_for_site(site:str):
    '''
    Given a site, retrieve the data set for that 
    
    :param site: Description
    :type site: str
    '''
    # df = pd.DataFrame()

    # #Figure out which data file we need
    year = site[0:4]
    # file_name = DATA_DIR / f"data {year}.csv"
    # headers = pd.read_csv(file_name, nrows=0).columns.tolist()
    # missing_columns = confirm_columns(data_col, headers, file_name)

    # #The set of columns we want to use are the basic info (filename, site, date), all songs, and all tags
    # usecols = [data_col[FILENAME], data_col[SITE], data_col[DATE]] 
    # for song in ALL_SONGS:
    #     usecols.append(data_col[song])
    # for tag in ALL_TAGS:
    #     usecols.append(data_col[tag])
    # #remove any columns that are missing from the data file, so we don't ask for them as that will cause
    # #an exception. Hopefully the rest of the code is robust enough to deal...
    # usecols = [item for item in usecols if item not in missing_columns]

    # # 0) Read the file
    # df = pd.read_csv(file_name, usecols=usecols)
    # df = df[df[SITE] == site]
    # # 1) Convert the date column explicitly
    # df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

    # # 2) Make it the index
    # df = df.set_index('date')

    pusecols = [data_col[FILENAME], data_col[SITE], data_col[DATE]] 
    for song in ALL_SONGS:
        pusecols.append(data_col[song])
    for tag in ALL_TAGS:
        pusecols.append(data_col[tag])

    pfile_name = DATA_DIR / f"data {year}.parquet"
    pusecols.append("dt")

    pdf = pd.read_parquet(pfile_name, columns=pusecols)
    pdf = pdf[pdf[SITE] == site]
    pdf = pdf.set_index("dt")
    pdf.index = pd.DatetimeIndex(pdf.index).normalize()
    df = clean_data(pdf, [site])
    df = df.rename_axis("date")
    
    return df


def load_pm_data(site:str) -> pd.DataFrame:
    # Load the pattern matching CSV files into a dataframe, validate that the columns are what we expect
    # These are the files from all the folders named by site. 
    # If there is a missing file, we want to have the data for that type of pattern be empty, adding columns with 
    # the right headers but empty data for any missing columns. Then make the graphing code robust enough
    # to deal with columns with zeros.

    # For each type of file for this site, try to load the file. 
    # Add a column to indicate which type it is. Then append it to the dataframe we're building. We end up with a 
    # table that has the site, date, and type columns with all the PM data in rows below. So, if there were 1000 PM 
    # events for each type, our table would have 5000 rows. 
    df = pd.DataFrame()
    usecols =[site_columns[SITE], site_columns['year'], site_columns['month'], 
            site_columns['day'], site_columns[validated]]

    # Add the site name so we look into the appropriate folder
    site_dir = PMJ_DATA_DIR / site
    if os.path.isdir(site_dir):
        for t in PM_FILE_TYPES:
            fname = f"{site} {t}.csv"
            full_file_name = site_dir / fname

            df_temp = pd.DataFrame()
            if is_non_zero_file(full_file_name):
                #Validate that all columns exist, and abandon ship if we're missing any
                headers = pd.read_csv(full_file_name, nrows=0).columns.tolist()
                
                df_temp = pd.read_csv(full_file_name, usecols=usecols)
                #make a new column that has the date in it, take into account that the table could be empty
                if not df_temp.empty:
                    date_str = (
                        df_temp["year"].astype("int64").astype(str)
                        + "-"
                        + df_temp["month"].astype("int64").astype(str).str.zfill(2)
                        + "-"
                        + df_temp["day"].astype("int64").astype(str).str.zfill(2)
                    )
                    df_temp[DATE] = pd.to_datetime(date_str, errors="coerce")
                    #BOOMdf_temp[DATE + "2"] = df_temp.apply(make_date, axis=1)

                else:
                    df_temp[DATE] = []

            else:
                df_temp[DATE] = []

            #Finally, add the table that we loaded to the end of the main one
            df_temp["type"] = t
            # Ensure all columns in df_temp have explicit dtypes to avoid warning
            df_temp = df_temp.astype("object")
            df = pd.concat([df, df_temp], ignore_index=True)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    return df



@st.cache_data
def load_summary_data() -> pd.DataFrame:
    #Load the summary data and prep it for graphing. 
    df = load_all_file()
    #If needed, can convert to date values as below, but it doesn't seem necessary
    #df[date_cols] = df[date_cols].apply(pd.to_datetime, errors='coerce')

    # Convert numeric columns to integers. As above, you have to force it this way if the types vary.
    # Empty values or strings are converted to NaN
    df[summary_numeric_cols] = df[summary_numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[summary_numeric_cols] = df[summary_numeric_cols].astype(pd.Int64Dtype())  # Keeps NaNs

    return df

# clean up the data for a particular site
DATE_FORMAT = "%m/%d/%Y"
def is_valid_date_string(date_string):
    potential_date = date_string
    if type(date_string) == str:
        # Strip leading '~' and trailing '*' markers around dates
        # e.g. "~8/24/2024" -> "8/24/2024", "7/14/2024*" -> "7/14/2024"
        potential_date = potential_date.lstrip("~").rstrip("*").strip()

    result = pd.to_datetime(potential_date, format=DATE_FORMAT, errors="coerce")
    if pd.isna(result):
        return False
    else:
        return True 


def convert_to_datetime(date_string):
    potential_date = date_string
    if type(date_string) == str:
        # Strip leading '~' and trailing '*' markers around dates
        # e.g. "~8/24/2024" -> "8/24/2024", "7/14/2024*" -> "7/14/2024"
        potential_date = potential_date.lstrip("~").rstrip("*").strip()

    ts = pd.to_datetime(potential_date, format=DATE_FORMAT, errors="coerce")
    return ts
    

def is_valid_date(timestamp):
    return pd.notna(timestamp)


def is_valid_date_pair(phase_data:dict) -> bool:
    result = False
    start = phase_data[START]
    end = phase_data[END]
    if is_valid_date(start) and is_valid_date(end):
        result = True
    return result


def count_valid_pulses(pulse_data:dict) -> int:
    #A pulse is considered valid if there is at least one "graphable" date pair
    count = 0
    for p in PULSES:
        result = False
        for phase in pulse_data[p]:
            if phase in PULSE_PHASES.keys(): #Need to skip Abandoned, as it doesn't have a pair of dates
                if is_valid_date_pair(pulse_data[p][phase]):
                    result = True
                    break
        count += 1 if result else 0

    return count


def get_val_from_df(df:pd.DataFrame, col) -> str:
    result = df.iloc[0,df.columns.get_loc(col)]
    return str(result)


def process_site_summary_data(summary_row:pd.DataFrame) -> dict:
    nd_string = "ND"
    first_rec = get_val_from_df(summary_row, SUMMARY_FIRST_REC)
    last_rec = get_val_from_df(summary_row, SUMMARY_LAST_REC)

    #TODO Is this the best way to handle the zero recording case?
    if pd.isna(first_rec):
        log_error("process_site: date of first recording was empty")
        return {}

    summary_dict = {
        SUMMARY_FIRST_REC   : convert_to_datetime(first_rec),
        SUMMARY_LAST_REC    : convert_to_datetime(last_rec),
    }

    for pulse in PULSES:
        pulse_result = {}
        error_prefix = f'process_site: {str(summary_row.iloc[0]["Name"])} at {pulse}'

        #Make our list of abandoned dates for later graphing purposes
        abandoned_date = convert_to_datetime(get_val_from_df(summary_row, f"{pulse}{ABANDONED}"))
        if is_valid_date(abandoned_date):
            pulse_result[ABANDONED] = abandoned_date 

        for phase in PULSE_PHASES:
            start, end = PULSE_PHASES[phase]

            target1 = f"{pulse}{start}"
            value1 = get_val_from_df(summary_row, target1) 
            result1 = pd.NaT

            target2 = f"{pulse}{end}"
            value2 = get_val_from_df(summary_row, target2)
            result2 = pd.NaT

            if is_valid_date_string(value1):
                #It's a good date, so format it
                result1 = convert_to_datetime(value1)

                if (value2.lower() not in [nd_string.lower(), "post"]) and not is_valid_date(value2):
                    log_error(f"{error_prefix}: {target1} is a valid date {value1}, but {target2} is {value2} and not ND, post, or a date")

            elif pd.notna(value1) and value1.startswith(ABANDONED):
                if not is_valid_date(abandoned_date):
                    log_error(f"{error_prefix}: Column Abandoned does not have a valid abandoned date")
                else:
                    result1 = pd.NaT
            #Check: if the phase = brooding and it is "Before Start, HBC Present" then we want to draw a left-pointing
            #arrow on the graph. So, if we find this, save it with a signal we can pass along to the graph maker
            elif value1.lower() == "before start, hbc present".lower():
                result1 = convert_to_datetime("6/1/1967") 
            elif value1.lower() == "before start, hbc absent".lower():
                pass #Don't do anything right now, maybe later
            elif value1 == "pre":
                #If the start date is "pre" then we want to indicate this so we can draw the arrow on the graph
                result1 = convert_to_datetime("6/1/1967") 
            elif value1 == "post":
                if value2 != "post":
                    log_error(f"{error_prefix}: {target1} is 'after end' but {target2} is not")
            elif value1 == nd_string or value1 == "" or pd.isna(value1):
                #this is OK, we aren't going to draw anything in this case
                pass
            else:
                #if not one of the above, then it's an error
                log_error(f"{error_prefix}: Found invalid data in {target1}: {value1}")

            if is_valid_date_string(value2):
                if (value1.lower() not in ["pre", nd_string.lower()]) and not is_valid_date_string(value1):
                    log_error(f"{error_prefix}: {target2} is a valid date, but {target1} is not ND, pre, or a valid date")
                #It's a good date, so format it
                if phase == PHASE_FLDG:
                    #For fledgling phase, don't subtract one from the end date
                    delta = pd.Timedelta(days=0)
                else:
                    delta = pd.Timedelta(days=1)
                result2 = convert_to_datetime(value2) - delta
            elif pd.notna(value2) and value2.startswith(ABANDONED):
                if not is_valid_date(abandoned_date):
                    log_error(f"{error_prefix}: Column Abandoned does not have a valid abandoned date")
                else:
                    result2 = abandoned_date - pd.Timedelta(days=1)
            elif value2.lower() == "pre".lower():
                #In this scenario, the start should be ND, throw an error if not
                if not value1 == nd_string:
                    log_error(f"{error_prefix}: In {target2} end date is 'pre' but start date is not 'ND'")
            elif value2.lower() == "post".lower():
                #See commentary above about "before start". It used to use the date of the last recording:
                #    result2 = summary_dict[SUMMARY_LAST_REC]
                pass
            elif value2 == nd_string:
                if not value1 == nd_string:
                    log_error(f"{error_prefix}: Second date is ND, but first date is not: {target1}:{value1}, {target2}:{value2}") 
            elif pd.isna(value2):
                # Blank cell, should be OK if value1 is also blank
                if pd.notna(value1):
                    log_error(f"{error_prefix}: Found {value2} in {target2}")
            else: #ND, empty, or any other values are not valid here
                log_error(f"{error_prefix}: Found {value2} in {target2}, which is invalid data")
            
            pulse_result[phase] = {"start":result1, "end":result2}

        #Add the sets of dates to our master dictionary
        summary_dict[pulse] = pulse_result

    #Calculate count of valid pulses. If there were zero, then set the count to 1 else we won't get a graph
    p_count = max(1, count_valid_pulses(summary_dict))
    summary_dict[PULSE_COUNT] = p_count

    #Save our abandoned dates, if any
#    summary_dict[abandoned] = abandoned_dates

    return summary_dict 


def get_pretty_name_for_site(site:str) -> str:
        name_column = "Pretty Site Name"
        name_dict = get_site_info(site, [name_column])
        return name_dict[name_column]


#Perform the following operations to clean up the data:
#   - Drop sites that aren't needed, so we're passing around less data
#   - Exclude any data where the year of the data doesn't match the target year
#   - Exclude any data where there aren't recordings on consecutive days  ##SEP2025 no longer doing this
@st.cache_data
def clean_data(df: pd.DataFrame, site_list: list) -> pd.DataFrame:
    # Drop sites we don't need
    df_clean = pd.DataFrame()
    for site in site_list:
        if SITE not in df.columns:
            break

        df_site = df[df[SITE] == site]

        #used to ensure anything outside this year gets dropped
        target_year = site[0:4]

        # Sort newest to oldest (backwards) and filter to this year
        df_site = df_site.sort_index(ascending=False)
        original_size = df_site.shape[0]
        df_site_filtered = df_site.query(f"date <= '{target_year}-12-31'")
        if df_site_filtered.shape[0] != original_size:
            log_error(f"clean_data: Data for site {site} has the wrong year in it, newer than its year")
            filtered_out = df_site.merge(df_site_filtered, how="left", indicator=True)
            filtered_out = filtered_out[filtered_out["_merge"] == "left_only"].drop(columns=["_merge"])
            if FILENAME in filtered_out.columns:
                log_error(filtered_out[FILENAME])
            else:
                log_error(filtered_out.sort_values("type"))

        #Sort oldest to newest, and filter to this year
        df_site = df_site.sort_index(ascending=True)
        original_size = df_site.shape[0]
        df_site_filtered = df_site.query(f"date >= '{target_year}-01-01'")
        if df_site_filtered.shape[0] != original_size:
            log_error(f"clean_data: Data for site {site} has the wrong year in it, older than its year")
            filtered_out = df_site.merge(df_site_filtered, how="left", indicator=True)
            filtered_out = filtered_out[filtered_out["_merge"] == "left_only"].drop(columns=["_merge"])
            if FILENAME in filtered_out.columns:
                log_error(filtered_out[FILENAME])
            else:
                log_error(filtered_out.sort_values("type"))

        df_clean = pd.concat([df_clean, df_site])
    
    # We need to preserve the diff between no data and 0 tags. But, we have to also make everything 
    # integers for later processing. So, we'll replace the hyphens with a special value and then just 
    # realize that we can't do math on this column any more without excluding it. Picked -100 (missing_data_flag) because 
    # if we do do math then the answer will be obviously wrong!
    df_clean = df_clean.replace('---', MISSING_DATA_FLAG)
    
    # For each type of song, convert its column to be numeric instead of a string so we can run pivots
    for s in ALL_SONGS + ALL_TAGS:
        if data_col[s] in df_clean.columns:
            df_clean[data_col[s]] = pd.to_numeric(df_clean[data_col[s]])
    return df_clean


#
#
# Data Analysis
# 
#  

import operator as op

_OPS = {
    ">": op.gt,
    ">=": op.ge,
    "<": op.lt,
    "<=": op.le,
    "==": op.eq,
    "!=": op.ne,
}
def filter_df_by_tags(df: pd.DataFrame, target_tags: list[str], filter_str: str = ">0", exclude_tags: list[str] | None = None) -> pd.DataFrame:
    missing = set(target_tags) - set(df.columns)
    if missing:
        #if the tags aren't there, then just return the whole thing?
        return df
    else:
        exclude_tags = exclude_tags or []

        op_token = filter_str[:2] if filter_str[:2] in _OPS else filter_str[:1]
        val_token = filter_str[len(op_token):]
        threshold = float(val_token)
        cmp = _OPS[op_token]

        target_mask = cmp(df[target_tags], threshold).any(axis=1)

        if exclude_tags:
            exclude_mask = cmp(df[exclude_tags], threshold).any(axis=1)
            target_mask &= ~exclude_mask

        return df.loc[target_mask]

# Add missing dates by creating the largest date range for our graph and then reindex to add missing entries
# Also, transpose to get the right shape for the graph
def normalize_pt(pt:pd.DataFrame, date_range_dict:dict) -> pd.DataFrame:
    date_range = pd.date_range(date_range_dict[START], date_range_dict[END]) 
    temp = pt.reindex(date_range)  #.fillna(0)
    temp = temp.transpose()
    temp = temp.astype(float) #convert all numeric data to floats

    return temp


def make_pivot_table(df, date_range_dict, preserve_edges=False, labels=None, label_dict=None):
    labels = labels or []
    label_dict = label_dict or {}
    if df.empty:
        return pd.DataFrame()

    if (set(label_dict.keys()) | set(label_dict.values())) - set(df.columns):
        #some columns are missing, so get out
        return pd.DataFrame()
    

    date_colname = data_col[DATE]

    if label_dict:
        out = {}
        for tag_col, value_col in label_dict.items():
            # rows where this tag is present
            m = df[tag_col].gt(0)   # same as >0
            if not m.any():
                continue
            # count occurrences where value_col >= 1
            ser = df.loc[m, value_col].ge(1)                    
            if df.index.name == date_colname:
                s = ser.groupby(level=date_colname).sum()
            else:
                s = ser.groupby(df.loc[m, date_colname]).sum()
                
            out[tag_col] = s

        aggregate_df = pd.DataFrame(out).fillna(0).astype(int)

    else:
        if not labels:
            return pd.DataFrame()
        value_cols = df[labels].select_dtypes(include="number").columns
        date_colname = data_col[DATE]
        if df.index.name == date_colname:
            aggregate_df = df[labels].ge(1).groupby(level=date_colname).sum()
        else:
            aggregate_df = df[labels].ge(1).groupby(df[date_colname]).sum()


    if preserve_edges:
        aggregate_df = aggregate_df.replace(0, PRESERVE_EDGES_FLAG)

    return normalize_pt(aggregate_df, date_range_dict)



# Pivot table for pattern matching is a little different
def make_pattern_match_pt(site_df: pd.DataFrame, type_name:str, date_range_dict:dict) -> pd.DataFrame:
    #If the value in 'validated' column is 'Present', count it.
    present = site_df[site_df[site_columns[validated]]=="present"]
    aggregate = present.pivot_table(index=DATE, values=site_columns[validated], aggfunc='count')
    aggregate = aggregate.rename(columns={validated:type_name})
    
    # If the pivot table is empty, ensure all dates exist with value 0
    if aggregate.empty:
        all_dates = site_df.index.unique()  # Get all dates from original df
        aggregate = pd.DataFrame(np.nan, index=all_dates, columns=[type_name])  # Fill with zeros
        aggregate.index.name = DATE  # Set the index name properly
    
    return normalize_pt(aggregate, date_range_dict)


def song_count_sufficient(value, threshold):
    return pd.notna(value) and value >= threshold

def find_pm_dates(row: pd.Series, pulse_gap:int, threshold: int) -> dict:
    # Scan through a row of pattern matching data and return pairs of dates such that the first date is preceeded by
    # NA values and is greater than threshold, while the second date is after the first date and there is no more than
    # one value less than the threshold or NA between it and the first date

    # Example:
    # If the row is: 0 0 1 5 6 1 7 0 0 8 1 9 0 10
    # Then the date pairs to be returned are the dates for 5, 7, 8, and 10

    dates = {}
    last_column = 0
    looking_for_first = True
    consecutive_dates_below_threshold = 0
    skip_ahead = False
    col = 0
    pulse = 1
    CONSECUTIVE_THRESHOLD = 2
    while col < len(row):       
        if looking_for_first:
            if song_count_sufficient(row.iloc[col], threshold):
                column_date = row.index[col]
                dates[pulse] = {
                    FIRST : column_date,
                    #If we're at the very beginning, then we don't actually know when it started, so note this
                    BEFORE_FIRST : col == 0
                }
                last_column = col
                looking_for_first = False
        else:
            # We're looking for two consecutive NA or less than threshold
            if song_count_sufficient(row.iloc[col], threshold):
                last_column = col
                consecutive_dates_below_threshold = 0            
            else: #No data, or at least there wasn't enough calls
                consecutive_dates_below_threshold += 1
                if consecutive_dates_below_threshold >= CONSECUTIVE_THRESHOLD: 
                    # Found enough consecutive dates below threshold to consider that the pulse ended
                    column_date = row.index[last_column]
                    dates[pulse].update({
                        LAST : column_date,
                        AFTER_LAST : False
                    })
                    consecutive_dates_below_threshold = 0
                    looking_for_first = True # Now that we found the end, we're looking for the first date in the next pulse
                    skip_ahead = True # Now that we've found a pair, skip forward by the pulse gap and start over

        #Either skip ahead by 1 for a normal case, or the pulse gap if we just found a pair
        if skip_ahead:
            col += pulse_gap
            skip_ahead = False
            pulse += 1
        else:
            col += 1 
    
    #Detect the case where the last phase ended on or after the recorder was pulled
    if dates and len(dates[len(dates)]) == 2:
        #We want to capture the last date in the row. Because of "pulse_gap", col could be beyond the end
        #of the table, so we'll use the value we know is good
        dates[len(dates)].update({
            LAST : row.index[len(row)-1],
            AFTER_LAST : True
        })

    return dates


def make_empty_summary_row() -> dict:
    # Create an empty row for a single pulse
    phases = PM_FILE_TYPES[1:] #Creates a new list except it drops "Male Song"
    base_dict = {}
    for phase in phases:
        #NOTE Dec 2024: added this if statement to limit the summarizing to just the bird songs
        if phase in PM_SONG_TYPES:
            base_dict[f"{phase}"] = {}
    return base_dict

def make_empty_summary_dict() -> dict:
    # Create the entire empty summary dict, so we don't get key errors
    base_dict = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
    for k in base_dict:
        base_dict[k] = make_empty_summary_row()
    return base_dict

def find_last_non_empty_key(d):
    # Walk through a dictionary backwards and return the first non-empty key 
    # Used to find the last key with data
    for key in reversed(d.keys()):
        if d[key]:  # Check if the value is non-empty
            return key
    return None  # Return None if all values are empty

def  find_first_non_empty_key(d):
    # Walk through a dictionary forwards and return the first non-empty key 
    # Used to find the first key with data
    for key in d.keys():
        if d[key]:  # Check if the value is non-empty
            return key
    return None  # Return None if all values are empty

#NOTE Dec 2024: this used to use "pm_file_types" and it attempted to auto-analyze all the types of calls
#       in the same way. However, now that we're adding insects, et al, I changed it to specifically look
#       only at the bird vocalizations by changing "pm_file_types" to "pm_song_types"
def find_correct_pulse(target_phase:str, target_date:pd.Timestamp, proposed_pulse:int, current_dates:dict):
    # Check to see if a pulse already has a date for a phase that is later than the current one.
    correct_pulse = proposed_pulse
    all_phases = PM_SONG_TYPES[1:] #Creates a new list except it drops "Male Song"
    target_position = all_phases.index(target_phase)

    while True:
        current_latest_phase = find_last_non_empty_key(current_dates[correct_pulse])

        if current_latest_phase in all_phases:
            latest_position = all_phases.index(current_latest_phase)
            if target_position <= latest_position:
                # The one we want to add is earlier or in the same position in the sequence as 
                # something already there, this means it's in the wrong pulse

                #BUT, if it's a Hatchling and the one that's after it is a Nestling, that's OK if the dates are close
                if target_phase == "Hatchling" and current_latest_phase == "Nestling":
                    if abs(current_dates[correct_pulse]["Nestling"][FIRST] - target_date) <= pd.Timedelta(days=6):
                        break

                correct_pulse += 1                
            else:
                break
        else:
            # The result was "None", so pulse is currently empty and it's OK to add to it
            break
    
    return correct_pulse


def correct_pulse_has_date_collision(target_phase:str, target_date:pd.Timestamp, target_pulse:dict):
    result = False
    if target_phase == "Fledgling":
        #This is the one that is problematic
        earlier_phase = find_last_non_empty_key(target_pulse)
        if earlier_phase is not None:
            #Any phase will be earlier than Nestling
            assert earlier_phase != "Fledgling", "Should never get a matching phase at this point"

            #Check that the start date is no closer that it should be
            earlier_phase_start = target_pulse[earlier_phase][FIRST]
            min_delta = 0 
            start_adding = False

            for item in PM_SONG_TYPES:
                if item == earlier_phase:
                    start_adding = True 
                min_delta += valid_pm_date_deltas[item] if start_adding else 0

            if (target_date - earlier_phase_start) <= pd.Timedelta(days=min_delta):
                #we have a problem!
                result = True

    return result

def clean_pm_dates(dates:dict):
    #Don't want Male Song in our results
    del dates["Male Song"]

    first_dates = []
    for phases, pulses in dates.items():
        for pulse, date in pulses.items():
            if "First" in date:
                first_dates.append((date[FIRST], f"{phases}{pulse}"))
    
    #TODO I'm not sure if this sorting is sufficient. Multiple sort passes may be necessary to get the pulses 
    #   in the right order
    first_dates.sort(key=lambda x: x[0]) 

    #Generate a blank dictionary so that we don't end up with any key errors
    temp_dict = make_empty_summary_dict()

    # We're now going to fill out the summary dict by walking through the dates in order and placing them where appropriate.
    # Note that this might require moving a key to a different pulse!
    for date in first_dates:
        proposed_pulse = int(date[1][-1:]) #Last digit off the value we built above, convert to int for easy comparison
        phase = date[1][:-1]

        # We need to ensure that everything is coming in the right order. If we go to add a phase and there is
        # a phase already present in that pulse that's AFTER the one we're working on, then we need to move the
        # new phase to the next pulse.
        correct_pulse = find_correct_pulse(phase, date[0], proposed_pulse, temp_dict)

        # We know which pulse it should go into, but need to check whether there is anything EARLIER...
        # If there is, it's in the wrong pulse and needs to move to the next pulse.
        if correct_pulse_has_date_collision(phase, date[0], temp_dict[correct_pulse]):
            # Copy the current pulse into the next one 
            # TODO: Need to worry about exceeding the valid number of pulses?
            temp_dict[correct_pulse+1] = temp_dict[correct_pulse]       
            #Reset the current pulse to blank
            temp_dict[correct_pulse] = make_empty_summary_row()

        temp_dict[correct_pulse][phase] = dates[phase][proposed_pulse]

    #Create a new dict by selecting any keys where the subkeys have a value
    result = {k: v for k, v in temp_dict.items() if v for k2, v2 in v.items() if v2}
    return result


def format_pm_dates(pm_dates:dict):
    # Convert the timestamp to a string
    formatted_dict = {}
    for pulse in pm_dates:
        pulse_str = f"Pulse {pulse}"
        formatted_dict[pulse_str] = {}

        for phase in pm_dates[pulse]:
            formatted_dict[pulse_str][phase] = {}
            if len(pm_dates[pulse][phase]): #Keys could be empty
                first_date = format_timestamp(pm_dates[pulse][phase][FIRST])
                last_date = format_timestamp(pm_dates[pulse][phase][LAST])
                message = ""

                message += "First: "
                if pm_dates[pulse][phase][BEFORE_FIRST]:
                    message += f"On or before {first_date}"
                else:
                    message += f"{first_date}"

                message += "<br>"
                message += "Last: "
                if pm_dates[pulse][phase][AFTER_LAST]:
                    message += f"On or after {last_date}"
                else:
                    message += f"{last_date}"

                formatted_dict[pulse_str][phase] = message
            else:
                #Empty key, put an appropriate message for display purposes
                formatted_dict[pulse_str][phase] = "No data"

    return formatted_dict

#NOTE Dec 2024: this used to analyze all the data, but since we're adding insects now and don't want 
#       them analyzed, it's changed to only work on bird vocalizations. Changed two things: 
#       1) Below, added "if idx in pm_song_types" to limit analysis to only songs
#       2) In make_empty_summary_row(), added the same if statement
def summarize_pm(pt_pm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # From pt_pm, get the first date that has a song count >= 4
    threshold = 4
    pulse_gap = 14
    
    #Get all the date pairs
    dates = {}    
    for idx, row in pt_pm.iterrows(): 
        if idx in PM_SONG_TYPES:
            dates[idx] = find_pm_dates(row, pulse_gap=pulse_gap, threshold=threshold)

    #Sanity check the data
    summary_dict = clean_pm_dates(dates)
    summary_dict = format_pm_dates(summary_dict)
    
    #Now format this for display. Make a new table where the "1" becomes "Pulse 1"
    result = pd.DataFrame.from_dict(summary_dict, orient='index')

    return result, dates


#
#
# UI and other setup
# 
#  
def get_site_to_analyze(site_list:list, my_sidebar) -> str:
    #Calculate the list of years, sort it backwards so most recent is at the top
    year_list = []
    for s in site_list:
        if s[0:4] not in year_list:
            year_list.append(s[0:4])
    year_list.sort(reverse=True)

    target_year = my_sidebar.selectbox('Site year', year_list)
    filtered_sites = sorted([s for s in site_list if target_year in s])
    return my_sidebar.selectbox('Site to summarize', filtered_sites)



# Helper function to parse various date formats
def parse_date(date_str):
    for fmt in ('%m/%d/%Y', '%m-%d-%Y'):
        try:
            return dt.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")

# Set the default date range to the first and last dates for which we have data. In the case that we're
# automatically generating all the sites, then stop there. Otherwise, show the UI for the date selection
# and if the user wants a specific range then update our range to reflect that.
# Assume that the data cleaning code has removed any extraneous dates, such as if data 
# is mistagged (i.e. data from 2019 shows up in the 2020 site)
def get_date_range(df:pd.DataFrame, graphing_all_sites:bool, standardize_dates:bool, my_sidebar) -> dict:
    date_range_dict = {}
    date_range_dict_from_sheet = {}
    dates_from_sheet = get_site_info(df["site"].iloc[0], ["First Recording", "Last Recording"])
    date_range_dict_from_sheet[START] = dates_from_sheet["First Recording"]
    date_range_dict_from_sheet[END] = dates_from_sheet["Last Recording"]

    if graphing_all_sites and standardize_dates:
        date_range_dict[START] = f"{STANDARD_START}/{date_range_dict_from_sheet[START][-4:]}"
        date_range_dict[END] = f"{STANDARD_END}/{date_range_dict_from_sheet[END][-4:]}"
        return date_range_dict
    
    if df.index.name == "date":
        date_range_dict_from_file = {START : df.index.min().strftime("%m-%d-%Y"), 
                                     END : df.index.max().strftime("%m-%d-%Y")}
    else:
        date_range_dict_from_file = {START : df["date"].min().strftime("%m-%d-%Y"), 
                             END : df["date"].max().strftime("%m-%d-%Y")}

    #Normalize the dates and then confirm that they are the same
    normalized_from_sheet = {k: parse_date(v) for k, v in date_range_dict_from_sheet.items()}
    normalized_from_data = {k: parse_date(v) for k, v in date_range_dict_from_file.items()}

    if normalized_from_sheet != normalized_from_data:
        #it's OK for the very first recording to have a different date, but we should log it
        #if the second one is also different. 
        if df.index[1].date() != normalized_from_sheet["start"]:
            log_error(f"Date for {df["site"].iloc[0]} in sheet and from data are different. From sheet: {normalized_from_sheet}; From data: {normalized_from_data}")

    #For now, use the sheet data
    date_range_dict = date_range_dict_from_sheet

    if not graphing_all_sites:
        months1 = {'First': '-1', 'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}
        months2 = {'Last': '-1',  'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}
        start_month = my_sidebar.selectbox("Start month", months1.keys(), index=0)
        end_month = my_sidebar.selectbox("End month", months2.keys(), index=0)

        #Update the date range if needed
        site_year = int(date_range_dict[START][-4:])
        if start_month != 'First':
            date_range_dict[START] = f'{months1[start_month]}-01-{site_year}'
        if end_month != 'Last':
            last_day = calendar.monthrange(site_year, int(months2[end_month]))[1]
            date_range_dict[END] = f'{months2[end_month]}-{last_day}-{site_year}'

    return date_range_dict


#
#
# Graphing
#
#
if BEING_DEPLOYED_TO_STREAMLIT:
    GRAPH_FONT = "sans serif"
    GRAPH_FONT_TTF = "comic.ttf" #never used in this case
else:
    GRAPH_FONT = "Franklin Gothic Book"
    GRAPH_FONT_TTF = "FRABK.TTF" #used for output to the file using PIL

TITLE_FONT_SIZE = 13
AXIS_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8

# Set up base theme
# See https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
#
# See here for color options: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
def set_global_theme():
    #https://matplotlib.org/stable/tutorials/introductory/customizing.html#matplotlib-rcparams
    line_color = 'black'
    line_width = '1.5'
    custom_params = {'figure.dpi':DPI, 
                     'font.family':GRAPH_FONT,                      
                     'font.size':AXIS_FONT_SIZE,
                     'font.stretch':'normal',
                     'font.weight':'light',
                     'xtick.bottom':'False',
                     'xtick.labelbottom':'False',
                     'ytick.left':'False',
                     'ytick.labelleft':'False',
                     'figure.frameon':'False',
                     'axes.spines.left':'False',
                     'axes.spines.right':'False',
                     'axes.spines.top':'False',
                     'axes.spines.bottom':'False',
                     'axes.edgecolor':line_color,
                     'axes.xmargin':0,
                     'axes.ymargin':0,
                     'lines.color':line_color,
                     'lines.linewidth':line_width,
                     'patch.edgecolor':line_color,
                     'patch.linewidth':line_width,
                     'savefig.facecolor':'white'
                     }
    mpl.rcParams.update(custom_params)


def output_cmap():
    #Save the legend if one doesn't exist; if I update the code, need to delete the file to regenerate it
    figure_path = FIGURE_DIR / LEGEND_NAME
    if not os.path.exists(figure_path):
        # os.remove(figure_path)
        plt.savefig(figure_path, dpi='figure',  bbox_inches='tight', pad_inches=0)   


def draw_legend(cmap: dict, make_all_graphs: bool, save_files: bool):
    # --- Geometry (all in ax.transAxes units) ---
    BAR_X = 0.00
    BAR_Y = 0.10
    BAR_W = 0.4
    BAR_H = 0.80

    LABEL_PAD = 0.05
    LABEL_X = BAR_X + BAR_W + LABEL_PAD
    LABEL_Y = 0.50

    # For swatch-only items (no gradient bar)
    SW_W = 0.18
    SW_H = 0.70
    SW_X = 0.00
    SW_Y = 0.50 - SW_H / 2.0

    # Gradient image used for all gradient legend items
    gradient = np.linspace(0, 1, 32)
    gradient = np.vstack((gradient, gradient))

    def imshow_rect(ax, img, *, x: float, y: float, w: float, h: float, **kwargs):
        ax.imshow(
            img,
            extent=(x, x + w, y, y + h),
            transform=ax.transAxes,
            **kwargs,
        )

    def draw_gradient_item(ax, cmap_name: str, label: str, scale = 1.0):
        if label.startswith("Hatchling"):
                bar_w = BAR_W * scale
        else:
                bar_w = BAR_W   

        imshow_rect(
            ax,
            gradient,
            x=BAR_X, y=BAR_Y, w=bar_w, h=BAR_H,
            aspect="auto",
            cmap=mpl.colormaps[cmap_name],
        )
        label_x = LABEL_X - (BAR_W - bar_w)
        ax.text(
            label_x, LABEL_Y, label,
            transform=ax.transAxes,
            va="center", ha="left",
            fontfamily=GRAPH_FONT,
            fontsize=LEGEND_FONT_SIZE,
        )
        ax.set_axis_off()

    def draw_swatch_item(ax, label: str, *, facecolor: str, 
                         hatch: str | None = None, 
                         hatch_edge_color: str | None = None):
        ax.add_patch(
            Rectangle(
                (SW_X, SW_Y),
                SW_W, SW_H,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor="black" if hatch_edge_color is None else hatch_edge_color,
                linewidth=0.2,
                hatch=hatch,
            )
        )
        ax.text(
            SW_X + SW_W + LABEL_PAD, LABEL_Y, label,
            transform=ax.transAxes,
            va="center", ha="left",
            fontfamily=GRAPH_FONT,
            fontsize=LEGEND_FONT_SIZE,
        )
        ax.set_axis_off()

    # --- Build 6 legend items in one row ---
    # First 4 = your gradient items (order from CMAP_NAMES)
    legend_items = []
    for key, label in CMAP_NAMES.items():
        label_text = label.replace(" ", "\n")
        label_text = label_text.replace("/", "\n")
        legend_items.append(("gradient", key, label_text))

    # Add the 2 new items at the end (or move them earlier if you prefer)
    legend_items.append(("swatch", "No analysis\ndone", None))
    legend_items.append(("hatch", "Missing\nrecordings", None))

    # Relative column widths (tweakable)
    width_ratios = [
        1.0,  # Male Song
        1.0,  # Male Chorus
        1.0,  # Female Chatter
        1.2,  # Hatchling / Nestling / Fledgling (wider)
        0.9,  # No data (narrow)
        0.9,  # Missing days
    ]

    ncols = len(legend_items)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(FIG_W * 0.8, FIG_H * 0.15),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    axs = axs.flatten()

    for i, (ax, item) in enumerate(zip(axs, legend_items)):
        kind = item[0]
        ratio = 1/width_ratios[i]
        
        if kind == "gradient":
            _, key, label = item
            cmap_name = cmap[key] if isinstance(cmap, dict) else cmap
            draw_gradient_item(ax, cmap_name, label, scale=ratio)
        elif kind == "swatch":
            _, label, _ = item
            draw_swatch_item(ax, label, facecolor=NO_DATA_COLOR, hatch=None)

        elif kind == "hatch":
            _, label, _ = item
            # White face so hatch reads clearly; NaN / missing gets the hatch signal
            draw_swatch_item(ax, label, 
                             facecolor=HATCH_BG_COLOR, 
                             hatch=HATCH_PATTERN,
                             hatch_edge_color=HATCH_DARK_COLOR)

        else:
            ax.set_axis_off()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if not make_all_graphs:
        st.pyplot(fig)
        #st_image_figure(fig)

    if save_files:
        output_cmap()

    return



def get_days_per_month(date_list:list) -> dict:
    # Make a list of all the values, but only use the month name. Then, count how many of each 
    # month names there are to get the number of days/month
    months = [pd.to_datetime(date).strftime('%B') for date in date_list]
    return Counter(months)



def draw_axis_labels(fig, month_lengths: dict, skip_month_names=False,
                     y: float = 0, bottom: float = 0, top: float = 0,
                     do_aligned_dates: bool = False):

    def approx_text_width_px(text: str, fontsize_pt: float) -> float:
        # Rule-of-thumb: average Latin glyph is ~0.5–0.6 em.
        # 1 pt = 1/72 inch, but we can stay relative because we compare widths in px.
        # In practice: width ≈ fontsize_px * 0.55 * n_chars
        return fontsize_pt * 1.333 * 0.55 * len(text)  # 1.333 ≈ 96/72 (px per pt at 96 dpi)


    def draw_month_label_if_fits(
        ax,
        text: str,
        x_start: float,
        x_end: float,
        y_axes: float,
        *,
        fontsize=AXIS_FONT_SIZE,
        fontfamily=GRAPH_FONT,
        pad_px=4,
        **text_kwargs,
    ):
        # available width in display pixels
        x0_disp, _ = ax.transData.transform((x_start, 0))
        x1_disp, _ = ax.transData.transform((x_end, 0))
        available_px = abs(x1_disp - x0_disp)

        text_px = approx_text_width_px(text, fontsize)

        if text_px + pad_px <= available_px:
            ax.text(
                (x_start + x_end) / 2,
                y_axes,
                text,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=fontsize,
                fontfamily=fontfamily,
                **text_kwargs,
            )

    ax = fig.get_axes()[-1]
    x_min, x_max = ax.get_xlim()

    x_fig = 0.0
    x_days = 0.0
    day_width = 1.0 / (x_max - x_min)

    months = list(month_lengths.items())
    total = len(months)

    for i, (month, n_days) in enumerate(months):
        if not skip_month_names and not do_aligned_dates:
            draw_month_label_if_fits(
                ax,
                month,
                x_days,
                x_days + n_days,
                y,
            )

        x_days += n_days
        x_fig += n_days * day_width
        
        #Don't draw the border for the last month
        if i < total - 1:
            line = mlines.Line2D(
                [x_fig, x_fig],
                [bottom + (BORDER_WIDTH / 200), top - (BORDER_WIDTH / 200)],
                transform=fig.transFigure,
                color="black",
                linewidth=BORDER_WIDTH,
                alpha=1,
            )
            fig.add_artist(line)


# For ensuring the title in the graph looks the same between weather and data graphs.
def plot_title(fig:Figure, title:str, x:float=0.0,y:float=1.0):
    fig.suptitle(' ' + title, x=x, y=y,
                 fontsize=TITLE_FONT_SIZE, fontfamily=GRAPH_FONT,
                 horizontalalignment="left", verticalalignment="top")


def draw_missing_day_boxes(
    fig,
    ax,
    missing_days: pd.DatetimeIndex,
    bottom: float,
    top: float,
    start_day: pd.Timestamp,   # first day of the heatmap
    last_day: pd.Timestamp,    # last day of the heatmap
    color="white",
    alpha=0.5,
):
    """
    Draw boxes for missing days on an index-based (heatmap) x-axis.
    """

    fig.canvas.draw()

    for day in missing_days:
        if day >= start_day and day <= last_day:
            # Map date -> column index
            i = (day - start_day).days
            x_start = i
            x_end = i + 1

            # data -> display
            x0_disp, _ = ax.transData.transform((x_start, 0))
            x1_disp, _ = ax.transData.transform((x_end, 0))

            # display -> figure
            x0_fig, _ = fig.transFigure.inverted().transform((x0_disp, 0))
            x1_fig, _ = fig.transFigure.inverted().transform((x1_disp, 0))

            # Safety guard (optional but wise)
            if not (-1 <= x0_fig <= 2):
                assert False
                continue

            rect = Rectangle(
                (x0_fig, bottom),
                x1_fig - x0_fig,
                top - bottom,
                transform=fig.transFigure,
                facecolor=HATCH_BG_COLOR,
                edgecolor=HATCH_DARK_COLOR,
                alpha=alpha,
                zorder=10,
                hatch=HATCH_PATTERN,
            )

            fig.add_artist(rect)

def overlay_missing_days_hatch(
    axs,
    missing_days,
    start_day,
    last_day,
    *,
    hatch=HATCH_PATTERN,
    color=HATCH_DARK_COLOR,
    facecolor=HATCH_BG_COLOR,
    zorder=10,
):
    for ax in axs:
        for d in missing_days:
            if start_day <= d <= last_day:
                i = (d - start_day).days
                ax.axvspan(
                    i,
                    i + 1,
                    ymin=0,
                    ymax=1,
                    facecolor=HATCH_BG_COLOR,
                    edgecolor=HATCH_DARK_COLOR,
                    hatch=HATCH_PATTERN,
                    linewidth=0,
                    zorder=10,
                    
                )


def file_missing(site, graph_type, type):
    if graph_type == GRAPH_PM:
        site_dir = PMJ_DATA_DIR / site
        if os.path.isdir(site_dir):
            fname = f"{site} {type}.csv"
            full_file_name = site_dir / fname
            if is_non_zero_file(full_file_name):
                return False
    
    return True


@st.cache_data(show_spinner=False)
def load_recordings_hourly(parquet_path: Path, site_col: str, date_col: str, hour_col: str, recordings_col: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=[site_col, date_col, hour_col, recordings_col])
    if not pd.api.types.is_string_dtype(df[site_col]):
        df[site_col] = df[site_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col], errors="raise").dt.normalize()
    df[hour_col] = pd.to_numeric(df[hour_col], errors="coerce")
    return df


def draw_event_date_marker(ax, x, add_arrow=False, date_type=PULSE_HATCH):
    # Cell center
    cx = x + 0.45
    cy = 0.45

    date_markers = {
        PULSE_MC_START : "M",
        PULSE_INC_START : "I",
        PULSE_HATCH : "B",
        PULSE_FIRST_FLDG : "F",
        PULSE_LAST_FLDG : "D"
    }

    # Draw "H" centered in the circle
    txt = ax.text(
        cx, cy,
        date_markers[date_type],
        ha="center", va="center",
        fontsize=8,
        color="black",
        zorder=16,
    )
    txt.set_in_layout(False)

    # cell center in data coords
    cx = x + 0.45
    cy = 0.41   

    # Draw circular outline marker (no fill)
    circ = ax.scatter(
        [cx], [cy],
        s=60, # marker area in points^2; tune this
        facecolors="white",
        edgecolors="black",
        linewidths=0.25,
        zorder=15,
    )
    circ.set_in_layout(False)

    if add_arrow:
        # Arrow parameters
        # arrow_end_x   = x          # how far arrow points
        # arrow_start_x = arrow_end_x + 1.5        # right edge of rectangle
        arrow_y       = 0.15              # vertical center of the rect

        arrow = FancyArrowPatch(
            (0.02, arrow_y),     # start inside the axes
            (-0.003, arrow_y),   # end exactly at the left border
            transform=ax.transAxes,
            # (arrow_start_x, arrow_y),
            # (arrow_end_x, arrow_y),
            # transform=ax.transData,
            arrowstyle="->",
            linewidth=0.5,
            color="black",
            mutation_scale=8,            # arrowhead size
            antialiased=True,
            zorder=18,
            clip_on=False,
        )
        arrow.set_in_layout(False)
        ax.add_patch(arrow)

    return

def calc_x_from_date(df, event_date) -> float:
    loc = df.columns.get_loc(event_date)
    if not isinstance(loc, int):
        raise ValueError(f"Expected unique column for {event_date}, got {type(loc)}")
    x = float(loc)
    return x


def add_event_date_marker(ax, df, date_type, event_date): 
    add_arrow = False
    if event_date == convert_to_datetime("6/1/1967"):
        #This is the new special case of a hatch date prior to graph start 
        x = 0
        add_arrow = True
    elif event_date >= df.columns[0] and event_date <= df.columns[-1]:
        x = calc_x_from_date(df, event_date)
    else:
        log_error(f"create_graph: {date_type} {event_date} is outside range of this year, which is {df.columns[0]} through {df.columns[-1]}")
        return
    draw_event_date_marker(ax, x, add_arrow=add_arrow, date_type=date_type)



# Create a graph, given a dataframe, list of row names, color map, and friendly names for the rows
def create_graph(site: str,
                 df: pd.DataFrame, 
                 row_names : list, 
                 cmap : dict,  
                 raw_data = pd.DataFrame(), 
                 draw_vert_rects:bool = False, draw_horiz_rects:bool = False,
                 title="", 
                 graph_type="",
                 draw_connectors:bool = False,
                 key_dates={},
                 missing_days=pd.DatetimeIndex([]),
                 denom_by_day: pd.Series = pd.Series(),
                 do_aligned_dates:bool = False,
) -> tuple[Figure, Axes]:
    plt.close() #close any prior graph that was open

    if len(df) == 0:
        #return an empty plot if nothing to graph
        fig, axs = plt.subplots(nrows=1, ncols=1)
        return fig, axs

    if graph_type == GRAPH_EDGE:
        row_count = 1 #All data should be drawn on the same axis for edge
    else:
        row_count = len(row_names)
    graph_drawn = []
    
    # --- inches-based spec ---
    if not do_aligned_dates:
        top_pad_in = 0.0       # Whitespace at the top
        title_band_in = 0.2    # Gap for the label
        top_band_in = top_pad_in + title_band_in

        # Height in inches of the actual graph, allot 0.2" per row
        row_height = 0.2

        label_height_in = 0.25  # Axis labels
        legend_height_in = 0.0  
        bottom_pad_in = 0.0     # Whitespace at the bottom
        bottom_band_in = label_height_in + legend_height_in + bottom_pad_in
        fig_w = FIG_W           # keep your width in inches
    else:
        top_pad_in = 0
        top_band_in = 0     #nothing goes at the top
        row_height = 0.05    #half height or so
        bottom_band_in = 0  #no labels or legend
        fig_w = 5.5            #inches

    plot_in = row_count * row_height
    fig_h = top_band_in + plot_in + bottom_band_in

    fig, axs = plt.subplots(
        nrows=row_count,
        ncols=1,
        sharex=True,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "height_ratios": np.ones(row_count),
            "hspace": 0.0,
        },
        squeeze=False,   #forces axs to be 2D array even if 1 row
    )
    axs = axs.flatten() #normalize axs to 1D 

    

    def disable_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])

        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(NullLocator())
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(NullFormatter())
            axis.set_minor_formatter(NullFormatter())

    for ax in axs:
        disable_ticks(ax)

    #This is the left gutter to allow the decorations to hang out
    fig_width_in = fig.get_size_inches()[0]
    left_margin = GRAPH_LEFT_PADDING / fig_width_in
    fig.subplots_adjust(left=left_margin)

    # Convert inches -> figure fractions for subplot rectangle
    bottom = bottom_band_in / fig_h
    top = 1.0 - (top_band_in / fig_h)
    fig.subplots_adjust(left=0, right=1, bottom=bottom, top=top)
    
    # If we have one, add the title for the graph and set appropriate formatting
    if len(title) and not do_aligned_dates:
        title_y = 1.0 - (top_pad_in / fig_h)
        plot_title(fig, title, y=title_y)
    
    # Ensure that we have a row for each index. If a row is missing, add it with NaN values
    for row in row_names:
        if row not in df.index:
            df.loc[row]=pd.Series(data=np.nan,index=df.columns)

    df_clean = df.copy()

    i=0
    for row in row_names:
        # plotting the heatmap
        # pull out the one row we want and transpose it to be wide
        if graph_type == GRAPH_EDGE:
            df_sel = df.loc[row_names].replace(-100, np.nan)   # keep all rows in df, select subset here
            df_to_graph = df_sel.bfill(axis=0).iloc[[0]]          # 1-row DataFrame
        else:
            df_to_graph = df_clean.loc[[row]].copy()

        #Adjust color maps to force the lowest value to white and gray for NaN data, except use white for NaN in PM graphs
        cmap_final = sns.color_palette(cmap[row] if len(cmap)>1 else cmap[0], as_cmap=True)
        cmap_final.set_under("white")
        no_data_color = "white" if graph_type==GRAPH_PM else NO_DATA_COLOR
        cmap_final.set_bad(no_data_color)   # light gray representing the days where analysis was not done 

        #Normalize all the data by the number of recordings that were used per day
        if graph_type == GRAPH_MINIMAN:
            df_norm = df_to_graph / 3  #3 recordings per day
        elif graph_type == GRAPH_MANUAL:
            df_norm = df_to_graph / 13  # 13 recordings per day
        elif graph_type == GRAPH_EDGE:
            df_norm = df_to_graph / 8  # 8 recordings per day 
        else:
            # Align denom to the same columns (dates)
            denom = denom_by_day.reindex(df_to_graph.columns)
            
            # Assume that columns are normalized Timestamps to match denom_by_day index, and are numeric 
            # otherwise we need these two lines
            # df_to_graph.columns = pd.to_datetime(df_to_graph.columns, errors="raise").normalize()
            # df_to_graph.iloc[0] = pd.to_numeric(df_to_graph.iloc[0], errors="coerce")
            df_norm = df_to_graph.div(denom, axis="columns")  # normalize by count of recordings per day

        # Adjust colors so that lighter values are more visible
        # gamma < 1 brightens lows, closer to 0 is more extreme
        gamma = float(st.session_state.get("gamma", 1.0))  # 1=default if slider not created yet
        vmin = 0.001 #slightly above 0 so that 0 values get the 'under' color 
        vmax = 1 #as we're normalizing the data, the ranges will all be 0-1
        norm = colors.PowerNorm(gamma = gamma, vmin=vmin, vmax=vmax)  

        arr = np.ma.masked_invalid(df_norm.to_numpy(dtype=float))
        n_rows, n_cols = arr.shape

        ax = axs[i]
        ax.imshow(
            arr,
            cmap=cmap_final,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(0, n_cols, 1, 0),  # forces the axis settings to be like the old seaborn one
            zorder=1, 
        )

        # If we drew an empty graph, write text on top to indicate that it is supposed to be empty
        # and not that it's just hard to read!
        if df_clean.loc[row].sum() == 0:
            #The conundrum: at least for edge, it's possible that a row we drew is blank, but the actual
            #row is going to get some boxes and lines. In this case, there will be -100s in the data, 
            #and if we find those, we should NOT draw the text that says there is no data 
            #THIS IS NOT WORKING?
            if graph_type == GRAPH_EDGE and df.loc[row].lt(0).any():
                pass
            else:
                pass
                # if file_missing(site, graph_type, row):
                #     label = PM_OTHER_TYPES[row] if row in PM_OTHER_TYPES.keys() else row 
                #     display_label = tag_name_map[label] if label in tag_name_map.keys() else label
                #     axs[i].text(0.5,0.5,f"No data for {display_label}", 
                #                 font = GRAPH_FONT, fontsize=8, fontstyle='italic', 
                #                 color='gray', verticalalignment='center')
        elif graph_type == GRAPH_PM and row in PM_OTHER_TYPES.keys():
            ax.text(0.5,0.5,f"{PM_OTHER_TYPES[row]}", 
                        font = GRAPH_FONT, fontsize=8, fontstyle='italic', 
                        color='black', verticalalignment='center')
        elif graph_type == GRAPH_EDGE:
            pass
            # ax.text(0.5,0.5,f"{row}", 
            #             font = GRAPH_FONT, fontsize=8, fontstyle='italic', 
            #             color='black', verticalalignment='center')

        # Track which graphs we drew, so we can put the proper ticks on later
        graph_drawn.append(i)
        if graph_type == GRAPH_PM and not do_aligned_dates:
            #Add the event markers if available
            for pulse in key_dates:
                for date_type, event_date in key_dates[pulse].items():
                    if row == "Male Chorus" and date_type == PULSE_MC_START or\
                       row == "Female" and date_type == PULSE_INC_START or\
                       row == "Hatchling" and date_type == PULSE_HATCH or\
                       row == "Fledgling" and (date_type == PULSE_FIRST_FLDG or date_type == PULSE_LAST_FLDG):
                        add_event_date_marker(ax, df, date_type, event_date)

                        
            #NOTE Dec 2024: Added extra lines to separate insects
            if row == PM_INSECT_SP30 or row == PM_FROG_PACTF:
                #Want to add a line above these two rows to separate them
                # Get the top y-limit
                top_y = ax.get_ylim()[1]
                xmin = ax.get_xlim()[0]
                xmax = ax.get_xlim()[1]

                # Draw a horizontal line at the top of the axis
                line = ax.hlines(y=top_y, xmin=xmin, xmax=xmax, colors='red',  linewidth=0.5)
                dashes = (0, (18, 2)) # 10 points on, 5 points off
                line.set_dashes(dashes)  # Apply the custom dash pattern


        # For edge: Add a rectangle around the regions of consective tags, and a line between 
        # non-consectutive if it's a N tag.
        if draw_horiz_rects and row in df_clean.index:
            df_col_nonzero = df.loc[[row]].T #pull out the row we want, it turns into a column as above
            df_col_nonzero = df_col_nonzero.reset_index()   #index by ints for easy graphing
            df_col_nonzero = df_col_nonzero[df_col_nonzero[row] != 0]

            if len(df_col_nonzero):
                #Scale the color maps so we get the same color but a little lighter
                c = mpl.colormaps[(cmap[row] if len(cmap) > 1 else cmap[0])](0.5)
                #n tags get boxes around each consecutive block
                idx = df_col_nonzero[row].dropna().index.to_numpy()
                if len(idx) == 0:
                    borders: list[tuple[int, int]] = []
                elif len(idx) == 1:
                    start_and_end = int(idx[0])
                    borders = [(start_and_end,start_and_end)]
                else:
                    # Find boundaries where contiguity breaks
                    breaks = np.where(np.diff(idx) > 1)[0]

                    starts = np.r_[idx[0], idx[breaks + 1]]
                    ends   = np.r_[idx[breaks], idx[-1]]
                    borders = [(int(a), int(b)) for a, b in zip(starts, ends)]
                # We now have a list of pairs of coordinates where we need a rect. For each pair, draw one.
                for start, end in borders:
                    left = start
                    width = (end-start) + 1
                    ax.add_patch(Rectangle((left,EDGE_GRAPH_BORDER_INSET), width, 1-2*EDGE_GRAPH_BORDER_INSET, 
                                           ec=c, fc=c, 
                                           lw=EDGE_GRAPH_BORDER_WIDTH,
                                           alpha=1, 
                                           fill=False))
                # For each pair of rects, draw a line between them.
                gaps = [(end1 + 1, start2 - 1) for (start1, end1), (start2, end2) in zip(borders, borders[1:])]
                for start_gap, end_gap in gaps:
                    left = start_gap
                    right = end_gap+1
                    line_distance = right - left
                    line_height = EDGE_GRAPH_BORDER_WIDTH/10
                    y_start_pos = 0.5
                    ax.add_patch(Rectangle((left, y_start_pos), line_distance, line_height, 
                                           ec=c, fc=c, 
                                           lw=0, 
                                           alpha=1,
                                           fill=True)) 

        #For edge, all data is drawn on the same axis so don't increment the counter here and just get out of the loop
        if graph_type == GRAPH_EDGE:
            break
        else:
            i += 1
    

    # For mini-manual: Add a rect around each day that has some data
    if graph_type == GRAPH_MINIMAN and len(raw_data)>0 and False:
        if draw_vert_rects :
            tagged_rows = filter_df_by_tags(raw_data, MINI_MANUAL_COLS)
            if len(tagged_rows):
                date_list = tagged_rows.index.unique()
                #I'm using df.columns[0] because it represents the date of the first day in the date range.
                #This accounts for the scenario where the user changed the Start Month.
                first = df.columns[0]
                box_pos = [(i - first)/pd.Timedelta(days=1) for i in date_list]

                _,top = fig.transFigure.inverted().transform(axs[0].transAxes.transform([0,1]))
                _,bottom = fig.transFigure.inverted().transform(axs[row_count-1].transAxes.transform([0,0]))
                trans = transforms.blended_transform_factory(axs[0].transData, fig.transFigure)
                for px in box_pos:
                    rect = Rectangle(xy=(px,bottom), width=1, height=top-bottom, transform=trans,
                                            fc='none', ec='C0', lw=0.5)
                    fig.add_artist(rect)
       
    #Draw a black box over every missing date
    start_day = pd.Timestamp(df.columns.min()).normalize()
    last_day = pd.Timestamp(df.columns.max()).normalize()
    overlay_missing_days_hatch(
        axs,
        missing_days,
        start_day=start_day,
        last_day=last_day,
    )

    # Add the vertical lines and month names
    text_offset_in = LABEL_OFFSET
    text_y = bottom - (text_offset_in / fig_h)
    text_y = -0.65
    draw_axis_labels(fig, get_days_per_month(df.columns.tolist()), y=text_y, bottom=bottom, top=top, do_aligned_dates=do_aligned_dates)

    # Draw a bounding rectangle around everything except the caption
    #b = axes_union_bbox(fig, "i") #Get the dimensions of the plot area
    b = Bbox.union([ax.get_position() for ax in fig.axes])

    border = Rectangle(
        (b.x0, b.y0),  # (x0, y0) in figure coords
        b.width,           # width (full figure width)
        b.height,   # height = plot area only
        transform=fig.transFigure,
        linewidth = BORDER_WIDTH, edgecolor="black",
        fill=False, 
        zorder=8,)    
    fig.add_artist(border)


    #if we want to add anything on top of the images, the time to do it is at the end
    #add_watermark(title)

    # return the final plotted heatmap
    return fig, axs

def axes_union_bbox(fig):
    # union of all axes positions in figure-fraction coords
    #b = Bbox.union([ax.get_position() for ax in fig.axes])

    fig_w, fig_h = fig.get_size_inches()
    b = Bbox.union([ax.get_position() for ax in fig.axes])  # figure fraction

    # axes bbox -> inches
    x0_in = b.x0 * fig_w
    y0_in = b.y0 * fig_h
    w_in  = b.width * fig_w
    h_in  = b.height * fig_h

    left_pad_in   = GRAPH_LEFT_PADDING
    right_pad_in  = 0.00
    top_pad_in    = 0.00
    bottom_pad_in = 0.00

    bbox_in = Bbox.from_bounds(
        x0_in - left_pad_in,
        y0_in - bottom_pad_in,
        w_in + left_pad_in + right_pad_in,
        h_in + top_pad_in + bottom_pad_in,
    )
    return bbox_in

def add_pulse_overlays(graph, summarized_data:dict, date_range:dict):
    # For each of the derived summary dates, draw a line on the graph
    # Top row of graph is Male Song, nothing goes there

    # For each of the other rows, we want to draw a bar to the left of the start date, a line to the end date, and then a line just 
    # after the end dates

    graph_start_date = pd.to_datetime(date_range["start"])
    for idx, phase_type in enumerate(PM_FILE_TYPES): 
        if phase_type in summarized_data:
            for pulse in summarized_data[phase_type]:
                pulse_dates = summarized_data[phase_type][pulse]
                assert FIRST in pulse_dates
                assert LAST in pulse_dates

                overlay_start = (pulse_dates[FIRST] - graph_start_date).days
                overlay_end = (pulse_dates[LAST] - graph_start_date).days + 1 
                target_ax = graph.axes[idx]
                # graph.axes[idx].axvspan(
                #     xmin=overlay_start,
                #     xmax=overlay_end,
                #     color="yellow",
                #     alpha=0.3  # Transparency
                # )

                # Get y-axis limits
                ymin, ymax = target_ax.get_ylim()

                # Create a rectangle spanning the range
                rect = Rectangle(
                    (overlay_start, ymin),           # Bottom-left corner (x, y)
                    overlay_end - overlay_start,     # Width (difference in dates)
                    ymax - ymin,                     # Height
                    edgecolor="red",                 # Outline color
                    facecolor="none",                # Transparent fill
                    linewidth=2                      # Outline width
                )

                # Add the rectangle to the axis
                target_ax.add_patch(rect)                            
        else:
            # do anything for missing rows?
            pass

    return



#Helper to ensure we make the filename consistently because this is done from multiple places
def make_img_filename(site:str, graph_type:str, extra="") ->str:
    filename = f"{site}_{graph_type}_{extra}.png"
    return filename

#Helper for when we need to remove a file
def remove_file(full_path:Path) -> bool:
    result = False
    try:
        os.remove(full_path)
        result = True
    except FileNotFoundError:
        result = True
    except OSError as e:
        print(f"Error {e} trying to remove file {full_path}")
        result = False
    return result
    
# Save the graphic to a different folder. All file-related options are managed from here.
def save_figure(site:str, graph_type:str, graph:Figure, delete_only=False, do_aligned_dates=False):
    #Do nothing if we're on the server, we can't save files there or download them without a lot of complexity
    if BEING_DEPLOYED_TO_STREAMLIT:
        return

    aligned_str = "aligned_" if do_aligned_dates else ""
    filename = make_img_filename(site, graph_type, extra=aligned_str)
    figure_path = FIGURE_DIR / filename
    # We aren't saving the "unclean" one any more, so technically this isn't necessary but doesn't hurt
    remove_file(figure_path)

    extra = aligned_str if do_aligned_dates else "Clean"
    cleaned_image_filename = make_img_filename(site, graph_type, extra=extra)    
    cleaned_figure_path = FIGURE_DIR / cleaned_image_filename
    remove_file(cleaned_figure_path)
    if not delete_only:
        #If we're making the graph where everything is aligned, we don't want the dates
        if do_aligned_dates:
            MONTH_NAMES = {
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            }
            for text in graph.texts[:]:
                if text.get_text() in MONTH_NAMES:
                    text.remove()
            bbox_inches = None

            # # We no longer have labels, so need to move up the legend if appropriate
            # if graph_type == GRAPH_WEATHER:
            #     ax = graph.get_axes()[0]

            #     # Get the legend and it's coordinates 
            #     legend = ax.get_legend()
            #     bb = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())

            #     # Shift the coords
            #     yOffset = 0.1
            #     new_bb = Bbox.from_bounds(
            #         bb.x0,
            #         bb.y0 + yOffset,
            #         bb.width,
            #         bb.height,
            #     )
            #     legend.set_bbox_to_anchor(new_bb, transform = ax.transAxes)

        #Now, need to trim off the bottom of the image that we don't need any more
        fig_w, fig_h = graph.get_size_inches()
        fig_w += 1/DPI  #Round up to prevent clipping on the right
        # Crop from the bottom
        if do_aligned_dates:
            trim_amount_in = -(1/DPI)
        else:
            trim_amount_in = 0.2 if graph_type==GRAPH_WEATHER else 0.1
        bbox_inches = Bbox.from_bounds(
            0,              # x0 (left), -0.25 preserves the margin
            trim_amount_in,     # y0 (bottom trim in inches)
            fig_w,              # width
            fig_h - trim_amount_in,  # height
            )
        plt.savefig(cleaned_figure_path, dpi='figure', bbox_inches=bbox_inches)   
        #save_with_reserved_margin(graph, cleaned_figure_path, dpi="figure", bottom_pad_in=trim_amount_in)

    else:
        #TODO If there is no data, what to do? The line below saves an empty image.
        #Image.new(mode="RGB", size=(1, 1)).save(figure_path)
        pass

    plt.close()


def save_with_reserved_margin(fig, path, left_pad_in=GRAPH_LEFT_PADDING, right_pad_in=0.0, top_pad_in=0.0, bottom_pad_in=0.0, dpi="figure"):
    fig_w, fig_h = fig.get_size_inches()

    b = axes_union_bbox(fig)  # figure fraction

    # Convert axes bbox (fraction) -> inches
    x0_in = b.x0 * fig_w
    y0_in = b.y0 * fig_h
    w_in  = b.width * fig_w
    h_in  = b.height * fig_h

    # Expand by reserved margins (inches)
    bbox_in = Bbox.from_bounds(
        x0_in - left_pad_in,
        y0_in - bottom_pad_in,
        w_in + left_pad_in + right_pad_in,
        h_in + top_pad_in + bottom_pad_in,
    )

    fig.savefig(path, dpi=dpi, bbox_inches=bbox_in)

def concat_images(*images: Image.Image, is_legend:bool = False) -> Image.Image:
    """Generate composite of all supplied images."""
    # Get the widest width. This will be a graph, not the legend
    width = max(image.width for image in images)
    # Add a little padding, so the border has space
    x_padding = 0
    width += x_padding

    # Add up all the heights.
    height = sum(image.height for image in images)

    #put some space between each graph
    y_padding = 25
    height += y_padding * len(images)

    composite = Image.new('RGB', (width, height), color='white')
    
    # Paste each image below the one before it.
    y = 0 + y_padding

    # Paste each image centered in the graphic
    for image in images:
        x = int((width - image.width)/2)
        composite.paste(image, (x, y))
        y += image.height + y_padding

    return composite


def apply_decorations_to_composite(site:str, composite:Image.Image, month_locs:dict) -> Image.Image:
    scale = DPI/300

    #Make a new image that's a little bigger so we can add the site name at the top
    width, height = composite.size
    title_height = 100 * scale
    month_row_height = 0
    border_width = 0
    border_height = border_width * 2  * scale
    margin_bottom = 20 * scale
    margin_left = 0 + 0 * scale
    margin_right = width - 0 * scale

    months_at_top = False
    if months_at_top:
        month_row_height = 80 * scale

    new_height = int(height + title_height + month_row_height + border_height + margin_bottom)

    title_font_size = 72 * scale
    month_font_size = 36 * scale
    fudge = 10 * scale #for descenders
    
    final = Image.new(composite.mode, (width, new_height), color='white')

    #Get the font path
    font_path = os.path.join(os.environ['WINDIR'], 'Fonts', GRAPH_FONT_TTF) 

    #Add the title
    draw = ImageDraw.Draw(final)
    font = ImageFont.truetype(font_path, size=title_font_size)
    graph_title = get_pretty_name_for_site(site)
    draw.text((width/2,title_height-fudge), graph_title, fill='black', anchor='ms', font=font)

    #Add the months
    if months_at_top:
        font = ImageFont.truetype(font_path, size=month_font_size)
        v_pos = title_height + month_row_height - fudge
        month_row_width = margin_right - margin_left
        
        total_days = sum(end - start + 1 for start, end in month_locs.values())
        day_width = month_row_width / total_days
        h_pos = margin_left
        for month in month_locs:
            days_in_month = month_locs[month][1] - month_locs[month][0] + 1
            m_center = days_in_month / 2
            text_pos = h_pos + (m_center * day_width)
            draw.text((text_pos, v_pos), month, fill='black', font=font, anchor='ms')
            h_pos += days_in_month * day_width

    #Paste in the composite
    max_height = int(title_height + month_row_height + border_width)
    final.paste(composite, box=(0,max_height)) 

    #Add the border
    border_top = title_height + month_row_height
    border_left = 0
    border_right = margin_right - border_width*2
    draw.rectangle([(border_left,border_top),(border_right,new_height-margin_bottom)], 
                    outline='black', width=int(border_width))

    return final

#Code from ChatGPT to draw the labels without clipping
def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """
    Word-wrap `text` into lines that fit within `max_width` pixels.
    Preserves existing newlines as paragraph breaks.
    """
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        words = para.split()
        if not words:
            lines.append("")  # blank line
            continue

        cur = words[0]
        for w in words[1:]:
            trial = f"{cur} {w}"
            if draw.textlength(trial, font=font) <= max_width:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def draw_text_box(
    img: Image.Image,
    text: str,
    box: tuple[int, int, int, int],   # (x0, y0, x1, y1)
    font: ImageFont.FreeTypeFont,
    fill=(0, 0, 0),
    align="left",
    line_spacing_px: int = 2,
    padding_px: int = 8,
    ellipsize: bool = True,
) -> bool:
    """
    Draw wrapped text within box. Returns True if all text fit, False if truncated.
    """
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    x0 += padding_px
    y0 += padding_px
    x1 -= padding_px
    y1 -= padding_px

    max_w = max(1, x1 - x0)
    max_h = max(1, y1 - y0)

    lines = wrap_text(draw, text, font, max_w)

    # Compute line height (font metrics are more reliable than guessing)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + line_spacing_px

    y = y0
    fit_all = True
    i = 100 #max
    for i, line in enumerate(lines):
        if y + line_h > y0 + max_h:
            fit_all = False
            break

        if align == "center":
            w = draw.textlength(line, font=font)
            x = x0 + (max_w - w) / 2
        elif align == "right":
            w = draw.textlength(line, font=font)
            x = x1 - w
        else:
            x = x0

        draw.text((x, y), line, font=font, fill=fill)
        y += line_h

    # If it didn't fit, optionally add ellipsis to the last visible line
    if not fit_all and ellipsize:
        # Determine where the last visible line was drawn
        last_y = y - line_h
        if last_y >= y0:
            # Clear and redraw the last line with ellipsis (simple overwrite approach)
            # (If you need true "clear", draw a filled rectangle behind the text area.)
            # Build truncated version of the *next* line or current line
            remaining_line = lines[i] if i < len(lines) else ""
            base = remaining_line
            ell = "…"

            # Find max text that fits with ellipsis
            while base and draw.textlength(base + ell, font=font) > max_w:
                base = base[:-1].rstrip()
            if not base:
                base = ell
            else:
                base = base + ell

            # Redraw ellipsized line at last_y
            if align == "center":
                w = draw.textlength(base, font=font)
                x = x0 + (max_w - w) / 2
            elif align == "right":
                w = draw.textlength(base, font=font)
                x = x1 - w
            else:
                x = x0

            draw.text((x, last_y), base, font=font, fill=fill)

    return fit_all


# Make the composite with the site name on the left and the graphic on the right
def concat_aligned_images(image_dict:dict, data_dict:dict):
    """Generate composite of all supplied images."""
    # Get the widest width. This will be a graph, not the legend
    img_width = max(image.width for image in image_dict.values())
    label_width = 250
    width = img_width + label_width

    # Add up all the heights.
    height = sum(image.height for image in image_dict.values())

    #put some space between each graph
    y_padding = 0
    height += y_padding * len(image_dict)

    composite = Image.new('RGB', (width, height), color='white')
    
    # Paste each image below the one before it.
    #Get the font path
    scale = DPI/300
    font_path = os.path.join(os.environ['WINDIR'], 'Fonts', GRAPH_FONT_TTF) 
    label_font_size = 24 * scale
    font = ImageFont.truetype(font_path, size=label_font_size)

    # Put the label on the left, and the image to the right
    y = 0 + y_padding
    graph_start = dt.strptime(f"2024/{STANDARD_START}", "%Y/%m/%d")
    graph_end   = dt.strptime(f"2024/{STANDARD_END}", "%Y/%m/%d")
    width_in_days = (graph_end - graph_start).days + 1
    pixels_per_day = img_width / width_in_days

    for label, image in image_dict.items():
        x = 0  # left edge of text
        text_box = (0, y, label_width, y+image.height)
        draw_text_box(composite, label, text_box, font, fill="black", align="left")

        img_x = x + label_width
        composite.paste(image, (img_x, y))

        #Add decorations for Abandoned dates
        for p in data_dict[label]:
            val = data_dict[label][p]
            if val == "ND":
                continue
            partial = False
            if val[-1:].lower() == "p":
                partial = True
                val = val[:-1]

            draw = ImageDraw.Draw(composite)
            abnd_date = dt.strptime(f"{val}", "%m/%d/%Y")
            year = dt.strptime(val, "%m/%d/%Y").year
            graph_start = dt.strptime(f"{year}/{STANDARD_START}", "%Y/%m/%d")

            # Pixel edges: length = width_in_days + 1
            # Using floor is usually the closest match to how raster bins “own” pixels
            edges = [int(math.floor(i * image.width / width_in_days)) for i in range(width_in_days + 1)]

            # For a given abandoned date:
            day_i = (abnd_date - graph_start).days  # 0-based

            # The constants below are extra rounding to fit into the same size day as the graph drew
            left  = img_x + edges[day_i] + 3
            right = img_x + edges[day_i + 1] - 3  

            top    = y + 1
            bottom = y + image.height - 2
            abnd_rect_color = "darkblue" if partial else "crimson" 
            draw.rectangle([(left, top), (right, bottom)], 
                           outline = abnd_rect_color, 
                           fill = abnd_rect_color,
                           width=0)
        y += image.height + y_padding

    return composite


def sort_by_latitude(files:list) -> list:
    ''' For each file, get the site name (the first characters up to the -) and then use that to get the latitude
        Put it all into a dict with latitude as keys
        Sort the dict by keys
        Return the dict values (the filenames) as a list
    '''
    sorted_files = {}
    if files:
        file_list = {}
        for filename in files:
            site = Path(filename).name.split("_",1)[0] 
            lat = get_site_info(site, ["Latitude"])
            if not "Latitude" in lat.keys():
                pass
            
            if pd.notna(pd.to_numeric(lat["Latitude"], errors="coerce")):
                file_list[float(lat["Latitude"])] = filename

        sorted_files = dict(sorted(file_list.items(), reverse=True))

    return list(sorted_files.values())

def combine_aligned_images():
    years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]

    for year in years:
        final_path = FIGURE_DIR / f"{year} Aligned.jpg"
        remove_file(final_path)    

        #Get all the matching images for this year by the string in the filename
        pattern = f"{year}*aligned*"
        matching_files = glob.glob(os.path.join(FIGURE_DIR, pattern))

        #Sort by the latitude of the site
        sorted_files = sort_by_latitude(matching_files)

        #Make a dict of the {pretty_name:file}
        images = {}
        data_dict = {}
        for filename in sorted_files:
            site = Path(filename).name.split("_",1)[0]
            pretty_name = get_pretty_name_for_site(site)
            with Image.open(filename) as im:
                images[pretty_name] = im.copy()
            data_dict[pretty_name] = get_site_info(site, ["p1abandon","p2abandon","p3abandon","p4abandon",])
        #Concat it all together
        final = concat_aligned_images(images, data_dict)
        final.save(final_path)

    return




# Load all the images that match the site name, combine them into a single composite,
# and then save that out
def combine_images(site:str, month_locs:dict, include_weather:bool):
    #if there are no months, then we didn't have any data to graph so don't make a composite
    if len(month_locs) == 0:
        return
    
    composite_filename = make_img_filename(site, "Composite")
    composite_path = FIGURE_DIR / composite_filename
    remove_file(composite_path)

    pattern = f"{site}_*Clean.png"
    matching_files = glob.glob(os.path.join(FIGURE_DIR, pattern))

    #Drop files we don't want
    if not SHOW_MANUAL_ANALYSIS:
        matching_files = [f for f in matching_files if not f == GRAPH_MANUAL]

    #clean_site_files = [file for file in matching_files if "clean" in file]  #Can use this if we need to do additional filtering
    site_fig_dict = {}
    for graph_type in GRAPH_TYPES:
        result = [f for f in matching_files if graph_type in f]
        assert len(result) <= 1
        if result:
            site_fig_dict[graph_type] = result[0]
    legend = FIGURE_DIR / LEGEND_NAME
     
    if len(site_fig_dict): 
        # exclude weather for now, we need to add it after the legend
        #images = [Image.open(filename) for graph_type,filename in site_fig_dict.items() if graph_type != GRAPH_WEATHER] 
        image_list = []
        for graph_type,filename in site_fig_dict.items():
            if graph_type != GRAPH_WEATHER:
                with Image.open(filename) as im:
                    image_list.append(im.copy())
                
        # add the legend
        with Image.open(legend) as im:
            image_list.append(im.copy())

        # add the weather graph at the end, if it's there
        if GRAPH_WEATHER in site_fig_dict.keys() and include_weather:
            with Image.open(site_fig_dict[GRAPH_WEATHER]) as im:
                image_list.append(im.copy())

        composite = concat_images(*image_list)

        final = apply_decorations_to_composite(site, composite, month_locs)
        final.save(composite_path)
    return

def output_graph(site:str, graph: Figure, graph_type:str, save_files=False, make_all_graphs=False, align_dates=False, data_to_graph=True):
    if data_to_graph:
        if make_all_graphs: #Don't write the graphs to the screen if we're doing them all to speed it up
            #st.write(f"Saving {graph_type} for {site}")
            pass
        else:
            #If there is data in the graph, then write it to the screen if we are doing one graphic at a time
            if graph.get_axes():
                #st_image_figure(graph)
                st.pyplot(graph)
        
        #Save it to disk if we are either doing all the graphs, or the Save checkbox is checked
        if make_all_graphs or save_files:
            save_figure(site, graph_type, graph, do_aligned_dates=make_all_graphs and align_dates)
    else:
        #No data, so show a message instead. 
        save_figure(site, graph_type, graph, delete_only=True)
        site_name_text = f'<p style="font-family:sans-serif; font-size: 16px;"><b>{graph_type}</b></p>' #used to also have color:Black; 
        st.write(site_name_text, unsafe_allow_html=True)

        # https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
        emoji = [':woman-shrugging:', ':crying_cat_face:', ':slightly_frowning_face:', 
                 ':see_no_evil:', ':no_entry_sign:', ':cry:', ':thumbsdown:']
        st.write('No data available ' + random.choice(emoji))


def output_text(text:str, make_all_graphs:bool):
    if make_all_graphs:
        st.write(text)
    else:
        st.subheader(text)


#
#
# Weather
#
#

#Load weather data from file
@st.cache_data
def load_weather_data_from_file() -> pd.DataFrame:
    df = pd.read_csv(FILES[WEATHER_FILE])

    # Select and rename relevant columns
    columns_to_keep = ['Name', 'date', 'tmax_F', 'tmin_F', 'precip_in', 'wind_speed_mean_m_s']
    new_names = {'Name':'site', 'tmax_F':'tmax', 'tmin_F':'tmin', 'precip_in':'prcp', 'wind_speed_mean_m_s':'wspd'}
    df = df[columns_to_keep].rename(columns=new_names)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Convert strings to numeric
    for col in ['tmax', 'tmin', 'prcp']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Transform the data to long format
    df_melted = df.melt(id_vars=['site', 'date'], 
                        value_vars=['tmax', 'tmin', 'prcp', 'wspd'],
                        var_name='datatype', value_name='value')
    df_melted.set_index('site', inplace=True)

    return df_melted

#Filter weather data down to just what we need for a site
def get_weather_data(site_name:str, date_range_dict:dict) -> dict:
    df = load_weather_data_from_file()    
    site_weather_by_type = {}
    
    #select only rows that match our site 
    if site_name in df.index:
        site_weather = df.loc[[site_name]]
        #select only rows that are in our date range
        mask = (site_weather['date'] >= date_range_dict[START]) & (site_weather['date'] <= date_range_dict[END])
        site_weather = site_weather.loc[mask]

        if not site_weather.empty:
            # For each type of weather, break out that type into a separate table and 
            # drop it into a dict. Then, reindex the table to match our date range and 
            # fill in empty values
            date_range = pd.date_range(date_range_dict[START], date_range_dict[END]) 
            for w in WEATHER_COLS:
                site_weather_by_type[w] = site_weather.loc[site_weather['datatype']==w]
                #reindex the table to match our date range and fill in empty values
                site_weather_by_type[w]  = site_weather_by_type[w].set_index('date')
                site_weather_by_type[w]  = site_weather_by_type[w].reindex(date_range, fill_value=0)         
    else:
        st.write(f"No weather available for {site_name}")

    return site_weather_by_type

# add the ticks and associated content for the weather graph
def add_weather_graph_ticks(ax1:Axes, ax2:Axes, ax3:Axes, wg_colors:dict, x_range:tuple):
    # TICK FORMATTING AND CONTENT
    x_min, x_max = x_range
    x_min -= 0.5
    x_max += 0.5
    temp_min = 32
    temp_max = 115
    prcp_min = 0
    prcp_max = 2
    
    # Adjust the axis limits so all graphs are consistent
    ax1.set_ylim(ymin=prcp_min, ymax=prcp_max) #Sets the max amount of precip to 1.5
    ax2.set_ylim(ymin=temp_min, ymax=temp_max) #Set temp range
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    # doing our own labels so we can customize positions
    tick1y = 100
    tick2y = temp_min+8
    tick_width = 0.004
    label_yoffset = -1

    #Using a transform to get the x-coords in axis units, so they stay the same size regardless
    #how much temperature data we are graphing
    #https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
    trans = transforms.blended_transform_factory(ax2.transAxes, ax2.transData)

    # line marking 100F
    ax2.hlines([100], x_min, x_max, color=wg_colors['high'], linewidth=0.5, linestyle='dotted', zorder=2)        
    
    #blank out part of the 100F line so the label is readable
    rect = Rectangle((1-0.005,tick1y-6), -0.03, 12, facecolor='white', 
                    fill=True, edgecolor='none', zorder=3, transform=trans)
    ax2.add_patch(rect)

    #add tick label and ticks
    x_pos = 1 - tick_width  #in axis coordinates, where 0 is far left and 1 is far right
    ax2.text(x_pos, tick1y+label_yoffset, "100F", 
            fontsize=6, color=wg_colors['high'], horizontalalignment='right', verticalalignment='center', zorder=4,
            transform=trans)
    ax2.text(x_pos, tick2y+label_yoffset, f"{temp_min+8}F", 
            fontsize=6, color=wg_colors['high'], horizontalalignment='right', verticalalignment='center',
            transform=trans)
    ax2.hlines([tick1y, tick2y], 1-tick_width, 1, colors=wg_colors['high'], linewidth=BORDER_WIDTH,
            transform=trans)
    
    #drawing this on the temp axis because drawing on the prcp axis blew up, so have to convert to that scale
    prcp_label_pos1 = (temp_max - temp_min)*(0.5/prcp_max) + temp_min
    ax2.text(0+tick_width, prcp_label_pos1, '0.5"',
            fontsize=6, color=wg_colors['prcp'], horizontalalignment='left', verticalalignment='center',
            transform=trans)
    prcp_label_pos2 = (temp_max - temp_min)*(1.5/prcp_max) + temp_min
    ax2.text(0+tick_width, prcp_label_pos2, '1.5"',
            fontsize=6, color=wg_colors['prcp'], horizontalalignment='left', verticalalignment='center',
            transform=trans)
    ax2.hlines([prcp_label_pos1, prcp_label_pos2], 0, tick_width, colors=wg_colors['prcp'], linewidth=BORDER_WIDTH,
            transform=trans)
    # To turn off all default y ticks
    ax1.tick_params(
        axis='y',
        which='both',      # both major and minor ticks are affected
        left=False, right=False,  # ticks along the sides are off
        labelleft=False, labelright=False) # labels on the Y are off 
    ax2.tick_params(
        axis='y',
        which='both',      # both major and minor ticks are affected
        left=False, right=False,  # ticks along the sides are off
        labelleft=False, labelright=False) # labels on the Y are off 
    ax3.tick_params(
        axis='y',
        which='both',      # both major and minor ticks are affected
        left=False, right=False,  # ticks along the sides are off
        labelleft=False, labelright=False) # labels on the Y are off 
    
    return

#Used below to get min temp that isn't zero
def min_above_zero(s:pd.Series):
    temps = (temp for temp in s if temp>0)
    
    try:
         min_temp = min(temps)
    except Exception:
        min_temp = 0

    return min_temp


def create_weather_graph(weather_by_type:dict, site_name:str) -> tuple[Figure, list[Axes]]:
    if len(weather_by_type) == 0:
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axes = [axs] #make a list because we have to return this as a list type
        return fig, axes

    # --- inches-based spec ---
    top_pad_in = 0.10       # Whitespace at the top
    title_band_in = 0.2    # Gap for the label
    top_band_in = top_pad_in + title_band_in

    # Height in inches of the actual graph
    plot_in = 1.5           
    
    label_height_in = 0.25  # Axis labels
    legend_height_in = 0.55 # Legend
    bottom_pad_in = 0    # Whitespace at the bottom
    bottom_band_in = legend_height_in + label_height_in + bottom_pad_in

    fig_w = FIG_W           # keep your width in inches
    fig_h = top_band_in + plot_in + bottom_band_in

    fig, ax1 = plt.subplots(figsize=(fig_w, fig_h))

    # Fractions for the plot rectangle
    bottom = bottom_band_in / fig_h
    top = 1.0 - (top_band_in / fig_h)
    fig.subplots_adjust(left=0, right=1, bottom=bottom, top=top)

    # Place title at the top of the title band
    title_y = 1.0 - (top_pad_in / fig_h)
    plot_title(fig, GRAPH_WEATHER, y=title_y)

    ax2 = ax1.twinx() # makes a second y axis on the same x axis 
    ax3 = ax1.twinx() # makes a third y axis on the same x axis for wind
    axes = [ax1, ax2, ax3]

    # Plot the data in the proper format on the correct axis.
    wg_colors = {'high':'#ff0000', 'low':'#ff8080', 'prcp':'blue', 'wspd':'gray'}
    line_width = TEMP_LINES  #For how thick the red lines are
    marker_size = 0   #For whether a little dot shows on the graph or not
    marker = ''       #If we want a marker, a "." gives a point
    for wt in WEATHER_COLS:
        w = weather_by_type[wt]
        x = (w.index.normalize()-w.index[0]).days #Convert our dates to a series
        if wt == WEATHER_PRCP:
            ax1.bar(x, w["value"].to_numpy(), color = wg_colors['prcp'], lw=0)
        elif wt == WEATHER_TMAX:
            ax2.plot(x, w["value"].to_numpy(), color = wg_colors['high'], lw=line_width)
        elif wt == WEATHER_TMIN: 
            ax2.plot(x, w["value"].to_numpy(), color = wg_colors['low'], lw=line_width)
        elif wt == WEATHER_WIND:
            wspd = w['value']
            ax3.bar(
                x,
                wspd.to_numpy(),
                width=0.8,                 # narrow daily bars
                color=wg_colors['wspd'],
                alpha=0.5,                  # light
                linewidth=0,
                zorder=1                    # behind temps
            )
            # Make wind bars occupy just the lower part of that axis
            max_wind = 15 #wspd.max() for just the max for this site, but 15 is the fastest speed across all sites so we have a fixed scale and the data can be visually compared 
            ax3.set_ylim(0, max_wind * 1.2)  # nice headroom
        else: 
            log_error(f"create_weather_graph: Unknown weather type {wt}")

    w=weather_by_type[WEATHER_PRCP]
    x = (w.index.normalize()-w.index[0]).days #Convert our dates to a series
    x_range = (x.min(), x.max())
    add_weather_graph_ticks(ax1, ax2, ax3, wg_colors, x_range)

    # HORIZONTAL TICKS AND LABLING 
    text_offset_in = LABEL_OFFSET
    text_y = bottom - (text_offset_in / fig_h)
    text_y = -0.09
    draw_axis_labels(fig, get_days_per_month(weather_by_type[WEATHER_TMAX].index.values), y=text_y, bottom=bottom, top=top)
    
    #Turn on the graph borders, these are off by default for other charts
    ax1.spines[:].set_linewidth(BORDER_WIDTH)
    ax1.spines[:].set_visible(True)

    # Add a legend for the figure
    # For more legend tips see here: https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    tmax_label = f"High temp\n({min_above_zero(weather_by_type[WEATHER_TMAX]['value']):.0f}-"\
                    f"{weather_by_type[WEATHER_TMAX]['value'].max():.0f}\u00B0F)"
    tmin_label = f"Low temp\n({min_above_zero(weather_by_type[WEATHER_TMIN]['value']):.0f}-"\
                    f"{weather_by_type[WEATHER_TMIN]['value'].max():.0f}\u00B0F)"
    prcp_label = f"Precipitation\n(0-"\
                    f"{weather_by_type[WEATHER_PRCP]['value'].max():.2f}\042)"
    wind_label = f"Avg Daily Wind Speed\n(0-"\
                    f"{weather_by_type[WEATHER_WIND]['value'].max():.2f} m/s)"
    
    legend_elements = [Line2D([0], [0], color=wg_colors['high'], lw=3, label=tmax_label),
                        Line2D([0], [0], color=wg_colors['low'], lw=3, label=tmin_label),
                        Line2D([0], [0], color=wg_colors['prcp'], lw=3, label=prcp_label),
                        Line2D([0], [0], color=wg_colors['wspd'], lw=3, label=wind_label)]
    
    #draw the legend below the chart
    #calculations for the bbox
    legend_height_frac = legend_height_in / plot_in #Percentage of the plot occupied by the legend
    legend_y_frac = -(legend_height_in + 0.2*legend_height_in + bottom_pad_in) / plot_in #negative to position it below the graph
    ax1.legend(handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.12, legend_y_frac, 0.8, legend_height_frac),  
        mode='expand',
        ncol=4,
        prop={'family': GRAPH_FONT, 'size': LEGEND_FONT_SIZE},
        frameon=False
    )

    return fig, axes

#
# Bonus functions
#

# In this case, we return specific formatting based on whether the cell is zero, non-zero but not 
# a date, or a date. This is to make non-zero values that aren't dates easier to see.
# Color options: https://www.w3schools.com/colors/colors_names.asp
def style_cells(v):
    zeroprops = 'background-color:GhostWhite;'
    nonzeroprops = 'color:DarkBlue;background-color:Ivory'
    if v == '':
        result = zeroprops
    elif isinstance(v, pd.Timestamp): #if it's a date, do nothing
        result = ''
    else: #it must be a non-date, non-zero value so format it to call it out
        result = nonzeroprops
    return result

# For pretty printing a table
def pretty_print_table(df:pd.DataFrame, body_alignment="center"):
    def format_date(x: object) -> str:
        if x is None:
            return ""
        if x is pd.NaT:
            return ""
        ts = cast(pd.Timestamp, x)
        return ts.strftime('%m-%d-%y')
    
    # Do this so that the original DF doesn't get edited, because of how Python handles parameters 
    output_df = df.copy()

    # The < and > signs in the headers seems to be confusing streamlit, so need to remove them
    for col in output_df.columns:
        new_name = col.replace('<',' ')
        new_name = new_name.replace('>', ' ')
        output_df.rename(columns={col:new_name},inplace=True)
        
    th_props = [
        ('font-size', '14px'),
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('color', '#6d6d6d'),
        ('background-color', '#f7ffff')
    ]
                                
    td_props = [
        ('font-size', '12px')
    ]

    th_props: Sequence[tuple[str, str]]
    td_props: Sequence[tuple[str, str]]
    styles = [
        dict(selector="th", props=th_props),
        dict(selector="td", props=td_props)
    ]

    # apply table formatting from above
    #output_df=output_df.style.set_properties(**{'text-align': body_alignment}).set_table_styles(styles)
    output_df = (
        output_df.style
        .set_properties(subset=None, **{"text-align": body_alignment})
        .set_table_styles(styles)
    )
    #If there is a Date column then format it correctly
    if 'Date' in output_df.columns:
        output_df = output_df.format(formatter={'Date': format_date})

    st.markdown(output_df.to_html(escape=False), unsafe_allow_html=True)


def get_site_info(site_name:str, site_info_fields:list) -> dict:
    site_info = {}
    df = load_all_file()

    #Make a dictionary, where the keys are in site_info_fields and the values are the values from the 
    #site info file in the columns that match site_info_fields, for the site==site_name
    if site_name in df["Name"].values:
        site_info = df.loc[df["Name"] == site_name, site_info_fields].iloc[0].to_dict()
        site_info = {k: ("N/A" if pd.isna(v) else v) for k, v in site_info.items()}  # Replace NaN

    return site_info


def show_station_info(site:pd.DataFrame):
    alt = site.at[site.index[0], "Altitude"]
    lat = site.at[site.index[0], "Latitude"]
    lng = site.at[site.index[0], "Longitude"]
    rec_count = site.at[site.index[0], "Number of Recordings"]
    #We can either open the map to a spot with a pin, or to a view with zoom + map type but no pin. Here's more documentation:
    #https://developers.google.com/maps/documentation/urls/get-started
    map = f"https://www.google.com/maps/search/?api=1&query={lat}%2C{lng}"
    st.write(f"About this site: [Open in Google Maps]({map}), elevation {alt} m, {rec_count} recordings.")

# If any tag column has "reviewed" in the title AND the value for a row (a recording) is 1, then 
#    check that all "val" columns have a number. 
#    If any of them have a "---" or not a number then print out the filename of that row.
def check_tags(df: pd.DataFrame):
    bad_rows = pd.DataFrame()                                               
    #Find rows where the columns (ws-m, mh-m) have data, but the song column is missing data
    non_zero_rows = filter_df_by_tags(df, [data_col[tag_mhm], 
                                            data_col[tag_wsm]])
    bad_rows = pd.concat([bad_rows,
                            filter_df_by_tags(non_zero_rows, SONG_COLS, f'=={MISSING_DATA_FLAG}')])


    #P1N, P2N throws an error if it's missing alternative song
    non_zero_rows = filter_df_by_tags(df, EDGE_N_COLS)
    bad_rows = pd.concat([bad_rows, 
                            filter_df_by_tags(non_zero_rows, [data_col[ALTSONG1]], f"=={MISSING_DATA_FLAG}")])       

    if len(bad_rows):
        for r in bad_rows[FILENAME]:
            log_error(f"check_tags: {r} missing song tag")

    if not(bad_rows.empty) or len(error_list):
        with st.expander('See errors'):
            st.write(error_list)
    else:
        st.write('No data errors found')
    
    return


def make_final_pt(site_pt: pd.DataFrame, columns: list, friendly_names: dict) -> pd.DataFrame:
    pt_temp = site_pt.transpose()

    # Build ordered dataframe with selected columns, numeric
    out_cols = []
    col_map = {}
    for col in columns:
        if col in pt_temp.columns:
            out_cols.append(col)
            col_map[col] = friendly_names[col]

    pt = pt_temp[out_cols].copy()

    # Force numeric (keeps NaN), do NOT inject ""
    for c in pt.columns:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")

    pt.rename(columns=col_map, inplace=True)
    return pt


#Calculate the first and last dates for each song type
def get_first_and_last_dates(pt_site: pd.DataFrame) -> dict:
    pt_site = pt_site.transpose()
    output = {}
    for song in SONG_COLS:
        output[song] = {}
        d = pt_site[pt_site[song]>0]
        if d.empty:
            output[song]['First'] = 'n/a'
            output[song]['First count'] = '0'
        else:
            output[song]['First'] = d.index[0].strftime('%x')
            output[song]['First count'] = str(d.iloc[0][song])
    
    pt_site.sort_index(ascending=False, inplace=True)
    for song in SONG_COLS:
        d = pt_site[pt_site[song]>0]
        if d.empty:
            output[song]['Last'] = 'n/a'
            output[song]['Last count'] = '0'
        else:
            output[song]['Last'] = d.index[0].strftime('%x')
            output[song]['Last count'] = str(d.iloc[0][song])
    return output


def get_ratio(site):
    '''
    Read the ratios file and retrieve the value for this site
    '''
    all_ratios = pd.read_csv(r"./TRBLSummarizer/nestling-to-female-ratios.csv")
    ratio_rows = all_ratios[all_ratios["Site_Name"]==site]
    if len(ratio_rows):
        ratio_str = "Nestling-to-Female Ratios: "
        for i, (index, row) in enumerate(ratio_rows.iterrows()): 
            ratio = "n/a" if pd.isna(row["ARI"]) else f"{row["ARI"]:.2f}"
            ratio_str += f"**{row["Pulse Name"][-2:].capitalize()}**: {ratio}"
            if i < (len(ratio_rows) - 1):
                ratio_str += ",  "

        ratio_str2 = "Nestling-to-Female Ratios: "
        ratio_str2 += "  \n" if len(ratio_rows) > 1 else ""
        for i, (index, row) in enumerate(ratio_rows.iterrows()): 
            ratio = "n/a" if pd.isna(row["ARI"]) else f"{row["ARI"]:.2f}"
            ratio = ratio.replace("inf", "∞")
            ratio_str2 += f"**{row["Pulse Name"][-2:].capitalize()}**: {ratio}, " + \
                          f"Inc days: {row["Incubation_Days"]} ({row["Total_Female_Calls"]} calls), " + \
                          f"Brood days: {row["Nestling_Days"]} ({row["Total_Nestling_Calls"]} calls)" 
            if i < (len(ratio_rows) - 1):
                ratio_str2 += "  \n"

    else:
        ratio_str = "None found"
        ratio_str2 = "None found"
    return ratio_str2




def do_pattern_matching(site:str, date_range_dict:dict, container_top) -> tuple[pd.DataFrame, dict, bool]:
    #
    # PATTERN MATCHING ANALYSIS
    #
    #Load all the PM files, any errors will return an empty table. For later graphing purposes, 
    df_pattern_match = load_pm_data(site)
    df_pattern_match = clean_data(df_pattern_match, [site]) #THIS NEEDS TO GET CHANGED BECAUSE FOR SITES THAT WERE MERGED, THEY DON'T HAVE THE SAME SITE
    pt_pm = pd.DataFrame()
    pm_date_range_dict = {}

    if not df_pattern_match.empty:
        #In the scenario where we have PM data but no other data, we need to generate the date range
        if date_range_dict:
            pm_date_range_dict = date_range_dict  
        else:
            pm_date_range_dict = get_date_range(df_pattern_match, MAKE_ALL_GRAPHS, ALIGN_DATES, container_top)

        if len(df_pattern_match):
            for t in PM_FILE_TYPES: 
                #For each file type, get the filtered range of just that type
                df_for_file_type = df_pattern_match[df_pattern_match['type']==t]
                #Build the pivot table for it
                pt_for_file_type = make_pattern_match_pt(df_for_file_type, t, pm_date_range_dict)
                #Concat as above
                pt_pm = pd.concat([pt_pm, pt_for_file_type])    

    else:
        log_error(f"{site}: All pattern matching data not available, missing some or all files")

    return pt_pm, pm_date_range_dict, not df_pattern_match.empty

def assert_df_equal(old_df: pd.DataFrame, new_df: pd.DataFrame, context:str):
    return pd.testing.assert_frame_equal(
        old_df.sort_index().sort_index(axis=1),
        new_df.sort_index().sort_index(axis=1),
        check_like=True,   # allows column order differences
    ), f"{context}: New and old pivot table results do not match"

def do_manual(df_site: pd.DataFrame, date_range_dict:dict) -> tuple[pd.DataFrame, bool]:
    #
    # Data Analysis
    # -------------
    # 
    # MANUAL ANALYSIS
    #   1. Select all rows where one of the following tags
    #       tag<reviewed-MH>, tag<reviewed-WS>, tag<reviewed>
    #   2. Make a pivot table with the following columns:
    #       The number of recordings from that set that have Common Song >= 1
    #       The number of recordings from that set that have Courtship Song >= 1
    #       The number of recordings from that set that have AltSong2 >= 1
    #       The number of recordings from that set that have AltSong >= 1 
    #     
    df_manual = filter_df_by_tags(df_site, MANUAL_COLS)
    pt_manual = make_pivot_table(df_manual,  date_range_dict, labels=SONG_COLS)
    return pt_manual, not df_manual.empty


def do_mini_manual(df_site: pd.DataFrame, date_range_dict:dict) -> tuple[pd.DataFrame, bool]:
    # 1. Select all rows with one of the following tags:
    #       tag<reviewed-MH-h>, tag<reviewed-MH-m>, tag<reviewed-WS-h>, tag<reviewed-WS-m>
    # 2. Make a pivot table as above
    #   
    df_mini_manual = filter_df_by_tags(df_site, MINI_MANUAL_COLS)
    pt_mini_manual = make_pivot_table(df_mini_manual, date_range_dict, labels=SONG_COLS)
    return pt_mini_manual, not df_mini_manual.empty


def do_edge(df_site: pd.DataFrame, date_range_dict:dict, site:str) -> tuple[pd.DataFrame, bool]:
    pt_edge = pd.DataFrame()    
    have_edge_data = False

    new_pn_tag_map = { # map of tag_pXn to ync tag
        data_col[tag_p1n] : data_col[ALTSONG1],
        data_col[tag_p2n] : data_col[tag_YNC_p2],
        data_col[tag_p3n] : data_col[tag_YNC_p3],
        data_col[tag_p4n] : data_col[tag_YNC_p4],
    }

    check_edge_cols_for_errors(df_site)

    for tag in new_pn_tag_map: 
        df_for_tag = filter_df_by_tags(df_site, [tag])
        have_edge_data = have_edge_data or len(df_for_tag)>0
        pt_for_tag = make_pivot_table(df_for_tag, date_range_dict, preserve_edges=True, label_dict = {tag:new_pn_tag_map[tag]})
        pt_edge = pd.concat([pt_edge, pt_for_tag])

    else:
        log_error(f"Site {site} has no edge tags")

    return pt_edge, have_edge_data


def get_missing_days(df_site: pd.DataFrame) -> pd.DatetimeIndex:
    # returns the set of days between start and end that don't have any recordings, i.e. are missing from the dataset
    df_temp = df_site.copy()
    df_temp.index = pd.to_datetime(df_temp.index)
    start = df_temp.index.min()
    end   = df_temp.index.max()
    all_days = pd.date_range(start, end, freq="D")
    idx = pd.DatetimeIndex(df_temp.index).normalize().unique()
    missing_days = all_days.difference(idx)
    return missing_days

def get_month_locs(cols: pd.Index) -> dict[str, list[int]]:
    def get_visible_month_day_ranges(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}

        # Snap start to the first day of its month
        first_month_start = start.replace(day=1)

        # Generate all months intersecting the interval
        month_starts = pd.date_range(
            start=first_month_start,
            end=end,
            freq="MS",
        )

        for ms in month_starts:
            me = ms + pd.offsets.MonthEnd(0)

            # Clip to visible interval
            visible_start = max(ms, start)
            visible_end   = min(me, end)

            month_name = ms.strftime("%B")  # "April"
            result[month_name] = [visible_start.day, visible_end.day]

        return result

    if not isinstance(cols, pd.DatetimeIndex):
        raise TypeError("get_month_locs requires a DatetimeIndex")
    start = cols.min().normalize()
    end   = cols.max().normalize()
    month_locs = get_visible_month_day_ranges(start, end) 
    return month_locs


# ===========================================================================================================
# ===========================================================================================================
#
#  Main
#
# ===========================================================================================================
# ===========================================================================================================

def main():
    global MAKE_ALL_GRAPHS


    with timed("Load summary data"):
        # Load all the summary data
        summary_df = load_summary_data()

    #combine_aligned_images()
    init_logging()

    # Set up the sidebar with three zones so it looks like we want
    container_top = st.sidebar.container()
    container_mid = st.sidebar.container(border=True)
    container_bottom = st.sidebar.container(border=True)
    container_top.title('TRBL Graphs')

    with container_mid:
        gamma = st.slider(
            "Heatmap contrast",
            min_value=0.1,
            max_value=1.0,
            value=0.65,
            step=0.05,
            key="gamma",
        )
        show_station_info_checkbox = st.checkbox('Show station info', value=True)
        show_weather_checkbox = st.checkbox('Show station weather', value=True)

    with container_bottom:
        st.write("Contact wendy.schackwitz@gmail.com with any questions")
        if not BEING_DEPLOYED_TO_STREAMLIT:
            MAKE_ALL_GRAPHS = st.checkbox('Make all graphs')
        else:
            MAKE_ALL_GRAPHS = False

    #Load all the data for most of the graphs
    # with timed("Load all tag data"):
    #     df_original = load_data()

    #Load the hourly groupings for the PM graphs
    parquet_path = DATA_DIR / "recordings_per_day_hour.parquet"
    recordings_df = load_recordings_hourly(parquet_path, "site", "date", "hour", "n_recordings")

    #Get the list of sites that we're going to do reports for, and then remove all the other data
    with timed("Clean data"):
        site_list = get_target_sites()
    #     df = clean_data(df_original, site_list)

    # with timed("Free memory"):
    #     # Nuke the original data, hopefully this frees up memory
    #     del df_original
    #     gc.collect()

    save_files = False
    save_composite = False

    # If we're doing all the graphs, then set our target to the entire list, else use the UI to pick
    if MAKE_ALL_GRAPHS:
        target_sites = site_list
        #Can use this to limit sites to just a particular year
        #target_sites = [string for string in target_sites if string.startswith("2024 ")]

        # This is the file where we write all the dates we extracted from the data
        if os.path.exists(FILES[DATES_FILE]):
            os.remove(FILES[DATES_FILE])
        
        # For now, I'm not saving all the files, only the composite because it's taking up too much space. 
        # When she needs all the files, we'll bring this back
        # Make sure to fix it here and in the Else statement below
        save_files = False
        save_composite = True
    else:
        target_sites = [get_site_to_analyze(site_list, container_top)]
        if not BEING_DEPLOYED_TO_STREAMLIT:
            save_composite = container_top.checkbox('Save as picture', value=False) #user decides to save the graphs as pics or not
            save_files = save_composite
        
        #debug: to get a specific site, put the name of the site below and uncomment
        #target_sites = ["2018 Rush Ranch"]

    # Set format shared by all graphs
    set_global_theme()

    df_site = pd.DataFrame()
    site_counter = 0
    pt_manual = pd.DataFrame()
    pt_mini_manual = pd.DataFrame()
    pt_edge = pd.DataFrame()
    pt_pm = pd.DataFrame()
    weather_by_type = {}

    do_aligned_dates = MAKE_ALL_GRAPHS and ALIGN_DATES
    # if do_aligned_dates:
    #         combine_aligned_images()

    for idx, site in enumerate(target_sites):
        if PROFILING:
            if idx > 5: 
                break
        error_msgs = []
        site_counter += 1
        # Select the site matching the one of interest
        #df_site = df[df[data_col[SITE]] == site]
        
        #Get the data for this site
        df_site = load_data_for_site(site)

        date_range_dict = {}
        pm_date_range_dict = {}
        missing_days = pd.DatetimeIndex([])
        have_pm_data = False
        have_mini_manual_data = False
        have_manual_data = False
        have_edge_data = False
        rec_norm = pd.Series(dtype=int)

        if not df_site.empty:
            #Get the data that we're going to graph
            rec_df_site = recordings_df.loc[recordings_df["site"] == site]
            mask = (rec_df_site["hour"] >= 5) & (rec_df_site["hour"] < 21)  # 05:00–20:59 
            rec_norm = (
                rec_df_site.loc[mask]
                .groupby("date")["n_recordings"]
                .sum()
            )

            #Using the site of interest, get the first & last dates and give the user the option to customize the range
            date_range_dict = get_date_range(df_site, MAKE_ALL_GRAPHS, ALIGN_DATES, container_top)

            #Get this list of days without data, for later graphing
            missing_days = get_missing_days(df_site)

            #Make the manual graphs if we're not aligning dates or we're going one-by-one             
            if not do_aligned_dates:
                # MANUAL ANALYSIS
                with timed("Manual analysis"):
                    pt_manual, have_manual_data = do_manual(df_site, date_range_dict)

                # MINI-MANUAL ANALYSIS
                with timed("Mini-manual analysis"):
                    pt_mini_manual, have_mini_manual_data = do_mini_manual(df_site, date_range_dict)

                # EDGE ANALYSIS
                with timed("Edge analysis"):
                    pt_edge, have_edge_data = do_edge(df_site, date_range_dict, site)

            # PATTERN MATCHING ANALYSIS
            with timed("Pattern matching analysis"):
                pt_pm, pm_date_range_dict, have_pm_data = do_pattern_matching(site, date_range_dict, container_top)


        # 
        #    Summary data
        #
        # What we want to do is break this into a dictionary, with one entry for each pulse. 
        # Each pulse should have a dictionary mapping any valid dates in the pulse to its column name, 
        # e.g. {"P1": {"P1 Inc Start":Timestamp('2023-05-01')}}

        #iloc[:,1] selects all the rows but only column 1 (which is the second column, as it's zero indexed)
        #== site selects the row that matches the site
        #Mar 2025: this actually creates a pd dataframe, not a dict, but if everything else works, don't change now!
        summary_row = summary_df[summary_df.iloc[:, 1] == site]

        # Process the summary data, i.e. figure out if it's correctly structured, adjust for 
        # "abandoned" and so on, and convert all dates to a date format so it's easy to graph later. 

        if len(summary_row):
            site_summary_dict = process_site_summary_data(summary_row)
        else:
            site_summary_dict = {}


        # ------------------------------------------------------------------------------------------------
        # DISPLAY
        pretty_site_name = get_pretty_name_for_site(site)
        if MAKE_ALL_GRAPHS:
            st.subheader(f"{pretty_site_name} [{str(site_counter)} of {str(len(target_sites))}]")
        else: 
            st.subheader(f"{pretty_site_name}")
        
        if not pd.isna(summary_row["Skip Site"].item()):
            st.write(f":red-background[Duplicate site: {summary_row["Comment for Skip Site"].item()}]")

        if len(error_msgs):
            for error_msg in error_msgs:
                st.write(f":red-background[{error_msg}]")

        if not MAKE_ALL_GRAPHS:
            if show_station_info_checkbox:
                show_station_info(summary_row)
            ratio_str = get_ratio(site)
            st.success(f"{ratio_str}")
            

        #list of month positions in the graphs
        month_locs = {} 

        # Pattern Matching Analysis
        # Everything has data, it didn't use to be the case. I'll leave the If in case we ever go back
        if True: #not pt_pm.empty:
            key_dates = {}
            for p in PULSES:
                if p in site_summary_dict:
                    key_dates[p] = {}
                    mc_date = site_summary_dict[p][PHASE_MALE_CHORUS]["start"]
                    inc_date = site_summary_dict[p][PHASE_INC]["start"]
                    hatch_date = site_summary_dict[p][PHASE_BROOD]["start"]
                    fledge_start_date = site_summary_dict[p][PHASE_FLDG]["start"]
                    dispersal = site_summary_dict[p][PHASE_FLDG]["end"]
                    if pd.notna(mc_date):
                        key_dates[p][PULSE_MC_START] = mc_date
                    if pd.notna(inc_date):
                        key_dates[p][PULSE_INC_START] = inc_date
                    if pd.notna(hatch_date):
                        key_dates[p][PULSE_HATCH] = hatch_date
                    if pd.notna(fledge_start_date):
                        key_dates[p][PULSE_FIRST_FLDG] = fledge_start_date
                    if pd.notna(dispersal):
                        key_dates[p][PULSE_LAST_FLDG] = dispersal


            with timed("Pattern matching graph"):
                graph, axs = create_graph(
                                    site = site,
                                    df = pt_pm, 
                                    row_names = PM_FILE_TYPES, 
                                    cmap = CMAP_PM, 
                                    title = GRAPH_PM,
                                    graph_type = GRAPH_PM,
                                    key_dates = key_dates,
                                    missing_days = missing_days,
                                    denom_by_day = rec_norm,
                                    do_aligned_dates=do_aligned_dates
                ) 

            # add this if we want to include the site name (site + ' ' if save_files else '')
            # Need to be able to build an image that looks like the graph labels so that it can be drawn
            # at the top of the composite. So, try to pull out the month positions for each graph as we don't 
            # know which graph will be non-empty. Once we have them, we don't need to get again (as we don't want)
            # to accidentally delete our list

            if month_locs=={}:
                month_locs = get_month_locs(pt_pm.columns)


            output_graph(site, graph, GRAPH_PM,
                        save_files=save_files, 
                        make_all_graphs=MAKE_ALL_GRAPHS, align_dates=ALIGN_DATES,
                        data_to_graph=have_pm_data)


        # MiniManual Analysis
        if not pt_mini_manual.empty and not do_aligned_dates:
            with timed("Mini manual graph"):
                graph, axs = create_graph(
                                site = site,
                                df = pt_mini_manual, 
                                row_names = SONG_COLS, 
                                cmap = CMAP, 
                                raw_data = df_site,
                                draw_vert_rects = True,
                                title = "Manual Analysis (Periodic)",
                                graph_type = GRAPH_MINIMAN,
                                missing_days = missing_days,
            )
            if month_locs=={}:
                month_locs = get_month_locs(pt_mini_manual.columns)

            output_graph(site, graph, GRAPH_MINIMAN, 
                        save_files=save_files, make_all_graphs=MAKE_ALL_GRAPHS, data_to_graph=have_mini_manual_data)


        # Manual analyisis graph    
        if not pt_manual.empty and SHOW_MANUAL_ANALYSIS and not do_aligned_dates:
            #SEPT2025- trying to hide COURT_SONG from the list of songs
            new_songs = [MALE_SONG, ALTSONG2, ALTSONG1]

            with timed("Manual graph"): 
                graph, axs = create_graph(
                                    site = site,
                                    df = pt_manual, 
                                    row_names = [data_col[s] for s in new_songs], #SEPT2025
                                    cmap = CMAP, 
                                    title = "Manual Analysis (Daily Review)",
                                    graph_type=GRAPH_MANUAL,
                                    missing_days = missing_days
                ) 
            if month_locs=={}:
                month_locs = get_month_locs(pt_manual.columns)

            output_graph(site, graph, GRAPH_MANUAL, 
                        save_files=save_files, make_all_graphs=MAKE_ALL_GRAPHS, data_to_graph=have_manual_data)


        # Edge Analysis
        if not pt_edge.empty and not do_aligned_dates:
            cmap_edge = {n:'Blues' for n in EDGE_N_COLS} # the |" is used to merge dicts
            graph, axs = create_graph(
                                site = site,
                                df = pt_edge, #was pt_edge
                                row_names = pt_edge.index.to_list(), #was EDGE_N_COLS, #was EDGE_COLS,
                                cmap = cmap_edge, 
                                raw_data = df_site,
                                draw_horiz_rects = True,
                                title = "Manual Analysis (Hatchlings Only)", # was GRAPH_EDGE,
                                graph_type=GRAPH_EDGE,
                                missing_days = missing_days
            )
            if month_locs=={}:
                month_locs = get_month_locs(pt_edge.columns)

            output_graph(site, graph, GRAPH_EDGE, 
                        save_files=save_files, make_all_graphs=MAKE_ALL_GRAPHS, data_to_graph=have_edge_data)
        
        if not do_aligned_dates:
            #Draw the single legend for the rest of the charts and save to a file if needed
            draw_legend(CMAP, MAKE_ALL_GRAPHS, save_composite)

            #Show weather, as needed and if available
            weather_by_type = {}
            if show_weather_checkbox:
                # If date_range_dict and pm_date_range dict are both defined, they will be the same. However, it's 
                # possible that there is only one of them. 
                if pm_date_range_dict or date_range_dict:
                    date_to_use = date_range_dict if date_range_dict else pm_date_range_dict
                    # Load and parse weather data
                    weather_by_type = get_weather_data(site, date_to_use)
                    if weather_by_type:
                        graph, axs = create_weather_graph(weather_by_type, site)
                        output_graph(site, graph, GRAPH_WEATHER, 
                                    save_files=save_files, make_all_graphs=MAKE_ALL_GRAPHS, data_to_graph=True)
            
        if not BEING_DEPLOYED_TO_STREAMLIT and (MAKE_ALL_GRAPHS or save_composite) and not do_aligned_dates:
            combine_images(site, month_locs, show_weather_checkbox)

    #If site_df is empty, then there were no recordings at all for the site and so we can skip all the summarizing
    if not MAKE_ALL_GRAPHS and len(df_site):
        # Show the table with all the raw data
        with st.expander("See raw data"):
            #Used for making the overview pivot table
            friendly_names = {data_col[MALE_SONG] : 'M-Male', 
                            data_col[COURT_SONG]: 'M-Chorus',
                            data_col[ALTSONG2] : 'M-Female', 
                            data_col[ALTSONG1] : 'M-Nestling'
            }
            overview = []
            overview.append(make_final_pt(pt_manual, SONG_COLS, friendly_names))
            
            friendly_names = {data_col[MALE_SONG] : 'MM-Male', 
                            data_col[COURT_SONG]: 'MM-Chorus',
                            data_col[ALTSONG2] : 'MM-Female', 
                            data_col[ALTSONG1] : 'MM-Nestling'
            }
            overview.append(make_final_pt(pt_mini_manual, SONG_COLS, friendly_names))

            friendly_names =   {data_col[tag_p1c]: 'P1C',
                                data_col[tag_p1n]: 'P1N',
                                data_col[tag_p2c]: 'P2C',
                                data_col[tag_p2n]: 'P2N',
                                data_col[tag_p3n]: 'P3N',
                                data_col[tag_p4n]: 'P4N',
            }
            overview.append(make_final_pt(pt_edge, EDGE_COLS, friendly_names))

            #Pattern Matching 
            overview.append(make_final_pt(pt_pm, PM_FILE_TYPES, pm_friendly_names))

            #Add weather at the end
            if len(weather_by_type):
                weather_data = pd.DataFrame()
                for t in WEATHER_COLS:
                    weather_data = pd.concat([weather_data, weather_by_type[t]['value']], axis=1)
                    weather_data.rename(columns={'value':t}, inplace=True)
                    if t != WEATHER_PRCP:
                        weather_data[t] = weather_data[t].fillna(0).astype(int)
                overview.append(weather_data)

            # The variable overview is a list of each dataframe. Now, take all the data and concat it into 
            #a single table
            union_pt = pd.concat(overview, axis=1)

            # enforce dtypes before index reset
            for c in union_pt.columns:
                if c == "prcp" or c == "PRCP":
                    union_pt[c] = pd.to_numeric(union_pt[c], errors="coerce")
                else:
                    union_pt[c] = pd.to_numeric(union_pt[c], errors="coerce").astype("Int64")

            # Pop the index out so that we can format it, do this by resetting the index so each 
            # row just gets a number index
            union_pt = union_pt.reset_index().rename(columns={"index": "Date"})        
            
            #Make a copy and clean it up for display
            df_display = union_pt.copy()
            # Keep Date readable; everything else becomes string with blanks for missing
            for c in df_display.columns:
                if c == "Date":
                    # optional formatting
                    df_display[c] = pd.to_datetime(df_display[c], errors="coerce").dt.strftime("%Y-%m-%d")
                else:
                    df_display[c] = df_display[c].astype("string").fillna("")

            sty_union = (
                df_display
                .style
                .map(style_cells)
                .set_properties(**{"text-align": "center"})
            )
        
            st.dataframe(
                sty_union,
                column_config={
                    "Date": st.column_config.DateColumn(format="MM-DD-YY"),
                    "PRCP": st.column_config.NumberColumn(format="%.2f"),
                    # numeric columns: right alignment is the default in Streamlit’s grid for numbers
                },
                use_container_width=True,
            )

        # Put a box with first and last dates for the Song columns, with counts on that date
        with st.expander("See overview of dates"): 
            st.write("Currently hiding this, let me know if you want it")
            # output = get_first_and_last_dates(make_pivot_table(df_site, date_range_dict, labels=song_cols))
            # pretty_print_table(pd.DataFrame.from_dict(output))

        # Scan the list of tags and flag any where there is "---" for the value.
        if container_mid.checkbox('Show errors', value=True): 
            check_tags(df_site)

        if st.button('Clear cache'):
            get_target_sites().clear()
            clean_data.clear()
            load_data.clear()
            load_weather_data_from_file.clear()
        
        plt.close("all")
    if do_aligned_dates:
            combine_aligned_images()
    return

# def profile_main():
#     prof = cProfile.Profile()
#     prof.enable()
#     main()  # or whatever generates your graphs
#     prof.disable()

#     out = Path("profile_main.prof")
#     prof.dump_stats(out)

#     stats = pstats.Stats(prof).sort_stats("cumtime")
#     stats.print_stats(40)  # top 40 by cumulative time


def profile_main():
    prof = cProfile.Profile()
    prof.enable()
    main()  # or whatever generates your graphs
    prof.disable()

    out = Path("profile_main.prof")
    prof.dump_stats(out)

    # Existing summary
    stats = pstats.Stats(prof)
    stats.strip_dirs().sort_stats("cumtime")
    stats.print_stats(40)  # top 40 by cumulative time

    # NEW: show who is calling the expensive matplotlib text-measurement paths
    print("\n=== CALLERS: get_text_width_height_descent ===")
    stats.print_callers("get_text_width_height_descent")

    print("\n=== CALLERS: _get_text_metrics_with_cache_impl ===")
    stats.print_callers("_get_text_metrics_with_cache_impl")

    print("\n=== CALLERS: _get_text_metrics_with_cache ===")
    stats.print_callers("_get_text_metrics_with_cache")

    # Optional: check common triggers
    print("\n=== CALLERS: tight_layout ===")
    stats.print_callers("tight_layout")

    print("\n=== CALLERS: constrained_layout ===")
    stats.print_callers("constrained_layout")

    print("\n=== CALLERS: Axis._update_ticks ===")
    stats.print_callers("_update_ticks")

    print("\n=== CALLERS: get_tightbbox ===")
    stats.print_callers("get_tightbbox")

    print("\n=== CALLERS: Figure.get_tightbbox ===")
    stats.print_callers("Figure.get_tightbbox")

    print("\n=== CALLERS: savefig ===")
    stats.print_callers("savefig")

    print("\n=== CALLERS: print_figure ===")
    stats.print_callers("print_figure")

if __name__ == "__main__":
    if PROFILING:
        profile_main()
    else:
        main()
    