import streamlit as st
import pandas as pd
import numpy as np
import math
import seaborn as sns

import matplotlib as mpl
mpl.use('WebAgg') #Have to select the backend before doing other imports
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from pathlib import Path
import os
import calendar
from collections import Counter
from itertools import tee
import random
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime as dt
from pyinstrument import Profiler
import glob
from datetime import timedelta

#to force garbage collection and reduce memory use
import gc

#Do we want profiling data?
profiling = False

#Set to true before I deploy
being_deployed_to_streamlit = True

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
#tag_p2a = 'tag_p2a'  #COMMENTING THIS OUT FOR NOW BUT IT WILL BE HERE SOON
tag_p2f = 'tag_p2f'
#tag_p2na= 'tag_p2na' doesn't exist any more
#tag_p3c = 'tag_p3c'  doesn't exist now, might exist in the future
#tag_p3n = 'tag_p3n'  doesn't exist now, might exist in the future
#tag_p3f = 'tag_p3f'  doesn't exist now, might exist in the future
#tag_p3na= 'tag_p3na' #DOESN'T EXIST NOW
tag_wsmc = 'tag_wsmc'
validated = 'validated'
tag_YNC_p2 = 'tag<YNC-p2>'
#tag_YNC_p3 = 'tag<YNC-p3>'  doesn't exist now, might exist in the future
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
    tag_p1a      : 'tag<p1a>',
    tag_p1c      : 'tag<p1c>',
    tag_p1f      : 'tag<p1f>',
    tag_p1n      : 'tag<p1n>',
    tag_p2c      : 'tag<p2c>',
    tag_p2f      : 'tag<p2f>',
    tag_p2n      : 'tag<p2n>',
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

songs = [MALE_SONG, COURT_SONG, ALTSONG2, ALTSONG1]
song_cols = [data_col[s] for s in songs]
all_songs = [MALE_SONG, COURT_SONG, ALTSONG2, ALTSONG1, SIMPLE_CALL2] 
all_song_cols = [data_col[s] for s in all_songs]

manual_tags = [tag_mh, tag_ws, tag_]
mini_manual_tags = [tag_mhh, tag_mhm, tag_wsm]

edge_c_tags = [tag_p1c, tag_p2c] #male chorus
edge_n_tags = [tag_p1n, tag_p2n] #nestlings, p1 = pulse 1, p2 = pulse 2
edge_tags = edge_c_tags + edge_n_tags + [tag_YNC_p2, tag_p1a, tag_p1f, tag_p2f]
edge_tag_map = {
    tag_p1n : [data_col[tag_p1f], data_col[tag_p1a]],
    tag_p2n : [data_col[tag_p2f]]#, data_col[tag_p2a]],
}

all_tags = manual_tags + mini_manual_tags + edge_tags

manual_cols = [data_col[t] for t in manual_tags]
mini_manual_cols = [data_col[t] for t in mini_manual_tags]
edge_c_cols = [data_col[t] for t in edge_c_tags]
edge_n_cols = [data_col[t] for t in edge_n_tags]
#all_tag_cols = manual_cols + mini_manual_cols + edge_c_cols + edge_n_cols

edge_cols = edge_c_cols + edge_n_cols #make list of the right length
edge_cols[::2] = edge_c_cols #assign C cols to the even indices (0, 2, ...)
edge_cols[1::2] = edge_n_cols #assign N cols to the odd indices (1, 3, ...)


#Constants for the graphing, so they can be shared across weather and blackbird graphs
#For setting figure width and height, values in inches
fig_w = 6.5
fig_h = 1

#constants for the weather data files
WEATHER_PRCP = 'PRCP'
WEATHER_TMAX = 'TMAX'
WEATHER_TMIN = 'TMIN'
weather_cols = [WEATHER_PRCP, WEATHER_TMAX, WEATHER_TMIN]

GRAPH_SUMMARY = "Summary"
GRAPH_MANUAL = 'Manual Analysis'
GRAPH_MINIMAN = 'Mini Man Analysis'
GRAPH_EDGE = 'Edge Analysis'
GRAPH_PM = 'Pattern Matching Analysis'
GRAPH_WEATHER = 'Weather'
graph_names = [GRAPH_SUMMARY, GRAPH_MANUAL, GRAPH_MINIMAN, GRAPH_PM, GRAPH_EDGE, GRAPH_WEATHER]
legend_name = 'legend.png'
legend_text = {GRAPH_SUMMARY: ["Settlement", "Incubation", "Brooding", "Fledgling"],
               GRAPH_MANUAL: ["Male Song", "Male Chorus", "Female Chatter", "Hatchling/Nestling"],
               GRAPH_MINIMAN: ["Male Song", "Male Chorus", "Female Chatter", "Hatchling/Nestling", "Fledgling"],
               GRAPH_EDGE: ["Male Chorus", "Hatchling Call"],
               GRAPH_PM: ["Male Song", "Male Chorus", "Female Chatter", "Hatchling/Nestling", "Fledgling", 
                          "Insect 30", "Insect 31", "Insect 32", "Insect 33", "Pacific Tree Frog", "Red-legged Frog", "Bull Frog"]
}

#default color map
cmap = {data_col[MALE_SONG]:'Greens', 
        data_col[COURT_SONG]:'Oranges', 
        data_col[ALTSONG2]:'Purples', 
        data_col[ALTSONG1]:'Blues', 
        "Fledgling":"Blues"
}

cmap_names = {data_col[MALE_SONG]:"Male Song",
              data_col[COURT_SONG]:"Male Chorus",
              data_col[ALTSONG2]:"Female Chatter",
              data_col[ALTSONG1]:"Hatchling/Nestling/Fledgling",
#   temporarily don't need this
#              "Fledgling":"Fledgling",
} 

#color map for pattern matching
cmap_pm = {"Male Song":         "Greens", 
           "Male Chorus":       "Oranges", 
           "Female":            "Purples", 
           "Hatchling":         "Blues",
           "Nestling" :         "Blues",
           "Fledgling":         "Blues",
           "Insect 30":         "Greys",
           "Insect 31":   	    "Greys",
           "Insect 32":         "Greys",	
           "Insect 33":         "Greys",
           "Pacific Tree Frog": "Greys",	
           "Red-legged Frog":   "Greys",
           "Bull Frog":         "Greys"}


#Files, paths, etc.
DATA_FOLDER = 'Data/'
FIG_FOLDER = 'Figures/'
data_dir = Path(__file__).parents[0] / DATA_FOLDER
figure_dir = Path(__file__).parents[0] / FIG_FOLDER
SITE_INFO_FILE = 'TRBL Analysis tracking - All.csv'
SHEET_HEADER_SIZE = 2 #number of rows to skip over
WEATHER_FILE = 'weather_history.csv'
DATA_OLD_FILE = 'data_old.csv'
error_file = Path(__file__).parents[0] / 'error.txt'
SUMMARY_FILE = 'TRBL Analysis tracking - All.csv'
DATES_FILE = 'analyzed dates.csv'

#This is everything except the data files, because those are auto-generated
files = {
    SITE_INFO_FILE : data_dir / SITE_INFO_FILE,
    WEATHER_FILE : data_dir / WEATHER_FILE,
    DATA_OLD_FILE : data_dir / DATA_OLD_FILE,
    SUMMARY_FILE : data_dir / SUMMARY_FILE, 
    DATES_FILE : Path(__file__).parents[0] / DATES_FILE
}

# Mar 2024: This is the new set of summary data that Wendy created
# Source data is from the Google Sheet
#TODO: For clarity, rename all symbols that are constants to be all caps.

PULSE_COUNT = "pulse_count"
ABANDONED = "abandon"
PULSES = ["p1", "p2", "p3", "p4"]
SUMMARY_FIRST_REC = "First Recording"
SUMMARY_LAST_REC = "Last Recording"
summary_edge_dates = [SUMMARY_FIRST_REC, SUMMARY_LAST_REC]
PULSE_MC_START = "mcstart"
PULSE_MC_END = "mcend"
PULSE_HATCH = "hatch"
PULSE_FIRST_FLDG = "fledgestart"
PULSE_LAST_FLDG = "fledgedisp"
pulse_date_types = [PULSE_MC_START, PULSE_MC_END, PULSE_HATCH, PULSE_FIRST_FLDG, PULSE_LAST_FLDG, ABANDONED]
#Mar 2025, these weren't needed to dropping them
#pulse_numeric_types = ["Inc Length", "Async Score", "Fldg Age"]
pulse_numeric_types = ["Site ID", "Altitude", "Number of Recordings"]
summary_date_cols = [p + ' ' + d for p in PULSES for d in pulse_date_types]
#summary_numeric_cols = [p + ' ' + n for p in PULSES for n in pulse_numeric_types]
summary_numeric_cols = pulse_numeric_types

PHASE_MALE_CHORUS = "Settlement"
PHASE_INC = "Incubation"
PHASE_BROOD = "Brooding"
PHASE_FLDG = "Fledgling"
pulse_phases = {PHASE_MALE_CHORUS : [PULSE_MC_START, PULSE_MC_END],
                PHASE_INC : [PULSE_MC_END, PULSE_HATCH],
                PHASE_BROOD : [PULSE_HATCH, PULSE_FIRST_FLDG],
                PHASE_FLDG : [PULSE_FIRST_FLDG, PULSE_LAST_FLDG]}


#
#Pattern Matching Files
#edit this if we add/remove file types
#Change: Color Map for Pattern Matching, Legend Text, plus File Types. Also, there are some lists
#of column names in summarize_pm() that likely need to change
pm_song_types = ["Male Song",
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
pm_other_types = [PM_INSECT_SP30,
                 "Insect 31",	
                 "Insect 32",	
                 "Insect 33",	
                 PM_FROG_PACTF,	
                 "Red-legged Frog",
                 "Bull Frog"]

#Abbreviations are used in the summary table, to reduce column width
pm_file_types = pm_song_types #+ pm_other_types
pm_abbreviations = ["PM-MS", "PM-MC", "PM-F", "PM-H", "PM-N", "PM-FL","PM-I30", "PM-I31", "PM-I32", "PM-I33", "PM-PTF", "PM-RLF", "PM-BF"]
pm_friendly_names = dict(zip(pm_file_types, pm_abbreviations))

FIRST = "First"
LAST = "Last"
BEFORE_FIRST = "Before First"
AFTER_LAST = "After Last"

valid_pm_date_deltas = {pm_song_types[1]:0, #Male Chorus to Female can be 0 days
                        pm_song_types[2]:5, #Female to Hatchling must be at least 5 days
                        pm_song_types[3]:0, #Hatchling to Nestling can be 0 days
                        pm_song_types[4]:3, #Nestling to Fledgling must be at least 3 days
                        pm_song_types[5]:0, #Nestling to Nestling is zero, here to make math easy
                        }

missing_data_flag = -100
preserve_edges_flag = -99

DPI = 300
scale = int(DPI/300)

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
    if not being_deployed_to_streamlit:
        remove_file(error_file)
        with error_file.open("a") as f:
            f.write(f"Logging started {my_time()}\n")    

def log_error(msg: str):
    global error_list
    error_list += f"{msg}\n\n"
    if not being_deployed_to_streamlit:
        with error_file.open("a") as f:
            f.write(f"{my_time()}: {msg}\n")

def show_error(msg: str):
    #Only show the error if we're doing one graph at a time, but log it
    if not make_all_graphs:
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

def make_date(row):
    s = f"{row['year']}-{row['month']:02}-{row['day']:02}"
    return np.datetime64(s)

#
#
#File handling and setup
#
#
@st.cache_resource
def get_target_sites() -> dict:
    #Load the list of unique site names, keep just the 'Name' column, and then convert that to a list
    all_sites = pd.read_csv(files[SITE_INFO_FILE], usecols = ["Name", "Skip Site"], skiprows=SHEET_HEADER_SIZE)

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

#Used by the two functions that follow to do file format validation
def confirm_columns(target_cols:dict, file_cols:list, file:str) -> bool:
    errors_found = []
    if len(target_cols) != len(file_cols):
        show_error(f"confirm_columns: File {file} has an unexpected number of columns, {len(file_cols)} instead of {len(target_cols)}")
    for col in target_cols:        
        if  target_cols[col] not in file_cols:
            errors_found.append(target_cols[col])
            show_error(f"confirm_columns: Column {target_cols[col]} missing from file {file}")
    
    return errors_found

# Confirm that a date has either a p1f tag or a p1n tag, but not both
#error cases
# p1c good with p1c, p1n, p1na,
#   error with p1f, any p2
#   error with p1c with any p3

# p1n with p1c, p2c are OK. All other combos are wrong:
#   p1n with p1f, p1na
#   p1n with p2n, p2na or p2f
#   p1n with any p3

# p1f allowed with p2c, p2n, p2na. All other combos are wrong:
#   p1f error with p1c, p1n, p1na
#   p1f error with p2f 
#   p1f error with any p3 tag

# p1na allowed with p1c, p2c
#   p1n, p1f
#   p2n, p2na, p2f
#   any p3

# p2c allowed with p1n, p1f, p1na, p2n, p2na, 
#   error with p1c
#   error with p2f
#   error with p3c, p3n, p3na, p3f

# p2n allowed with p1f, p2c, p3c, 
#   error with p1c, p1n, p1na
#   error with p2na, p2f
#   error with p3n, p3na, p3f

# p2na allowed with p2c, p1f, p3c
#   p1c, p1n, p1na not allowed
#   p2n, p2f
#   p3n, p3na, p3f

# p2f allowed with p3c, p3n, p3na
#   p1c, p1n, p1na, p1f
#   p2c, p2n, p2na
#   p3f

# p3c allowed with p2n, p2na, p2f, p3n, p3na, p3f
#   error with and p1
#   error with p2c

# p3n allowed with p2f, p3c, 
#   error with and p1
#   error with p2c, p2n, p2na 
#   error with p3na, p3f

# p3na allowed with p2f, p3c
#   p1 not allowed
#   p2c, p2n, p2na, 
#   p3n, p3na, p3f

# p3f allowed with p3f
#   p1 not allowed
#   p2 not allowed
#   p3 except p3f

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

    # For each day, there should be only either P1F or P1N, never both
    tag_errors = df.loc[(df[data_col[tag_p1f]]>=1) & (df[data_col[tag_p1n]]>=1)]

    if len(tag_errors):
        error_found = True
        show_error("check_edge_cols_for_errors: Found recordings that have both P1F and P1N tags, see log")
        for f in tag_errors[FILENAME]: 
            log_error(f"check_edge_cols_for_errors: {f}\tRecording has both P1F and P1N tags")

    return error_found 

# Load the main data.csv file into a dataframe, validate that the columns are what we expect
@st.cache_resource
def load_data() -> pd.DataFrame:
    files_to_load = [data_dir / f"data {year}.csv" for year in range(2017, 2025)]
    combined_df = pd.DataFrame()
    for file_name in files_to_load:
        #Validate the data file format
        headers = pd.read_csv(file_name, nrows=0).columns.tolist()
        missing_columns = confirm_columns(data_col, headers, file_name)  

        #The set of columns we want to use are the basic info (filename, site, date), all songs, and all tags
        usecols = [data_col[FILENAME], data_col[SITE], data_col[DATE]]
        for song in all_songs:
            usecols.append(data_col[song])
        for tag in all_tags:
            usecols.append(data_col[tag])

        #remove any columns that are missing from the data file, so we don't ask for them as that will cause
        #an exception. Hopefully the rest of the code is robust enough to deal...
        usecols = [item for item in usecols if item not in missing_columns]

        df = pd.read_csv(file_name, 
                        usecols = usecols,
                        parse_dates = [data_col[DATE]],
                        index_col = [data_col[DATE]])
        combined_df = pd.concat([combined_df, df]) #NOTE This assumes the files don't have overlapping dates

    return combined_df


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
    site_dir = data_dir / site
    if os.path.isdir(site_dir):
        for t in pm_file_types:
            fname = f"{site} {t}.csv"
            full_file_name = site_dir / fname

            df_temp = pd.DataFrame()
            if is_non_zero_file(full_file_name):
                #Validate that all columns exist, and abandon ship if we're missing any
                headers = pd.read_csv(full_file_name, nrows=0).columns.tolist()
                missing_columns = confirm_columns(site_columns, headers, fname)
                if len(missing_columns) == 0: 
                    df_temp = pd.read_csv(full_file_name, usecols=usecols)
                    #make a new column that has the date in it, take into account that the table could be empty
                    if len(df_temp):
                        df_temp[DATE] = df_temp.apply(lambda row: make_date(row), axis=1)
                    else:
                        df_temp[DATE] = []
                else:
                    #columns are missing so can't do anything!
                    log_error(f"load_pm_data: Columns {missing_columns} are missing from pattern matching file!")
                    return pd.DataFrame()
            else:
                #NOTE: Dec 2024, removing the error logging because there will be a ton of these. Consider adding back
                #       error checking for only the blackbird songs, although that also probably isn't necessary
                #       given that these are being downloaded automatically
                #log_error(f"Missing or empty pattern matching file {full_file_name}")
                #Add an empty date column so we don't have a mismatch for the concat
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



@st.cache_resource
def load_summary_data() -> pd.DataFrame:
    #Load the summary data and prep it for graphing. 
    #This assumes that all validation (e.g. column names, values, etc.) is done in the script that downloads the csv file
    data_csv = Path(__file__).parents[0] / files[SUMMARY_FILE]

    #Load up the file 
    #Skiprows is because the All file has junk at the top we want to ignore
    df = pd.read_csv(data_csv, skiprows=SHEET_HEADER_SIZE)

    #If needed, can convert to date values as below, but it doesn't seem necessary
    #df[date_cols] = df[date_cols].apply(pd.to_datetime, errors='coerce')

    # Convert numeric columns to integers. As above, you have to force it this way if the types vary.
    # Empty values or strings are converted to NaN
    df[summary_numeric_cols] = df[summary_numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[summary_numeric_cols] = df[summary_numeric_cols].astype(pd.Int64Dtype())  # Keeps NaNs

    # If we want to make those "NaN" or "NaT" into a string we can do this:
    #for d in date_cols:
    #    df[d] = df[d].fillna("ND")

    return df

# clean up the data for a particular site
def is_valid_date_string(date_string):
    result = pd.to_datetime(date_string, format="%m/%d/%Y", errors="coerce")
    if pd.isna(result):
        return False
    else:
        return True 

def convert_to_datetime(date_string):
    date_format = "%m/%d/%Y"
    result = pd.to_datetime(date_string, format=date_format, errors="coerce")
    return result  # Will return pd.NaT for missing or null datetime values

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
            if phase in pulse_phases.keys(): #Need to skip Abandoned, as it doesn't have a pair of dates
                if is_valid_date_pair(pulse_data[p][phase]):
                    result = True
                    break
        count += 1 if result else 0

    return count

def get_val_from_df(df:pd.DataFrame, col):
    result = df.iloc[0,df.columns.get_loc(col)]
    return result

def process_site_summary_data(summary_row:pd.DataFrame) -> dict:
    nd_string = "ND"
    # This function takes a row from the summary spreadsheet. The goal is to process it as follows:
    first_rec = get_val_from_df(summary_row, SUMMARY_FIRST_REC)
    last_rec = get_val_from_df(summary_row, SUMMARY_LAST_REC)

    #TODO Is this the best way to handle the zero recording case?
    if pd.isna(first_rec): #TODO: Log an error in this case
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

        for phase in pulse_phases:
            start, end = pulse_phases[phase]
            target1 = f"{pulse}{start}"
            value1 = get_val_from_df(summary_row, target1) #TODO Test case where it's actually ND in the table instead of blank
            result1 = pd.NaT
            if is_valid_date_string(value1):
                #It's a good date, so format it
                result1 = convert_to_datetime(value1)             
            elif pd.notna(value1) and value1.endswith(ABANDONED):
                if not is_valid_date(abandoned_date):
                    log_error(f"{error_prefix}: Column Abandoned does not have a valid abandoned date")
                else:
                    result1 = pd.NaT
            elif value1 == "before start":
                #If the start date is "before start" then we don't know when it was exactly. I used to use the
                #date of the first recording as "before start" but that's not correct, so we took this line 
                #out:
                #   result1 = summary_dict[SUMMARY_FIRST_REC]
                #
                #Not sure if there's anything to do here...
                pass
            elif value1 == "after end":
                result1 = summary_dict[SUMMARY_LAST_REC]
            elif value1 == nd_string or value1 == "" or pd.isna(value1):
                #this is OK, we aren't going to draw anything in this case
                pass
            else:
                #if not one of the above, then it's an error
                log_error(f"{error_prefix}: Found invalid data in {target1}")

            target2 = f"{pulse}{end}"
            value2 = get_val_from_df(summary_row, target2)
            result2 = pd.NaT
            if is_valid_date_string(value2):
                #It's a good date, so format it
                if phase == PHASE_FLDG:
                    #For fledgling phase, don't subtract one from the end date
                    delta = pd.Timedelta(days=0)
                else:
                    delta = pd.Timedelta(days=1)
                result2 = convert_to_datetime(value2) - delta
            elif pd.notna(value2) and value2.endswith("abandoned"):
                if not is_valid_date(abandoned_date):
                    log_error(f"{error_prefix}: Column Abandoned does not have a valid abandoned date")
                else:
                    result2 = abandoned_date - pd.Timedelta(days=1)
            elif value2 == "before start":
                #In this scenario, the start should be ND, throw an error if not
                if not value1 == nd_string:
                    log_error(f"{error_prefix}: In {target2} end date is 'before start' but start date is not 'ND'")
            elif value2 == "after end":
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
                log_error(f"process_site_summary_data: Found {value2} in {target2}, which is invalid data")
            
            pulse_result[phase] = {"start":result1, "end":result2}

        #Add the sets of dates to our master dictionary
        summary_dict[pulse] = pulse_result

    #Calculate count of valid pulses. If there were zero, then set the count to 1 else we won't get a graph
    p_count = max(1, count_valid_pulses(summary_dict))
    summary_dict[PULSE_COUNT] = p_count

    #Save our abandoned dates, if any
#    summary_dict[abandoned] = abandoned_dates

    return summary_dict 





#Perform the following operations to clean up the data:
#   - Drop sites that aren't needed, so we're passing around less data
#   - Exclude any data where the year of the data doesn't match the target year
#   - Exclude any data where there aren't recordings on consecutive days  
@st.cache_resource
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


        # Now, find first two consecutive items and drop everything after.
        dates = df_site.index.unique()
        for x,y in pairwise(dates):
            if abs((x-y).days) == 1:
                #found a match, need to keep only what's after this
                df_site = df_site.query(f"date <= '{x.strftime('%Y-%m-%d')}'")
                break

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
        # Find first two consecutive items and drop everything before
        dates = df_site.index.unique()
        for x,y in pairwise(dates):
            if abs((x-y).days) == 1:
                #found a match, need to keep only what's after this
                df_site = df_site.query(f"date >= '{x.strftime('%Y-%m-%d')}'")
                break

        df_clean = pd.concat([df_clean, df_site])
    
    # We need to preserve the diff between no data and 0 tags. But, we have to also make everything 
    # integers for later processing. So, we'll replace the hyphens with a special value and then just 
    # realize that we can't do math on this column any more without excluding it. Picked -100 because 
    # if we do do math then the answer will be obviously wrong!
    df_clean = df_clean.replace('---', missing_data_flag)
    
    # For each type of song, convert its column to be numeric instead of a string so we can run pivots
    for s in all_songs + all_tags:
        if data_col[s] in df_clean.columns:
            df_clean[data_col[s]] = pd.to_numeric(df_clean[data_col[s]])
    return df_clean


#
#
# Data Analysis
# 
#  

# Get the subset of rows where there's at least one tag, i.e. the count of any tag is greater than zero
# See here for an explanation of the next couple lines: 
# https://stackoverflow.com/questions/45925327/dynamically-filtering-a-pandas-dataframe
# filter_str is '>0' by default because that's what most queries involve, but if a 
# different string is passed in then we use that instead. 
def filter_df_by_tags(df:pd.DataFrame, target_tags:list, filter_str='>0', exclude_tags=[]) -> pd.DataFrame:
    # This is an alternative to: tagged_rows = site_df[((site_df[columns[tag_wse]]>0) | (site_df[columns[tag_mhh]]>0) ...
    query  =  '(' + ' | '.join([f'`{tag}`{filter_str}' for tag in target_tags]) + ')' 
    query += ' & ~'.join([f'`{tag}`{filter_str}' for tag in exclude_tags])
    filtered_df = df.query(query)
    return filtered_df

# Add missing dates by creating the largest date range for our graph and then reindex to add missing entries
# Also, transpose to get the right shape for the graph
def normalize_pt(pt:pd.DataFrame, date_range_dict:dict) -> pd.DataFrame:
    date_range = pd.date_range(date_range_dict[START], date_range_dict[END]) 
    temp = pt.reindex(date_range).fillna(0)
    temp = temp.transpose()
    return temp

# Generate the pivot table for the site
def make_pivot_table(df: pd.DataFrame, date_range_dict:dict, preserve_edges=False, labels=[], label_dict={}) -> pd.DataFrame:
    if len(df):
        if len(label_dict):
            # Assumes dict is: {"column to filter on": "column to count"}
            # In this case, we filter to only columns that match the key (because the DF being passed in has
            # columns matching any key in the dict), and then count the columns in that subset that are non-zero.
            # And then, the result is all merged together
            aggregate_df = pd.DataFrame()
            for tag in label_dict:
                temp = filter_df_by_tags(df, [tag])
                # If the value in a column is >=1, count it. To achieve this, the aggfunc below sums up 
                # the number of times that the test 'x>=1' is true
                temp_pt = pd.pivot_table(temp, values = [label_dict[tag]], index = [data_col[DATE]], 
                                    aggfunc = lambda x: (x>=1).sum())
                if len(temp_pt):
                    temp_pt.rename(columns={label_dict[tag]:'temp'}, inplace=True) #rename so that in the merge the cols are added
                    aggregate_df = pd.concat([aggregate_df, temp_pt]).groupby('date').sum()
            #rename the index so that it's the song name            
            aggregate_df.rename(columns={'temp':list(label_dict.keys())[0]}, inplace=True) 
        else:
            #If we were passed a list of labels instead of a dict, then use the same logic to count songs
            aggregate_df = pd.pivot_table(df, values = labels, index = [data_col[DATE]], 
                                    aggfunc = lambda x: (x>=1).sum()) 

        if preserve_edges:
            # For every date where there is a tag, make sure that the value is non-zero. Then, when we do the
            # graph later, we'll use this to show where the edges of the analysis were
            aggregate_df = aggregate_df.replace(to_replace=0, value=preserve_edges_flag)

        return normalize_pt(aggregate_df, date_range_dict)
    else:
        return pd.DataFrame()


# Pivot table for pattern matching is a little different
def make_pattern_match_pt(site_df: pd.DataFrame, type_name:str, date_range_dict:dict) -> pd.DataFrame:
    #If the value in 'validated' column is 'Present', count it.
    present = site_df[site_df[site_columns[validated]]=="present"]
    aggregate = present.pivot_table(index=DATE, values=site_columns[validated], aggfunc='count')
    #aggregate = pd.pivot_table(site_df, values=[site_columns[validated]], index = [data_col[date_str]], 
    #                          aggfunc = lambda x: (x==present).sum())
    aggregate = aggregate.rename(columns={validated:type_name})
    
    # If the pivot table is empty, ensure all dates exist with value 0
    if aggregate.empty:
        all_dates = site_df.index.unique()  # Get all dates from original df
        aggregate = pd.DataFrame(0, index=all_dates, columns=[type_name])  # Fill with zeros
        aggregate.index.name = DATE  # Set the index name properly
    
    return normalize_pt(aggregate, date_range_dict)


def song_count_sufficient(value, threshold):
    return pd.notna(value) and value >= threshold

def find_pm_dates(row: pd.Series, pulse_gap:int, threshold: int) -> list:
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
    phases = pm_file_types[1:] #Creates a new list except it drops "Male Song"
    base_dict = {}
    for phase in phases:
        #NOTE Dec 2024: added this if statement to limit the summarizing to just the bird songs
        if phase in pm_song_types:
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
    all_phases = pm_song_types[1:] #Creates a new list except it drops "Male Song"
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
    # Is there anything earlier in the target_pulse that is earlier in order than the target_phase?
    # If so, do the dates make sense?

    # NOTE: It's theoretically possible that we could have female, hatching, etc. be in the wrong
    # pulse, but Wendy says that in practice it never happens, and it's only Male Chorus that we're 
    # worried about. 

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

            #NOTE Dec 2024: this used to use "pm_file_types" and it attempted to auto-analyze all the types of calls
            #       in the same way. However, now that we're adding insects, et al, I changed it to specifically look
            #       only at the bird vocalizations by changing "pm_file_types" to "pm_song_types"
            for item in pm_song_types:
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
def summarize_pm(pt_pm: pd.DataFrame) -> pd.DataFrame:
    # From pt_pm, get the first date that has a song count >= 4
    threshold = 4
    pulse_gap = 14
    
    #Get all the date pairs
    dates = {}    
    for idx, row in pt_pm.iterrows(): 
        if idx in pm_song_types:
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

# Set the default date range to the first and last dates for which we have data. In the case that we're
# automatically generating all the sites, then stop there. Otherwise, show the UI for the date selection
# and if the user wants a specific range then update our range to reflect that.
# Assume that the data cleaning code has removed any extraneous dates, such as if data 
# is mistagged (i.e. data from 2019 shows up in the 2020 site)
def get_date_range(df:pd.DataFrame, graphing_all_sites:bool, my_sidebar) -> dict:
    if df.index.name == "date":
        date_range_dict = {START : df.index.min().strftime("%m-%d-%Y"), 
                             END : df.index.max().strftime("%m-%d-%Y")}
    else:
        date_range_dict = {START : df["date"].min().strftime("%m-%d-%Y"), 
                             END : df["date"].max().strftime("%m-%d-%Y")}

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


# Set up base theme
# See https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
#
# See here for color options: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
def set_global_theme():
    #https://matplotlib.org/stable/tutorials/introductory/customizing.html#matplotlib-rcparams
    line_color = 'gray'
    line_width = '0.75'
    custom_params = {'figure.dpi':DPI, 
#                     'font.family':'Arial',                      
                     'font.family':'sans serif', 
                     'font.size':'12',
                     'font.stretch':'normal',
                     'font.weight':'light',
                     'xtick.labelsize':'medium',
                     'xtick.major.size':'12',
                     'xtick.color':line_color,
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
    #Save the legend
    figure_path = figure_dir / legend_name
    if os.path.exists(figure_path):
        os.remove(figure_path)
    plt.savefig(figure_path, dpi='figure', bbox_inches='tight', pad_inches=0)    


def draw_legend(cmap:dict, make_all_graphs:bool, save_files:bool):
    gradient = np.linspace(0, 1, 32)
    gradient = np.vstack((gradient, gradient))

    n = len(cmap_names)
    # Create one axis per legend item
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(fig_w*0.6, fig_h*0.1))

    for ax, (key, label) in zip(axs, cmap_names.items()):
        # Set the axis coordinate system to [0,1] in both directions
        # Draw the color block: we reserve x from 0 to 0.35 for the block.
        ax.imshow(gradient, extent=[0, 0.35, 0.1, 0.9], aspect="auto", cmap=mpl.colormaps[cmap[key]], transform=ax.transAxes)
        # Draw a border around the color block
        #ax.add_patch(Rectangle((0, 0), 0.35, 1, transform=ax.transAxes,
        #                       edgecolor="black", linewidth=0.5, fill=False))
        
        # Add the label text immediately to the right of the color block.
        # Here, x=0.37 places the text just to the right of the block.
        ax.text(0.37, 0.5, label, transform=ax.transAxes, 
                va="center", ha="left", 
                fontsize=5)
        
        # Remove the axis visuals
        ax.set_axis_off()
 
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # color_ax = axs[i * 2]   # Color box axis
        # text_ax = axs[i * 2 + 1]  # Text label axis
        
        # #Draw the gradient
        # color_ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[cmap[key]])

        # #Add a border
        # color_ax.add_patch(Rectangle((0, 0), 1, 1, ec="black", fill=False, transform=color_ax.transAxes))

        # #Add the name
        # # ax.text(1.03, 0.5, cmap_names[call_type], 
        # #         verticalalignment='center', horizontalalignment='left',
        # #         fontsize=4, 
        # #         transform=ax.transAxes)
        # text_ax.text(-0.5, 0.5, label, 
        #              va="center", ha="left", 
        #              fontsize=5)

        # text_ax.set_axis_off()
        # color_ax.set_axis_off()

    if not make_all_graphs:
        st.pyplot(fig)

    if save_files:
        output_cmap()

    return


def month_days_between_dates(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    start_of_month = pd.Timestamp(start_date) - pd.DateOffset(days=pd.Timestamp(start_date).day - 1)

    result = {}
    for month_start in pd.date_range(start=start_of_month, end=end_date, freq='MS'):
        month_name = month_start.strftime('%B')
        days_in_month = (date_range >= month_start) & (date_range < (month_start + pd.DateOffset(months=1)))
        result[month_name] = days_in_month.sum()
    
    return result

def get_days_per_month(date_list:list) -> dict:
    # Make a list of all the values, but only use the month name. Then, count how many of each 
    # month names there are to get the number of days/month
    months = [pd.to_datetime(date).strftime('%B') for date in date_list]
    return Counter(months)

# The axis already has all the dates in it, but they need to be formatted. 
# 27 Jan 2023: This isn't really doing anything, because we're currently hiding all the 
# x-ticks, but I'm leaving this code because it doesn't hurt and we might want it later.
def format_xdateticks(date_axis:plt.Axes, mmdd = False):
    if mmdd:
        fmt = '%d-%b'
        rot = 30
        weight = 'light'
        fontsize = 9
    else:
        fmt = '%d'
        rot = 0
        weight = 'light'
        fontsize = 10
    
    #Make a list of all the values currently in the graph
    date_values = [value for value in date_axis.xaxis.get_major_formatter().func.args[0].values()]

    #Make a list of all the possible ticks with their strings formatted correctly
    ticks = [pd.to_datetime(value).strftime(fmt) for value in date_values]

    #Actually set the ticks and then apply font format as needed
    date_axis.xaxis.set_ticklabels(ticks, fontweight=weight, fontsize=fontsize)
    date_axis.tick_params(axis = 'x',labelrotation = rot, length=6, width=0.5) 
    return


#Take the list of month length counts we got from the function above, and draw lines at those positions. 
#Skip the last one so we don't draw over the border
def draw_axis_labels(month_lengths:dict, axs:np.ndarray, weather_graph=False, summary_graph=False, skip_month_names=False):
    font_size = 8
    target_ax = -1
    if weather_graph:
        y = -0.4
    elif summary_graph:
        y = 0.2
        target_ax = -2
        target_ax = -1  #TEMPORARILY
        font_size = 7  #TODO figure out why the fonts need to be adjusted like this
    else:
        y = 1.9+(0.25 if len(axs)>4 else 0)

    month_count = len(month_lengths)
    n = 0
    x_min, x_max = axs[len(axs)-1].get_xlim()
    x = x_min
    for month in month_lengths:
        # Center the label on the middle of the month, which is the #-days-in-the-month/2
        #   -1 because the count is 1-based but the axis is 0-based, e.g. if there are 12 days
        #   in the month
        center_pt = int((month_lengths[month])/2)
        mid = x + center_pt
        if not skip_month_names:
            axs[target_ax].text(x=mid, y=y, s=month, 
                        fontsize=font_size, va="bottom", ha="center") 

        x += month_lengths[month]
        if n < month_count:
            if summary_graph:
                for chart in range(len(axs)+target_ax):
                    axs[chart].axvline(x=x, color="darkgray")

                pass
            else:
                for ax in axs:
                    ax.axvline(x=x)
    

# For ensuring the title in the graph looks the same between weather and data graphs.
# note that if the fontsize is too big, then the title will become the largest thing 
# in the figure which causes the graph to shrink!
def plot_title(title:str):
    plt.suptitle(' ' + title, x=0, y=1,
                 fontsize=10, horizontalalignment='left')

# Mar 2024, this is unused so I'm going to comment it out for now, but keeping it just in case...
# def add_watermark(title:str):
#     # Function that adds additional text to the right of the existing suptitle
#     title = ' '+title
#     # Find the suptitle in the figure
#     suptitle = [t for t in plt.gcf().texts if t.get_text() == title][0]

#     # Get the figure coordinates of the suptitle without changing its position
#     suptitle_extent = suptitle.get_window_extent(renderer=plt.gcf().canvas.get_renderer())

#     # Convert suptitle extent to figure coordinates
#     fig_position = suptitle_extent.transformed(plt.gcf().transFigure.inverted())

#     # Add additional red text to the right of the suptitle
#     additional_text = dt.now().strftime("%Y-%m-%d %H:%M:%S")
#     figtext_x = fig_position.x1 + 0.02  # Adjust the value to position the text
#     figtext_y = fig_position.y0 + (fig_position.y1 - fig_position.y0) / 2  # Center vertically
#     plt.figtext(figtext_x, figtext_y, additional_text, ha='left', va='center', color='gray', fontsize=8)
    

#
# Custom graphing code to make the summary     
#
def create_summary_graph(pulse_data:dict, date_range:dict, make_all_graphs:bool) -> plt.figure:
    pc = pulse_data[PULSE_COUNT]
    #Rows_for_labels allows 1 row for the labels, and 1 row for the legend
    rows_for_labels = 2
    rows_for_labels = 1 #TEMPORARILY GETTING RID OF DATE
    legend_row = -1  # means it's the last row
    label_row = -2   # means it's the second-to-last row
    total_rows = pc + rows_for_labels    

    #This number is the percentage of the height, starting from the botom of the chart that we're going to reserve
    #for the charts themselves. 
    gap_for_title = 0.62 + (pc * 0.06)  #TEMPORARILY was 0.72, 0.04
#    gap_for_title = 0.88 #Good for 4 rows
#    gap_for_title = 0.76 #Good for 1 row

    chart_height = 0.25  # In inches
    figsize = (fig_w, (pc + rows_for_labels) * chart_height) 

    #create a chart that has pc+rows_for_labels rows, and 1 column
    fig, axs = plt.subplots(nrows = total_rows, ncols=1, 
                            figsize=figsize, sharex=True, sharey=True, 
                            gridspec_kw={'height_ratios': np.repeat(1,total_rows), 
                                         'left':0, 'right':1, 'bottom':0, 'top':gap_for_title,
                                         'hspace':0},  #hspace is row spacing (gap between rows)
    ) 

    plot_title("Inferred Timing of Breeding Stages")
    plt.subplots_adjust(left=0.0)

    # Color options here: https://matplotlib.org/stable/gallery/color/named_colors.html
    phase_color = {
        PHASE_MALE_CHORUS : "seagreen",
        PHASE_INC : "mediumpurple",
        PHASE_BROOD : "steelblue",
        PHASE_FLDG : "black",
        ABANDONED : "red"
    }
    background_color = "white"
    
    nesting_start_date = pulse_data[SUMMARY_FIRST_REC]
    nesting_end_date = pulse_data[SUMMARY_LAST_REC]

    days_per_month = month_days_between_dates(date_range[START], date_range[END])
    start_date = pd.Timestamp(date_range[START])
    end_date = pd.Timestamp(date_range[END])
    for ax in axs:
        ax.set_xlim(start_date, end_date + timedelta(days=1))
    
    #Add patch covering the background of the entire nesting phase, except for the date row
    rect_height = 1
    for row in range(len(axs)+label_row):
       rect = Rectangle((nesting_start_date, 0), nesting_end_date - nesting_start_date +timedelta(days=1), 
                     rect_height, color=background_color, alpha=1, zorder=1)
       axs[row].add_patch(rect)

    #Draw all our boxes
    pulses_graphed = {}
    #As of now, she wants all phases in the legend, not just the ones that actually occurred. If she changes her mind, 
    #then uncomment the line below plus the rest as appropriate
    #legend_elements = {}  
    row_count = 0
    abandoned_dict = {}
    for i, (p, data) in enumerate(pulse_data.items()):
        graphed_something = False
        if p in PULSES:
            for phase in pulse_phases:
                if is_valid_date_pair(pulse_data[p][phase]):
                    #Given that the x axis starts at start_date, we need to calculate everything as an offset from there
                    phase_start = pulse_data[p][phase][START]
                    phase_end = pulse_data[p][phase][END]
                    width = phase_end - phase_start + timedelta(days=1)
                    color = phase_color[phase]
                    #Note width is a Timedelta, so use .days to get the int value, but phase_start is a Timestamp, so use .day instead 
                    axs[row_count].barh(y=0, height=1,
                                        left=phase_start, width=width.days,  
                                        label=phase, color=color, align="edge", alpha=1)
                    graphed_something = True

                    if p not in pulses_graphed:
                        pulses_graphed[p] = row_count   # save the pulse name plus the axis that it went into

                    # if phase not in legend_elements:
                    #      legend_elements[phase] = color

            #Apply the marker for abandonded colony on top of the data after we've drawn the row
            if ABANDONED in pulse_data[p]:
                abandoned_dict[p] = pulse_data[p][ABANDONED]
                start_point = (pulse_data[p][ABANDONED],0)
                width = timedelta(days=1)
                height = 1
                rect = Rectangle(start_point, width, height, 
                            color=phase_color[ABANDONED], alpha=1, zorder=5,
                            label="Abandonded")
                axs[row_count].add_patch(rect)
                graphed_something = True

                # if abandoned not in legend_elements:
                #     legend_elements[abandoned] = 'red'

            #Not all pulses will have graphable data, so we only want to change axis if there was something to graph
            if graphed_something:
                row_count += 1

    # Legendary!
    # box_height = 0.6 #axis coordinates
    # box_buffer = (1 - box_height)/2
    # box_width = 0.06
    # text_width = 0.135
    # gap = 0.005
    # width = box_width + gap + text_width #Comes to 0.2, or 20% of width, as we are doing up to 5 labels
    # total_legend_width = len(phase_color)*width
    # xpos = (1 - total_legend_width)/2 + 0.03 #Give it a little push to the right
    # for i, (caption, color) in enumerate(phase_color.items()):
    #     legend_box = Rectangle(xy=(xpos,box_buffer), width=box_width, height=box_height, facecolor=color, transform=ax.transAxes)
    #     axs[legend_row].add_patch(legend_box)
    #     axs[legend_row].text(xpos+box_width+gap, y=0.5, s=caption, transform=ax.transAxes, 
    #                          fontsize=6, color="black", verticalalignment="center")
    #     xpos += width

    # Add the vertical lines and month names
    draw_axis_labels(days_per_month, axs, summary_graph=True, skip_month_names=True)

    # Draw the labels for each pulse
    for p in pulses_graphed:
        axs[pulses_graphed[p]].text(x=start_date + timedelta(days=0.5), y=0.5, s=p, 
                                    horizontalalignment='left', verticalalignment='center', color='black', fontsize=6)

    #Get the count of rows we ended up making and add spines to each except the last
    ax_count = len(axs)
    #If need to adjust anything else about all the charts, do it here. Right now, we're just making sure there
    #is one border drawn around the outside of everything
    for row in range(ax_count):
        if row == 0:
            left=True
            right=True
            top=True
            bottom=False
        elif row > 0 and row < ax_count+label_row:
            left=True
            right=True
            top=False
            bottom=False
        #WAS ...
        # elif row == ax_count+label_row:
        #     left=False
        #     right=False
        #     top=True
        #     bottom=False
        elif row == ax_count+legend_row:
            left=False
            right=False
            top=True
            bottom=False
        else:
            pass #ERROR!!!
        
        axs[row].spines['left'].set_visible(left)
        axs[row].spines['right'].set_visible(right)
        axs[row].spines['top'].set_visible(top)
        axs[row].spines['bottom'].set_visible(bottom)

        for spine in axs[row].spines.values():
            spine.set_linewidth(0.4)

    # Adjust vertical spacing between subplots
    plt.subplots_adjust(hspace=0)

    debugging = False
    if debugging:
        #To help with Debugging, draw lines on the graph for each day
        for row in range(ax_count-label_row):
            for day in pd.date_range(start=start_date,
                                end=end_date,
                                freq='D'):
                axs[row].axvline(x=day, color='gray', linestyle='--', linewidth=0.5)
        #Draw ticks only on the bottom one
        axs[ax_count-1].set_facecolor('none') #make the bottom graph transparent so we can see the ticks from above 
        bottom_chart = ax_count - 2
        day_numbers = pd.date_range(start=start_date, end=end_date, freq='3D')
        axs[bottom_chart].set_xticks(day_numbers)
        axs[bottom_chart].set_xticklabels([day.strftime('%m-%d') for day in day_numbers], fontsize=5, rotation=90, ha='center')
        axs[bottom_chart].tick_params(axis='x', direction='inout', pad=0)
        axs[bottom_chart].tick_params(labelbottom=True, bottom=True)

    # Output the data as text if we're doing one graph at a time
    if not make_all_graphs:
        st.write(f"First recording: {start_date.strftime('%m-%d')}, Last recording: {end_date.strftime('%m-%d')}")
        report = ""
        found_valid_dates=0
        for p in PULSES:
            empty_pulse = 1
            for phase in pulse_phases:
                if is_valid_date_pair(pulse_data[p][phase]):
                    #First time only per pulse, add the title
                    if empty_pulse:
                        empty_pulse = 0
                        report += f"-----Pulse {p}-----<br>"

                    phase_start = pulse_data[p][phase][START].strftime("%m-%d")
                    phase_end = pulse_data[p][phase][END].strftime("%m-%d")
                    report += f"{phase} start: {phase_start}, end: {phase_end}<br>"
                    found_valid_dates+=1

        if found_valid_dates or len(abandoned_dict):
            # with st.expander("Show pulse dates"):
            #     st.write("<b>Automatically derived dates:</b>", unsafe_allow_html=True)
            #     pretty_print_table(summarize_pm(pt_pm), body_alignment="left")
            with st.expander("Show manually derived dates:"):
                st.write("<br><b>Manually derived dates:</b>", unsafe_allow_html=True)
                st.write(report, unsafe_allow_html=True)
                if len(abandoned_dict):
                    report = "Abandoned:<br>"
                    for a in abandoned_dict:
                        report += f"{a}: {abandoned_dict[a].strftime('%m-%d')}"
                    st.write(report, unsafe_allow_html=True)

    return fig


# Create a graph, given a dataframe, list of row names, color map, and friendly names for the rows
def create_graph(df: pd.DataFrame, row_names:list, cmap:dict, draw_connectors=False, raw_data=pd.DataFrame, 
                 draw_vert_rects=False, draw_horiz_rects=False,title='', hatch_dates={}) -> plt.figure:
    plt.close() #close any prior graph that was open

    if len(df) == 0:
        return

    row_count = len(row_names)
    graph_drawn = []
    
    #distance between top of plot space and chart
    if title == GRAPH_PM:
        gap_for_title = 0.8 # removed insects so don't need extra height, was 0.9
    else:
        gap_for_title = 0.8 if title else 1

    #tick_spacing is how many days apart the tick marks are. If set to 0 then it turns off all ticks and labels except for month name
    tick_spacing = 0

    #NOTE Dec 2024 adding this to accomodate all the insect calls in the PM graph
    fig_height = fig_h if not title == GRAPH_PM else fig_h # removed insects so don't need extra height, was *2

    # Create the base figure for the graphs
    fig, axs = plt.subplots(nrows = row_count, ncols = 1,
                            sharex = 'col', 
                            gridspec_kw={'height_ratios': np.repeat(1,row_count), 
                                         'left':0, 'right':1, 'bottom':0, 'top':gap_for_title,
                                         'hspace':0},  #hspace is row spacing (gap between rows)
                            figsize=(fig_w,fig_height))

    # If we have one, add the title for the graph and set appropriate formatting
    if len(title) :
        plot_title(title)
    
    # Ensure that we have a row for each index. If a row is missing, add it with zero values
    for row in row_names:
        if row not in df.index:
            df.loc[row]=pd.Series(0,index=df.columns)

    # Set a mask ("NaN" since the value isn't specified) on the zero values so that we can force them 
    # to display as white. Keep the original data as we need it for drawing later. Use '<=0' because negative
    # numbers are used to differentiate no data from data with zero value
    df_clean = df.mask(df <= 0)

    i=0
    for row in row_names:
        # plotting the heatmap
        heatmap_max = df_clean.loc[row].max()

        # pull out the one row we want. When we do this, it turns into a series, so we then need to convert it back to a DF and transpose it to be wide
        df_to_graph = df_clean.loc[row].to_frame().transpose()

        axs[i] = sns.heatmap(data = df_to_graph,
                        ax = axs[i],
                        cmap = cmap[row] if len(cmap) > 1 else cmap[0],
                        vmin = 0, vmax = heatmap_max if heatmap_max > 0 else 1,
                        cbar = False,
                        xticklabels = tick_spacing,
                        yticklabels = False)
        
        # If we drew an empty graph, write text on top to indicate that it is supposed to be empty
        # and not that it's just hard to read!
        if df_clean.loc[row].sum() == 0:
            #The conundrum: at least for edge, it's possible that a row we drew is blank, but the actual
            #row is going to get some boxes and lines. In this case, there will be -99s in the data, 
            #and if we find those, we should NOT draw the text that says there is no data 
            #THIS IS NOT WORKING?
            if title == GRAPH_EDGE and df.loc[row].lt(0).any():
                pass
            else:
                axs[i].text(0.5,0.5,f"No data for {row}", 
                            fontsize='xx-small', fontstyle='italic', color='gray', verticalalignment='center')
        elif title == GRAPH_PM and row in pm_other_types:
                axs[i].text(0.5,0.5,f"{row}", 
                            fontsize='xx-small', fontstyle='italic', color='black', verticalalignment='center')


        # Track which graphs we drew, so we can put the proper ticks on later
        graph_drawn.append(i)

        if title == GRAPH_PM:
            #NOTE Add dates of first hatching if they exist
            if row == "Hatchling":
                for pulse in hatch_dates:
                    hatch_date = hatch_dates[pulse]
                    if hatch_date >= df_to_graph.columns[0] and hatch_date <= df_to_graph.columns[-1]:
                        hatch_index = df_to_graph.columns.get_loc(hatch_date)

                        # Determine marker size proportional to cell dimensions
                        marker_size = 7  # Adjust multiplier if needed

                        # Plot the "X" centered in the cell
                        axs[i].plot(hatch_index+0.7, 0.5, 
                                    marker='>', color='black', markersize=marker_size, mew=0.5,
                                    transform=axs[i].get_xaxis_transform())
                    else:
                        log_error(f"create_graph: Hatch date {hatch_date} is outside range of this year, which is {df_to_graph.columns[0]} through {df_to_graph.columns[-1]}")
                        
            #NOTE Dec 2024: Added extra lines to separate insects
            if row == PM_INSECT_SP30 or row == PM_FROG_PACTF:
                #Want to add a line above these two rows to separate them
                # Get the top y-limit
                top_y = axs[i].get_ylim()[1]
                xmin = axs[i].get_xlim()[0]
                xmax = axs[i].get_xlim()[1]

                # Draw a horizontal line at the top of the axis
                line = axs[i].hlines(y=top_y, xmin=xmin, xmax=xmax, colors='red',  linewidth=0.5)
                dashes = (0, (18, 2)) # 10 points on, 5 points off
                line.set_dashes(dashes)  # Apply the custom dash pattern


        # For edge: Add a rectangle around the regions of consective tags, and a line between 
        # non-consectutive if it's a N tag.
        if draw_horiz_rects and row in df_clean.index:
            df_col_nonzero = df.loc[row].to_frame()  #pull out the row we want, it turns into a column as above
            df_col_nonzero = df_col_nonzero.reset_index()   #index by ints for easy graphing
            df_col_nonzero = df_col_nonzero.query(f"`{row}` != 0")  #get only the nonzero values. 

            if len(df_col_nonzero):
                c = mpl.colormaps[(cmap[row] if len(cmap) > 1 else cmap[0])](0.85)
                if row in edge_c_cols: #these tags get a box around the whole block
                    first = df_col_nonzero.index[0]
                    last  = df_col_nonzero.index[len(df_col_nonzero)-1]+1
                    axs[i].add_patch(Rectangle((first,0), last-first, 0.99, 
                                     ec=c, fc=c, fill=False))
                    
                else: #n tags get boxes around each consecutive block
                    borders = []
                    borders.append(df_col_nonzero.index[0]) #block always starts with the first point
                    # Get the non-contiguous blocks
                    for x,y in pairwise(df_col_nonzero.index):
                        if y-x != 1:
                            borders.append(x)
                            borders.append(y)
                    borders.append(df_col_nonzero.index[len(df_col_nonzero)-1]+1) #always ends with the last one
                    #debug code -- show the dates
                    #for x in range(0,len(borders),2):
                    #    d1 = borders[x]
                    #    d2 = borders[x+1] if borders[x+1] < df_col_nonzero.index[len(df_col_nonzero)-1] else borders[x+1]-1
                    #    st.error("Blue: " + str(df_col_nonzero.loc[d1].values[0].date()) + "->" + str(df_col_nonzero.loc[d2].values[0].date()))                    

                    # We now have a list of pairs of coordinates where we need a rect. For each pair, draw one.
                    for x in range(0,len(borders),2):
                        extra = 1 if x != ((len(borders)/2)-1)*2 else 0
                        axs[i].add_patch(Rectangle((borders[x],0), borders[x+1]-borders[x] + extra, 0.99, ec=c, fc=c, fill=False))
                    # For each pair of rects, draw a line between them.
                    for x in range(1,len(borders)-1,2):
                        # The +1/-1 are because we don't want to draw on top of the days, just between the days
                        axs[i].add_patch(Rectangle((borders[x]+1,0.48), borders[x+1]-borders[x]-1, 0.04, ec=c, fc=c, fill=True)) 
        i += 1
        
    # For mini-manual: Add a rect around each day that has some data
    if draw_vert_rects and len(raw_data)>0:
        tagged_rows = filter_df_by_tags(raw_data, mini_manual_cols)
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
    
    # Clean up the ticks on the axis we're going to use
    # Set x-ticks
    #axs[len(row_names)-1].set_xticks(np.arange(0,len(df_to_graph.columns)), labels=df_to_graph.columns)
    #format_xdateticks(axs[len(row_names)-1])
    
    # Add the vertical lines and month names
    draw_axis_labels(get_days_per_month(df.columns.tolist()), axs)

    #Hide the ticks as we don't want them, we're just using them
    for i in graph_drawn:
        axs[i].tick_params(labelbottom=False, bottom=False)

    # Draw a bounding rectangle around everything except the caption
    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.0, 0.0), 1.0, gap_for_title, 
        linewidth = 0.5, fill=False, zorder=1000, transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])

    #if we want to add anything on top of the images, the time to do it is at the end
    #add_watermark(title)

    # return the final plotted heatmap
    return fig


def add_pulse_overlays(graph, summarized_data:pd.DataFrame, date_range:dict):
    # For each of the derived summary dates, draw a line on the graph
    # Top row of graph is Male Song, nothing goes there

    # For each of the other rows, we want to draw a bar to the left of the start date, a line to the end date, and then a line just 
    # after the end dates

    graph_start_date = pd.to_datetime(date_range["start"])
    for idx, phase_type in enumerate(pm_file_types): 
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
    filename = f"{site} - {graph_type}{extra}.png"
    return filename

#Helper for when we need to remove a file
def remove_file(full_path:str) -> bool:
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
def save_figure(site:str, graph_type:str, delete_only=False):
    #Do nothing if we're on the server, we can't save files there or download them without a lot of complexity
    if being_deployed_to_streamlit:
        return

    filename = make_img_filename(site, graph_type)
    figure_path = figure_dir / filename
    remove_file(figure_path)

    cleaned_image_filename = make_img_filename(site, graph_type, extra=' clean')    
    cleaned_figure_path = figure_dir / cleaned_image_filename
    remove_file(cleaned_figure_path)
    
    if not delete_only and plt.gcf().get_axes():
        # save the original image
        #plt.savefig(figure_path, dpi='figure', bbox_inches='tight')
        
        #Create a different version of the image that we'll use for the compilation
        #plt.suptitle('')  #if we want to remove the titles but I don't think we do

        #Figure out where the labels are. There's probably a way to do this in one call ...
        #maybe check the last axis?
        if graph_type == GRAPH_SUMMARY:
            #for the summary graph, we dont want to do anything i don't think
            pass
        else:
            if graph_type == GRAPH_WEATHER:
                ax = plt.gcf().get_axes()[0] #in the weather graph, the labels are in the first axis
                legend = ax.get_legend()
                bb = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
                yOffset = 0.25
                bb.y0 += yOffset
                bb.y1 += yOffset
                legend.set_bbox_to_anchor(bb, transform = ax.transAxes)
            else:
                ax = plt.gca() #in the other graphs, it's in the last axis
            
            #Go find the month labels and remove them
            for ge in ax.texts:
                #if the word 'data' is there then it's one of the error messages, otherwise it's a month
                if 'data' not in ge.get_text():
                    ge.remove()
            # Now save the cleaned up version
        plt.savefig(cleaned_figure_path, dpi='figure', bbox_inches='tight')    
    
    else:
        #TODO If there is no data, what to do? The line below saves an empty image.
        #Image.new(mode="RGB", size=(1, 1)).save(figure_path)
        pass

    plt.close()

def get_month_locs_from_graph() -> dict:
    locs = {}
    months = []
    #This only works for the data graphs, not the weather graph. But if all we have is a weather 
    #graph then we don't care what the composite looks like.
    ax = plt.gca() 
    for t in ax.texts:
        if 'data' not in t.get_text():
            # This pulls out the month string for the key of the dict
            months.append(t.get_text())
    x = 0
    m = 0 
    for line in ax.get_lines():
        locs[months[m]] = (x, line.get_xdata()[0])
        x = line.get_xdata()[0]
        assert type(x) is np.float64
        m+=1
    if x>0:
        locs['max']=x    
    return locs

def concat_images(*images, is_legend=False):
    """Generate composite of all supplied images."""
    # Get the widest width.
    #TODO Why aren't the images all the same width?
    width = max(image.width for image in images)
    # Add up all the heights.
    height = sum(image.height for image in images)
    composite = Image.new('RGB', (width, height), color='white')

    # Paste each image below the one before it.
    y = 0
    if is_legend:
        # In this case,  there will be exactly two images. The first one is the main image 
        # and the second is the legend, which should be centered.
        composite.paste(images[0], (0,0))
        y += images[0].height
        x = int((width-images[1].width)/2)
        composite.paste(images[1], (x,y))
    else:
        for image in images:
            composite.paste(image, (0, y))
            y += image.height
    return composite

def apply_decorations_to_composite(composite:Image, month_locs:dict) -> Image:
    #Make a new image that's a little bigger so we can add the site name at the top
    width, height = composite.size
    title_height = 125 * scale
    month_row_height = 80 * scale
    border_width = 4 * scale
    border_height = border_width * 2  * scale
    margin_bottom = 20 * scale
    new_height = height + title_height + month_row_height + border_height + margin_bottom

    title_font_size = 72 * scale
    month_font_size = 36 * scale
    fudge = 10 * scale #for descenders
    
    final = Image.new(composite.mode, (width, new_height), color='white')

    #Add the title
    draw = ImageDraw.Draw(final)
    if being_deployed_to_streamlit:
        font = ImageFont.load_default(size=title_font_size)
    else:
        font = ImageFont.truetype("arialbd.ttf", size=title_font_size)
    draw.text((width/2,title_height-fudge), site, fill='black', anchor='ms', font=font)

    #Add the months
    margin_left = 27 * scale
    margin_right = 1982 * scale
    if being_deployed_to_streamlit:
        font = ImageFont.load_default(size=month_font_size)
    else:
        font = ImageFont.truetype("arial.ttf", size=month_font_size)
    v_pos = title_height + month_row_height - fudge
    month_row_width = margin_right - margin_left
    
    max_width = month_locs['max']
    del month_locs['max'] #This entry has a dif't data type than the rest, so nuke it so we don't crash

    for month in month_locs:
        m_left = month_locs[month][0]
        m_right = month_locs[month][1]
        m_center = (m_right - m_left) * 0.5
        row_center = (m_left + m_center)/max_width 
        h_pos =  (row_center * month_row_width) + margin_left
        draw.text((h_pos, v_pos), month, fill='black', font=font, anchor='ms')

    #Paste in the composite
    final.paste(composite, box=(0,title_height + month_row_height + border_width)) 

    #Add the border
    border_top = title_height + month_row_height
    border_left = margin_left
    border_right = margin_right
    draw.rectangle([(border_left,border_top),(border_right,new_height-margin_bottom)], 
                    outline='black', width=border_width)

    return final

# Load all the images that match the site name, combine them into a single composite,
# and then save that out
def combine_images(site:str, month_locs:dict, include_weather:bool):
    #if there are no months, then we didn't have any data to graph so don't make a composite
    if len(month_locs) == 0:
        return
    
    composite_filename = make_img_filename(site, "composite")
    composite_path = figure_dir / composite_filename
    remove_file(composite_path)

    pattern = f"{site} -*clean.png"
    matching_files = glob.glob(os.path.join(figure_dir, pattern))
    #clean_site_files = [file for file in matching_files if "clean" in file]  #Can use this if we need to do additional filtering
    site_fig_dict = {}
    for graph_type in graph_names:
        result = [f for f in matching_files if graph_type in f]
        assert len(result) <= 1
        if result:
            site_fig_dict[graph_type] = result[0]
    legend = figure_dir / legend_name
     
    if len(site_fig_dict): 
        # exclude weather for now, we need to add it after the legend
        images = [Image.open(filename) for graph_type,filename in site_fig_dict.items() if graph_type != GRAPH_WEATHER] 
        composite = concat_images(*images)
        if True: #TODO decide if there is any logic needed about when to save the legend
            composite = concat_images(*[composite, Image.open(legend)], is_legend=True)
        #Add the weather graph only if it exists, to prevent an error if we haven't obtained it yet
        if GRAPH_WEATHER in site_fig_dict.keys() and include_weather:
            composite = concat_images(*[composite, Image.open(site_fig_dict[GRAPH_WEATHER])])
        final = apply_decorations_to_composite(composite, month_locs)
        final.save(composite_path)
    return

def output_graph(site:str, graph_type:str, save_files:bool, make_all_graphs:bool, data_to_graph=True):
    if data_to_graph:
        if make_all_graphs: #Don't write the graphs to the screen if we're doing them all to speed it up
            #st.write(f"Saving {graph_type} for {site}")
            pass
        else:
            #If there is data in the graph, then write it to the screen if we are doing one graphic at a time
            if graph.get_axes():
                st.write(graph)
        
        #Save it to disk if we are either doing all the graphs, or the Save checkbox is checked
        if make_all_graphs or save_files:
            save_figure(site, graph_type)
    else:
        #No data, so show a message instead. 
        save_figure(site, graph_type, delete_only=True)
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
@st.cache_resource
def load_weather_data_from_file() -> pd.DataFrame:
    #Validate the data file format
    try:
        headers = pd.read_csv(files[WEATHER_FILE], nrows=0).columns.tolist()
        weather_cols = {'row':'row','date':'date', 'datatype':'datatype', 'value':'value', 'site':'site'}

        #Removed error checking, assuming it's right        
        #This will show an error if something is wrong with the data 
        #missing_columns = confirm_columns(weather_cols, headers, WEATHER_FILE)
        
        df = pd.read_csv(files[WEATHER_FILE], 
                        parse_dates = [weather_cols['date']],
                        index_col = [weather_cols['site']])
    
    except: #something went wrong trhing to get the data, so just return an empty frame
        df = pd.DataFrame()

    return df

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
            for w in weather_cols:
                site_weather_by_type[w] = site_weather.loc[site_weather['datatype']==w]
                #reindex the table to match our date range and fill in empty values
                site_weather_by_type[w]  = site_weather_by_type[w].set_index('date')
                site_weather_by_type[w]  = site_weather_by_type[w].reindex(date_range, fill_value=0)         
    else:
        st.write(f"No weather available for {site_name}")

    return site_weather_by_type

# add the ticks and associated content for the weather graph
def add_weather_graph_ticks(ax1:plt.axes, ax2:plt.axes, wg_colors:dict, x_range:pd.Series):
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

    # line marking 100F
    ax2.hlines([100], x_min, x_max, color=wg_colors['high'], linewidth=0.5, linestyle='dotted', zorder=1)        
    
    # doing our own labels so we can customize positions
    tick1y = 100
    tick2y = temp_min+8
    tick_width = 0.004
    label_yoffset = -1

    #Using a transform to get the x-coords in axis units, so they stay the same size regardless
    #how much temperature data we are graphing
    #https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
    trans = transforms.blended_transform_factory(ax2.transAxes, ax2.transData)

    #blank out part of the 100F line so the label is readable
    rect = Rectangle((1-0.005,tick1y-6), -0.03, 12, facecolor='white', 
                    fill=True, edgecolor='none', zorder=2, transform=trans)
    ax2.add_patch(rect)

    #add tick label and ticks
    x_pos = 1 - tick_width  #in axis coordinates, where 0 is far left and 1 is far right
    ax2.text(x_pos, tick1y+label_yoffset, "100F", 
            fontsize=6, color=wg_colors['high'], horizontalalignment='right', verticalalignment='center', zorder=3,
            transform=trans)
    ax2.text(x_pos, tick2y+label_yoffset, f"{temp_min+8}F", 
            fontsize=6, color=wg_colors['high'], horizontalalignment='right', verticalalignment='center',
            transform=trans)
    ax2.hlines([tick1y, tick2y], 1-tick_width, 1, colors=wg_colors['high'], linewidth=0.5,
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
    ax2.hlines([prcp_label_pos1, prcp_label_pos2], 0, tick_width, colors=wg_colors['prcp'], linewidth=0.5,
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
    return

#Used below to get min temp that isn't zero
def min_above_zero(s:pd.Series):
    temps = (temp for temp in s if temp>0)
    
    try:
         min_temp = min(temps)
    except Exception:
        min_temp = 0

    return min_temp

#
# Something funky here...the weather graph is slightly offset from the others even though when viewed in the
# Paint they look the same. However, when I show the composite from the debugger after adding the weather
# graph, you can see that the border on the outside of the weather graph is slightly different. So, there 
# is some scaling that needs to be done that i'm missing... 
#
# Know:
# - number of data points is the same across all graphs
# - the error gets greater further we go across the graph
# - the error doesn't seem to be bigger as DPI increases
# - Line Width of the external frame is smaller on the weather graph (like 0.5pt instead of 0.75pt)
# -  


def create_weather_graph(weather_by_type:dict, site_name:str) -> plt.figure:
    if len(weather_by_type)>0:
        # The use of rows, cols, and gridspec is to force the graph to be drawn in the same 
        # proportions and size as the heatmaps
        fig, ax1 = plt.subplots(nrows = 1, ncols = 1, 
            gridspec_kw={'left':0, 'right':1, 'bottom':0, 'top':0.8},
            figsize=(fig_w,fig_h))
        ax2 = ax1.twinx() # makes a second y axis on the same x axis 

        plot_title(GRAPH_WEATHER) #site_name + ' ' +  to include site

        # Plot the data in the proper format on the correct axis.
        wg_colors = {'high':'red', 'low':'pink', 'prcp':'blue'}
        for wt in weather_cols:
            w = weather_by_type[wt]
            if wt == WEATHER_PRCP:
                ax1.bar(w.index.values, w['value'], color = wg_colors['prcp'], linewidth=0)
            elif wt == WEATHER_TMAX:
                ax2.plot(w.index.values, w['value'], color = wg_colors['high'], marker='.', markersize=2)
            else: #TMIN
                ax2.plot(w.index.values, w['value'], color = wg_colors['low'], marker='.', markersize=2)
        
        x_range = (mpl.dates.date2num(w.index.min()), mpl.dates.date2num(w.index.max()))
        add_weather_graph_ticks(ax1, ax2, wg_colors, x_range)

        # HORIZONTAL TICKS AND LABLING 
        #x_min = ax1.get_xlim()[0]
        #x_max = ax1.get_xlim()[1]
        # Need to set xlim so that we don't get an extra gap on either side
        # Get the list of ticks and set them --only needed if we ever want individual dates on the axis
#        axis_dates = list(weather_by_type[weather_tmax].index.values.astype(str))
        ax1.axes.set_xticks([])
        draw_axis_labels(get_days_per_month(weather_by_type[WEATHER_TMAX].index.values), [ax1], weather_graph=True)
        
        #Turn on the graph borders, these are off by default for other charts
        ax1.spines[:].set_linewidth(0.5)
        ax1.spines[:].set_visible(True)

        # Add a legend for the figure
        # For more legend tips see here: https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
        tmax_label = f"High temp ({min_above_zero(weather_by_type[WEATHER_TMAX]['value']):.0f}-"\
                     f"{weather_by_type[WEATHER_TMAX]['value'].max():.0f}\u00B0F)"
        tmin_label = f"Low temp ({min_above_zero(weather_by_type[WEATHER_TMIN]['value']):.0f}-"\
                     f"{weather_by_type[WEATHER_TMIN]['value'].max():.0f}\u00B0F)"
        prcp_label = f"Precipitation (0-"\
                     f"{weather_by_type[WEATHER_PRCP]['value'].max():.2f}\042)"
        legend_elements = [Line2D([0], [0], color=wg_colors['high'], lw=3, label=tmax_label),
                           Line2D([0], [0], color=wg_colors['low'], lw=3, label=tmin_label),
                           Line2D([0], [0], color=wg_colors['prcp'], lw=3, label=prcp_label)]
        
        #draw the legend below the chart. that's what the bbox_to_anchor with -0.5 does
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3,
                   fontsize=5, frameon=False)

    else:
        fig = plt.figure()

    return fig

#
# Bonus functions
#


#Used for formatting output table
#The function can take at most one paramenter. In this case, if there is a param and the 
# value passed in is zero, then we use the props that are passed in, otherwise none. In this
# way, the cell in question gets formatted using props if the value is zero. 
# It would be called like this:
#        #union_pt = union_pt.style.applymap(style_zero, props='color:gray;')
#
#https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
def style_zero(v, props=''):
    return props if v == 0 else None

# In this case, we return specific formatting based on whether the cell is zero, non-zero but not 
# a date, or a date. This is to make non-zero values that aren't dates easier to see.
# Color options: https://www.w3schools.com/colors/colors_names.asp
def style_cells(v):
    zeroprops = 'color:gray;'
    nonzeroprops = 'color:black;background-color:YellowGreen;text-align:center;'
    result = ''
    if v == 0:
        result = zeroprops
    elif isinstance(v, pd.Timestamp): #if it's a date, do nothing
        result = ''
    else: #it must be a non-date, non-zero value so format it to call it out
        result = nonzeroprops
    return result

def style_center_align(s, props='text-align: center;'):
    return props

# For pretty printing a table
def pretty_print_table(df:pd.DataFrame, body_alignment="center"):
    # Do this so that the original DF doesn't get edited, because of how Python handles parameters 
    output_df = df

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
                                    
    styles = [
    dict(selector="th", props=th_props),
    dict(selector="td", props=td_props)
    ]

    # apply table formatting from above
    output_df=output_df.style.set_properties(**{'text-align': body_alignment}).set_table_styles(styles)
    #If there is a Date column then format it correctly
    if 'Date' in output_df.columns:
        output_df.format(formatter={'Date':lambda x:x.strftime('%m-%d-%y')})

    st.markdown(output_df.to_html(escape=False), unsafe_allow_html=True)

def get_site_info(site_name:str, site_info_fields:list) -> dict:
    site_info = {}
    df = pd.read_csv(files[SITE_INFO_FILE], skiprows=SHEET_HEADER_SIZE)

    #Make a dictionary, where the keys are in site_info_fields and the values are the values from the 
    #site info file in the columns that match site_info_fields, for the site==site_name
    if site_name in df["Name"].values:
        site_info = df.loc[df["Name"] == site_name, site_info_fields].iloc[0].to_dict()
        site_info = {k: ("N/A" if pd.isna(v) else v) for k, v in site_info.items()}  # Replace NaN

    # for f in site_info_fields:
    #     value = df.loc[df['Name'] == site_name,f].values[0]
    #     site_info[f] = "N/A" if pd.isna(value) else value 
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
                            filter_df_by_tags(non_zero_rows, song_cols, f'=={missing_data_flag}')])

    #P1C, P2C throws an error if it's missing courtsong song
    non_zero_rows = filter_df_by_tags(df, edge_c_cols)
    bad_rows = pd.concat([bad_rows,
                            filter_df_by_tags(non_zero_rows, [data_col[COURT_SONG]], f"=={missing_data_flag}")])

    #P1N, P2N throws an error if it's missing alternative song
    non_zero_rows = filter_df_by_tags(df, edge_n_cols)
    bad_rows = pd.concat([bad_rows, 
                            filter_df_by_tags(non_zero_rows, [data_col[ALTSONG1]], f"=={missing_data_flag}")])       

    if len(bad_rows):
        for r in bad_rows[FILENAME]:
            log_error(f"check_tags: {r} missing song tag")

    if not(bad_rows.empty) or len(error_list):
        with st.expander('See errors'):
            st.write(error_list)
    else:
        st.write('No tag errors found')
    
    return


# Clean up a pivottable so we can display it as a table
def make_final_pt(site_pt: pd.DataFrame, columns:list, friendly_names:dict) -> pd.DataFrame:
    pt = pd.DataFrame()
    pt_temp = site_pt.transpose()
    #Build the column name mapping (<ugly-tag-name> to 'My tag')
    #While we're at it, copy the columns into a temp DF in the
    #correct order (i.e. same order as the songs).
    col_map = {}
    for col in columns:
        if col in pt_temp:
            col_map[col] = friendly_names[col]
            pt_temp[col] = pt_temp[col].astype(int)
            pt = pd.concat([pt, pt_temp[col]], axis=1)

    #rename the columns
    pt.rename(columns=col_map, inplace=True)
    return pt


# Retrieve the start and end dates for the analysis from the sidebar and format them appropriately
def get_first_and_last_dates(pt_site: pd.DataFrame) -> dict:
    pt_site = pt_site.transpose()
    output = {}
    for song in song_cols:
        output[song] = {}
        d = pt_site[pt_site[song]>0]
        if d.empty:
            output[song]['First'] = 'n/a'
            output[song]['First count'] = '0'
        else:
            output[song]['First'] = d.index[0].strftime('%x')
            output[song]['First count'] = str(d.iloc[0][song])
    
    pt_site.sort_index(ascending=False, inplace=True)
    for song in song_cols:
        d = pt_site[pt_site[song]>0]
        if d.empty:
            output[song]['Last'] = 'n/a'
            output[song]['Last count'] = '0'
        else:
            output[song]['Last'] = d.index[0].strftime('%x')
            output[song]['Last count'] = str(d.iloc[0][song])
    return output






# Function to draw hatching manually
def liang_barsky_clip(x1, y1, x2, y2, rx, ry, rwidth, rheight):
    """
    Liang–Barsky line clipping algorithm
    Clips the line segment (x1,y1)-(x2,y2) against the rectangle defined by
    rx, ry, rwidth, rheight.
    Returns (Xc1, Yc1, Xc2, Yc2) for the clipped line or None if fully outside.
    """
    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0
    x_min, y_min = rx, ry
    x_max, y_max = rx + rwidth, ry + rheight

    def clip(p, q):
        nonlocal t0, t1
        if p == 0:
            if q < 0:
                return False
            # else no update
        else:
            r = q/p
            if p < 0:
                if r > t1:
                    return False
                elif r > t0:
                    t0 = r
            else: # p > 0
                if r < t0:
                    return False
                elif r < t1:
                    t1 = r
        return True

    # Left boundary
    if not clip(-dx, x1 - x_min):
        return None
    # Right boundary
    if not clip(dx, x_max - x1):
        return None
    # Bottom boundary
    if not clip(-dy, y1 - y_min):
        return None
    # Top boundary
    if not clip(dy, y_max - y1): 
        return None

    if t1 < t0:
        return None

    Xc1, Yc1 = x1 + t0*dx, y1 + t0*dy
    Xc2, Yc2 = x1 + t1*dx, y1 + t1*dy
    return (Xc1, Yc1, Xc2, Yc2)

def transform_points(xs, ys, angle, aspect_ratio):
    theta = math.radians(angle)
    Xs = xs
    Ys = ys * aspect_ratio
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    Xr = Xs*cos_t - Ys*sin_t
    Yr = Xs*sin_t + Ys*cos_t
    return Xr, Yr

def inverse_transform_points(Xr, Yr, angle, aspect_ratio):
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    Xs = Xr*cos_t - Yr*sin_t
    Ys = Xr*sin_t + Yr*cos_t
    Xo = Xs
    Yo = Ys / aspect_ratio
    return Xo, Yo

def create_local_grid(spacing, width, height):
    # Create lines with (0,0) at upper-left corner
    lines = []
    max_dim = max(width, height) + spacing*10
    x_min, x_max = -max_dim, max_dim
    y_min, y_max = -max_dim, max_dim

    # # Vertical lines
    # xv_start = math.floor(x_min/spacing)*spacing
    # xv_end = math.ceil(x_max/spacing)*spacing
    # for xv in np.arange(xv_start, xv_end+spacing, spacing):
    #     lines.append(((xv, y_min), (xv, y_max)))

    # Horizontal lines
    yv_start = math.floor(y_min/spacing)*spacing
    yv_end = math.ceil(y_max/spacing)*spacing
    for yv in np.arange(yv_start, yv_end+spacing, spacing):
        lines.append(((x_min, yv), (x_max, yv)))

    return lines

def add_pattern(ax, x, y, width, height, spacing, angle_h=0, angle_v=90, aspect_ratio=1, color="black", linewidth=1):
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # aspect_ratio = (ylim[1]-ylim[0])/(xlim[1]-xlim[0])

    # Upper-left corner of cell
    cell_ul_x, cell_ul_y = x, y+height
    local_lines = create_local_grid(spacing, width, height)

    def generate_lines_for_angle(angle):
        transformed_lines = []
        theta = math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        for (Xl1, Yl1), (Xl2, Yl2) in local_lines:
            # Translate local coords (UL corner as origin)
            # Upper-left corner = (0,0)
            # Rotate around UL corner by angle
            Xr1 = Xl1*cos_t - Yl1*sin_t
            Yr1 = Xl1*sin_t + Yl1*cos_t
            Xr2 = Xl2*cos_t - Yl2*sin_t
            Yr2 = Xl2*sin_t + Yl2*cos_t

            # Translate back
            Xf1 = cell_ul_x + Xr1
            Yf1 = cell_ul_y + Yr1
            Xf2 = cell_ul_x + Xr2
            Yf2 = cell_ul_y + Yr2

            clipped = liang_barsky_clip(Xf1, Yf1, Xf2, Yf2, x, y, width, height)
            if clipped:
                Xc1, Yc1, Xc2, Yc2 = clipped
                transformed_lines.append(Line2D([Xc1, Xc2],[Yc1, Yc2], color=color, linewidth=linewidth, zorder=2))
        return transformed_lines

    lines_h = generate_lines_for_angle(angle_h)
    lines_v = []#generate_lines_for_angle(angle_v)

    # Add lines to the axis
    for ln in lines_h + lines_v:
        ax.add_line(ln)


def make_one_row_pm_summary(df: pd.DataFrame):
    #Trial of new graphing approach

    phases = ["Male", "Female", "Hatch", "Fledge"]
    phase_colors = {
        phases[0] : "#ED7D31",
        phases[1] : "#7030A0",
        phases[2] : "#58A1CF",
        phases[3] : "#08519C"
    }

    male = df.loc["Male Chorus"]
    female = df.loc["Female"]
    hatch = df.loc[["Hatchling", "Nestling"]].sum()
    fledge = df.loc["Fledgling"]

    data = pd.DataFrame({
        phases[0] : male,
        phases[1] : female,
        phases[2] : hatch,
        phases[3] : fledge
    }).T

    #Use the default width and double the height since we're graphing so much data
    fig, ax = plt.subplots(figsize=(fig_w, fig_h/2))
    aspect_ratio = fig_w / (fig_h/2)

    #We're not using this so commenting it out now
    for date_idx, date in enumerate(data):
        min_value = 2
        total_phases = sum(data[date] > min_value)
        if total_phases > 0:
            # Compute rectangle width (split evenly based on total phases for the day)
            width_per_phase = 1 / total_phases
            x_start = date_idx

            angle_counter=0
            # Loop through phases
            for phase_idx, phase in enumerate(phases):
                value = data[date][phase]
                if value > min_value: #If there is something worth graphing
                    if total_phases == 1 or (total_phases > 1 and angle_counter==0):
                        # If there's just one phase or there's more than 1 phase, we draw a solid rect
                        ax.add_patch(Rectangle(
                            (date_idx, 0),       # Bottom-left corner
                            1,                   # Full width
    #                        (x_start, 0),       # Bottom-left corner
    #                        width_per_phase,    # Width
                            1,                  # Full height
                            facecolor=phase_colors[phase],
                            linewidth=0,
                            edgecolor="none", #phase_colors[phase],
                            alpha=1,
                            zorder=1,
                        ))
                        x_start += width_per_phase  # Increment y_start for the next phase
                        angle_counter+=1
                    else: #multiple phases so need to overlap them
    #                     ax.add_patch(Rectangle(
    #                         (date_idx, 0),       # Bottom-left corner
    #                         1,                   # Full width
    # #                        (x_start, 0),       # Bottom-left corner
    # #                        width_per_phase,    # Width
    #                         1,                  # Full height
    #                         facecolor="none", #phase_colors[phase],
    #                         edgecolor="none", #phase_colors[phase],
    #                         alpha=1,
    #                         zorder=1,
    #                     ))

                        # Draw pattern with horizontal (0°) and vertical (90°) lines
                        if phase_idx>=1 and date_idx>=0:
                            angles = [90,0,15,165]
                            if angles[angle_counter] == 0: #more horizontal lines
                                spacing = 0.1
                            else:
                                spacing = 0.1
                            #In the case where we have exactly 2 phases, make the horizontals wider
                            if total_phases == 2:
                                width = 1.2
                            else:
                                width= 0.6
                            add_pattern(ax, date_idx, 0, 1, 1, 
                                        angle_h=angles[angle_counter], angle_v=angles[angle_counter], 
                                        color=phase_colors[phase],
                                        spacing=spacing, linewidth=width, 
                                        aspect_ratio = aspect_ratio)
                            angle_counter+=1

                        # # Add custom hatching for this rectangle
                        # hatch_lines = add_rotated_hatching(
                        #     x=x_start,
                        #     y=0,
                        #     width=1,
                        #     height=1,
                        #     spacing=0.25,  # Control spacing here
                        #     linewidth=0.5,
                        #     angle=45*phase_idx,
                        #     color=phase_colors[phase],
                        #     aspect_ratio=aspect_ratio
                        # )
                        # for line in hatch_lines:
                        #     ax.add_line(line)
        #Code to add a divider line after each day
        # ax.add_line(Line2D(
        #             [date_idx, date_idx], # Bottom-left corner
        #             [0, 1],             # Full height
        #             color="black",
        #             linewidth=0.25,
        # ))
    # Configure plot
    x_axis_len = len(data.columns)
    ax.set_xlim(0, x_axis_len)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(x_axis_len))
    #ax.grid(axis='x', color='black', linewidth=0.25)  # Grid lines as borders

    ax.set_yticks([])
    ax.set_title(" Phase Frequencies by Date", loc='left', fontsize=8, ha='left')
    #exterior border
    ax.add_patch(Rectangle((0, 0), x_axis_len, 1, edgecolor="black", facecolor="none", linewidth=0.75, zorder=99))
    st.write(fig)

    return


# ===========================================================================================================
# ===========================================================================================================
#
#  Main
#
# ===========================================================================================================
# ===========================================================================================================

init_logging()

# Set up the sidebar with three zones so it looks like we want
container_top = st.sidebar.container()
container_mid = st.sidebar.container(border=True)
container_bottom = st.sidebar.container(border=True)

with container_mid:
    show_station_info_checkbox = st.checkbox('Show station info', value=True)
    show_weather_checkbox = st.checkbox('Show station weather', value=True)
    show_PM_dates = st.checkbox('Graph derived pulse dates', value=False)

with container_bottom:
    st.write("Contact wendy.schackwitz@gmail.com with any questions")
    if not being_deployed_to_streamlit:
        make_all_graphs = st.checkbox('Make all graphs')
    else:
        make_all_graphs = False

container_top.title('TRBL Graphs')


#Load all the data for most of the graphs
df_original = load_data()

#Get the list of sites that we're going to do reports for, and then remove all the other data
site_list = get_target_sites()
df = clean_data(df_original, site_list)

# Nuke the original data, hopefully this frees up memory
del df_original
gc.collect()

# Load all the summary data
summary_df = load_summary_data()

save_files = False
save_composite = False

# If we're doing all the graphs, then set our target to the entire list, else use the UI to pick
if make_all_graphs:
    target_sites = site_list
    #Can use this to limit sites to just a particular year
    #target_sites = [string for string in target_sites if string.startswith("2024 ")]

    # This is the file where we write all the dates we extracted from the data
    if os.path.exists(files[DATES_FILE]):
        os.remove(files[DATES_FILE])
    
    # For now, I'm not saving all the files, only the composite because it's taking up too much space. 
    # When she needs all the files, we'll bring this back
    # Make sure to fix it here and in the Else statement below
    save_files = False
    save_composite = True

else:
    target_sites = [get_site_to_analyze(site_list, container_top)]
    if not being_deployed_to_streamlit:
        save_files = True
        save_composite = container_top.checkbox('Save as picture', value=True) #user decides to save the graphs as pics or not
    
    #debug: to get a specific site, put the name of the site below and uncomment
    #target_sites = ["2023 Hale Road"]

# Set format shared by all graphs
set_global_theme()

if profiling:
    profiler = Profiler()
    profiler.start()

site_counter = 0
for site in target_sites:
    error_msgs = []
    site_counter += 1
    # Select the site matching the one of interest
    df_site = df[df[data_col[SITE]] == site]
    date_range_dict = {}
    pt_manual = pd.DataFrame()
    pt_mini_manual = pd.DataFrame()
    pt_edge = pd.DataFrame()

    if not df_site.empty:
        #Using the site of interest, get the first & last dates and give the user the option to customize the range
        date_range_dict = get_date_range(df_site, make_all_graphs, container_top)

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
        df_manual = filter_df_by_tags(df_site, manual_cols)
        pt_manual = make_pivot_table(df_manual,  date_range_dict, labels=song_cols)

        # MINI-MANUAL ANALYSIS
        # 1. Select all rows with one of the following tags:
        #       tag<reviewed-MH-h>, tag<reviewed-MH-m>, tag<reviewed-WS-h>, tag<reviewed-WS-m>
        # 2. Make a pivot table as above
        #   
        df_mini_manual = filter_df_by_tags(df_site, mini_manual_cols)
        pt_mini_manual = make_pivot_table(df_mini_manual, date_range_dict, labels=song_cols)

        # EDGE ANALYSIS
        # Goal: 
        #   1. Draw a rectangle around the outside of all the P_C tags from the first day that has 
        #      at least 1 recording to the last (orange)
        #           1. if the data is null, then the rectangles are going to be drawn as connecting lines
        #   2. Draw a rectangle starting at the first P_N that has at least one recording, and ending
        #      at the latest date that has either a P_A or P_F
        #           1. if there is a YNC_P2 then only count recordings that have YNC_P2, do not count the actual P_N
        #   
        # Steps:
        #   1. Select all rows where one of the following tags
        #       P1C, P1N, P2C, P2N [later: , P3C, P3N]
        #   2. For tags that end in C, make a pivot table with the number of recordings that have CourtshipSong
        #   3. For tags that end in N, make a pivot table that follows more complicated logic, described below
        #   4. Merge all the tables together so we get one block of heatmaps

        have_edge_data = False
        has_ync = 'has_ync'
        ync_tag = 'ync_tag'
        pf_tag = 'pf_tag'
        abandon_tag = 'na_tag'
        sc_tag = 'sc_tag'

        # The dict below captures all the various tags that need to be factored into the the edge
        # analysis assocated with P_N phase:
        #   has_ync: Does this site have any YNC tags of this type?
        #   ync_tag: The name of the YNC Tag column associated with this stage (i.e. the number after P)
        #   na_tag : The name of the P_NA tag associated with this stage
        #   pf_tag : THe name of the P_F tag associated with this stage
        pn_tag_map = {
            data_col[tag_p1n] : {has_ync : False,  #YNC_P1 tag not currently being used
                                ync_tag : '',  #YNC_P1 tag not currently being used
                                abandon_tag  : data_col[tag_p1a], 
                                pf_tag  : data_col[tag_p1f],
            },
            data_col[tag_p2n] : {has_ync : not filter_df_by_tags(df_site, [tag_YNC_p2]).empty, 
                                ync_tag : data_col[tag_YNC_p2],
                                abandon_tag  : "",#data_col[tag_p2a],
                                pf_tag  : data_col[tag_p2f],
            },
    #        data_col[tag_p3n] : {has_ync : not filter_df_by_tags(df_site, [tag_YNC_p3]).empty, 
    #                             ync_tag : data_col[tag_YNC_p3],
    #                             na_tag  : '', #P3NA not currently being used
    #                             pf_tag  : data_col[tag_p3f],
    #        }
        }

        check_edge_cols_for_errors(df_site)

        # 
        # [  P1C  ]
        #            [P1N attaches to either P1A | P1F] 
        #                                [  P2C  ]                   
        #                                           [P2N attaches to either P2A | P2F] 
        # 
        for tag in edge_cols: # tag_p1c, tag_p1n, tag_p2c, tag_p2n, tag_p3c, tag_p3n
            tag_dict = {}

            if tag in edge_c_cols: #P1C, P2C, P3C
                tag_dict[tag] = data_col[COURT_SONG]

            else: #P1N, P2N, P3N
                # For p?n, if there's a YNC_p? then count YNC_p? tags, else count altsong1 
                if pn_tag_map[tag][has_ync]: 
                    #Count YNC tags
                    tag_dict[tag] = pn_tag_map[tag][ync_tag]  #will be tag<YNC-p2> for p2n, tag<YNC-p3> for p3n
                else:
                    #Count altsong1 
                    tag_dict[tag] = data_col[ALTSONG1]

                # For p?na, count altsong1 
                if len(pn_tag_map[tag][abandon_tag]):
                    tag_dict[pn_tag_map[tag][abandon_tag]] = data_col[ALTSONG1]

                # P1F, P2F, P3F: count simplecall2
                tag_dict[pn_tag_map[tag][pf_tag]] = data_col[SIMPLE_CALL2]
    
            # At this point, the dictionary "tag_dict" has a mapping of all the tags and the columns
            # that need to be counted for them. It will either be: 
            #       {"P_C" : "courtsong"}
            # or
            #       {"p_n" : "altsong1" OR "YNC",
            #        "p_a" : "altsong1",
            #        "P_f" : "simplecall2"} 
            # We need to get all the rows that have at least one of those keys, and then count the appropriate song 
            df_for_tag = filter_df_by_tags(df_site, list(tag_dict.keys()))
            have_edge_data = have_edge_data or len(df_for_tag)>0

            # Make_pivot_table takes the dataframe that we've already filtered to the correct tag,
            #    and it further filters it to the columns that have a non-zero value in the target_col
            # "preserve_edges" causes the zero values in the data we pass in to be replaced with -1 
            #    this way, in the graph, we can tell the difference between a day that had no tags vs. one that 
            #    had tags but no songs
            pt_for_tag = make_pivot_table(df_for_tag, date_range_dict, preserve_edges=True, label_dict = tag_dict)
            pt_edge = pd.concat([pt_edge, pt_for_tag])

    else:
        error_msgs.append("Site has no manual annotations")


    #
    # PATTERN MATCHING ANALYSIS
    #
    #Load all the PM files, any errors will return an empty table. For later graphing purposes, 
    df_pattern_match = load_pm_data(site)
    df_pattern_match = clean_data(df_pattern_match, [site])
    pm_data_empty = False
    pt_pm = pd.DataFrame()
    pm_date_range_dict = {}

    if not df_pattern_match.empty:
        #In the scenario where we have PM data but no other data, we need to generate the date range
        if date_range_dict:
            pm_date_range_dict = date_range_dict  
        else:
            #TODO GET THE DATES FROM THE SHEET INSTEAD??? May be irrelevant after we get the new XL files
            pm_date_range_dict = get_date_range(df_pattern_match, make_all_graphs, container_top)

        if len(df_pattern_match):
            for t in pm_file_types: 
                #For each file type, get the filtered range of just that type
                df_for_file_type = df_pattern_match[df_pattern_match['type']==t]
                pm_data_empty = pm_data_empty or len(df_for_file_type)
                #Build the pivot table for it
                pt_for_file_type = make_pattern_match_pt(df_for_file_type, t, pm_date_range_dict)
                #Concat as above
                pt_pm = pd.concat([pt_pm, pt_for_file_type])
    

    else: #TODO Should just graph what we get unless the data is completely missing
        error_msgs.append(f"{site}: All pattern matching data not available, missing some or all files")


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
    # The graphing should just be executing the plan, not figuring out if there are errors.
    # 11/2024 - the graph could be empty, so need to handle that
    if len(summary_row):
        site_summary_dict = process_site_summary_data(summary_row)
    else:
        site_summary_dict = {}


    # ------------------------------------------------------------------------------------------------
    # DISPLAY
    if make_all_graphs:
        st.subheader(f"{site} [{str(site_counter)} of {str(len(target_sites))}]")
    else: 
        st.subheader(site)
    
    if len(error_msgs):
        for error_msg in error_msgs:
            st.write(f":red-background[{error_msg}]")

    if show_station_info_checkbox:
        show_station_info(summary_row)

    #list of month positions in the graphs
    month_locs = {} 

    #Summary graph -- new 3/2024
    #TODO Make a version of this that creates ALL the summary graphs, and only the summary graphs, then 
    #puts them together into one big picture
    include_summary_graph = False
    if (include_summary_graph and 
        len(site_summary_dict) and 
        pd.notna(site_summary_dict[SUMMARY_FIRST_REC]) and 
        pd.notna(site_summary_dict[SUMMARY_LAST_REC])):
        target_date_range_dict = {START:site_summary_dict[SUMMARY_FIRST_REC].strftime('%m-%d-%Y'),
                                  END:  site_summary_dict[SUMMARY_LAST_REC].strftime('%m-%d-%Y')}
        graph = create_summary_graph(pulse_data=site_summary_dict, date_range=target_date_range_dict, make_all_graphs=make_all_graphs)
        output_graph(site, GRAPH_SUMMARY, save_files, make_all_graphs, len(site_summary_dict))
    else:
        pass
        #was:   log_error(f"{site} didn't have any site summary data")

    # Manual analyisis graph
    if not pt_manual.empty:
        graph = create_graph(df = pt_manual, 
                            row_names = song_cols, 
                            cmap = cmap, 
                            title = GRAPH_MANUAL) # add this if we want to include the site name (site + ' ' if save_files else '')
        # Need to be able to build an image that looks like the graph labels so that it can be drawn
        # at the top of the composite. So, try to pull out the month positions for each graph as we don't 
        # know which graph will be non-empty. Once we have them, we don't need to get again (as we don't want)
        # to accidentally delete our list
        if len(month_locs)==0:
            month_locs = get_month_locs_from_graph() 
        output_graph(site, GRAPH_MANUAL, save_files, make_all_graphs, len(df_manual))

    # MiniManual Analysis
    if not pt_mini_manual.empty:
        graph = create_graph(df = pt_mini_manual, 
                            row_names = song_cols, 
                            cmap = cmap, 
                            raw_data = df_site,
                            draw_vert_rects = True,
                            title = 'Mini Manual Analysis')
        if len(month_locs)==0:
            month_locs = get_month_locs_from_graph() 
        output_graph(site, GRAPH_MINIMAN, save_files, make_all_graphs, len(df_mini_manual))

    # Pattern Matching Analysis
    if True: #not pt_pm.empty:
        hatch_dates = {}
        for p in PULSES:
            if p in site_summary_dict:
                hatch_date = site_summary_dict[p]["Brooding"]["start"]
                if pd.notna(hatch_date):
                    hatch_dates[p] = hatch_date

        graph = create_graph(df = pt_pm, 
                            row_names = pm_file_types, 
                            cmap = cmap_pm, 
                            title = GRAPH_PM,
                            hatch_dates = hatch_dates) 
        
        if len(month_locs)==0:
            month_locs = get_month_locs_from_graph() 

        summarized_data, raw_pm_dates = summarize_pm(pt_pm)
        if show_PM_dates:
            add_pulse_overlays(graph, raw_pm_dates, pm_date_range_dict)
                    
        with st.expander("Show pulse dates from Pattern Matching"):
            if make_all_graphs:
                append_to_csv(summarized_data, site, files[DATES_FILE])
            else:
                pretty_print_table(summarized_data, body_alignment="left")


        output_graph(site, GRAPH_PM, save_files, make_all_graphs, pm_data_empty)

        # We're not using this now so commenting it out
        # if not make_all_graphs:
        #     make_one_row_pm_summary(df = pt_pm)

    # Edge Analysis
    if not pt_edge.empty and False:
        cmap_edge = {c:'Oranges' for c in edge_c_cols} | {n:'Blues' for n in edge_n_cols} # the |" is used to merge dicts
        graph = create_graph(df = pt_edge, 
                            row_names = edge_cols,
                            cmap = cmap_edge, 
                            raw_data = df_site,
                            draw_horiz_rects = True,
                            title = GRAPH_EDGE)
        if len(month_locs)==0:
            month_locs = get_month_locs_from_graph() 
        output_graph(site, GRAPH_EDGE, save_files, make_all_graphs, have_edge_data)
    
    #Draw the single legend for the rest of the charts and save to a file if needed
    draw_legend(cmap, make_all_graphs, save_composite)

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
                graph = create_weather_graph(weather_by_type, site)
                output_graph(site, GRAPH_WEATHER, save_files, make_all_graphs)
    
    if not being_deployed_to_streamlit or make_all_graphs or save_composite:
        combine_images(site, month_locs, show_weather_checkbox)

#If site_df is empty, then there were no recordings at all for the site and so we can skip all the summarizing
if not make_all_graphs and len(df_site):
    # Show the table with all the raw data
    with st.expander("See raw data"):
        #Used for making the overview pivot table
        friendly_names = {data_col[MALE_SONG] : 'M-Male', 
                          data_col[COURT_SONG]: 'M-Chorus',
                          data_col[ALTSONG2] : 'M-Female', 
                          data_col[ALTSONG1] : 'M-Nestling'
        }
        overview = []
        overview.append(make_final_pt(pt_manual, song_cols, friendly_names))
        
        friendly_names = {data_col[MALE_SONG] : 'MM-Male', 
                          data_col[COURT_SONG]: 'MM-Chorus',
                          data_col[ALTSONG2] : 'MM-Female', 
                          data_col[ALTSONG1] : 'MM-Nestling'
        }
        overview.append(make_final_pt(pt_mini_manual, song_cols, friendly_names))

        friendly_names =   {data_col[tag_p1c]: 'E-P1C',
                            data_col[tag_p1n]: 'E-P1N',
                            data_col[tag_p2c]: 'E-P2C',
                            data_col[tag_p2n]: 'E-P2N',
#                            data_col[tag_p3c]: 'E-P3C',
#                            data_col[tag_p3n]: 'E-P3N'
        }
        overview.append(make_final_pt(pt_edge, edge_cols, friendly_names))

        #Pattern Matching 
        overview.append(make_final_pt(pt_pm, pm_file_types, pm_friendly_names))

        #Add weather at the end
        if len(weather_by_type):
            weather_data = pd.DataFrame()
            for t in weather_cols:
                weather_data = pd.concat([weather_data, weather_by_type[t]['value']], axis=1)
                weather_data.rename(columns={'value':t}, inplace=True)
                if t != WEATHER_PRCP:
                    weather_data[t] = weather_data[t].astype(int)
            overview.append(weather_data)

        # The variable overview is a list of each dataframe. Now, take all the data and concat it into 
        #a single table
        union_pt = pd.concat(overview, axis=1)

        # Pop the index out so that we can format it, do this by resetting the index so each 
        # row just gets a number index
        union_pt.reset_index(inplace=True)
        union_pt.rename(columns={'index':'Date'}, inplace=True)

        # Format the overview table so it's easy to read and output it 
        # Learn about formatting
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#
        union_pt = union_pt.style.map(style_cells)
    
        #Apply formatting to the weather if the weather data is there
        if len(weather_by_type):
            union_pt.format(formatter={'PRCP':'{:.2f}', 'Date':lambda x:x.strftime('%m-%d-%y')})
        st.dataframe(union_pt)

    # Put a box with first and last dates for the Song columns, with counts on that date
    with st.expander("See overview of dates"):  
        output = get_first_and_last_dates(make_pivot_table(df_site, date_range_dict, labels=song_cols))
        pretty_print_table(pd.DataFrame.from_dict(output))

    # Scan the list of tags and flag any where there is "---" for the value.
    if container_mid.checkbox('Show errors', value=True): 
        check_tags(df_site)

    if st.button('Clear cache'):
        get_target_sites().clear()
        clean_data.clear()
        load_data.clear()
        load_weather_data_from_file.clear()
    
    plt.close("all")

if profiling:
    profiler.stop()
    profiler.print()