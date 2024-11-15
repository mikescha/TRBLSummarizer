import streamlit as st
import pandas as pd
import numpy as np
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

#Set to true before we deploy
being_deployed_to_streamlit = True


# Constants and Globals
#
#
bad_files = 'bad'
filename_str = 'filename'
site_str = 'site'
date_str = 'date'
hour_str = 'hour'
tag_wse = 'tag_edge'
tag_wsm = 'tag_wsm'
tag_wsh = 'tag_wsh'
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
malesong = 'malesong'
altsong2 = 'altsong2'
altsong1 = 'altsong1'
courtsong = 'courtsong'
simplecall2 = 'simplecall2'

present = 'present'

start_str = 'start'
end_str = 'end'

# Master list of all the columns I need. If columns get added/removed then this needs to update
# The dictionary values MUST map to what's in the data file. 
data_col = {
    filename_str : 'filename', 
    site_str     : 'site', 
    'day'        : 'day',
    'month'      : 'month',
    'year'       : 'year',
    hour_str     : 'hour', 
    date_str     : 'date',
    tag_YNC_p2   : 'tag<YNC-p2>', #Young nestling call pulse 2
#    tag_YNC_p3   : 'tag<YNC-p3>', #Young nestling call pulse 3
    tag_p1c      : 'tag<p1c>',
    tag_p1f      : 'tag<p1f>',
    tag_p1n      : 'tag<p1n>',
    tag_p1a      : 'tag<p1a>',
    tag_p2c      : 'tag<p2c>',
    tag_p2f      : 'tag<p2f>',
    tag_p2n      : 'tag<p2n>',
#    tag_p2a      : 'tag<p2a>',
#    tag_p3c      : 'tag<p3c>',
#    tag_p3f      : 'tag<p3f>',
#    tag_p3n      : 'tag<p3n>',
    tag_mhe2     : 'tag<reviewed-MH-e2>', 
    tag_mhe      : 'tag<reviewed-MH-e>',
    tag_mhh      : 'tag<reviewed-MH-h>',
    tag_mhm      : 'tag<reviewed-MH-m>',
    tag_mh       : 'tag<reviewed-MH>',
    tag_wse      : 'tag<reviewed-WS-e>', #WENDY this is in DF but not being used
    tag_wsh      : 'tag<reviewed-WS-h>',
    tag_wsm      : 'tag<reviewed-WS-m>',
    tag_ws       : 'tag<reviewed-WS>',
    tag_         : 'tag<reviewed>',
    malesong     : 'val<Agelaius tricolor/Common Song>',
    altsong1     : 'val<Agelaius tricolor/Alternative Song>',
    altsong2     : 'val<Agelaius tricolor/Alternative Song 2>',
    courtsong    : 'val<Agelaius tricolor/Courtship Song>',
    simplecall2  : 'val<Agelaius tricolor/Simple Call 2>',
}

site_columns = {
    'id'        : 'id',
    'recording' : 'recording',
    site_str    : 'site', 
    'day'       : 'day',
    'month'     : 'month',
    'year'      : 'year',
    hour_str    : 'hour', 
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

songs = [malesong, courtsong, altsong2, altsong1]
song_cols = [data_col[s] for s in songs]
all_songs = [malesong, courtsong, altsong2, altsong1, simplecall2] 
all_song_cols = [data_col[s] for s in all_songs]

manual_tags = [tag_mh, tag_ws, tag_]
mini_manual_tags = [tag_mhh, tag_wsh, tag_mhm, tag_wsm]
#if we get 3rd pulse back, add tag_p3c and tag_p3n to the two lines below
edge_c_tags = [tag_p1c, tag_p2c] #male chorus
edge_n_tags = [tag_p1n, tag_p2n] #nestlings, p1 = pulse 1, p2 = pulse 2
#if we get 3rd pulse back, change line below to this:
#edge_tags = edge_c_tags + edge_n_tags + [tag_YNC_p2, tag_YNC_p3, tag_p1f, tag_p2f, tag_p3f, tag_p2na] 
edge_tags = edge_c_tags + edge_n_tags + [tag_YNC_p2, tag_p1a, tag_p1f, tag_p2f] #, tag_p2a]
edge_tag_map = {
    tag_p1n : [data_col[tag_p1f], data_col[tag_p1a]],
    tag_p2n : [data_col[tag_p2f]]#, data_col[tag_p2a]],
#    tag_p3n : [data_col[tag_p3f]],
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
weather_prcp = 'PRCP'
weather_tmax = 'TMAX'
weather_tmin = 'TMIN'
weather_cols = [weather_prcp, weather_tmax, weather_tmin]

graph_summary = "Summary"
graph_man = 'Manual Analysis'
graph_miniman = 'Mini Man Analysis'
graph_edge = 'Edge Analysis'
graph_pm = 'Pattern Matching Analysis'
graph_weather = 'Weather'
graph_names = [graph_summary, graph_man, graph_miniman, graph_pm, graph_edge, graph_weather]
legend_name = 'legend.png'
legend_text = {graph_summary: ["Settlement", "Incubation", "Brooding", "Fledgling"],
               graph_man: ["Male Song", "Male Chorus", "Female Chatter", "Hatchling/Nestling/Fledgling Call"],
               graph_miniman: ["Male Song", "Male Chorus", "Female Chatter", "Hatchling/Nestling/Fledgling Call", "Fledgling Call"],
               graph_edge: ["Male Chorus", "Hatchling Call"],
               graph_pm: ["Male Song", "Male Chorus", "Female Chatter", "Hatchling/Nestling Call", "Fledgling Call"]
}

#default color map
cmap = {data_col[malesong]:'Greens', 
        data_col[courtsong]:'Oranges', 
        data_col[altsong2]:'Purples', 
        data_col[altsong1]:'Blues', 
        "Fledgling":"Greys"}

cmap_names = {data_col[malesong]:"Male Song",
              data_col[courtsong]:"Male Chorus",
              data_col[altsong2]:"Female Chatter",
              data_col[altsong1]:"Hatchling/Nestling/\nFledgling Call",
              "Fledgling":"Fledgling Call"} 

#color map for pattern matching
cmap_pm = {"Male Song":"Greens", 
           "Male Chorus":"Oranges", 
           "Female":"Purples", 
           "Hatchling":"Blues",
           "Nestling" :"Blues",
           "Fledgling":'Greys',
            }


#Files, paths, etc.
data_foldername = 'Data/'
figure_foldername = 'Figures/'
data_dir = Path(__file__).parents[0] / data_foldername
figure_dir = Path(__file__).parents[0] / figure_foldername
data_file = 'data 2021-2023.csv'
site_info_file = 'sites.csv'
weather_file = 'weather_history.csv'
data_old_file = 'data_old.csv'
error_file = figure_dir / 'error.txt'
summary_file = 'summary.csv'

files = {
    data_file : data_dir / data_file,
    site_info_file : data_dir / site_info_file,
    weather_file : data_dir / weather_file,
    data_old_file : data_dir / data_old_file,
    summary_file : data_dir / summary_file 
}

# Mar 2024: This is the new set of summary data that Wendy created
# Source data is from the Google Sheet
pulse_count = "pulse_count"
abandoned = "Abandoned"
pulses = ["P1", "P2", "P3", "P4"]
summary_first_rec = "First Rec"
summary_last_rec = "Last Rec"
summary_edge_dates = [summary_first_rec, summary_last_rec]
pulse_MC_start = "MC Start"
pulse_MC_end = "MC End"
pulse_hatch = "Hatch"
pulse_first_fldg = "First Fldg Call"
pulse_last_fldg = "Last Fldg Call"
pulse_date_types = [pulse_MC_start, pulse_MC_end, pulse_hatch, "Last FS > 2", pulse_first_fldg, pulse_last_fldg, abandoned]
pulse_numeric_types = ["Inc Length", "Async Score", "Fldg Age"]
summary_date_cols = [p + ' ' + d for p in pulses for d in pulse_date_types]
summary_numeric_cols = [p + ' ' + n for p in pulses for n in pulse_numeric_types]

phase_mcs = "Settlement"
phase_inc = "Incubation"
phase_brd = "Brooding"
phase_flg = "Fledgling"
pulse_phases = {phase_mcs : [pulse_MC_start, pulse_MC_end],
                phase_inc : [pulse_MC_end, pulse_hatch],
                phase_brd : [pulse_hatch, pulse_first_fldg],
                phase_flg : [pulse_first_fldg, pulse_last_fldg]}


#
#Pattern Matching Files
#edit this if we add/remove file types
#Change: Color Map for Pattern Matching, Legend Text, plus File Types. Also, there are some lists
#of column names in summarize_pm() that likely need to change
pm_file_types = ['Male Song',
                 'Male Chorus', 
                 'Female', 
                 'Hatchling', 
                 'Nestling',
                 'Fledgling', 
]
pm_abbreviations = ["PM-MS", "PM-MC", "PM-F", "PM-H", "PM-N", "PM-FL"]
pm_friendly_names = dict(zip(pm_file_types, pm_abbreviations))

first_str = "First"
last_str = "Last"
before_first_str = "Before First"
after_last_str = "After Last"

valid_pm_date_deltas = {pm_file_types[1]:0, #Male Chorus to Female can be 0 days
                        pm_file_types[2]:5, #Female to Hatchling must be at least 5 days
                        pm_file_types[3]:0, #Hatchling to Nestling can be 0 days
                        pm_file_types[4]:3, #Nestling to Fledgling must be at least 3 days
                        pm_file_types[5]:0, #Nestling to Nestling is zero, here to make math easy
                        }

missing_data_flag = -100
preserve_edges_flag = -99

dpi = 300
scale = int(dpi/300)

error_list = ''

#
#
# Helper functions
#
#
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
    s = '{}-{}-{}'.format(row['year'], format(row['month'],'02'), format(row['day'],'02'))
    return np.datetime64(s)

#
#
#File handling and setup
#
#
@st.cache_resource
def get_target_sites() -> dict:
    results = {}
    for t in pm_file_types:
        results[t] = []
    results[bad_files] = []
    results[site_str] = []

    #Load the list of unique site names, keep just the 'Name' column, and then convert that to a list
    all_site_data = pd.read_csv(files[site_info_file], usecols = ['Name', 'Recordings_Count'])

    #Clean it up. Only keep names that start with a 4-digit number. 
    all_sites = []
    for s in all_site_data['Name'].tolist():
        if s[0:4].isdigit():
            all_sites.append(s)
    
    #2/7/24: New approach -- we don't care whether the PM folders have the wrong number of files, 
    #We will deal with it later in the code. So, just go ahead and add everything, and then flag
    #the ones that do have errors
    for s in all_sites:
        results[site_str].append(s)

    #Now, go through all the folders and check them
    top_items = os.scandir(data_dir)
    if any(top_items):
        for item in top_items:
            if item.is_dir():
                #Check that the directory name is in our site list. If yes, continue. If not, then add it to the bad list
                s=item.name
                if s in all_sites:
                    #NOTE: Now that the PM files are downloaded automatically, much of this is no longer necessary. However, it's not 
                    #      a bad thing to have checks for things like subfolders where they shouldn't be. But if this becomes a 
                    #      maintenance issue, then I should be able to just kill everything except checking that the count is right. 
                    
                    #Get a list of all files in that directory, scan for files that match our pattern
                    if any(os.scandir(item)):
                        #Check that each type of expected file is there:
                        if len(pm_file_types) != count_files_in_folder(item):
                            results[bad_files].append('Wrong number of files: ' + item.name)

                        for t in pm_file_types:
                            found_file = False
                            found_dir_in_subfolder = False
                            sub_items = os.scandir(item)
                            for f in sub_items:
                                empty_dir = False #if the sub_items constructor is empty, we won't get here

                                if f.is_file():
                                    f_type = f.name[len(s)+1:len(f.name)] # Cut off the site name
                                    if t.lower() == f_type[0:len(t)].lower():
                                        results[t].append(f.name)
                                        if s not in results[site_str]: 
                                            results[site_str].append(s)
                                        found_file = True
                                        break
                                else:
                                    if not found_dir_in_subfolder and f.name.lower() != 'old files': # if this is the first time here, then log it
                                        results[bad_files].append('Found subfolder in data folder: ' + s)
                                    found_dir_in_subfolder = True
                            sub_items.close()
                    
                            if not found_file and not empty_dir:
                                results[bad_files].append('Missing file: ' + s + ' ' + t)

                    else:
                        results[bad_files].append('Empty folder: ' + item.name)
        
                else:
                    if item.name.lower() != 'hide' and item.name.lower() != 'old files':
                        results[bad_files].append('Bad folder name: ' + item.name)
            
            else: 
                # If it's not a directory, it's a file. If the file we found isn't one of the exceptions to 
                # our pattern, then mark it as Bad.
                if item.name.lower() not in files.keys():
                    results[bad_files].append(item.name)

    top_items.close()
    
    if len(results[site_str]):
        results[site_str].sort()
    else:
        show_error('No site files found')

#    if len(results[bad_files]):
#        show_error('File errors were found')

    return results

#Used by the two functions that follow to do file format validation
def confirm_columns(target_cols:dict, file_cols:list, file:str) -> bool:
    errors_found = []
    if len(target_cols) != len(file_cols):
        show_error('File {} has an unexpected number of columns, {} instead of {}'.
                   format(file, len(file_cols), len(target_cols)))
    for col in target_cols:        
        if  target_cols[col] not in file_cols:
            errors_found.append(target_cols[col])
            show_error('Column {} missing from file {}'.format(target_cols[col], file))
    
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
            log_error(f'Column {col} contains "---"')
            df[col] = df[col].replace(-100, 0)

def check_edge_cols_for_errors(df:pd.DataFrame) -> bool:
    error_found = False

    #Remove any -100 (were "---" in the original file, converted to numbers in the first cleaning pass) and log it, if there are any
    fix_bad_values(df)

    # For each day, there should be only either P1F or P1N, never both
    tag_errors = df.loc[(df[data_col[tag_p1f]]>=1) & (df[data_col[tag_p1n]]>=1)]

    if len(tag_errors):
        error_found = True
        show_error("Found recordings that have both P1F and P1N tags, see log")
        for f in tag_errors[filename_str]: 
            log_error(f"{f}\tRecording has both P1F and P1N tags")

    return error_found 

# Load the main data.csv file into a dataframe, validate that the columns are what we expect
@st.cache_resource
def load_data() -> pd.DataFrame:
    data_csv = files[data_file]

    #Validate the data file format
    headers = pd.read_csv(files[data_file], nrows=0).columns.tolist()
    missing_columns = confirm_columns(data_col, headers, data_file)  

    #The set of columns we want to use are the basic info (filename, site, date), all songs, and all tags
    usecols = [data_col[filename_str], data_col[site_str], data_col[date_str]]
    for song in all_songs:
        usecols.append(data_col[song])
    for tag in all_tags:
        usecols.append(data_col[tag])

    #remove any columns that are missing from the data file, so we don't ask for them as that will cause
    #an exception. Hopefully the rest of the code is robust enough to deal...
    usecols = [item for item in usecols if item not in missing_columns]

    df = pd.read_csv(data_csv, 
                     usecols = usecols,
                     parse_dates = [data_col[date_str]],
                     index_col = [data_col[date_str]])
    return df


# Load the pattern matching CSV files into a dataframe, validate that the columns are what we expect
# These are the files from all the folders named by site. 
# If there is a missing file, we want to have the data for that type of pattern be empty, adding columns with 
# the right headers but empty data for any missing columns. Then make the graphing code robust enough
# to deal with columns with zeros.
def load_pm_data(site:str) -> pd.DataFrame:
    # For each type of file for this site, try to load the file. 
    # Add a column to indicate which type it is. Then append it to the dataframe we're building. We end up with a 
    # table that has the site, date, and type columns with all the PM data in rows below. So, if there were 1000 PM 
    # events for each type, our table would have 5000 rows. 
    df = pd.DataFrame()
    usecols =[site_columns[site_str], site_columns['year'], site_columns['month'], 
            site_columns['day'], site_columns[validated]]

    # Add the site name so we look into the appropriate folder
    site_dir = data_dir / site
    if os.path.isdir(site_dir):
        for t in pm_file_types:
            fname = site + ' ' + t + '.csv'
            full_file_name = site_dir / fname

            df_temp = pd.DataFrame()
            if is_non_zero_file(full_file_name): #should never be non-zero, but just in case...
                #Validate that all columns exist, and abandon ship if we're missing any
                headers = pd.read_csv(full_file_name, nrows=0).columns.tolist()
                missing_columns = confirm_columns(site_columns, headers, fname)
                if len(missing_columns) == 0: 
                    df_temp = pd.read_csv(full_file_name, usecols=usecols)
                    #make a new column that has the date in it, take into account that the table could be empty
                    if len(df_temp):
                        df_temp[date_str] = df_temp.apply(lambda row: make_date(row), axis=1)
                    else:
                        df_temp[date_str] = []
                else:
                    #columns are missing so can't do anything!
                    log_error(f"Columns {missing_columns} are missing from pattern matching file!")
                    return pd.DataFrame()
            else:
                log_error(f"Missing or empty pattern matching file {full_file_name}")
                #Add an empty date column so we don't have a mismatch for the concat
                df_temp[date_str] = []

            #Finally, add the table that we loaded to the end of the main one
            df_temp['type'] = t
            df = pd.concat([df, df_temp], ignore_index=True)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    return df



@st.cache_resource
def load_summary_data() -> pd.DataFrame:
    #Load the summary data and prep it for graphing. 
    #This assumes that all validation (e.g. column names, values, etc.) is done in the script that downloads the csv file
    data_csv = Path(__file__).parents[0] / files[summary_file]

    #Load up the file
    df = pd.read_csv(data_csv)

    #If needed, can convert to date values as below, but it doesn't seem necessary
    #df[date_cols] = df[date_cols].apply(pd.to_datetime, errors='coerce')

    # Convert numeric columns to integers. As above, you have to force it this way if the types vary.
    # Empty values or strings are converted to NaN
    df[summary_numeric_cols] = df[summary_numeric_cols].apply(pd.to_numeric, errors='coerce', downcast='integer')

    # If we want to make those "NaN" or "NaT" into a string we can do this:
    #for d in date_cols:
    #    df[d] = df[d].fillna("ND")

    return df

# clean up the data for a particular site
def is_valid_date_string(date_string):
    try:
        pd.to_datetime(date_string, format="%m/%d/%Y")
        return True
    except ValueError:
        return False

def convert_to_datetime(date_string):
    date_format = "%m/%d/%Y"
    try:
        return pd.to_datetime(date_string, format=date_format)
    except ValueError as e:
        return pd.NaT  # Use pd.NaT for missing or null datetime values

def is_valid_date(timestamp):
    return not pd.isna(timestamp)

def is_valid_date_pair(phase_data:dict) -> bool:
    result = False
    start = phase_data[start_str]
    end = phase_data[end_str]
    if is_valid_date(start) and is_valid_date(end):
        result = True
    return result

def count_valid_pulses(pulse_data:dict) -> int:
    #A pulse is considered valid if there is at least one "graphable" date pair
    count = 0
    for p in pulses:
        result = False
        for phase in pulse_data[p]:
            if phase in pulse_phases.keys(): #Need to skip Abandoned, as it doesn't have a pair of dates
                if is_valid_date_pair(pulse_data[p][phase]):
                    result = True
                    break
        count += 1 if result else 0

    return count

def get_val_from_df(df:pd.DataFrame, col):
    return df.iloc[0,df.columns.get_loc(col)]

def process_site_summary_data(summary_row:pd.DataFrame) -> dict:
    nd_string = "ND"
    # This function takes a row from the summary spreadsheet. The goal is to process it as follows:
    first_rec = get_val_from_df(summary_row, summary_first_rec)
    last_rec = get_val_from_df(summary_row, summary_last_rec)
    summary_dict = {
        summary_first_rec   : convert_to_datetime(first_rec),
        summary_last_rec    : convert_to_datetime(last_rec),
    }

#    abandoned_dates = {}

    for pulse in pulses:
        pulse_result = {}

        #Make our list of abandoned dates for later graphing purposes
        abandoned_date = convert_to_datetime(get_val_from_df(summary_row, f"{pulse} {abandoned}"))
        if is_valid_date(abandoned_date):
            pulse_result[abandoned] = abandoned_date 
#            abandoned_dates[pulse] = abandoned_date

        for phase in pulse_phases:
            start, end = pulse_phases[phase]
            target1 = f"{pulse} {start}"
            value1 = get_val_from_df(summary_row, target1)
            result1 = pd.NaT
            if is_valid_date_string(value1):
                #It's a good date, so format it
                result1 = convert_to_datetime(value1)             
            elif value1 == "abandoned":
                if not is_valid_date(abandoned_date):
                    log_error("A column says Abandoned, but there is not a valid abandoned date")
                else:
                    result1 = pd.NaT
            elif value1 == "before start":
                result1 = summary_dict[summary_first_rec]
            elif value1 == "after end":
                result1 = summary_dict[summary_last_rec]
            elif value1 == nd_string or value1 == "":
                #this is OK, we aren't going to draw anything in this case
                pass
            else:
                #if not one of the above, then it's an error
                log_error("Found invalid data in site summary data")

            target2 = f"{pulse} {end}"
            value2 = get_val_from_df(summary_row, target2)
            result2 = pd.NaT
            if is_valid_date_string(value2):
                #It's a good date, so format it
                if phase == phase_flg:
                    #For fledgling phase, don't subtract one from the end date
                    delta = pd.Timedelta(days=0)
                else:
                    delta = pd.Timedelta(days=1)
                result2 = convert_to_datetime(value2) - delta
            elif value2 == "abandoned":
                if not is_valid_date(abandoned_date):
                    log_error("A column says Abandoned, but there is not a valid abandoned date")
                else:
                    result2 = abandoned_date - pd.Timedelta(days=1)
            elif value2 == "before start":
                #In this scenario, the start should be ND, throw an error if not
                if not value1 == nd_string:
                    log_error("Found case where end date is 'before start' but start date is not 'ND'")
            elif value2 == "after end":
                result2 = summary_dict[summary_last_rec]
            elif value2 == nd_string:
                if not value1 == nd_string:
                    log_error(f"Second date is ND, but first date is not: {target1}:{value1}, {target2}:{value2}") 
            else: #ND, empty, or any other values are not valid here
                log_error("Found invalid data in site summary data")
            
            pulse_result[phase] = {"start":result1, "end":result2}

        #Add the sets of dates to our master dictionary
        summary_dict[pulse] = pulse_result

    #Calculate count of valid pulses. If there were zero, then set the count to 1 else we won't get a graph
    p_count = max(1, count_valid_pulses(summary_dict))
    summary_dict[pulse_count] = p_count

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
        if site_str not in df.columns:
            break

        df_site = df[df[site_str] == site]

        #used to ensure anything outside this year gets dropped
        target_year = site[0:4]

        # Sort newest to oldest (backwards) and filter to this year
        df_site = df_site.sort_index(ascending=False)
        original_size = df_site.shape[0]
        df_site = df_site.query(f"date <= '{target_year}-12-31'")
        if df_site.shape[0] != original_size:
            log_error(f"Data for site {site} has the wrong year in it, newer than its year")

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
        df_site = df_site.query(f"date >= '{target_year}-01-01'")
        if df_site.shape[0] != original_size:
            log_error(f"Data for site {site} has the wrong year in it, older than its year")

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
    date_range = pd.date_range(date_range_dict[start_str], date_range_dict[end_str]) 
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
                temp_pt = pd.pivot_table(temp, values = [label_dict[tag]], index = [data_col[date_str]], 
                                    aggfunc = lambda x: (x>=1).sum())
                if len(temp_pt):
                    temp_pt.rename(columns={label_dict[tag]:'temp'}, inplace=True) #rename so that in the merge the cols are added
                    aggregate_df = pd.concat([aggregate_df, temp_pt]).groupby('date').sum()
            #rename the index so that it's the song name            
            aggregate_df.rename(columns={'temp':list(label_dict.keys())[0]}, inplace=True) 
        else:
            #If we were passed a list of labels instead of a dict, then use the same logic to count songs
            aggregate_df = pd.pivot_table(df, values = labels, index = [data_col[date_str]], 
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
    aggregate = present.pivot_table(index=date_str, values=site_columns[validated], aggfunc='count')
    #aggregate = pd.pivot_table(site_df, values=[site_columns[validated]], index = [data_col[date_str]], 
    #                          aggfunc = lambda x: (x==present).sum())
    aggregate = aggregate.rename(columns={validated:type_name})
    return normalize_pt(aggregate, date_range_dict)



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
    nan_count = 0
    skip_ahead = False
    col = 0
    pulse = 1
    while col < len(row):
        # Go through the entire row
        if looking_for_first:
            if pd.notna(row[col]) and row[col] >= threshold:
                dates[pulse] = {}
                dates[pulse][first_str] = row.index[col]
                #If we're at the very beginning, then we don't actually know when it started, so note this
                dates[pulse][before_first_str] = col == 0

                last_column = col
                looking_for_first = False
        else:
            # We're looking for two consecutive NA or less than threshold
            if pd.notna(row[col]) and row[col] >= threshold:
                last_column = col
                nan_count = 0            
            elif pd.isna(row[col]) or (pd.notna(row[col]) and row[col] < threshold):
                nan_count += 1
                if nan_count > 1:
                    # Found it
                    dates[pulse][last_str] = row.index[last_column]
                    dates[pulse][after_last_str] = False
                    nan_count = 0
                    looking_for_first = True
                    skip_ahead = True # Now that we've found a pair, skip forward by the pulse gap and start over
            else:
                assert True # I don't think it's possible to get here

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
        dates[len(dates)][last_str] = row.index[len(row)-1]
        dates[len(dates)][after_last_str] = True

    return dates


def make_empty_summary_row() -> dict:
    # Create an empty row for a single pulse
    phases = pm_file_types[1:] #Creates a new list except it drops "Male Song"
    base_dict = {}
    for phase in phases:
        base_dict[f"{phase}"] = {}
    return base_dict

def make_empty_summary_dict() -> dict:
    # Create the entire empty summary dict, so we don't get key errors
    base_dict = {1:{}, 2:{}, 3:{}, 4:{}}
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


def find_correct_pulse(target_phase:str, target_date:pd.Timestamp, proposed_pulse:int, current_dates:dict):
    # Check to see if a pulse already has a date for a phase that is later than the current one.
    correct_pulse = proposed_pulse
    all_phases = pm_file_types[1:] #Creates a new list except it drops "Male Song"
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
                    if abs(current_dates[correct_pulse]["Nestling"][first_str] - target_date) <= pd.Timedelta(days=6):
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
            earlier_phase_start = target_pulse[earlier_phase][first_str]
            min_delta = 0 
            start_adding = False
            for item in pm_file_types:
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
                first_dates.append((date[first_str], f"{phases}{pulse}"))
    
    first_dates.sort(key=lambda x: x[0]) ###IS THIS SORTING ENOUGH, IF NEST AND FLEDG ARE ON SAME DATE THEN NEST SHOULD BE FIRST

    previous = None
    for date in first_dates:
        if previous:
            if previous[0] == date[0]:
                previous_pos = pm_file_types[previous[1][:-1]]
                date_pos = pm_file_types[date[1][:-1]]
                assert previous_pos<date_pos, "Sorting needs to be improved"
            previous = date

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
                first_date = format_timestamp(pm_dates[pulse][phase][first_str])
                last_date = format_timestamp(pm_dates[pulse][phase][last_str])
                message = ""

                message += "First: "
                if pm_dates[pulse][phase][before_first_str]:
                    message += f"On or before {first_date}"
                else:
                    message += f"{first_date}"

                message += "<br>"
                message += "Last: "
                if pm_dates[pulse][phase][after_last_str]:
                    message += f"On or after {last_date}"
                else:
                    message += f"{last_date}"

                formatted_dict[pulse_str][phase] = message
            else:
                #Empty key, put an appropriate message for display purposes
                formatted_dict[pulse_str][phase] = "No data"

                
    return formatted_dict


def summarize_pm(pt_pm: pd.DataFrame) -> pd.DataFrame:
    # From pt_pm, get the first date that has a song count >= 4
    threshold = 4
    pulse_gap = 14
    
    #Get all the date pairs
    dates = {}    
    for idx, row in pt_pm.iterrows():
        dates[idx] = find_pm_dates(row, pulse_gap=pulse_gap, threshold=threshold)

    #Sanity check the data
    summary_dict = clean_pm_dates(dates)
    summary_dict = format_pm_dates(summary_dict)
    #Prep the data for display    
    # summary_dict = make_empty_summary_dict()
    # for phase in dates:
    #     if phase != "Male Song":
    #         pass

    # for index, (start, end) in enumerate(zip(dates[::2], dates[1::2]), start=1):
    #     summary_dict[f"Pulse {index}"][f"First {row.name}"] = format_timestamp(start)
    #     summary_dict[f"Pulse {index}"][f"Last {row.name}"] = format_timestamp(end)
    #     if index > max_pulse:
    #         max_pulse = index
    
    #Now format this for display. Make a new table where the "1" becomes "Pulse 1"
    result = pd.DataFrame.from_dict(summary_dict, orient='index')

    return result


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
    date_range_dict_new = {}
    if df.index.name == "date":
        date_range_dict = {start_str : df.index.min().strftime("%m-%d-%Y"), 
                             end_str : df.index.max().strftime("%m-%d-%Y")}
    else:
        date_range_dict = {start_str : df["date"].min().strftime("%m-%d-%Y"), 
                             end_str : df["date"].max().strftime("%m-%d-%Y")}

    # #OLD WAY
    # df.sort_index(inplace=True)

    # #Need to determine whether we are indexed by date or not
    # date_range_dict = {}
    # if hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
    #     date_range_dict = {start_str : df.index[0].strftime("%m-%d-%Y"), 
    #                        end_str : df.index[len(df)-1].strftime("%m-%d-%Y")}
    # else:
    #     #No index, but it should be findable by the date column    
    #     if "date" in df.columns.to_list():
    #         date_range_dict = {start_str : df.iloc[0,df.columns.get_loc('date')].strftime("%m-%d-%Y"), 
    #                            end_str : df.iloc[len(df)-1,df.columns.get_loc('date')].strftime("%m-%d-%Y")}
    
    # if date_range_dict == {}:
    #     log_error("Couldn't find date range!!")

    if not graphing_all_sites:
        months1 = {'First': '-1', 'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}
        months2 = {'Last': '-1',  'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}
        start_month = my_sidebar.selectbox("Start month", months1.keys(), index=0)
        end_month = my_sidebar.selectbox("End month", months2.keys(), index=0)

        #Update the date range if needed
        site_year = int(date_range_dict[start_str][-4:])
        if start_month != 'First':
            date_range_dict[start_str] = f'{months1[start_month]}-01-{site_year}'
        if end_month != 'Last':
            last_day = calendar.monthrange(site_year, int(months2[end_month]))[1]
            date_range_dict[end_str] = f'{months2[end_month]}-{last_day}-{site_year}'

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
    custom_params = {'figure.dpi':dpi, 
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
    figure_path = figure_dir / legend_name
    plt.savefig(figure_path, dpi='figure', bbox_inches='tight')    


def draw_legend(cmap:dict, make_all_graphs:bool, save_files:bool):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    #Create the figure, 25% of the height of the other graphs
    fig = plt.figure(figsize=(fig_w, fig_h*0.25), layout='constrained')
    #Add one grid square for each colormap
    gs = fig.add_gridspec(nrows=1, ncols=len(cmap_names), wspace=0)
    axs = gs.subplots()

    for ax, call_type in zip(axs, cmap_names):
        #Draw the gradient
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[cmap[call_type]])
        #Add a border
        ax.add_patch(Rectangle((0,0), 1, 1, ec='black', fill=False, transform=ax.transAxes))
        #Add the name
        ax.text(1.03, 0.5, cmap_names[call_type], verticalalignment='center', horizontalalignment='left',
                fontsize=6, transform=ax.transAxes)
        
    # Turn off all axes, so we don't draw any ticks, spines, labels, etc.
    for ax in axs:
        ax.set_axis_off()

    if not make_all_graphs:
        st.pyplot(fig)

    if save_files and not make_all_graphs:
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
    pc = pulse_data[pulse_count]
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
        phase_mcs : "seagreen",
        phase_inc : "mediumpurple",
        phase_brd : "steelblue",
        phase_flg : "black",
        "Abandoned" : "red"
    }
    background_color = "white"
    
    nesting_start_date = pulse_data[summary_first_rec]
    nesting_end_date = pulse_data[summary_last_rec]

    days_per_month = month_days_between_dates(date_range[start_str], date_range[end_str])
    start_date = pd.Timestamp(date_range[start_str])
    end_date = pd.Timestamp(date_range[end_str])
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
        if p in pulses:
            for phase in pulse_phases:
                if is_valid_date_pair(pulse_data[p][phase]):
                    #Given that the x axis starts at start_date, we need to calculate everything as an offset from there
                    phase_start = pulse_data[p][phase][start_str]
                    phase_end = pulse_data[p][phase][end_str]
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
            if abandoned in pulse_data[p]:
                abandoned_dict[p] = pulse_data[p][abandoned]
                start_point = (pulse_data[p][abandoned],0)
                width = timedelta(days=1)
                height = 1
                rect = Rectangle(start_point, width, height, 
                            color=phase_color["Abandoned"], alpha=1, zorder=5,
                            label=abandoned)
                axs[row_count].add_patch(rect)
                graphed_something = True

                # if abandoned not in legend_elements:
                #     legend_elements[abandoned] = 'red'

            #Not all pulses will have graphable data, so we only want to change axis if there was something to graph
            if graphed_something:
                row_count += 1

    # Legendary!
    box_height = 0.6 #axis coordinates
    box_buffer = (1 - box_height)/2
    box_width = 0.06
    text_width = 0.135
    gap = 0.005
    width = box_width + gap + text_width #Comes to 0.2, or 20% of width, as we are doing up to 5 labels
    total_legend_width = len(phase_color)*width
    xpos = (1 - total_legend_width)/2 + 0.03 #Give it a little push to the right
    for i, (caption, color) in enumerate(phase_color.items()):
        legend_box = Rectangle(xy=(xpos,box_buffer), width=box_width, height=box_height, facecolor=color, transform=ax.transAxes)
        axs[legend_row].add_patch(legend_box)
        axs[legend_row].text(xpos+box_width+gap, y=0.5, s=caption, transform=ax.transAxes, 
                             fontsize=6, color="black", verticalalignment="center")
        xpos += width

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
        for p in pulses:
            empty_pulse = 1
            for phase in pulse_phases:
                if is_valid_date_pair(pulse_data[p][phase]):
                    #First time only per pulse, add the title
                    if empty_pulse:
                        empty_pulse = 0
                        report += f"-----Pulse {p}-----<br>"

                    phase_start = pulse_data[p][phase][start_str].strftime("%m-%d")
                    phase_end = pulse_data[p][phase][end_str].strftime("%m-%d")
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
                 draw_vert_rects=False, draw_horiz_rects=False,title='') -> plt.figure:
    if len(df) == 0:
        return

    row_count = len(row_names)
    graph_drawn = []
    
    #distance between top of plot space and chart
    gap_for_title = 0.8 if title else 1
    #tick_spacing is how many days apart the tick marks are. If set to 0 then it turns off all ticks and labels except for month name
    tick_spacing = 0

    # Create the base figure for the graphs
    fig, axs = plt.subplots(nrows = row_count, ncols = 1,
                            sharex = 'col', 
                            gridspec_kw={'height_ratios': np.repeat(1,row_count), 
                                         'left':0, 'right':1, 'bottom':0, 'top':gap_for_title,
                                         'hspace':0},  #hspace is row spacing (gap between rows)
                            figsize=(fig_w,fig_h))

    # If we have one, add the title for the graph and set appropriate formatting
    if len(title)>0:
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
            if title == "Edge Analysis" and df.loc[row].lt(0).any():
                pass
            else:
                axs[i].text(0.5,0.5,'No data for ' + row, 
                            fontsize='xx-small', fontstyle='italic', color='gray', verticalalignment='center')

        # Track which graphs we drew, so we can put the proper ticks on later
        graph_drawn.append(i)
            
        # For edge: Add a rectangle around the regions of consective tags, and a line between 
        # non-consectutive if it's a N tag.
        if draw_horiz_rects and row in df_clean.index:
            df_col_nonzero = df.loc[row].to_frame()  #pull out the row we want, it turns into a column as above
            df_col_nonzero = df_col_nonzero.reset_index()   #index by ints for easy graphing
            df_col_nonzero = df_col_nonzero.query('`{}` != 0'.format(row))  #get only the nonzero values. 

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
        plt.savefig(figure_path, dpi='figure', bbox_inches='tight')
        
        #Create a different version of the image that we'll use for the compilation
        #plt.suptitle('')  #if we want to remove the titles but I don't think we do

        #Figure out where the labels are. There's probably a way to do this in one call ...
        #maybe check the last axis?
        if graph_type == graph_summary:
            #for the summary graph, we dont want to do anything i don't think
            pass
        else:
            if graph_type == graph_weather:
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
        images = [Image.open(filename) for graph_type,filename in site_fig_dict.items() if graph_type != graph_weather] 
        composite = concat_images(*images)
        composite = concat_images(*[composite, Image.open(legend)], is_legend=True)
        #Add the weather graph only if it exists, to prevent an error if we haven't obtained it yet
        if graph_weather in site_fig_dict.keys() and include_weather:
            composite = concat_images(*[composite, Image.open(site_fig_dict[graph_weather])])
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
        site_name_text = '<p style="font-family:sans-serif; font-size: 16px;"><b>{}</b></p>'.format(graph_type) #used to also have color:Black; 
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
        headers = pd.read_csv(files[weather_file], nrows=0).columns.tolist()
        weather_cols = {'row':'row','date':'date', 'datatype':'datatype', 'value':'value', 'site':'site', 
                        'lat':'lat', 'lng':'lng', 'alt':'alt'}
        
        #This will show an error if something is wrong with the data 
        missing_columns = confirm_columns(weather_cols, headers, weather_file)
        
        if not missing_columns:
            df = pd.read_csv(files[weather_file], 
                            parse_dates = [weather_cols['date']],
                            index_col = [weather_cols['site']])
        else: #there was an error where we didn't get the right columns so don't try to do anything with the data
            df = pd.DataFrame()

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
        mask = (site_weather['date'] >= date_range_dict[start_str]) & (site_weather['date'] <= date_range_dict[end_str])
        site_weather = site_weather.loc[mask]

        if not site_weather.empty:
            # For each type of weather, break out that type into a separate table and 
            # drop it into a dict. Then, reindex the table to match our date range and 
            # fill in empty values
            date_range = pd.date_range(date_range_dict[start_str], date_range_dict[end_str]) 
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

        plot_title(graph_weather) #site_name + ' ' +  to include site

        # Plot the data in the proper format on the correct axis.
        wg_colors = {'high':'red', 'low':'pink', 'prcp':'blue'}
        for wt in weather_cols:
            w = weather_by_type[wt]
            if wt == weather_prcp:
                ax1.bar(w.index.values, w['value'], color = wg_colors['prcp'], linewidth=0)
            elif wt == weather_tmax:
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
        draw_axis_labels(get_days_per_month(weather_by_type[weather_tmax].index.values), [ax1], weather_graph=True)
        
        #Turn on the graph borders, these are off by default for other charts
        ax1.spines[:].set_linewidth(0.5)
        ax1.spines[:].set_visible(True)

        # Add a legend for the figure
        # For more legend tips see here: https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
        tmax_label = f"High temp ({min_above_zero(weather_by_type[weather_tmax]['value']):.0f}-"\
                     f"{weather_by_type[weather_tmax]['value'].max():.0f}\u00B0F)"
        tmin_label = f"Low temp ({min_above_zero(weather_by_type[weather_tmin]['value']):.0f}-"\
                     f"{weather_by_type[weather_tmin]['value'].max():.0f}\u00B0F)"
        prcp_label = f"Precipitation (0-"\
                     f"{weather_by_type[weather_prcp]['value'].max():.2f}\042)"
        legend_elements = [Line2D([0], [0], color=wg_colors['high'], lw=4, label=tmax_label),
                           Line2D([0], [0], color=wg_colors['low'], lw=4, label=tmin_label),
                           Line2D([0], [0], color=wg_colors['prcp'], lw=4, label=prcp_label)]
        
        #draw the legend below the chart. that's what the bbox_to_anchor with -0.5 does
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3,
                   fontsize='x-small')

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
    df = pd.read_csv(files[site_info_file])
    for f in site_info_fields:
        site_info[f] = df.loc[df['Name'] == site_name,f].values[0]
    return site_info

def show_station_info(site_name:str):
    site_info_fields = ['Latitude', 'Longitude', 'Altitude', 'Recordings_Count']
    site_info = get_site_info(site_name, site_info_fields)

    #We can either open the map to a spot with a pin, or to a view with zoom + map type but no pin. Here's more documentation:
    #https://developers.google.com/maps/documentation/urls/get-started
    map = 'https://www.google.com/maps/search/?api=1&query={}%2C{}'.format(site_info['Latitude'], site_info['Longitude'])
    st.write(f"About this site: [Google Maps Link]({map}), elevation {site_info['Altitude']} ft, {site_info['Recordings_Count']} recordings")

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
                            filter_df_by_tags(non_zero_rows, [data_col[courtsong]], '=={}'.format(missing_data_flag))])

    #P1N, P2N throws an error if it's missing alternative song
    non_zero_rows = filter_df_by_tags(df, edge_n_cols)
    bad_rows = pd.concat([bad_rows, 
                            filter_df_by_tags(non_zero_rows, [data_col[altsong1]], '=={}'.format(missing_data_flag))])       

    if len(bad_rows):
        for r in bad_rows[filename_str]:
            log_error(f"{r} missing song tag")

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


# ===========================================================================================================
# ===========================================================================================================
#
#  Main
#
# ===========================================================================================================
# ===========================================================================================================

init_logging()
error_msgs = []

#Load all the data for most of the graphs
df_original = load_data()

#Get the list of sites that we're going to do reports for, and then remove all the other data
site_list = get_target_sites()
df = clean_data(df_original, site_list[site_str])

# Nuke the original data, hopefully this frees up memory
del df_original
gc.collect()

# Load all the summary data, note that this doesn't exist for 2024 and beyond right now (Nov 2024)
summary_df = load_summary_data()

# Set up the sidebar with three zones so it looks like we want
container_top = st.sidebar.container()
container_mid = st.sidebar.container(border=True)
container_bottom = st.sidebar.container(border=True)

with container_mid:
    show_station_info_checkbox = st.checkbox('Show station info', value=True)
    show_weather_checkbox = st.checkbox('Show station weather', value=True)

with container_bottom:
    if not being_deployed_to_streamlit:
        make_all_graphs = st.checkbox('Make all graphs')
    else:
        make_all_graphs = False

container_top.title('TRBL Graphs')

save_files = False

# If we're doing all the graphs, then set our target to the entire list, else use the UI to pick
if make_all_graphs:
    target_sites = site_list[site_str]
    target_sites = [string for string in target_sites if string.startswith("2024 ")]
else:
    target_sites = [get_site_to_analyze(site_list[site_str], container_top)]
    if not being_deployed_to_streamlit:
        save_files = container_top.checkbox('Save as picture', value=True) #user decides to save the graphs as pics or not

    #debug: to get a specific site, put the name of the site below and uncomment
    #target_sites = ["2023 Hale Road"]

# Set format shared by all graphs
set_global_theme()

if profiling:
    profiler = Profiler()
    profiler.start()

site_counter = 0
for site in target_sites:
    site_counter += 1
    # Select the site matching the one of interest
    df_site = df[df[data_col[site_str]] == site]
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
                tag_dict[tag] = data_col[courtsong]

            else: #P1N, P2N, P3N
                # For p?n, if there's a YNC_p? then count YNC_p? tags, else count altsong1 
                if pn_tag_map[tag][has_ync]: 
                    #Count YNC tags
                    tag_dict[tag] = pn_tag_map[tag][ync_tag]  #will be tag<YNC-p2> for p2n, tag<YNC-p3> for p3n
                else:
                    #Count altsong1 
                    tag_dict[tag] = data_col[altsong1]

                # For p?na, count altsong1 
                if len(pn_tag_map[tag][abandon_tag]):
                    tag_dict[pn_tag_map[tag][abandon_tag]] = data_col[altsong1]

                # P1F, P2F, P3F: count simplecall2
                tag_dict[pn_tag_map[tag][pf_tag]] = data_col[simplecall2]
    
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
    
            #TODO PUT summarize_pm here
    else: #TODO Should just graph what we get unless the data is completely missing
        error_msgs.append("All pattern matching data not available, missing some or all files")


    # 
    #    Summary data
    #
    # What we want to do is break this into a dictionary, with one entry for each pulse. 
    # Each pulse should have a dictionary mapping any valid dates in the pulse to its column name, 
    # e.g. {"P1": {"P1 Inc Start":Timestamp('2023-05-01')}}
    summary_row = summary_df[summary_df.iloc[:, 0] == site]

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
        show_station_info(site)

    #list of month positions in the graphs
    month_locs = {} 

    #Summary graph -- new 3/2024
    #TODO Make a version of this that creates ALL the summary graphs, and only the summary graphs, then 
    #puts them together into one big picture
    if len(site_summary_dict):
        target_date_range_dict = {start_str:site_summary_dict[summary_first_rec].strftime('%m-%d-%Y'),
                                end_str:site_summary_dict[summary_last_rec].strftime('%m-%d-%Y')}
        graph = create_summary_graph(pulse_data=site_summary_dict, date_range=target_date_range_dict, make_all_graphs=make_all_graphs)
        output_graph(site, graph_summary, save_files, make_all_graphs, len(site_summary_dict))

    # Manual analyisis graph
    if not pt_manual.empty:
        graph = create_graph(df = pt_manual, 
                            row_names = song_cols, 
                            cmap = cmap, 
                            title = graph_man) # add this if we want to include the site name (site + ' ' if save_files else '')
        # Need to be able to build an image that looks like the graph labels so that it can be drawn
        # at the top of the composite. So, try to pull out the month positions for each graph as we don't 
        # know which graph will be non-empty. Once we have them, we don't need to get again (as we don't want)
        # to accidentally delete our list
        if len(month_locs)==0:
            month_locs = get_month_locs_from_graph() 
        output_graph(site, graph_man, save_files, make_all_graphs, len(df_manual))

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
        output_graph(site, graph_miniman, save_files, make_all_graphs, len(df_mini_manual))

    # Pattern Matching Analysis
    if not pt_pm.empty:
        graph = create_graph(df = pt_pm, 
                            row_names = pm_file_types, 
                            cmap = cmap_pm, 
                            title = graph_pm) 
        if len(month_locs)==0:
            month_locs = get_month_locs_from_graph() 

        with st.expander("Show pulse dates from Pattern Matching"):
            #st.write("<b>Automatically derived dates:</b>", unsafe_allow_html=True)
            pretty_print_table(summarize_pm(pt_pm), body_alignment="left")

        output_graph(site, graph_pm, save_files, make_all_graphs, pm_data_empty)

    # Edge Analysis
    if not pt_edge.empty:
        cmap_edge = {c:'Oranges' for c in edge_c_cols} | {n:'Blues' for n in edge_n_cols} # the |" is used to merge dicts
        graph = create_graph(df = pt_edge, 
                            row_names = edge_cols,
                            cmap = cmap_edge, 
                            raw_data = df_site,
                            draw_horiz_rects = True,
                            title = graph_edge)
        if len(month_locs)==0:
            month_locs = get_month_locs_from_graph() 
        output_graph(site, graph_edge, save_files, make_all_graphs, have_edge_data)
    
    #Draw the single legend for the rest of the charts and save to a file if needed
    draw_legend(cmap, make_all_graphs, save_files)

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
                output_graph(site, graph_weather, save_files, make_all_graphs)
    
    if not being_deployed_to_streamlit or make_all_graphs or save_files:
        combine_images(site, month_locs, show_weather_checkbox)

#If site_df is empty, then there were no recordings at all for the site and so we can skip all the summarizing
if not make_all_graphs and len(df_site):
    # Show the table with all the raw data
    with st.expander("See raw data"):
        #Used for making the overview pivot table
        friendly_names = {data_col[malesong] : 'M-Male', 
                          data_col[courtsong]: 'M-Chorus',
                          data_col[altsong2] : 'M-Female', 
                          data_col[altsong1] : 'M-Nestling'
        }
        overview = []
        overview.append(make_final_pt(pt_manual, song_cols, friendly_names))
        
        friendly_names = {data_col[malesong] : 'MM-Male', 
                          data_col[courtsong]: 'MM-Chorus',
                          data_col[altsong2] : 'MM-Female', 
                          data_col[altsong1] : 'MM-Nestling'
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
                if t != weather_prcp:
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

    if len(site_list[bad_files]) > 0:
        with st.expander("See possibly bad filenames"):  
            st.write(site_list[bad_files])

    if st.button('Clear cache'):
        get_target_sites().clear()
        clean_data.clear()
        load_data.clear()
        load_weather_data_from_file.clear()

if profiling:
    profiler.stop()
    profiler.print()