import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib as mpl
mpl.use('WebAgg') 

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

#to force garbage collection and reduce memory use
import gc

profiling = True

#Set to true before we deploy
being_deployed_to_streamlit = False

#TODO
# 2022 Foley Ranch A, Edge Analysis, tag p1n is getting the message that there's no data but 
#   also gets the boxes drawn like there is data 
# 
# Consider adding the code to write to temp files and then allow the composite image to be
#   downloaded when the script is deployed to streamlite

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

graph_man = 'Manual Analysis'
graph_miniman = 'Mini Man Analysis'
graph_edge = 'Edge Analysis'
graph_pm = 'Pattern Matching Analysis'
graph_weather = 'Weather'
graph_names = [graph_man, graph_miniman, graph_pm, graph_edge, graph_weather]

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
files = {
    data_file : data_dir / data_file,
    site_info_file : data_dir / site_info_file,
    weather_file : data_dir / weather_file,
    data_old_file : data_dir / data_old_file
}

pm_file_types = ['Male', 'Female', 'Young Nestling', 'Mid Nestling', 'Old Nestling']

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
def my_time():
    return dt.now().strftime('%d-%b-%y %H:%M:%S')

def init_logging():
    if not being_deployed_to_streamlit:
        if os.path.isfile(error_file):
            os.remove(error_file)
        with error_file.open("a") as f:
            f.write(f"Logging started {my_time()}/n")    

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
    file_summary = {}
    for t in pm_file_types:
        file_summary[t] = []
    file_summary[bad_files] = []
    file_summary[site_str] = []

    #Load the list of unique site names, keep just the 'Name' column, and then convert that to a list
    all_site_data = pd.read_csv(files[site_info_file], usecols = ['Name', 'Recordings_Count'])

    #Clean it up. Only keep names that start with a 4-digit number. 
    #TODO More validation to be done?
    all_sites = []
    for s in all_site_data['Name'].tolist():
        if s[0:4].isdigit():
            all_sites.append(s)
    
    #Get a list of all items in the directory, then check whether a site has a matching folder. 
    #If it doesn't, there is no Pattern Matching data available, so go ahead and add it.
    for s in all_sites:
        if not os.path.isdir(data_dir / s):
            file_summary[site_str].append(s)

    #Now, go through all the folders and check them
    top_items = os.scandir(data_dir)
    if any(top_items):
        for item in top_items:
            if item.is_dir():
                #Check that the directory name is in our site list. If yes, continue. If not, then add it to the bad list
                if item.name in all_sites:
                    s = item.name
                    #Get a list of all files in that directory, scan for files that match our pattern
                    if any(os.scandir(item)):
                        #Check that each type of expected file is there:

                        if len(pm_file_types) != count_files_in_folder(item):
                            file_summary[bad_files].append('Wrong number of files: ' + item.name)

                        for t in pm_file_types:
                            found_file = False
                            found_dir_in_subfolder = False
                            sub_items = os.scandir(item)
                            for f in sub_items:
                                empty_dir = False #if the sub_items constructor is empty, we won't get here

                                if f.is_file():
                                    f_type = f.name[len(s)+1:len(f.name)] # Cut off the site name
                                    if t.lower() == f_type[0:len(t)].lower():
                                        file_summary[t].append(f.name)
                                        if s not in file_summary[site_str]: 
                                            file_summary[site_str].append(s)
                                        found_file = True
                                        break
                                else:
                                    if not found_dir_in_subfolder and f.name.lower() != 'old files': # if this is the first time here, then log it
                                        file_summary[bad_files].append('Found subfolder in data folder: ' + s)
                                    found_dir_in_subfolder = True
                    
                            if not found_file and not empty_dir:
                                file_summary[bad_files].append('Missing file: ' + s + ' ' + t)

                    else:
                        file_summary[bad_files].append('Empty folder: ' + item.name)

                    sub_items.close()
                
                else:
                    if item.name.lower() != 'hide' and item.name.lower() != 'old files':
                        file_summary[bad_files].append('Bad folder name: ' + item.name)
            
            else: 
                # If it's not a directory, it's a file. If the file we found isn't one of the exceptions to 
                # our pattern, then mark it as Bad.
                if item.name.lower() not in files.keys():
                    file_summary[bad_files].append(item.name)

    top_items.close()
    
    if len(file_summary[site_str]):
        file_summary[site_str].sort()
    else:
        show_error('No site files found')

#    if len(file_summary[bad_files]):
#        show_error('File errors were found')

    return file_summary

#Used by the two functions that follow to do file format validation
def confirm_columns(target_cols:dict, file_cols:list, file:str) -> bool:
    #TODO Make every place that uses this function handle the scenario where a list of missing columns is returned
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

def check_edge_cols_for_errors(df:pd.DataFrame) -> bool:
    error_found = False

    # For each day, there should be only either P1F or P1N, never both
    tag_errors = df.loc[(df[data_col[tag_p1f]]>=1) & (df[data_col[tag_p1n]]>=1)]

    if len(tag_errors):
        error_found = True
        show_error("Found recordings that have both P1F and P1N tags, see log")
        for f in tag_errors[filename_str]: 
            log_error(f"{f}\tRecording has both P1F and P1N tags")
    
    # If there's a P1C or P2C tag, then the value for CourtshipSong should be zero or 1, any other value is an error and should be flagged 
    # If there's a P1N or P2N tag, then the value for AlternativeSong should be zero or 1, any other value is an error and should be flagged 
    error_case = {data_col[tag_p1c] : data_col[courtsong],
                  data_col[tag_p2c] : data_col[courtsong],
                  data_col[tag_p1n] : data_col[altsong1],
                  data_col[tag_p2n] : data_col[altsong1],
    }
    tag_errors = pd.DataFrame()

    #TODO Any columns that start "val" can have counts greater than 1 or "---", do not flag those as errors 
    for e in error_case:
        tag_errors = pd.concat([tag_errors, df.loc[(df[e] >= 1) & (df[error_case[e]] > 1)]])

    if len(tag_errors):
        error_found = True
        show_error("Found recordings that have song count > 1 for edge analysis, see log")
        for f in tag_errors[filename_str]:
            log_error(f"{f}\tRecording has soung count > 1")

    return error_found 

# Load the main data.csv file into a dataframe, validate that the columns are what we expect
@st.cache_resource
def load_data() -> pd.DataFrame:
    data_csv = Path(__file__).parents[0] / files[data_file]

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

#TODO Is this needed
def set_pm_data_date_range(pm_data:pd.DataFrame, date_range_dict:dict) -> pd.DataFrame:
    df_temp = pm_data

    if len(df_temp):
        pass #anything to do?
    else: # if the file is empty, make an empty table so the graphing code has something to work with
        df_temp[date_str] = pd.date_range(date_range_dict[start_str], date_range_dict[end_str])
        df_temp[validated] = 0
        #pt = pt.reindex(date_range).fillna(0)

    return df_temp

# Load the pattern matching CSV files into a dataframe, validate that the columns are what we expect
# These are the files from all the folders named by site. 
# Note that if there is no data, then there will be an empty file
# However, if there are any missing files then we should return an empty DF
def load_pm_data(site:str) -> pd.DataFrame:
    # For each type of file for this site, try to load the file. 
    # Add a column to indicate which type it is. Then append it to the dataframe we're building.
    df = pd.DataFrame()

    # Add the site name so we look into the appropriate folder
    site_dir = data_dir / site
    if os.path.isdir(site_dir):
        for t in pm_file_types:
            fname = site + ' ' + t + '.csv'
            full_file_name = site_dir / fname
            usecols =[site_columns[site_str], site_columns['year'], site_columns['month'], 
                    site_columns['day'], site_columns[validated]]

            df_temp = pd.DataFrame()
            if is_non_zero_file(full_file_name):
                #Validate that all columns exist
                headers = pd.read_csv(full_file_name, nrows=0).columns.tolist()
                #TODO: what if the number of columns is wrong, just continue and get an exception or somehow fail?
                confirm_columns(site_columns, headers, fname)

                df_temp = pd.read_csv(full_file_name, 
                                      usecols=usecols)
                
                #make a new column that has the date in it
                df_temp[date_str] = df_temp.apply(lambda row: make_date(row), axis=1)
                #df_temp['date'] = pd.to_datetime(df[['year', 'month', 'day']])
                #need to make this the index so it looks like the main dataframe

                df_temp = df_temp.set_index(date_str)
            else:
                #missing file, so log it and break, because if anything is missing then we abandon ship  
                log_error(f"Missing pattern matching file {full_file_name}")
                return pd.DataFrame()
            
            df_temp['type'] = t
            df = pd.concat([df, df_temp])

    return df


#Perform the following operations to clean up the data:
#   - Drop sites that aren't needed, so we're passing around less data
#   - Exclude any data where the year of the data doesn't match the target year
#   - Exclude any data where there aren't recordings on consecutive days  
@st.cache_resource
def clean_data(df: pd.DataFrame, site_list: list) -> pd.DataFrame:
    # Drop sites we don't need
    df_clean = pd.DataFrame()
    for site in site_list:
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
    for s in all_songs:
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
            summary = pd.DataFrame()
            for tag in label_dict:
                temp = filter_df_by_tags(df, [tag])
                # If the value in a column is >=1, count it. To achieve this, the aggfunc below sums up 
                # the number of times that the test 'x>=1' is true
                temp_pt = pd.pivot_table(temp, values = [label_dict[tag]], index = [data_col[date_str]], 
                                    aggfunc = lambda x: (x>=1).sum())
                if len(temp_pt):
                    temp_pt.rename(columns={label_dict[tag]:'temp'}, inplace=True) #rename so that in the merge the cols are added
                    summary = pd.concat([summary, temp_pt]).groupby('date').sum()
            #rename the index so that it's the song name            
            summary.rename(columns={'temp':list(label_dict.keys())[0]}, inplace=True) 
        else:
            #If we were passed a list of labels instead of a dict, then use the same logic to count songs
            summary = pd.pivot_table(df, values = labels, index = [data_col[date_str]], 
                                    aggfunc = lambda x: (x>=1).sum()) 

        if preserve_edges:
            # For every date where there is a tag, make sure that the value is non-zero. Then, when we do the
            # graph later, we'll use this to show where the edges of the analysis were
            summary = summary.replace(to_replace=0, value=preserve_edges_flag)

        return normalize_pt(summary, date_range_dict)
    else:
        return pd.DataFrame()


# Pivot table for pattern matching is a little different
def make_pattern_match_pt(site_df: pd.DataFrame, type_name:str, date_range_dict:dict) -> pd.DataFrame:
    #If the value in 'validated' column is 'Present', count it.
    summary = pd.pivot_table(site_df, values=[site_columns[validated]], index = [data_col[date_str]], 
                              aggfunc = lambda x: (x==present).sum())
    summary = summary.rename(columns={validated:type_name})
    return normalize_pt(summary, date_range_dict)


#
#
# UI and other setup
# 
#  
def get_site_to_analyze(site_list:list, my_sidebar) -> str:
    #debug: to get a specific site, put the name of the site below and uncomment
    #return('2022 Baja 1')

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
def get_date_range(df:pd.DataFrame, graphing_all_sites:bool, my_sidebar) -> dict:
    df.sort_index(inplace=True)

    # Set the default date range to the first and last dates that we have data
    # Assume that the data cleaning code has removed any extraneous dates, such as if data 
    # is mistagged (i.e. data from 2019 shows up in the 2020 site)
    date_range_dict = {start_str : df.index[0].strftime("%m-%d-%Y"), 
                       end_str : df.index[len(df)-1].strftime("%m-%d-%Y")}
    
    if not graphing_all_sites:
        months1 = {'First': '-1', 'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}
        months2 = {'Last': '-1',  'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}
        start_month = my_sidebar.selectbox("Start month", months1.keys(), index=0)
        end_month = my_sidebar.selectbox("End month", months2.keys(), index=0)

        #Update the date range if needed
        site_year = df.index[0].year
        if start_month != 'First':
            date_range_dict[start_str] = '{}-01-{}'.format(months1[start_month], site_year)
        if end_month != 'Last':
            last_day = calendar.monthrange(site_year, int(months2[end_month]))[1]
            date_range_dict[end_str] = '{}-{}-{}'.format(months2[end_month], last_day, site_year)
    
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
                     'font.family':'Arial', #'sans-serif'
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
    filename = 'legend.png'
    figure_path = figure_dir / filename
    plt.savefig(figure_path, dpi='figure', bbox_inches='tight')    


def draw_cmap(cmap:dict, make_all_graphs:bool):
    good_cmap = cmap.copy()
    del good_cmap["bad"]

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig = plt.figure(figsize=(fig_w*0.75, fig_h*0.28), layout='constrained')
    gs = fig.add_gridspec(nrows=1, ncols=len(good_cmap), wspace=0)
    axs = gs.subplots()

    friendly_names={data_col[malesong]:'Male\nSong', 
                    data_col[courtsong]:'Male\nChorus', 
                    data_col[altsong2]:'Female\nSong', 
                    data_col[altsong1]:'Nestling and\nFledgling Call'}

    for ax, call_type in zip(axs, good_cmap):
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[good_cmap[call_type]])

        ax.text(1.02, 0.5, friendly_names[call_type], verticalalignment='center', horizontalalignment='left',
                fontsize=8, transform=ax.transAxes)
        ax.add_patch(Rectangle((0,0), 1, 1, ec='black', fill=False, transform=ax.transAxes))
        
    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
    
#    st.pyplot(fig)
    col1, col3, col5= st.columns([1,4, 1])
    with col3:
        if not make_all_graphs:
            st.pyplot(fig)
    return



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
def draw_axis_labels(month_lengths:dict, axs:np.ndarray, weather_graph = False):
    if weather_graph:
        y = -0.4
    else:
        y = 1.9+(0.25 if len(axs)>4 else 0)

    month_count = len(month_lengths)
    n = 0
    x = axs[len(axs)-1].get_xlim()[0]
    for month in month_lengths:
        # Center the label on the middle of the month, which is the #-days-in-the-month/2
        #   -1 because the count is 1-based but the axis is 0-based, e.g. if there are 12 days
        #   in the month
        center_pt = int((month_lengths[month])/2)
        mid = x + center_pt

        axs[len(axs)-1].text(x=mid, y=y, s=month, fontdict={'fontsize':'small', 'horizontalalignment':'center'})
        x += month_lengths[month]
        if n < month_count:
            for ax in axs:
                ax.axvline(x=x) 

# For ensuring the title in the graph looks the same between weather and data graphs.
# note that if the fontsize is too big, then the title will become the largest thing 
# in the figure which causes the graph to shrink!
def plot_title(title:str):
    plt.suptitle(' '+title, fontsize=10, x=0, horizontalalignment='left')
            
# Create a graph, given a dataframe, list of row names, color map, and friendly names for the rows
def create_graph(df: pd.DataFrame, row_names:list, cmap:dict, draw_connectors=False, raw_data=pd.DataFrame, 
                 draw_vert_rects=False, draw_horiz_rects=False,title='') -> plt.figure:
    if len(df) == 0:
        return

    row_count = len(row_names)
    graph_drawn = []
    
    #distance between top of plot space and chart
    top_gap = 0.8 if title != '' else 1
    #tick_spacing is how many days apart the tick marks are. If set to 0 then it turns off all ticks and labels except for month name
    tick_spacing = 0

    # Create the base figure for the graphs
    fig, axs = plt.subplots(nrows = row_count, ncols = 1,
                            sharex = 'col', 
                            gridspec_kw={'height_ratios': np.repeat(1,row_count), 
                                         'left':0, 'right':1, 'bottom':0, 'top':top_gap,
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
        #x_max = len(df_to_graph.columns)*scale
        #axs[i].imshow(df_to_graph, 
        #                    cmap = cmap[row] if len(cmap) > 1 else cmap[0],
        #                    vmin = 0, vmax = heatmap_max if heatmap_max > 0 else 1,
        #                    interpolation='nearest',
        #                    origin='upper', extent=(0,x_max,0,x_max/fig_w))
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
                c = mpl.colormaps[(cmap[row] if len(cmap) > 1 else cmap[0])](1)
                #c = cm.get_cmap(cmap[row] if len(cmap) > 1 else cmap[0], 1)(1) #This is deprecated, changed it to above to avoid errors
                #for debug
                #c = cm.get_cmap('prism',1)(1)
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
            first = raw_data.index[0]
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
        (0.0, 0.0), 1.0, top_gap, 
        linewidth = 0.5, fill=False, zorder=1000, transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])

    # return the final plotted heatmap
    return fig

#Helper to ensure we make the filename consistently because this is done from multiple places
def make_img_filename(site:str, graph_type:str) ->str:
    return site + ' - ' + graph_type + '.png'

# Save the graphic to a different folder. All file-related options are managed from here.
def save_figure(site:str, graph_type:str, delete_only=False):
    #Do nothing if we're on the server for now
    if being_deployed_to_streamlit:
        return

    filename = make_img_filename(site, graph_type)
    figure_path = figure_dir / filename

    #If the file exists then delete it, so that we make sure a new one is written
    if os.path.isfile(figure_path):
        os.remove(figure_path)
    
    if not delete_only and plt.gcf().get_axes():
        # save the original image
        plt.savefig(figure_path, dpi='figure', bbox_inches='tight')
        
        #Create a different version of the image that we'll use for the compilation
        #plt.suptitle('')  #if we want to remove the titles but I don't think we do

        #Figure out where the labels are. There's probably a way to do this in one call ...
        #maybe check the last axis?
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
        filename = site + ' - ' + graph_type + ' clean.png'
        figure_path = figure_dir / filename
        plt.savefig(figure_path, dpi='figure', bbox_inches='tight')    
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
    font = ImageFont.truetype("arialbd.ttf", size=title_font_size)
    draw.text((width/2,title_height-fudge), site, fill='black', anchor='ms', font=font)

    #Add the months
    margin_left = 27 * scale
    margin_right = 1982 * scale
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
def combine_images(site:str, month_locs:dict):
    #if there are no months, then we didn't have any data to graph so don't make a composite
    if len(month_locs) == 0:
        return
    
    composite = site + ' - composite.png'
    composite_path = figure_dir / composite

    # Get the list of all files that match this site
    site_fig_dict = {}
    figures = os.scandir(figure_dir)
    if any(figures):
        # Get all the figures for this site
        for fig in figures:
            if site in fig.name and 'clean' in fig.name and composite not in fig.name:
                for graph_type in graph_names:
                    #TODO Need to add some error checking to not crash if somehow none of the graph names are present 
                    if graph_type in fig.name and os.path.getsize(fig)>1000:
                        site_fig_dict[graph_type] = fig.path
            elif 'legend' in fig.name:
                legend = fig.path

        # Now have the list of figures.  
        if len(site_fig_dict) > 0:
            # Building the list of the figures but it needs to be in a specific order so that 
            # the composite looks right
            site_fig_list = []
            graph_names.append('legend')
            for g in graph_names:
                if g in site_fig_dict and g != graph_weather:
                    site_fig_list.append(site_fig_dict[g])
            images = [Image.open(f) for f in site_fig_list]
            composite = concat_images(*images)
            composite = concat_images(*[composite, Image.open(legend)], is_legend=True)
            #Add the weather graph only if it exists, to prevent an error if we haven't obtained it yet
            if graph_weather in site_fig_dict.keys():
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
            if graph.get_axes():
                st.write(graph)
            
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

#TODO Use the sites file from here in the weather

#Load weather data from file
@st.cache_resource
def load_weather_data_from_file() -> pd.DataFrame:
    #Validate the data file format
    headers = pd.read_csv(files[weather_file], nrows=0).columns.tolist()
    weather_cols = {'date':'date', 'datatype':'datatype', 'value':'value', 'site':'site', 
                    'lat':'lat', 'lng':'lng', 'alt':'alt'}
    
    #TODO what to do if the number of columns is wrong?
    confirm_columns(weather_cols, headers, weather_file)
    
    df = pd.read_csv(files[weather_file], 
                     parse_dates = [weather_cols['date']],
                     index_col = [weather_cols['site']])
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
        show_error('No weather available for ' + site_name)

    return site_weather_by_type

# add the ticks and associated content for the weather graph
def add_weather_graph_ticks(ax1:plt.axes, ax2:plt.axes, wg_colors:dict, x_range:pd.Series):
    # TODO:
    # 1) Clean up where the x-axis formatting code goes, here or in another function

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
def pretty_print_table(df:pd.DataFrame):
    # Do this so that the original DF doesn't get edited, because of how Python handles parameters 
    output_df = df
    # Formatting note
    # Note that the line that should work to set alignment doesn't if we are outputting
    # the table using st.dataframe or st.table. I can set color this way, but alignment
    # appears to be ignored.
    #  
    # The only way to get it to work is to write the table out using st.write, which is
    # OK for a little table like this but bad for the data table because st.write doesn't
    # give the interactive table where you scroll to see rows and columns, it puts all 
    # the data on the page at once!  
    #
    # So, eventually all this formatting junk should be moved to a functiona and cleaned up.
    # style

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
    output_df=output_df.style.set_properties(**{'text-align': 'center'}).set_table_styles(styles)
    #If there is a Date column then format it correctly
    if 'Date' in output_df.columns:
        output_df.format(formatter={'Date':lambda x:x.strftime('%m-%d-%y')})

    #Currently the centering isn't working. The following does center the text but then we lose
    #the scrolling feature
    #st.write(output_df.to_html(), unsafe_allow_html=True)
    st.table(output_df)

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
    st.write('About this site: [Location in Google Maps]({}), Elevation {} ft, {} Recordings'.
             format(map, site_info['Altitude'], site_info['Recordings_Count']))


# If any tag column has "reviewed" in the title AND the value for a row (a recording) is 1, then 
#    check that all "val" columns have a number. 
#    If any of them have a "---" or not a number then print out the filename of that row.
def check_tags(df: pd.DataFrame):
    bad_rows = pd.DataFrame()                                               
    #Find rows where the columns (ws-m, mh-m) have data, but the song column is missing data
    non_zero_rows = filter_df_by_tags(df, [data_col[tag_mhm], 
                                            data_col[tag_wsm]])
    bad_rows = pd.concat([bad_rows, 
                            filter_df_by_tags(non_zero_rows, song_cols, '=={}'.format(missing_data_flag))])

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
            #This code prints a formatted table of just the missing song tag rows, not currently
            #necessary but here just in case I want to bring this back.
#                if not(bad_rows.empty):
#                    bad_rows.sort_values(by='filename', inplace=True)
#                    #Pull out date so we can format it
#                    bad_rows.reset_index(inplace=True)
#                    bad_rows.rename(columns={'index':'Date'}, inplace=True)
#
#                    pretty_print_table(bad_rows)
                
            st.write(error_list)

    else:
        st.write('No tag errors found')
    
    return


# Clean up a pivottable so we can display it as a table
def make_summary_pt(site_pt: pd.DataFrame, columns:list, friendly_names:dict) -> pd.DataFrame:
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
    #TODO Need to ensure the rows aren't missing
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

#Load all the data for most of the graphs
df_original = load_data()

#Get the list of sites that we're going to do reports for, and then remove all the other data
site_list = get_target_sites()
df = clean_data(df_original, site_list[site_str])

# Nuke the original data, hopefully this frees up memory
del df_original
gc.collect()

container_top = st.sidebar.container()
container_mid = st.sidebar.container(border=True)
container_bottom = st.sidebar.container(border=True)

with container_mid:
    show_station_info_checkbox = st.checkbox('Show station info', value=True)
    show_weather_checkbox = st.checkbox('Show station weather', value=True)

with container_bottom:
    make_all_graphs = st.checkbox('Make all graphs')

container_top.title('TRBL Graphs')

save_files = False

# If we're doing all the graphs, then set our target to the entire list, else use the UI to pick
if make_all_graphs:
    target_sites = site_list[site_str]
else:
    target_sites = [get_site_to_analyze(site_list[site_str], container_top)]
    if not being_deployed_to_streamlit:
        save_files = container_top.checkbox('Save as picture', value=True) #user decides to save the graphs as pics or not

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
        st.write('Site {} has no recordings'.format(site))

    #
    # PATTERN MATCHING ANALYSIS
    #
    #Load all the PM files, any errors will return an empty table. For later graphing purposes, 
    df_pattern_match = load_pm_data(site)
    pm_data_empty = False
    pt_pm = pd.DataFrame()

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
    else:
        st.write("All pattern matching data not available -- missing some or all files")


    # ------------------------------------------------------------------------------------------------
    # DISPLAY
    if make_all_graphs:
        st.subheader(f"{site} [{str(site_counter)} of {str(len(target_sites))}]")
    else: 
        st.subheader(site)

    if show_station_info_checkbox:
        show_station_info(site)

    #list of month positions in the graphs
    month_locs = {} 
    #default color map
    cmap = {data_col[malesong]:'Greens', data_col[courtsong]:'Oranges', data_col[altsong2]:'Purples', data_col[altsong1]:'Blues', 'bad':'Black'}

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
        cmap_pm = {'Male':'Greens', 'Female':'Purples', 'Young Nestling':'Blues', 'Mid Nestling':'Blues', 'Old Nestling':'Blues'}
        graph = create_graph(df = pt_pm, 
                            row_names = pm_file_types, 
                            cmap = cmap_pm, 
                            title = graph_pm) 
        if len(month_locs)==0:
            month_locs = get_month_locs_from_graph() 
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

    #Create a graphic for our legend, only draw it if needed
    draw_cmap(cmap, make_all_graphs)
    if save_files and not make_all_graphs:
        output_cmap()

    #Show weather, as needed and if available
    weather_by_type = {}
    if show_weather_checkbox:
        # If date_range_dict and pm_date_range dict are both defined, they will be the same. However, it's 
        # possible that there is only one of them. 
        if pm_date_range_dict or date_range_dict:
            date_to_use = date_range_dict if date_range_dict else pm_date_range_dict
            # Load and parse weather data
            weather_by_type = get_weather_data(site, date_to_use)
            graph = create_weather_graph(weather_by_type, site)
            output_graph(site, graph_weather, save_files, make_all_graphs)
    
    if not being_deployed_to_streamlit or make_all_graphs or save_files:
        combine_images(site, month_locs)
        #TODO clean up by deleting all the files with "clean" in their name?

#If site_df is empty, then there were no recordings at all for the site and so we can skip all the summarizing
if not make_all_graphs and len(df_site):
    # Show the table with all the raw data
    with st.expander("See raw data"):
        #Used for making the summary pivot table
        friendly_names = {data_col[malesong] : 'M-Male', 
                          data_col[courtsong]: 'M-Chorus',
                          data_col[altsong2] : 'M-Female', 
                          data_col[altsong1] : 'M-Nestling'
        }
        summary = []
        summary.append(make_summary_pt(pt_manual, song_cols, friendly_names))
        
        friendly_names = {data_col[malesong] : 'MM-Male', 
                          data_col[courtsong]: 'MM-Chorus',
                          data_col[altsong2] : 'MM-Female', 
                          data_col[altsong1] : 'MM-Nestling'
        }
        summary.append(make_summary_pt(pt_mini_manual, song_cols, friendly_names))

        friendly_names =   {data_col[tag_p1c]: 'E-P1C',
                            data_col[tag_p1n]: 'E-P1N',
                            data_col[tag_p2c]: 'E-P2C',
                            data_col[tag_p2n]: 'E-P2N',
#                            data_col[tag_p3c]: 'E-P3C',
#                            data_col[tag_p3n]: 'E-P3N'
        }
        summary.append(make_summary_pt(pt_edge, edge_cols, friendly_names))

        friendly_names =   {pm_file_types[0]: 'PM-M',
                            pm_file_types[1]: 'PM-F',
                            pm_file_types[2]: 'PM-YN',
                            pm_file_types[3]: 'PM-MN',
                            pm_file_types[4]: 'PM-ON'
        }
        summary.append(make_summary_pt(pt_pm, pm_file_types, friendly_names))

        #Add weather at the end
        if len(weather_by_type):
            weather_data = pd.DataFrame()
            for t in weather_cols:
                weather_data = pd.concat([weather_data, weather_by_type[t]['value']], axis=1)
                weather_data.rename(columns={'value':t}, inplace=True)
                if t != weather_prcp:
                    weather_data[t] = weather_data[t].astype(int)
            summary.append(weather_data)

        # The variable Summary is a list of each dataframe. Now, take all the data and concat it into 
        #a single table
        union_pt = pd.concat(summary, axis=1)

        # Pop the index out so that we can format it, do this by resetting the index so each 
        # row just gets a number index
        union_pt.reset_index(inplace=True)
        union_pt.rename(columns={'index':'Date'}, inplace=True)

        # Format the summary table so it's easy to read and output it 
        # Learn about formatting
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#
        union_pt = union_pt.style.map(style_cells)
    
        #Apply formatting to the weather if the weather data is there
        if len(weather_by_type):
            union_pt.format(formatter={'PRCP':'{:.2f}', 'Date':lambda x:x.strftime('%m-%d-%y')})
        st.dataframe(union_pt)

    # Put a box with first and last dates for the Song columns, with counts on that date
    with st.expander("See summary of first and last dates"):  
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

if profiling:
    profiler.stop()
    profiler.print()