import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
from matplotlib import cm
from pathlib import Path
import os
import calendar
from collections import Counter
from itertools import tee
import random
from PIL import Image, ImageDraw, ImageFont

#to force garbage collection and reduce memory use
import gc

#
#
# Constants and Globals
#
#
bad_files = 'bad'
filename_str = 'filename'
site_str = 'site'
malesong = 'malesong'
altsong2 = 'altsong2'
altsong1 = 'altsong1'
courtsong = 'courtsong'
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
tag_p2c = 'tag_p2c'
tag_p2n = 'tag_p2n'
tag_wsmc = 'tag_wsmc'
validated = 'validated'
tag_ONC_p1 = 'tag<ONC-p1>'
tag_YNC_p1 = 'tag<YNC-p1>'
tag_YNC_p2 = 'tag<YNC-p2>'

present = 'present'

start_str = 'start'
end_str = 'end'

#Master list of all the columns I need. If columns get added/removed then this needs to update
data_columns = {
    filename_str : 'filename', 
    site_str     : 'site', 
    'day'        : 'day',
    'month'      : 'month',
    'year'       : 'year',
    hour_str     : 'hour', 
    date_str     : 'date',
    tag_ONC_p1   : 'tag<ONC-p1>', #Older nestling call pulse 1
    "tag_ONC_p2"  : 'tag<YNC-p1>', #REMOVE WHEN WE GET A NEW DATAFILE
    tag_YNC_p2   : 'tag<YNC-p2>', #Young nestling call pulse 2
    tag_p1c      : 'tag<p1c>',
    tag_p1n      : 'tag<p1n>',
    tag_p2c      : 'tag<p2c>',
    tag_p2n      : 'tag<p2n>',
    tag_mhe      : 'tag<reviewed-MH-e>', #WENDY this is still in the data file, should it be?
    tag_mhe2     : 'tag<reviewed-MH-e2>', #WENDY this is still in the data file, should it be?
    tag_mhh      : 'tag<reviewed-MH-h>',
    tag_mhm      : 'tag<reviewed-MH-m>',
    tag_mh       : 'tag<reviewed-MH>',
    tag_wse      : 'tag<reviewed-WS-e>', #WENDY this is in DF but not being used
    tag_wsh      : 'tag<reviewed-WS-h>',
    tag_wsm      : 'tag<reviewed-WS-m>',
    tag_wsmc     : 'tag<reviewed-WS-mc>', #WENDY this is in DF but not being used
    tag_ws       : 'tag<reviewed-WS>',
    tag_         : 'tag<reviewed>',
    malesong     : 'val<Agelaius tricolor/Common Song>',
    altsong1     : 'val<Agelaius tricolor/Alternative Song>',
    altsong2     : 'val<Agelaius tricolor/Alternative Song 2>',
    courtsong    : 'val<Agelaius tricolor/Courtship Song>',
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
song_cols = [data_columns[s] for s in songs]

#TODO When we get a new data file, update all these lists
manual_tags = [tag_mh, tag_ws, tag_]
mini_manual_tags = [tag_mhh, tag_wsh, tag_mhm, tag_wsm]
edge_c_tags = [tag_p1c, tag_p2c] #male chorus
edge_n_tags = [tag_p1n, tag_p2n] #nestlings, p1 = pulse 1, p2 = pulse 2
edge_tags = edge_c_tags + edge_n_tags
all_tags = manual_tags + mini_manual_tags + edge_tags

manual_cols = [data_columns[t] for t in manual_tags]
mini_manual_cols = [data_columns[t] for t in mini_manual_tags]
edge_c_cols = [data_columns[t] for t in edge_c_tags]
edge_n_cols = [data_columns[t] for t in edge_n_tags]
all_tag_cols = manual_cols + mini_manual_cols + edge_c_cols + edge_n_cols

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
data_file = 'data.csv'
site_info_file = 'sites.csv'
weather_file = 'weather_history.csv'
data_old_file = 'data_old.csv'
files = {
    data_file : data_dir / data_file,
    site_info_file : data_dir / site_info_file,
    weather_file : data_dir / weather_file,
    data_old_file : data_dir / data_old_file
}

pm_file_types = ['Male', 'Female', 'Young Nestling', 'Mid Nestling', 'Old Nestling']

missing_data_flag = -100
preserve_edges_flag = -99
#
#
# Helper functions
#
#
def show_error(msg: str):
    st.error("Whoops! " + msg + "! This may not work correctly.")

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
@st.experimental_singleton(suppress_st_warning=True)
def get_target_sites() -> dict:
    file_summary = {}
    for t in pm_file_types:
        file_summary[t] = []
    file_summary[bad_files] = []
    file_summary[site_str] = []

    #Load the list of unique site names, keep just the 'Name' column, and then convert that to a list
    all_site_data = pd.read_csv(files[site_info_file], usecols = ['Name', 'Recordings_Count'])

    #Clean it up. Only keep names that start with a 4-digit number. More validation to be done?
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
                if not(item.name.lower() in files.keys()):
                    file_summary[bad_files].append(item.name)

    top_items.close()
    
    if len(file_summary[site_str]):
        file_summary[site_str].sort()
    else:
        show_error('No site files found')

    if len(file_summary[bad_files]):
        show_error('File errors were found')

    return file_summary

#Used by the two functions that follow to do file format validation
def confirm_columns(target_cols:dict, file_cols:list, file:str) -> bool:
    error_found = False
    if len(target_cols) != len(file_cols):
        error_found = True
        show_error('File {} has an unexpected number of columns, {} instead of {}'.
                   format(file, len(file_cols), len(target_cols)))
    for col in target_cols:
        error_found = True
        if not target_cols[col] in file_cols:
            show_error('Column {} missing from file {}'.format(target_cols[col], file))
    
    return error_found

# Load the main data.csv file into a dataframe, validate that the columns are what we expect
@st.experimental_singleton(suppress_st_warning=True)
def load_data() -> pd.DataFrame:
    data_csv = Path(__file__).parents[0] / files[data_file]

    #Validate the data file format
    headers = pd.read_csv(files[data_file], nrows=0).columns.tolist()
    confirm_columns(data_columns, headers, data_file)
    
    #The set of columns we want to use are the basic info (filename, site, date), all songs, and all tags
    usecols = [data_columns[filename_str], data_columns[site_str], data_columns[date_str]]
    for song in songs:
        usecols.append(data_columns[song])
    for tag in all_tags:
        usecols.append(data_columns[tag])

    df = pd.read_csv(data_csv, 
                     usecols = usecols,
                     parse_dates = [data_columns[date_str]],
                     index_col = [data_columns[date_str]])
    return df

# Load the pattern matching CSV files into a dataframe, validate that the columns are what we expect
# These are the files from all the folders named by site. 
# Note that if there is no data, then there will be an empty file
# However, if there any missing files then we should return an empty DF
def load_pm_data(site:str, date_range_dict:dict) -> pd.DataFrame:

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

                df_temp = pd.read_csv(full_file_name, usecols=usecols)
                df_temp[date_str] = df_temp.apply(lambda row: make_date(row), axis=1)

            else: # if the file is empty, make an empty table so the graphing code has something to work with
                df_temp[date_str] = pd.date_range(date_range_dict[start_str], date_range_dict[end_str])
                df_temp[validated] = 0
                #pt = pt.reindex(date_range).fillna(0)

            df_temp['type'] = t
            df = pd.concat([df, df_temp])

    return df


#Perform the following operations to clean up the data:
#   - Drop sites that aren't needed, so we're passing around less data
#   - Exclude any data where the year of the data doesn't match the target year
#   - Exclude any data where there aren't recordings on consecutive days  
@st.experimental_singleton(suppress_st_warning=True)
def clean_data(df: pd.DataFrame, site_list: list) -> pd.DataFrame:
    # Drop sites we don't need
    df_clean = pd.DataFrame()
    for site in site_list:
        df_site = df[df[site_str] == site]

        #Sort descending, find first two consecutive items and drop everything after.
        df_site = df_site.sort_index(ascending=False)
        dates = df_site.index.unique()
        for x,y in pairwise(dates):
            if abs((x-y).days) == 1:
                #found a match, need to keep only what's after this
                df_site = df_site.query("date <= '{}'".format(x.strftime('%Y-%m-%d')))
                break

        #Sort ascending, find first two consecutive items and drop everything before
        df_site = df_site.sort_index(ascending=True)
        dates = df_site.index.unique()
        for x,y in pairwise(dates):
            if abs((x-y).days) == 1:
                #found a match, need to keep only what's after this
                df_site = df_site.query("date >= '{}'".format(x.strftime('%Y-%m-%d')))
                break

        df_clean = pd.concat([df_clean, df_site])
    
    # We need to preserve the diff between no data and 0 tags. But, we have to also make everything 
    # integers for later processing. So, we'll replace the hyphens with a special value and then just 
    # realize that we can't do math on this column any more without excluding it. Picked -100 because 
    # if we do do math then the answer will be obviously wrong!
    df_clean = df_clean.replace('---', missing_data_flag)
    
    # For each type of song, convert its column to be numeric instead of a string so we can run pivots
    for s in songs:
        df_clean[data_columns[s]] = pd.to_numeric(df_clean[data_columns[s]])
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
def filter_df_by_tags(site_df:pd.DataFrame, target_tags:list, filter_str='>0') -> pd.DataFrame:
    # This is an alternative to: tagged_rows = site_df[((site_df[columns[tag_wse]]>0) | (site_df[columns[tag_mhh]]>0) ...
    query = ' | '.join([f'`{tag}`{filter_str}' for tag in target_tags])
    filtered_df = site_df.query(query)
    return filtered_df

# Add missing dates by creating the largest date range for our graph and then reindex to add missing entries
# Also, transpose to get the right shape for the graph
def normalize_pt(pt:pd.DataFrame, date_range_dict:dict) -> pd.DataFrame:
    date_range = pd.date_range(date_range_dict[start_str], date_range_dict[end_str]) 
    pt = pt.reindex(date_range).fillna(0)
    pt = pt.transpose()
    return pt

# Generate the pivot table for the site
def make_pivot_table(site_df: pd.DataFrame, labels:list, date_range_dict:dict, preserve_edges=False) -> pd.DataFrame:
    #If the value in a column is >=1, count it. To achieve this, the aggfunc below sums up the number of times 
    #that the test 'x>=1' is true.
    summary = pd.pivot_table(site_df, values = labels, index = [data_columns[date_str]], 
                              aggfunc = lambda x: (x>=1).sum()) 

    if preserve_edges:
        # For every date where there is a tag, make sure that the value is non-zero. Then, when we do the
        # graph later, we'll use this to show where the edges of the analysis were
        summary = summary.replace(to_replace=0, value=preserve_edges_flag)

    return normalize_pt(summary, date_range_dict)


def make_pattern_match_pt(site_df: pd.DataFrame, type_name:str, date_range_dict:dict) -> pd.DataFrame:
    #If the value in 'validated' column is 'Present', count it.
    summary = pd.pivot_table(site_df, values=[site_columns[validated]], index = [data_columns[date_str]], 
                              aggfunc = lambda x: (x==present).sum())
    summary = summary.rename(columns={validated:type_name})
    return normalize_pt(summary, date_range_dict)


#
#
# UI and other setup
# 
#  
def get_site_to_analyze(site_list:list) -> str:
    #debug: to get a specific site, put the name of the site below and uncomment
    return('2019 Rush Ranch')

    #Calculate the list of years, sort it backwards so most recent is at the top
    year_list = []
    for s in site_list:
        if s[0:4] not in year_list:
            year_list.append(s[0:4])
    year_list.sort(reverse=True)

    target_year = st.sidebar.selectbox('Site year', year_list)
    filtered_sites = sorted([s for s in site_list if target_year in s])
    return st.sidebar.selectbox('Site to summarize', filtered_sites)

# Set the default date range to the first and last dates for which we have data. In the case that we're
# automatically generating all the sites, then stop there. Otherwise, show the UI for the date selection
# and if the user wants a specific range then update our range to reflect that.
def get_date_range(df:pd.DataFrame, graphing_all_sites:bool) -> dict:
    df.sort_index(inplace=True)
    #Set the default date range to the first and last dates that we have data
    date_range_dict = {start_str : df.index[0].strftime("%m-%d-%Y"), end_str : df.index[len(df)-1].strftime("%m-%d-%Y")}
    
    if not graphing_all_sites:
        months1 = {'First': '-1', 'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}
        months2 = {'Last': '-1',  'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}
        start_month = st.sidebar.selectbox("Start month", months1.keys(), index=0)
        end_month = st.sidebar.selectbox("End month", months2.keys(), index=0)

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
    custom_params = {'figure.dpi':'300', 
                     'font.family':'Arial', #'sans-serif'
                     'font.size':'12',
                     'font.stretch':'normal',
                     'font.weight':'light',
                     'xtick.labelsize':'medium',
                     'xtick.major.size':'12',
                     'xtick.color':'black',
                     'xtick.bottom':'True',
                     'ytick.left':'False',
                     'ytick.labelleft':'False',
                     'figure.frameon':'False',
                     'axes.spines.left':'False',
                     'axes.spines.right':'False',
                     'axes.spines.top':'False',
                     'axes.spines.bottom':'False',
                     'savefig.facecolor':'white'
                     }
    #The base context is "notebook", and the other contexts are "paper", "talk", and "poster".
    sns.set_theme(context = 'paper', 
                  style = 'white',
                  rc = custom_params)

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
        y = -0.33
    else:
        y = 1.9+(0.25 if len(axs)>4 else 0)

    max = len(month_lengths)
    n = 0
    x = 0
    for month in month_lengths:
        center_pt = int(month_lengths[month]/2)

        #TODO If the month_length is too short to fit the text, then don't draw the text or maybe center it differently
        #The line below shifts the label to the left a little bit to better center it on the month space. 
        center_pt -= len(month)/4
        mid = x + center_pt

        axs[len(axs)-1].text(x=mid, y=y, s=month, fontdict={'fontsize':'small'})
        x += month_lengths[month]
        if n<max:
            for ax in axs:
                ax.axvline(x=x+0.5, color='black', lw=0.5) #The "0.5" puts it in the middle of the day, so it aligns with the tick

# For ensuring the title in the graph looks the same between weather and data graphs.
# note that if the fontsize is too big, then the title will become the largest thing 
# in the figure which causes the graph to shrink!
def plot_title(title:str):
    plt.suptitle(title, fontsize=14, x=0, horizontalalignment='left')
            
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
    fig, axs = plt.subplots(nrows = row_count, 
                            ncols = 1,
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
        data_points = df_clean.loc[row].max()

        # pull out the one row we want. When we do this, it turns into a series, so we then need to convert it back to a DF and transpose it to be wide
        df_to_graph = df_clean.loc[row].to_frame().transpose()
        axs[i] = sns.heatmap(data = df_to_graph,
                        ax = axs[i],
                        cmap = cmap[row] if len(cmap) > 1 else cmap[0],
                        vmin = 0, vmax = data_points if data_points > 0 else 1,
                        cbar = False,
                        xticklabels = tick_spacing,
                        yticklabels = False)
        
        # If we drew an empty graph, write text on top to indicate that it is supposed to be empty
        # and not that it's just hard to read!
        if df_clean.loc[row].sum() == 0:
            axs[i].text(0.5,0.8,'No data for ' + row, fontsize='x-small', fontstyle='italic', color='gray')

        # Track which graphs we drew, so we can put the proper ticks on later
        graph_drawn.append(i)
            
        # For edge: Add a rectangle around the regions of consective tags, and a line between 
        # non-consectutive if it's a N tag.
        if draw_horiz_rects and row in df_clean.index:
            df_col_nonzero = df.loc[row].to_frame()  #pull out the row we want, it turns into a column as above
            df_col_nonzero = df_col_nonzero.reset_index()   #index by ints for easy graphing
            df_col_nonzero = df_col_nonzero.query('`{}` != 0'.format(row))  #get only the nonzero values. 

            if len(df_col_nonzero):
                c = cm.get_cmap(cmap[row] if len(cmap) > 1 else cmap[0], 1)(1)
                #for debug
                #c = cm.get_cmap('prism',1)(1)
                if row in edge_c_cols: #these tags get a box around the whole block
                    first = df_col_nonzero.index[0]
                    last  = df_col_nonzero.index[len(df_col_nonzero)-1]+1
                    axs[i].add_patch(patches.Rectangle((first,0), last-first, 0.99, ec=c, fc=c, fill=False))
                    
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
                        axs[i].add_patch(patches.Rectangle((borders[x],0), borders[x+1]-borders[x] + extra, 0.99, ec=c, fc=c, fill=False))
                    # For each pair of rects, draw a line between them.
                    for x in range(1,len(borders)-1,2):
                        # The +1/-1 are because we don't want to draw on top of the days, just between the days
                        axs[i].add_patch(patches.Rectangle((borders[x]+1,0.48), borders[x+1]-borders[x]-1, 0.04, ec=c, fc=c, fill=True)) 
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
                rect = patches.Rectangle(xy=(px,bottom), width=1, height=top-bottom, transform=trans,
                                         fc='none', ec='C0', lw=0.5)
                fig.add_artist(rect)
    
    if len(graph_drawn):
        # Clean up the ticks on the axis we're going to use
        format_xdateticks(axs[len(row_names)-1])
        draw_axis_labels(get_days_per_month(df.columns.tolist()), axs)

        #Hide the ticks on the top graphs
        for i in range(0,len(row_names)-1):
            axs[i].tick_params(bottom = False)
    else: 
        #Need to hide the ticks, although I don't think this will get called anymore since I now create
        #an empty row for each index, so we always have something to graph
        axs[len(row_names)-1].tick_params(bottom = False, labelbottom = False)

    # Draw a bounding rectangle around everything except the caption
    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.0, 0.0), 1.0, top_gap, fill=False, color='black', lw=0.5, 
        zorder=1000, transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])

    # return the final plotted heatmap
    return fig

#Helper to ensure we make the filename consistently because this is done from multiple places
def make_img_filename(site:str, graph_type:str) ->str:
    return site + ' - ' + graph_type + '.png'

# Save the graphic to a different folder. All file-related options are managed from here.
def save_figure(site:str, graph_type:str, delete_only=False):
    filename = make_img_filename(site, graph_type)
    figure_path = figure_dir / filename

    #If the file exists then delete it, so that we make sure a new one is written
    if os.path.isfile(figure_path):
        os.remove(figure_path)
    
    if not delete_only:
        plt.savefig(figure_path, dpi='figure', bbox_inches='tight')
        
        #Create a different version of the image that we'll use for the compilation
        plt.suptitle('')
        x=plt.gca()
        for s in x.texts:
            if not('data' in s.get_text()):
                s.remove()
        filename = site + ' - ' + graph_type + '_clean.png'
        figure_path = figure_dir / filename
        plt.savefig(figure_path, dpi='figure', bbox_inches='tight')
        

    plt.close()


def concat_images(*images):
    """Generate composite of all supplied images."""
    # Get the widest width.
    #TODO Why aren't the images all the same width?
    width = max(image.width for image in images)
    # Add up all the heights.
    height = sum(image.height for image in images)
    composite = Image.new('RGB', (width, height), color='white')
    # Paste each image below the one before it.
    y = 0
    for image in images:
        composite.paste(image, (0, y))
        y += image.height
    return composite


# Load all the images that match the site name, combine them into a single composite,
# and then save that out
def combine_images(site):
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
                    print("figure:       " + fig.name)
                    print("current type: " + graph_type)
                    if graph_type in fig.name:
                        print('Found')
                        print(fig.path)
                        site_fig_dict[graph_type] = fig.path

        # Now have the list of figures.  
        if len(site_fig_dict) > 0:
            #Building the list of the figures but it needs to be in a specific order so that the composite looks right
            site_fig_list = []
            for g in graph_names:
                if g in site_fig_dict:
                    site_fig_list.append(site_fig_dict[g])
            images = [Image.open(f) for f in site_fig_list]
            composite = concat_images(*images)

            #Make a new image that's a little bigger so we can add the site name at the top
            width, height = composite.size
            caption_height = 125      
            new_height = height + caption_height #for the string
            final = Image.new(composite.mode, (width, new_height), color='white')
            draw = ImageDraw.Draw(final)
            font = ImageFont.truetype("arialbd.ttf", size=72)
            draw.text((25,25), site, fill='black', font=font)
            final.paste(composite, box=(0,caption_height)) 
            final.save(composite_path)


def output_graph(site:str, graph_type:str, save_files:bool, make_all_graphs:bool, data_to_graph=True):
    if data_to_graph:
        if make_all_graphs:
            pass
        else:
            st.write(graph)
            
        if make_all_graphs or save_files:
            #TODO if we want to put _ in the names then a) it neds to happen elsewhere, as this is 
            #just putting them into the graph type and not the site name, and b) this breaks the
            #code in the image compilation section because the filename it has (without underscore)
            #doesn't match ones with the underscore
            #graph_type = graph_type.replace(' ', '_')
            save_figure(site, graph_type)
    else:
        #No data, so show a message instead. 
        #TODO write an empty file in this case
        save_figure(site, graph_type, delete_only=True)
        site_name_text = '<p style="font-family:sans-serif; color:Black; font-size: 16px;"><b>{}</b></p>'.format(graph_type)
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
@st.experimental_singleton(suppress_st_warning=True)
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
    
    #select only rows that match our site
    if site_name in df.index:
        site_weather = df.loc[[site_name]]
        #select only rows that are in our date range
        mask = (site_weather['date'] >= date_range_dict[start_str]) & (site_weather['date'] <= date_range_dict[end_str])
        site_weather = site_weather.loc[mask]

        # For each type of weather, break out that type into a separate table and 
        # drop it into a dict. Then, reindex the table to match our date range and 
        # fill in empty values
        site_weather_by_type = {}
        date_range = pd.date_range(date_range_dict[start_str], date_range_dict[end_str]) 
        for w in weather_cols:
            site_weather_by_type[w] = site_weather.loc[site_weather['datatype']==w]
            #reindex the table to match our date range and fill in empty values
            site_weather_by_type[w]  = site_weather_by_type[w].set_index('date')
            site_weather_by_type[w]  = site_weather_by_type[w].reindex(date_range, fill_value=0)         
    else:
        show_error('No weather available for ' + site_name)

    return site_weather_by_type

#Used below to get min temp that isn't zero
def min_above_zero(s:pd.Series):
    return min(i for i in s if i > 0)

def create_weather_graph(weather_by_type:dict, site_name:str) -> plt.figure:
    if len(weather_by_type)>0:
        # The use of rows, cols, and gridspec is to force the graph to be drawn in the same 
        # proportions and size as the heatmaps
        fig, ax1 = plt.subplots(nrows = 1, ncols = 1, 
            gridspec_kw={'left':0, 'right':1, 'bottom':0, 'top':0.8},
            figsize=(fig_w,fig_h))
        ax2 = ax1.twinx() # makes a second y axis on the same x axis 
        ax1.margins(0,tight=True)
        ax2.margins(0,tight=True)

        plot_title(site_name + ' ' + graph_weather)

        # Plot the data in the proper format on the correct axis.
        temp_color = 'red'
        prcp_color = 'blue'
        for wt in weather_cols:
            w = weather_by_type[wt]
            if wt == weather_prcp:
                ax1.bar(w.index.values.astype(str), w['value'], color = prcp_color)
            elif wt == weather_tmax:
                ax2.plot(w.index.values.astype(str), w['value'], color = temp_color)
            else: #TMIN
                ax2.plot(w.index.values.astype(str), w['value'], color = 'pink')

        # VERTICAL TICK FORMATTING AND CONTENT
        # Adjust the axis limits
        ax1.set_ylim(ymax=1.5) #Sets the max amount of precip to 1.5
        ax2.set_ylim(ymin=32,ymax=115) #Set temp range
        ax2.axline((0,100),slope=0, color='red', lw=0.5, ls='dotted')        

        ax1.tick_params(axis='y', direction='in', color=prcp_color, labelsize=6, length=3, width=0.5, 
                        labelcolor=prcp_color, labelleft=True, pad=-14)
        prcp_ticks = [0.2,0.5,1.0]
        prcp_tlabel = ['{}\"'.format(t) for t in prcp_ticks]
        ax1.set_yticks(prcp_ticks)
        ax1.set_yticklabels(prcp_tlabel, fontdict={'horizontalalignment':'right'})

        temp_ticks = [50,75,100]
        temp_tlabel = ['{}°F'.format(t) for t in temp_ticks]
        ax2.tick_params(axis='y', direction='in', color=temp_color, labelsize=6, length=3, width=0.5,
                        labelcolor=temp_color, labelright=True, pad=-4)
        ax2.set_yticks(temp_ticks)
        ax2.set_yticklabels(temp_tlabel, fontdict={'horizontalalignment':'right'})

        # To turn off all y ticks
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

        # HORIZONTAL TICKS AND LABLING 
        # Get the list of ticks and set them 
        axis_dates = list(weather_by_type[weather_tmax].index.values.astype(str))
#        tick_pos = list(range(len(weather_by_type[weather_tmax])))
#        weather_tick_spacing = 14
#        ax1.axes.set_xticks(tick_pos[::weather_tick_spacing], axis_dates[::weather_tick_spacing])
        ax1.axes.set_xticks([])
#        format_xdateticks(ax1) #, mmdd=True)
        draw_axis_labels(get_days_per_month(weather_by_type[weather_tmax].index.values), [ax1], weather_graph=True)

        #Turn on the graph borders, these are off by default for other charts
        ax1.spines[:].set_linewidth(0.5)
        ax1.spines[:].set_visible(True)

        # Add a legend for the figure
        # For more legend tips see here: https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
        tmax_label = 'High temp ({}-{}F)'.format(min_above_zero(weather_by_type[weather_tmax]['value']),
                                                 weather_by_type[weather_tmax]['value'].max())
        tmin_label = 'Low temp ({}-{}F)'.format(min_above_zero(weather_by_type[weather_tmin]['value']),
                                                weather_by_type[weather_tmin]['value'].max())
        prcp_label = 'Precipitation ({}-{}\")'.format(min_above_zero(weather_by_type[weather_prcp]['value']),
                                                      weather_by_type[weather_prcp]['value'].max())
        legend_elements = [Line2D([0], [0], color='red', lw=4, label=tmax_label),
                           Line2D([0], [0], color='pink', lw=4, label=tmin_label),
                           Line2D([0], [0], color='blue', lw=4, label=prcp_label)]
        
        #draw the legend below the chart. that's what the bbox_to_anchor with -0.5 does
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
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
    elif type(v) == type(pd.Timestamp.now()): #if it's a date, do nothing
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

    if True or st.sidebar.checkbox('Show errors'):
        bad_rows = pd.DataFrame()                                               
        #Find rows where the columns (ws-m, mh-m) have data, but the song column is missing data
        non_zero_rows = filter_df_by_tags(df, [data_columns[tag_mhm], 
                                               data_columns[tag_wsm]])
        bad_rows = pd.concat([bad_rows, 
                              filter_df_by_tags(non_zero_rows, song_cols, '=={}'.format(missing_data_flag))])

        #P1C, P2C throws an error if it's missing courtsong song
        non_zero_rows = filter_df_by_tags(df, edge_c_cols)
        bad_rows = pd.concat([bad_rows,
                              filter_df_by_tags(non_zero_rows, [data_columns[courtsong]], '=={}'.format(missing_data_flag))])

        #P1N, P2N throws an error if it's missing alternative song
        non_zero_rows = filter_df_by_tags(df, edge_n_cols)
        bad_rows = pd.concat([bad_rows, 
                              filter_df_by_tags(non_zero_rows, [data_columns[altsong1]], '=={}'.format(missing_data_flag))])       

        if not(bad_rows.empty):
            with st.expander('See errors'):
                bad_rows.sort_values(by='filename', inplace=True)
                st.write('Total errors: {}'.format(len(bad_rows)))
                
                #Pull out date so we can format it
                bad_rows.reset_index(inplace=True)
                bad_rows.rename(columns={'index':'Date'}, inplace=True)

                pretty_print_table(bad_rows)
        else:
            st.write('No tag errors found')

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


#
#
# Main
#
#
st.sidebar.title('TRBL Summary')

#Load all the data for most of the graphs
df_original = load_data()

#Get the list of sites that we're going to do reports for, and then remove all the other data
site_list = get_target_sites()
df = clean_data(df_original, site_list[site_str])

# Nuke the original data, hopefully this frees up memory
del df_original
gc.collect()

#TODO Make this a UI option
make_all_graphs = False
# If we're doing all the graphs, then set our target to the entire list, else use the UI to pick
if make_all_graphs:
    target_sites = site_list[site_str]
    save_files = False #True if we want to save all the image files
else:
    target_sites = [get_site_to_analyze(site_list[site_str])]
    save_files = st.sidebar.checkbox('Save as picture', value=False) #user decides to save the graphs as pics or not

# Set format shared by all graphs
set_global_theme()

site_counter = 0
for site in target_sites:
    site_counter += 1
    # Select the site matching the one of interest
    df_site = df[df[data_columns[site_str]] == site]

    if df_site.empty:
        st.write('Site {} has no recordings'.format(site))
        break

    #Using the site of interest, get the first & last dates and give the user the option to customize the range
    date_range_dict = get_date_range(df_site, make_all_graphs)

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
    pt_manual = make_pivot_table(df_manual, song_cols, date_range_dict)

    # MINI-MANUAL ANALYSIS
    # 1. Select all rows with one of the following tags:
    #       tag<reviewed-MH-h>, tag<reviewed-MH-m>, tag<reviewed-WS-h>, tag<reviewed-WS-m>
    # 2. Make a pivot table as above
    #   
    df_mini_manual = filter_df_by_tags(df_site, mini_manual_cols)
    pt_mini_manual = make_pivot_table(df_mini_manual, song_cols, date_range_dict)

    # EDGE ANALYSIS
    #   1. Select all rows where one of the following tags
    #       P1C, P1N, P2C, P2N
    
    #TODO We found at least one row that should have kicked up an error given the description
    #   below, why didn't it?

    #   2. If there's a P1C or P2C tag, then the value for CourtshipSong should be zero or 1, any other value is an error and should be flagged 
    #      If there's a P1N or P2N tag, then the value for AlternativeSong should be zero or 1, any other value is an error and should be flagged 
    #   3. Make a pivot table with the number of recordings that have CourtshipSong for the tags ending in C
    #   4. Make another pivot table with the number of recordings that have AlternativeSong for the tags ending in N
    #   5. Merge the tables together so we get one block of heatmaps
    #   
    #TODO add the new things having to do with ONC and YNC tags

    pt_edge = pd.DataFrame()
    edge_data_empty = False
    for tag in edge_cols:
        df_for_tag = filter_df_by_tags(df_site, [tag])
        edge_data_empty = edge_data_empty or len(df_for_tag)
        if tag in edge_c_cols:
            target_col = data_columns[courtsong]
        else:
            target_col = data_columns[altsong1]

        #TODO
        # Validate that all values in target_col are values and not --- or NaN
        #if len(df_for_tag.query('`{}` < 0 | `{}` > 1'.format(target_col,target_col))):
        #    show_error('In edge analysis, tag {} has values in {} that are not 0 or 1'.format(tag, target_col))

        # Make our pivot. "preserve_edges" causes the zero values in the data we pass in to be replaced with -1 
        # this way, in the graph, we can tell the difference between a day that had no tags vs. one that 
        # had tags but no songs
        pt_for_tag = make_pivot_table(df_for_tag, [target_col], date_range_dict, preserve_edges=True)
        pt_for_tag = pt_for_tag.rename({target_col:tag}) #rename the index so that it's the tag, not the song name
        pt_edge = pd.concat([pt_edge, pt_for_tag])

    # PATTERN MATCHING ANALYSIS
    #
    df_pattern_match = load_pm_data(site, date_range_dict)
    pt_pm = pd.DataFrame()
    pm_data_empty = False
    if len(df_pattern_match):
        for t in pm_file_types:
            #For each file type, get the filtered range of just that type
            df_for_file_type = df_pattern_match[df_pattern_match['type']==t]
            pm_data_empty = pm_data_empty or len(df_for_file_type)
            #Build the pivot table for it
            pt_for_file_type = make_pattern_match_pt(df_for_file_type, t, date_range_dict)
            #Concat as above
            pt_pm = pd.concat([pt_pm, pt_for_file_type])

    # ------------------------------------------------------------------------------------------------
    # DISPLAY
    if make_all_graphs:
        st.subheader(site + ' [' + str(site_counter) + ' of ' + str(len(target_sites)) + ']')
    else: 
        st.subheader(site)

    if st.sidebar.checkbox('Show station info', value=True):
        show_station_info(site)

    # Manual analyisis graph
    cmap = {data_columns[malesong]:'Greens', data_columns[courtsong]:'Oranges', data_columns[altsong2]:'Purples', data_columns[altsong1]:'Blues', 'bad':'Black'}
    graph = create_graph(df = pt_manual, 
                        row_names = song_cols, 
                        cmap = cmap, 
                        title = (site + ' ' if save_files else '') + graph_man)
    output_graph(site, graph_man, save_files, make_all_graphs, len(df_manual))

    # MiniManual Assisted Analysis
    graph = create_graph(df = pt_mini_manual, 
                        row_names = song_cols, 
                        cmap = cmap, 
                        raw_data = df_site,
                        draw_vert_rects = True,
                        title = (site + ' ' if save_files else '') + graph_miniman)
    output_graph(site, graph_miniman, save_files, make_all_graphs, len(df_mini_manual))

    # Pattern Matching Analysis
    cmap_pm = {'Male':'Greens', 'Female':'Purples', 'Young Nestling':'Blues', 'Mid Nestling':'Blues', 'Old Nestling':'Blues'}
    graph = create_graph(df = pt_pm, 
                        row_names = pm_file_types, 
                        cmap = cmap_pm, 
                        title = (site + ' ' if save_files else '') + graph_pm)
    output_graph(site, graph_pm, save_files, make_all_graphs, pm_data_empty)

    # Edge Analysis
    cmap_edge = {c:'Oranges' for c in edge_c_cols} | {n:'Blues' for n in edge_n_cols} # the |" is used to merge dicts
    graph = create_graph(df = pt_edge, 
                        row_names = edge_cols,
                        cmap = cmap_edge, 
                        raw_data = df_site,
                        draw_horiz_rects = True,
                        title = (site + ' ' if save_files else '') + graph_edge)
    output_graph(site, graph_edge, save_files, make_all_graphs, edge_data_empty)

    #Show weather, as needed            
    if st.sidebar.checkbox('Show station weather', True):
        # Load and parse weather data
        weather_by_type = get_weather_data(site, date_range_dict)
        graph = create_weather_graph(weather_by_type, site)
        output_graph(site, graph_weather, save_files, make_all_graphs)
    
    #TODO remove the "True" below when we're done debugging
    if True or make_all_graphs or save_files:
        combine_images(site)

#If site_df is empty, then there were no recordings at all for the site and so we can skip all the summarizing
if not make_all_graphs and len(df_site):
    # Show the table with all the raw data
    with st.expander("See raw data"):
        #Used for making the summary pivot table
        friendly_names = {data_columns[malesong] : 'M-Male', 
                          data_columns[courtsong]: 'M-Chorus',
                          data_columns[altsong2] : 'M-Female', 
                          data_columns[altsong1] : 'M-Nestling'
        }
        summary = []
        summary.append(make_summary_pt(pt_manual, song_cols, friendly_names))
        
        friendly_names = {data_columns[malesong] : 'MM-Male', 
                          data_columns[courtsong]: 'MM-Chorus',
                          data_columns[altsong2] : 'MM-Female', 
                          data_columns[altsong1] : 'MM-Nestling'
        }
        summary.append(make_summary_pt(pt_mini_manual, song_cols, friendly_names))

        friendly_names =   {data_columns[tag_p1c]: 'E-P1C',
                            data_columns[tag_p1n]: 'E-P1N',
                            data_columns[tag_p2c]: 'E-P2C',
                            data_columns[tag_p2n]: 'E-P2N'
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
        union_pt = union_pt.style.applymap(style_cells)
        union_pt.format(formatter={'PRCP':'{:.2f}', 'Date':lambda x:x.strftime('%m-%d-%y')})
        st.dataframe(union_pt)

    # Put a box with first and last dates for the Song columns, with counts on that date
    with st.expander("See summary of first and last dates"):  
        output = get_first_and_last_dates(make_pivot_table(df_site, song_cols, date_range_dict))
        pretty_print_table(pd.DataFrame.from_dict(output))

    # Scan the list of tags and flag any where there is "---" for the value. 
    check_tags(df_site)

    if len(site_list[bad_files]) > 0:
        with st.expander("See possibly bad filenames"):  
            st.write(site_list[bad_files])

    if st.button('Clear cache'):
        st.experimental_singleton.clear()
