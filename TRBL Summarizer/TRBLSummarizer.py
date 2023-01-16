import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib import cm
from pathlib import Path
import os
import calendar
from collections import Counter
from itertools import tee
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
#tag_p1bc = 'tag_p1bc'
#tag_p1bn = 'tag_p1bn'
tag_p2c = 'tag_p2c'
tag_p2n = 'tag_p2n'
#tag_p2bc = 'tag_p2bc' 
#tag_p2bn = 'tag_p2bn'
tag_wsmc = 'tag_wsmc'
validated = 'validated'

present = 'present'

start_str = 'start'
end_str = 'end'

#Master list of all the columns I need. If columns get added/removed then this needs to update
columns = {filename_str : 'filename', 
           site_str     : 'site', 
           'day'        : 'day',
           'month'      : 'month',
           'year'       : 'year',
           hour_str     : 'hour', 
           date_str     : 'date',
#           tag_wse       : 'tag<reviewed-WS-e>',
           tag_wsm      : 'tag<reviewed-WS-m>',
           tag_wsh      : 'tag<reviewed-WS-h>',
#           tag_mhe       : 'tag<reviewed-MH-e>',
           tag_mhm      : 'tag<reviewed-MH-m>',
           tag_mhh      : 'tag<reviewed-MH-h>',
#           tag_mhe2      : 'tag<reviewed-MH-e2>',
           tag_ws       : 'tag<reviewed-WS>',
           tag_mh       : 'tag<reviewed-MH>',
           tag_         : 'tag<reviewed>',
           tag_p1c      : 'tag<p1c>',
           tag_p1n      : 'tag<p1n>',
           tag_p2c      : 'tag<p2c>',
           tag_p2n      : 'tag<p2n>',
#           tag_wsmc      : 'tag<reviewed-WS-mc>',
#           tag_p1bc     : 'tag<p1bc>',
#           tag_p1bn     : 'tag<p1bn>',
#           tag_p2bc     : 'tag<p2bc>',
#           tag_p2bn     : 'tag<p2bn>',
           malesong     : 'val<Agelaius tricolor/Common Song>',
           altsong1     : 'val<Agelaius tricolor/Alternative Song>',
           altsong2     : 'val<Agelaius tricolor/Alternative Song 2>',
           courtsong    : 'val<Agelaius tricolor/Courtship Song>',
           validated    : 'validated',
           }

songs = [malesong, courtsong, altsong2, altsong1]
song_columns = [columns[s] for s in songs]

manual_tags = [tag_mh, tag_ws, tag_]
mini_manual_tags = [tag_mhh, tag_wsh, tag_mhm, tag_wsm]
#old version is the next two lines:
#edge_c_tags = [tag_p1c, tag_p1bc, tag_p2c, tag_p2bc]
#edge_n_tags = [tag_p1n, tag_p1bn, tag_p2n, tag_p2bn]
edge_c_tags = [tag_p1c, tag_p2c]
edge_n_tags = [tag_p1n,  tag_p2n]
tags = manual_tags + mini_manual_tags + edge_c_tags + edge_n_tags

manual_cols = [columns[t] for t in manual_tags]
mini_manual_cols = [columns[t] for t in mini_manual_tags]
edge_c_cols = [columns[t] for t in edge_c_tags]
edge_n_cols = [columns[t] for t in edge_n_tags]

edge_cols = edge_c_cols + edge_n_cols #make list of the right length
edge_cols[::2] = edge_c_cols #assign C cols to the even indices (0, 2, ...)
edge_cols[1::2] = edge_n_cols #assign N cols to the odd indices (1, 3, ...)

#For setting figure width and height, values in inches
fig_w = 16
fig_h = 3

#Files, paths, etc.
data_foldername = 'Data/'
data_dir = Path(__file__).parents[0] / data_foldername
data_file = 'data.csv'
site_info_file = 'sites.csv'
data_fullfilename = data_dir / data_file
site_info_fullfilename = data_dir / site_info_file
file_types = ['Male', 'Female', 'Young Nestling', 'Mid Nestling', 'Old Nestling']
weather_filename = data_foldername + '/' + 'weather_history.csv'

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
#
#
#File handling and setup
#
#
@st.experimental_singleton(suppress_st_warning=True)
def get_target_sites() -> dict:
    file_summary = {}
    for t in file_types:
        file_summary[t] = []
    file_summary[bad_files] = []
    file_summary[site_str] = set()

    #Load the list of unique site names, keep just the 'Name' column, and then convert that to a list
    site_list = pd.read_csv(site_info_fullfilename, usecols = ['Name'])
    site_list = site_list['Name'].tolist()

    #Clean it up. Everything must start with a 4-digit number. More validation to be done?
    for s in site_list:
        if not s[0:3].isdigit():
            site_list.remove(s)
    
    #Get a list of all items in the directory, then check each folder we find
    top_items = os.scandir(data_dir)
    if any(top_items):
        for item in top_items:
            if item.is_dir():
                #Check that the directory name is in our site list. If yes, continue. If not, then add it to the bad list
                if item.name in site_list:
                    s = item.name
                    #Get a list of all files in that directory, scan for files that match our pattern
                    if any(os.scandir(item)):
                        #Check that each type of expected file is there:

                        if len(file_types) != count_files_in_folder(item):
                            file_summary[bad_files].append('Wrong number of files: ' + item.name)

                        for t in file_types:
                            found_file = False
                            found_dir_in_subfolder = False
                            sub_items = os.scandir(item)
                            for f in sub_items:
                                empty_dir = False #if the sub_items constructor is empty, we won't get here

                                if f.is_file():
                                    f_type = f.name[len(s)+1:len(f.name)] # Cut off the site name
                                    if t.lower() == f_type[0:len(t)].lower():
                                        file_summary[t].append(f.name)
                                        file_summary[site_str].add(s)
                                        found_file = True
                                        break
                                else:
                                    if not found_dir_in_subfolder: # if this is the first time here, then log it
                                        file_summary[bad_files].append('Found subfolder in data folder: ' + s)
                                    found_dir_in_subfolder = True
                    
                            if not found_file and not empty_dir:
                                file_summary[bad_files].append('Missing file: ' + s + ' ' + t)

                    else:
                        file_summary[bad_files].append('Empty folder: ' + item.name)

                    sub_items.close()
                
                else:
                    if item.name.lower() != 'hide':
                        file_summary[bad_files].append('Bad folder name: ' + item.name)
            
            else: #If it's not a directory, it's a file. If the file we found isn't one of the exceptions to our pattern, then mark it as Bad.
                if (item.name.lower() != data_file.lower() and item.name.lower() != site_info_file.lower() and
                        item.name.lower() != 'data_old.csv'): 
                    file_summary[bad_files].append(item.name)

    top_items.close()
    
    #Confirm that there are the same set of files for each type
    if len(file_summary[site_str]) > 0:
        for t in file_types:
            if len(file_summary[site_str]) != len(file_summary[t]):
                if len(file_summary[t]) == 0:
                    show_error('Missing all files of type ' + t)
                else:
                    show_error('Wrong number of files of type ' + t)
    else:
        show_error('No site files found')

    if len(file_summary[bad_files]):
        show_error('File errors were found')

    return file_summary

# Load the CSV file into a dataframe, validate that the columns are what we expect
@st.experimental_singleton(suppress_st_warning=True)
def load_data() -> pd.DataFrame:
    data_csv = Path(__file__).parents[0] / data_fullfilename

    #Validate the data file format
    headers = pd.read_csv(data_fullfilename, nrows=0).columns.tolist()
    if len(headers) != len(columns):
        show_error('Data file {} has an unexpected number of columns, {} instead of {}'.
                   format(data_fullfilename, len(headers), len(columns)))
    for col in columns:
        if not columns[col] in headers:
            #TODO there is at least one column in the set of columns that does not exist in the 
            #big data file -- 'validated'. Should I remove it from this dictionary or just ignore it?
            show_error('Column {} missing from the data file {}'.format(columns[col], data_fullfilename))
    
    #The set of columns we want to use are the basic info (filename, site, date), all songs, and all tags
    usecols = [columns[filename_str], columns[site_str], columns[date_str]]
    for song in songs:
        usecols.append(columns[song])
    for tag in tags:
        usecols.append(columns[tag])

    df = pd.read_csv(data_csv, 
                     usecols = usecols,
                     parse_dates = [columns[date_str]],
                     index_col = [columns[date_str]])
    return df


def make_date(row):
    s = '{}-{}-{}'.format(row['year'], format(row['month'],'02'), format(row['day'],'02'))
#    s = '{}-{}-{}T{}:{}'.format(row['year'], 
#                                format(row['month'],'02'), 
#                                format(row['day'],'02'), 
#                                format(row['hour'],'02'), 
#                                format(row['minute'],'02'))
    return np.datetime64(s)


# Load the pattern matching CSV files into a dataframe, validate that the columns are what we expect
#@st.experimental_singleton(suppress_st_warning=True)
def load_pm_data(site:str, date_range_dict:dict) -> pd.DataFrame:

    # For each type of file for this site (which has already been validated that they exist, but should
    # probably check here, too, eventually), load the file. Add a column to indicate which type it is. 
    # Then append it to the dataframe we're building.
    df = pd.DataFrame()

    # Add the site name so we look into the appropriate folder
    site_dir = data_dir / site
    for t in file_types:
        fname = site + ' ' + t + '.csv'
        full_file_name = site_dir / fname
        usecols = [site_str, 'year', 'month', 'day', validated] 
    
        #TODO Validate the data file format

        df_temp = pd.DataFrame()
        if is_non_zero_file(full_file_name):
            df_temp = pd.read_csv(full_file_name, usecols=usecols)
            df_temp[date_str] = df_temp.apply(lambda row: make_date(row), axis=1)
        else: # if the file is non-zero, make an empty table so the graphing code has something to work with
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
    
    # We need to preserve the diff between no data and 0 tags. But, we have to also make everything integers for
    # later processing. So, we'll replace the hyphens with -100 and then just realize that we can't do math on this
    # column any more without excluding the -100s. Picked -100 because if we do do math then the answer will be obviously wrong!
    df_clean = df_clean.replace('---', -100)
    
    # For each type of song, convert its column to be numeric instead of a string so we can run pivots
    for s in songs:
        df_clean[columns[s]] = pd.to_numeric(df_clean[columns[s]])
    return df_clean


#
#
# Data Analysis
# 
#  

# Get the subset of rows where there's at least one tag, i.e. the count of any tag is greater than zero
# See here for an explanation of the next couple lines: https://stackoverflow.com/questions/45925327/dynamically-filtering-a-pandas-dataframe
def filter_site(site_df:pd.DataFrame, target_tags:list) -> pd.DataFrame:
    # This is an alternative to: tagged_rows = site_df[((site_df[columns[tag_wse]]>0) | (site_df[columns[tag_mhh]]>0) ...
    query = ' | '.join([f'`{tag}`>0' for tag in target_tags])
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
    summary = pd.pivot_table(site_df, values = labels, index = [columns[date_str]], 
                              aggfunc = lambda x: (x>=1).sum()) 

    if preserve_edges:
        # For every date where there is a tag, make sure that the value is non-zero. Then, when we do the
        # graph later, we'll use this to show where the edges of the analysis were
        summary = summary.replace(to_replace=0, value=-1)

    return normalize_pt(summary, date_range_dict)


def make_pattern_match_pt(site_df: pd.DataFrame, type_name:str, date_range_dict:dict) -> pd.DataFrame:
    #If the value in 'validated' column is 'Present', count it.
    summary = pd.pivot_table(site_df, values=[columns[validated]], index = [columns[date_str]], 
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
    #return('2021 Markham Ravine')
    site_list = sorted(site_list)
    return st.sidebar.selectbox('Site to summarize', site_list, index=1)

def get_date_range(df:pd.DataFrame, doing_all_sites:bool) -> dict:
    df.sort_index(inplace=True)
    #Set the default date range to the first and last dates that we have data
    date_range_dict = {start_str : df.index[0].strftime("%m-%d-%Y"), end_str : df.index[len(df)-1].strftime("%m-%d-%Y")}
    
    if not doing_all_sites:
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
    custom_params = {'figure.dpi':'1200',
                     'font.family':'Corbel', #'sans-serif'
                     'font.size':'12',
                     'font.weight':'600',
                     'font.stretch':'semi-condensed',
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
                     }
    #The base context is "notebook", and the other contexts are "paper", "talk", and "poster".
    sns.set_theme(context = 'paper', 
                  style = 'white',
                  rc = custom_params)

def get_days_per_month(df:pd.DataFrame) -> dict:
    date_list = df.columns.tolist()
    #Make a list of all the values, but only use the month name. Then, count how many of each month names there are, to get the number of days/mo
    months = [pd.to_datetime(date).strftime('%B') for date in date_list]
    return Counter(months)


#The axis already has all the dates in it, but they need to be formatted. 
def format_xdateticks(date_axis:plt.Axes,mmdd = False):
    if mmdd:
        fmt = '%d-%b'
        rot = 30
        weight = 'light'
    else:
        fmt = '%d'
        rot = 0
        weight = 'bold'

    #Make a list of all the values
    date_values = [value for value in date_axis.xaxis.get_major_formatter().func.args[0].values()]

    #Make a list of all the ticks where they have the day number only.
    ticks = [pd.to_datetime(value).strftime(fmt) for value in date_values]

    #Actually set the ticks and then format them as needed
    date_axis.xaxis.set_ticklabels(ticks, fontweight=weight)
    date_axis.tick_params(axis = 'x',labelrotation = rot)
    return


#Take the list of month length counts we got from the function above, and draw lines at those positions. 
#Skip the last one so we don't draw over the border
def draw_axis_labels(month_lengths:dict, axs:np.ndarray, gap:float):
    max = len(month_lengths)
    n = 0
    x = 0
    for month in month_lengths:
        center_pt = int(month_lengths[month]/2)
        #The line below shifts the label to the left a little bit to better center it on the month space. 
        center_pt -= len(month)/4
        mid = x + center_pt
        axs[len(axs)-1].text(x=mid, y=gap, s=month, size='x-large')
        x += month_lengths[month]
        if n<max:
            for ax in axs:
                ax.axvline(x=x+0.5, color='black', lw=0.5) #The "0.5" puts it in the middle of the day, so it aligns with the tick
            
# Create a graph, given a dataframe, list of row names, color map, and friendly names for the rows
def create_graph(df: pd.DataFrame, items:list, cmap:dict, draw_connectors=False, raw_data=pd.DataFrame, 
                 draw_vert_rects=False, draw_horiz_rects=False,title='') -> plt.figure:
    max = len(items)
    graph_drawn = []

    #distance between top of plot space and chart
    top_gap = 0.8 if title != '' else 1
    #tick_spacing is how many days apart the tick marks are. If set to 0 then it turns off all ticks and labels except for month name
    tick_spacing = 0

    # Create the base figure for the graphs
    fig, axs = plt.subplots(nrows = max, 
                            ncols = 1,
                            sharex = 'col', 
                            gridspec_kw={'height_ratios': np.repeat(1,max), 
                                         'left':0, 'right':1, 'bottom':0, 'top':top_gap,
                                         'hspace':0},  #hspace is row spacing (gap between rows)
                            figsize=(fig_w,fig_h))
    # Draw the title https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html#matplotlib.pyplot.suptitle
    if len(title)>0:
        plt.suptitle(title, fontsize=36, fontweight='bold')
    
    #Clean up the data, make sure that we have a row for each index.
    for item in items:
        if item not in df.index:
            #Add the row and make it all zeroes
            df.loc[item]=pd.Series(0,index=df.columns)

    #Set a mask on the zero values so that we can force them to display as white. Keep the original data as we
    #need it for drawing later. Use '<=0' because -100 is use to differentiate no data from data with zero value
    df_clean = pd.DataFrame()
    for col in df:
        df_clean[col] = df[col].mask(df[col] <= 0)

    i=0
    for item in items:
        # plotting the heatmap
        max_count = 0
        max_count = df_clean.loc[item].max()
        # pull out the one row we want. When we do this, it turns into a series, so we then need to convert it back to a DF and transpose it to be wide
        df_to_graph = df_clean.loc[item].to_frame().transpose()
        axs[i] = sns.heatmap(data = df_to_graph,
                        ax = axs[i],
                        cmap = cmap[item] if len(cmap) > 1 else cmap[0],
                        vmin = 0, vmax = max_count if max_count > 0 else 1,
                        cbar = False,
                        xticklabels = tick_spacing,
                        yticklabels = False)
        #track which graphs we drew, so we can put the proper ticks on later
        graph_drawn.append(i)
            
        #Add a rectangle around the regions of consective tags, and a line between non-consectutive if it's a N tag
        if draw_horiz_rects and item in df_clean.index:
            df_col_nonzero = df.loc[item].to_frame()  #pull out the row we want, it turns into a column as above
            df_col_nonzero = df_col_nonzero.reset_index()   #index by ints for easy graphing
            df_col_nonzero = df_col_nonzero.query('`{}` != 0'.format(item))  #get only the nonzero values. 

            if len(df_col_nonzero):
                c = cm.get_cmap(cmap[item] if len(cmap) > 1 else cmap[0], 1)(1)
                #for debug
                #c = cm.get_cmap('prism',1)(1)
                if item in edge_c_cols: #these tags get a box around the whole block
                    first = df_col_nonzero.index[0]
                    last  = df_col_nonzero.index[len(df_col_nonzero)-1]+1
                    axs[i].add_patch(patches.Rectangle((first,0), last-first, 0.99, ec=c, fc=c, fill=False))
                    #st.error("MC First: " + str(df_col_nonzero.loc[first].values[0].date()) + 
                    #         ", MC Last: " + str(df_col_nonzero.loc[last-1].values[0].date()))
                    
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
    
        
    # add a rect around each day that has some data
    if draw_vert_rects and len(raw_data)>0:
        tagged_rows = filter_site(raw_data, mini_manual_cols)
        if len(tagged_rows):
            date_list = tagged_rows.index.unique()
            first = raw_data.index[0]
            box_pos = [(i - first)/pd.Timedelta(days=1) for i in date_list]

            _,top = fig.transFigure.inverted().transform(axs[0].transAxes.transform([0,1]))
            _,bottom = fig.transFigure.inverted().transform(axs[max-1].transAxes.transform([0,0]))
            trans = transforms.blended_transform_factory(axs[0].transData, fig.transFigure)
            for px in box_pos:
                rect = patches.Rectangle(xy=(px,bottom), width=1, height=top-bottom, transform=trans,
                                         fc='none', ec='C0', lw=0.5)
                fig.add_artist(rect)
    
    if len(graph_drawn):
        # Clean up the ticks on the axis we're going to use
        format_xdateticks(axs[len(items)-1])
        month_counts = get_days_per_month(df)
        draw_axis_labels(month_counts, axs, 2 if max==4 else 3)

        #Hide the ticks on the top graphs
        for i in range(0,len(items)-1):
            axs[i].tick_params(bottom = False)
    else: 
        #Need to hide the ticks, although I don't think this will get called anymore since I now create
        #an empty row for each index, so we always have something to graph
        axs[len(items)-1].tick_params(bottom = False, labelbottom = False)

    # draw a bounding rectangle around everything except the caption
    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.0, 0.0), 1.0, top_gap, fill=False, color='black', lw=0.5, 
        zorder=1000, transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])

    # return the final plotted heatmap
    return fig

# Save the graphic to a different folder. All file-related options are managed from here.
def save_figure(site:str, graph_type:str):
    filename = site + ' - ' + graph_type + '.png'
    figure_path = Path(__file__).parents[0] / 'Figures/' / filename
    #If the file exists then delete it, so that we make sure a new one is written
    if os.path.isfile(figure_path):
        os.remove(figure_path)
    plt.savefig(figure_path, dpi='figure', bbox_inches='tight')
    #plt.close()

def output_graph(site:str, graph_type:str, save_files:bool, make_all_graphs:bool):
    if make_all_graphs:
        st.write(graph_type)
    else:
        st.subheader(graph_type)

    if not make_all_graphs:
        st.write(graph)

    if make_all_graphs or save_files:
        graph_type = graph_type.replace(' ', '_')
        save_figure(site, graph_type)

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
@st.experimental_singleton(suppress_st_warning=True)
def load_weather_data_from_file() -> pd.DataFrame:
    weather_csv = Path(__file__).parents[0] / weather_filename

    #Validate the data file format
    headers = pd.read_csv(weather_csv, nrows=0).columns.tolist()
    weather_cols = {'date':'date', 'datatype':'datatype', 'value':'value', 'site':'site', 
                    'lat':'lat', 'lng':'lng', 'alt':'alt'}
    if len(headers) != len(weather_cols):
        show_error('File {} has an unexpected number of columns, {} instead of {}.'.
                   format(weather_filename, len(headers), len(weather_cols)))
    for col in weather_cols:
        if not weather_cols[col] in headers:
            show_error('Column {} missing from {}.'.format(weather_cols[col], weather_filename))
    
    df = pd.read_csv(weather_csv, 
                     parse_dates = [weather_cols['date']],
                     index_col = [weather_cols['site']])
    return df

#Filter weather data down to just what we need for a site
def get_weather_data(site_name:str, date_range_dict:dict) -> pd.DataFrame:
    df = load_weather_data_from_file()    

    #select only rows that are in our date range
    mask = (df['date'] >= date_range_dict[start_str]) & (df['date'] <= date_range_dict[end_str])
    df = df.loc[mask]
    
    #select only rows that match our site
    site_weather = pd.DataFrame
    if site_name in df.index:
        site_weather = df.loc[[site_name]]
        site_weather = site_weather.set_index('date')
    else:
        show_error('No weather available for ' + site_name)
        
    return site_weather

def create_weather_graph(site_name:str, date_range_dict:dict) -> plt.figure:
    # Load and parse weather data
    df = get_weather_data(site_name, date_range_dict)
    
    if not df.empty:
        # Break into three groups for cleaner code
        prcp = df.loc[df['datatype']=='PRCP']
        tmax = df.loc[df['datatype']=='TMAX']
        tmin = df.loc[df['datatype']=='TMIN']

        # Build graph for data
        fig, ax1 = plt.subplots(figsize=(fig_w,fig_h)) # initializes figure and plots
        ax2 = ax1.twinx() # makes a second y axis on the same x axis 

        # plots the first set of data, and sets it to ax1
        ax1.bar(prcp.index.values.astype(str), prcp['value'], color = 'blue')
        ax2.plot(tmax.index.values.astype(str), tmax['value'], color = 'red')
        ax2.plot(tmin.index.values.astype(str), tmin['value'], color = 'pink')

        #Add the annotations for the plot 
        ax1.set_ylabel('Precipitation', color='blue', fontweight='light', fontsize=16)
        ax2.set_ylabel('High & Low Temperature', color='red', fontweight='light', fontsize=16)

        #Get the list of ticks and set them 
        axis_dates = list(tmax.index.values.astype(str))
        tick_pos = list(range(len(tmax)))
        weather_tick_spacing = 14
        ax1.axes.set_xticks(tick_pos[::weather_tick_spacing], axis_dates[::weather_tick_spacing])
        format_xdateticks(ax1, mmdd=True)

        #Turn on the graph borders, these are off by default for other charts
        ax1.spines[:].set_visible(True)

    return fig




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
    save_files = False
else:
    target_sites = [get_site_to_analyze(site_list[site_str])]

    #Decide if we're going to save the graphs as pics or not
    save_files = st.sidebar.checkbox('Save as picture', value=False)

# Set format shared by all graphs
set_global_theme()

site_counter = 0
for site in target_sites:
    site_counter += 1
    # Select the site matching the one of interest
    site_df = df[df[columns[site_str]] == site]

    #Using the site of interest, get the first & last dates and give the user the option to customize the range
    date_range_dict = get_date_range(site_df, make_all_graphs)

    #
    # Data Analysis
    # -------------
    # We want a series of charts. The first chart is:
    #   1. Select all rows where one of the following tags
    #       tag<reviewed-MH>, tag<reviewed-WS>, tag<reviewed>
    #   2. Make a pivot table with the following columns:
    #       The number of recordings from that set that have Common Song >= 1
    #       The number of recordings from that set that have Courtship Song >= 1
    #       The number of recordings from that set that have AltSong2 >= 1
    #       The number of recordings from that set that have AltSong >= 1 
    #     
    df_manual = filter_site(site_df, manual_cols)
    manual_pt = make_pivot_table(df_manual, song_columns, date_range_dict)

    # 
    # COMMENT
    #  
    df_mini_manual = filter_site(site_df, mini_manual_cols)
    mini_manual_pt = make_pivot_table(df_mini_manual, song_columns, date_range_dict)


    #   1. Select all rows where one of the following tags
    #       P1C, P1N, P2C, P2N
    #   2. If there's a P1C or P2C tag, then the value for CourtshipSong should be zero or 1, any other value is an error and should be flagged 
    #      If there's a P1N or P2N tag, then the value for AlternativeSong should be zero or 1, any other value is an error and should be flagged 
    #   3. Make a pivot table with the number of recordings that have CourtshipSong for the tags ending in C
    #   4. Make another pivot table with the number of recordings that have AlternativeSong for the tags ending in N
    #   5. Merge the tables together so we get one block of heatmaps
    #   

    edge_pt = pd.DataFrame()
    for tag in edge_cols:
        df_for_tag = filter_site(site_df, [tag])
        if tag in edge_c_cols:
            target_col = columns[courtsong]
        else:
            target_col = columns[altsong1]

        # Validate that all values in target_col are values and not --- or NaN
        #if len(df_for_tag.query('`{}` < 0 | `{}` > 1'.format(target_col,target_col))):
        #    show_error('In edge analysis, tag {} has values in {} that are not 0 or 1'.format(tag, target_col))

        # Make our pivot. "preserve_edges" causes the zero values in the data we pass in to be replaced with -1 for future graphing needs
        pt_for_tag = make_pivot_table(df_for_tag, [target_col], date_range_dict, preserve_edges=True)
        pt_for_tag = pt_for_tag.rename({target_col:tag}) #rename the index so that it's the tag, not the song name
        edge_pt = pd.concat([edge_pt, pt_for_tag])


    # Load and process the pattern matching tag files
    df_pattern_match = load_pm_data(site, date_range_dict)
    pm_pt = pd.DataFrame()
    # Check that we got some PM data before building the pivot tables. It's possible that there is zero data for some sites
#    if len(df_pattern_match):
    for t in file_types:
        #For each file type, get the filtered range of just that type
        df_for_file_type = df_pattern_match[df_pattern_match['type']==t]
        #Build the pivot table for it
        pt_for_file_type = make_pattern_match_pt(df_for_file_type, t, date_range_dict)
        #Concat as above
        pm_pt = pd.concat([pm_pt, pt_for_file_type])

    # ------------------------------------------------------------------------------------------------
    # DISPLAY
    if make_all_graphs:
        st.subheader(site + ' [' + str(site_counter) + ' of ' + str(len(target_sites)) + ']')
    else: 
        st.header(site)

    # Manual analyisis graph
    cmap = {columns[malesong]:'Greens', columns[courtsong]:'Oranges', columns[altsong2]:'Purples', columns[altsong1]:'Blues', 'bad':'Black'}
    graph = create_graph(df = manual_pt, 
                        items = song_columns, 
                        cmap = cmap, 
                        title = site + ' Manual Analysis')
    output_graph(site, 'Manual Analysis', save_files, make_all_graphs)

    # Computer Assisted Analysis
    graph = create_graph(df = mini_manual_pt, 
                        items = song_columns, 
                        cmap = cmap, 
                        raw_data = site_df,
                        draw_vert_rects = True,
                        title = site + ' Mini Manual Analysis')
    output_graph(site, 'Mini Manual', save_files, make_all_graphs)

    # Edge Analysis
    cmap_edge = {c:'Oranges' for c in edge_c_cols} | {n:'Blues' for n in edge_n_cols} # the |" is used to merge dicts
    graph = create_graph(df = edge_pt, 
                        items = edge_cols,
                        cmap = cmap_edge, 
                        raw_data = site_df,
                        draw_horiz_rects = True,
                        title = site + ' Edge Analysis')
    output_graph(site, 'Edge Analysis', save_files, make_all_graphs)


    # Pattern Matching Analysis
    cmap_pm = {'Male':'Greens', 'Female':'Purples', 'Young Nestling':'Blues', 'Mid Nestling':'Blues', 'Old Nestling':'Blues'}
    graph = create_graph(df = pm_pt, 
                        items = file_types, 
                        cmap = cmap_pm, 
                        title = site + ' Pattern Matching Analysis')
    output_graph(site, 'Pattern Matching Analysis', save_files, make_all_graphs)

    #We are all done with graphs. This must be the last thing before weather and data tables
    plt.close(graph)

    #Show weather, as needed            
    if st.sidebar.checkbox('Show station weather'):
        st.subheader('Weather Data')
        st.write(create_weather_graph(site, date_range_dict))




if len(site_list[bad_files]) > 0:
    with st.expander("See possibly bad filenames"):  
        st.write(site_list[bad_files])

