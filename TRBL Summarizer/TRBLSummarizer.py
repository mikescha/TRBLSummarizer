import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib import cm
from pathlib import Path
import os
import calendar
from collections import Counter

#
#
# Constants and Globals
#
#
data_foldername = 'Data'
data_filename = data_foldername + '/' + 'data.csv'
site_info_filename = data_foldername + '/' + 'sites.csv'
weather_filename = data_foldername + '/' + 'weather_history.csv'

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
tag_mhh = 'tag_fs'
tag_wsm = 'tag_m'
tag_mhe = 'tag_mc'
tag_wsh = 'tag_wsh'
tag_mhe2= 'tag_mhe2'
tag_ws  = 'tag_ws'
tag_mh  = 'tag_mh'
tag_    = 'tag_'

start_str = 'start'
end_str = 'end'
songs = [malesong, courtsong, altsong2, altsong1]
tags = [tag_wse, tag_wsm, tag_mhe, tag_mhh, tag_wsh, tag_mhe2, tag_ws, tag_mh, tag_]

columns = {filename_str  : 'filename', 
           site_str      : 'site', 
           'day'         : 'day',
           'month'       : 'month',
           'year'        : 'year',
           hour_str      : 'hour', 
           date_str      : 'date',
           tag_wse       : 'tag<reviewed-WS-e>',
           tag_wsm       : 'tag<reviewed-WS-m>',
           tag_mhe       : 'tag<reviewed-MH-e>',
           tag_mhh       : 'tag<reviewed-MH-h>',
           tag_wsh       : 'tag<reviewed-WS-h>',
           tag_mhe2      : 'tag<reviewed-MH-e2>',
           tag_ws        : 'tag<reviewed-WS>',
           tag_mh        : 'tag<reviewed-MH>',
           tag_          : 'tag<reviewed>',
           malesong      : 'val<Agelaius tricolor/Common Song>',
           altsong1      : 'val<Agelaius tricolor/Alternative Song>',
           altsong2      : 'val<Agelaius tricolor/Alternative Song 2>',
           courtsong     : 'val<Agelaius tricolor/Courtship Song>'}

friendly_names = {malesong : 'Male', 
                  courtsong: 'Chorus',
                  altsong2 : 'Female', 
                  altsong1 : 'Nestling',
                  tag_wsm  : 'WS MM',
                  tag_wse  : 'WS Edge',              
                  tag_mhh  : 'MH MM',
                  tag_mhe  : 'MH Edge',
                  tag_wsh  : 'WS-h',
                  tag_mhe2 : 'MH-e2',
                  tag_ws   : 'WS',
                  tag_mh   : 'MH',
                  tag_     : 'WS orig'}


#Make the map of column names to friendly names
col_map = {}
for song in songs:
    col_map[columns[song]] = friendly_names[song]

for song in songs:
    col_map[columns[song]] = friendly_names[song]

#Make the map of column names to friendly names
col_tag_map = {}
for tag in tags:
    col_tag_map[columns[tag]] = friendly_names[tag]

data_foldername = 'Data/'
data_dir = Path(__file__).parents[0] / data_foldername
data_filename = data_dir / 'data.csv'
site_info_filename = data_dir / 'sites.csv'
file_types = ["Young Nestling", "Mid Nestling", "Old Nestling", "Female", "Male"]

#
#
# Error logging, such as it is
#
#
def show_error(msg: str):
    st.error("Whoops! " + msg + "! This may not work correctly.")


#
#
#File handling and setup
#
#
def get_target_sites() -> dict:
    file_summary = {}
    for t in file_types:
        file_summary[t] = []
    file_summary[bad_files] = []
    file_summary[site_str] = set()

    #Load the list of unique site names, keep just the 'Name' column, and then convert that to a list
    site_list = pd.read_csv(site_info_filename, usecols = ['Name'])
    site_list = site_list['Name'].tolist()

    #Clean it up. Everything must start with a 4-digit number. More validation to be done?
    for s in site_list:
        if not s[0:3].isdigit():
            site_list.remove(s)

    #Get a list of all files in the Data directory, scan for files that match our pattern
    for f in os.listdir(data_dir):
        found = False
        if f[-4:] == '.csv':  #must be CSV
            for s in site_list:
                if s.lower() == f[0:len(s)].lower() and not found: # If the first part of the filename matches a site
                    f_type = f[len(s)+1:len(f)] # Cut off the site name
                    for t in file_types:
                        if t.lower() == f_type[0:len(t)].lower():
                            file_summary[t].append(f)
                            file_summary[site_str].add(s)
                            found = True
                            break
        if not found:
            if f != data_filename and f != site_info_filename: 
                file_summary[bad_files].append(f)
    
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

    return file_summary

# Load the CSV file into a dataframe, validate that the columns are what we expect
@st.experimental_singleton(suppress_st_warning=True)
def load_data() -> pd.DataFrame:
    data_csv = Path(__file__).parents[0] / data_filename

    #Validate the data file format
    headers = pd.read_csv(data_filename, nrows=0).columns.tolist()
    if len(headers) != len(columns):
        show_error('Data file {} has an unexpected number of columns, {} instead of {}'.
                   format(data_filename, len(headers), len(columns)))
    for col in columns:
        if not columns[col] in headers:
            show_error('Column {} missing from the data file {}'.format(columns[col], data_filename))
    
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

#TODO what other cleaning might be necessary?
@st.experimental_singleton(suppress_st_warning=True)
def clean_data(df: pd.DataFrame, site_list: list) -> pd.DataFrame:
    # Drop rows we don't need
    df_clean = pd.DataFrame()
    for site in site_list:
        df_clean = pd.concat([df_clean, df[df[site_str] == site]])

    # Interpret the "---" as zero
    df_clean = df_clean.replace('---', 0)

    # For each type of song, convert its column to be numeric instead of a string
    for s in songs:
        df_clean[columns[s]] = pd.to_numeric(df_clean[columns[s]])

    return df_clean


#
#
# Data Analysis
# 
#  


# Generate the pivot table for the site
def make_pivot_table(site_df: pd.DataFrame, labels:list, show_count:bool, date_range_dict:dict) -> pd.DataFrame:
    summary = dict()
    for label in labels:
        summary[label] = pd.pivot_table(site_df, 
                                        values = columns[label],  
                                        index = [columns[date_str]], 
                                        aggfunc = (lambda x: (x>0).sum()) if show_count else sum)

    # Add missing dates by creating the largest date range for our graph and then reindex to add missing entries
    date_range = pd.date_range(date_range_dict[start_str], date_range_dict[end_str])
    summary_pt = dict()
    for label in labels: 
        summary_pt[label] = summary[label].reindex(date_range).fillna(0)
    
    return summary_pt


# Generate the table that's a union of all data
def union_the_data(sitesummary_pt: pd.DataFrame) -> pd.DataFrame:
    union_pt = pd.DataFrame()
    for song in songs:
        union_pt = pd.concat([union_pt, sitesummary_pt[song]], axis=1)
    # rename columns to friendly names
    union_pt.rename(columns = col_map, inplace=True)
    # convert float to int
    for song in songs:
        union_pt[friendly_names[song]] = union_pt[friendly_names[song]].astype(int)

    return union_pt



#
#
# UI and other setup
# 
#  

def get_site_to_analyze(site_list:list) -> str:
    return st.sidebar.selectbox('Site to summarize', site_list)

# Set up base theme
# See https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
def set_global_theme():
    #https://matplotlib.org/stable/tutorials/introductory/customizing.html#matplotlib-rcparams
    custom_params = {'figure.dpi':'600',
                     'font.family':'Corbel', #'sans-serif'
                     'font.size':'12',
                     'font.weight':'600',
                     'font.stretch':'semi-condensed',
                     'xtick.labelsize':'medium',
#                     'ytick.labelsize':'medium',
                     'xtick.major.size':'12',
                     'xtick.color':'black',
                     'xtick.bottom':'True',
#                     'axes.labelsize':'large',
                     }
    #The base context is "notebook", and the other contexts are "paper", "talk", and "poster".
    sns.set_theme(context = 'paper', 
                  style = 'white',
                  rc = custom_params)

def get_date_range(site:str) -> dict:
    date_range_dict = {start_str : '02/01/2022', end_str : '09/01/2022'}
    months = {'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09'}

    start_month = st.sidebar.selectbox("Start month", months.keys(), index=1)
    end_month = st.sidebar.selectbox("End month", months.keys(), index=5)

    #Get the date range for the selected site, first four chars are the year name
    site_year = site[0:4]
    if site_year[0].isdigit():
        date_range_dict[start_str] = '{}-01-{}'.format(months[start_month], site_year)
        last_day = calendar.monthrange(int(site_year), int(months[end_month]))[1]
        date_range_dict[end_str] = '{}-{}-{}'.format(months[end_month], last_day, site_year)
    else:
        show_error("Site doesn't start with the year: {}".format(site))
    
    return date_range_dict


#
#
# Graphing
#
#

#The axis already has all the dates in it, but they need to be formatted. 
def format_xdateticks(date_axis:plt.Axes) -> dict:
    #Make a list of all the values
    date_values = [value for value in date_axis.xaxis.get_major_formatter().func.args[0].values()]
    #Make a list of all the values, but only use the month name. Then, count how many of each month names there are, to get the number of days/mo
    months = [pd.to_datetime(date).strftime('%B') for idx, date in enumerate(date_values)]
    counts = Counter(months)

    #Make a list of all the ticks where they have the day number only, then go through add the month to the middle ones.
    ticks = [pd.to_datetime(value).strftime('%d') for value in date_values]
    pos = 0
    for c in counts:
        mid = pos + int(counts[c]/2)
        ticks[mid] = ticks[mid] + '\n' + months[mid]
        pos += counts[c]

    #Actually set the ticks and then format them as needed
    date_axis.xaxis.set_ticklabels(ticks)
    date_axis.tick_params(axis = 'x',labelrotation = 0)
    return counts

#Take the list of month length counts we got from the function above, and draw lines at those positions. 
#Skip the last one so we don't draw over the border
def draw_overlays(month_lengths:dict, date_axis:plt.Axes):
    max = len(month_lengths)
    n = 0
    x = 0
    for m in month_lengths:
        if n<max:
            x += month_lengths[m]
            date_axis.axvline(x=x, color='black', lw=0.5)





# Create a graph, given a dataframe, list of row names, color map, and friendly names for the rows
def create_graph(df: pd.DataFrame, items:list, cmap:dict, row_names:dict, short_rows:bool, use_color_blocks:bool) -> plt.figure:

# Problems:
# How to get the rectangle to draw entirely around the graphic, including the axis labels
# How to draw vertical lines on the month start?
# Figure DPI doesn't seem to work
# For date, add a "Min/max data" value where we calculate what the correct range should be

    max = len(items)
    # Set figure size, values in inches
    w = 16
    h = 6 if short_rows else 2
    tick_spacing = 14

    #Set a mask on the zero values so that we can force them to display as white
    for col in df:
        df[col] = df[col].mask(df[col] == 0)

    # Create the base figure for the graphs
    fig, axs = plt.subplots(nrows = max, 
                            ncols = 1,
                            sharex = 'col', 
                            gridspec_kw={'height_ratios': np.repeat(1,max), 
                                         'left':0, 'right':1, 'bottom':0, 'top':1,
                                         'hspace':0},  #hspace is row spacing (gap between rows)
                            figsize=(w,h))

    i=0
    for item in items:
        # plotting the heatmap
        max_count = df[item].max(axis=1).values[0]
        axs[i] = sns.heatmap(data = df[item],
                        ax = axs[i],
                        cmap = cmap[item] if len(cmap) > 1 else cmap[0],
                        vmin = 0, vmax = max_count if max_count > 0 else 1,
                        cbar = False,
                        yticklabels = False,
                        )
        # hide the axis if there's nothing in the graph. but, we need to draw the graph so we have data for the tick labels
        if max_count == 0:
            axs[i].set_visible(False)
        
        month_counts = format_xdateticks(axs[i])
        draw_overlays(month_counts, axs[i])
        # clear the ticks on the top graphs, only show them on the bottom one
        if i < max-1:
            axs[i].set_xticks([])
            axs[i].tick_params(bottom = False)

        rect = plt.Rectangle(
            # (lower-left corner), width, height
            (0.0, 0.0), 1.0, 1.0, fill=False, color='black', lw=0.5, 
            zorder=1000, transform=fig.transFigure, figure=fig)
        fig.patches.extend([rect])

        #Add a rectangle around the data from the first non-zero day to the last
        #df[item] = the row of data we're currently graphing. want to find the first and last non-zero value in this vector
#        df_col = df[item].transpose()  #pivot to be vertical so the values are in rows instead of columns
#        df_col = df_col.reset_index()  #index by ints for easy graphing
#        df_col_nonzero = df_col[df_col[columns[item]]>0]  #get only the non-zero values

#        if len(df_col_nonzero):
#            c = cm.get_cmap(cmap[item] if len(cmap) > 1 else cmap[0], 1)(1)
#            first = df_col_nonzero.index[0]
#            last  = df_col_nonzero.index[len(df_col_nonzero)-1]
#            axs[i][0].add_patch(patches.Rectangle((first,0), last-first, 0.99, 
#                                                  ec=c, 
#                                                  fc=c, fill=use_color_blocks)) 
#            axs[i][0].add_patch(patches.Rectangle((first,0.48), last-first, 0.04, 
#                                                  ec='r', 
#                                                  fc='r', fill=True)) 
#            axs[i][0].add_patch(patches.Ellipse((first,0.5), 10, 0.5, 
#                                                  ec='b', 
#                                                  fc='b', fill=True)) 

        i += 1

    # return the final plotted heatmap
    return fig



#
#
# Main
#
#
st.title('TRBL Summary')

#Get the list of sites that we're going to do reports for
site_list = get_target_sites()
site = get_site_to_analyze(site_list[site_str])
date_range_dict = get_date_range(site)
st.subheader(site)

#Load all the data for most of the graphs
df_original = load_data()
df = clean_data(df_original, site_list[site_str])

#nuke the original data, hopefully this frees up memory
df_original = pd.DataFrame()

# Set format shared by all graphs
set_global_theme()

# Select the site matching the one of interest
site_df = df[df[columns[site_str]] == site]

# Pivot that site
sitesummary_pt = make_pivot_table(site_df, songs, False, date_range_dict)

# Create the summary of all data
union_pt = union_the_data(sitesummary_pt)

# Flip rows and columns so that we're running horizontally
sitesummary_wide = dict()
for song in songs:
    sitesummary_wide[song] = sitesummary_pt[song].transpose()

# Analyze the tags
# Get the subset of rows where there's at least one tag, i.e. the count of tags is greater than zero
# See here for an explanation of the next couple lines: https://stackoverflow.com/questions/45925327/dynamically-filtering-a-pandas-dataframe
# This is an alternative to: tagged_rows = site_df[((site_df[columns[tag_wse]]>0) | (site_df[columns[tag_mhh]]>0) ...
query = ' | '.join([f'`{columns[tag]}`>0' for tag in tags])
tagged_rows = site_df.query(query)

rowsummary_wide = dict()
if len(tagged_rows):
    rowsummary_pt = make_pivot_table(tagged_rows, tags, False, date_range_dict)    
    for tag in tags:
        rowsummary_wide[tag] = rowsummary_pt[tag].transpose()

    # Add a column where the count of recordings that contain at least one tag
    # So if a recording at 10a has two tags, that only counts as 1, because we're counting recordings not tags
    #Make a pivot table out of it
    tagged2_pt = pd.pivot_table(tagged_rows, values = columns[filename_str], 
                    index = [columns[date_str]], 
                    aggfunc = 'count')
    date_range = pd.date_range(date_range_dict['start'], date_range_dict['end'])
    tagged2_pt = tagged2_pt.reindex(date_range).fillna(0)
    #Rename the columns from the strings in the file to the friendly names
    tagged2_pt.rename(columns = {columns[filename_str]:'Tagged Files'}, inplace=True)
    #Concat this column to the original raw data
    union_pt = pd.concat([union_pt, tagged2_pt.astype(int)], axis=1)


# ------------------------------------------------------------------------------------------------
# DISPLAY
#
# See here for color options: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html

cmap = {malesong:'Greens', courtsong:'Oranges', altsong2:'Purples', altsong1:'Blues', 'bad':'Black'}
st.write(create_graph(df = sitesummary_wide, 
                        items = songs, 
                        cmap = cmap, 
                        row_names = friendly_names,
                        short_rows = False,
                        use_color_blocks = False))

#If there are any tags, then plot them, otherwise don't
if len(rowsummary_wide) > 0:
    st.write(create_graph(df = rowsummary_wide, 
                            items = tags,
                            cmap = ['Greys'],
                            row_names = friendly_names,
                            short_rows = True,
                            use_color_blocks = False))
else:
    st.write('No tags to plot')


if len(site_list[bad_files]) > 0:
    with st.expander("See possibly bad filenames"):  
        st.write(site_list[bad_files])

