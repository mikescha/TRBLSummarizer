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

start_str = 'start'
end_str = 'end'

columns = {filename_str  : 'filename', 
           site_str      : 'site', 
           'day'         : 'day',
           'month'       : 'month',
           'year'        : 'year',
           hour_str      : 'hour', 
           date_str      : 'date',
           tag_wse       : 'tag<reviewed-WS-e>',
           tag_wsm       : 'tag<reviewed-WS-m>',
           tag_wsh       : 'tag<reviewed-WS-h>',
           tag_mhe       : 'tag<reviewed-MH-e>',
           tag_mhm       : 'tag<reviewed-MH-m>',
           tag_mhh       : 'tag<reviewed-MH-h>',
           tag_mhe2      : 'tag<reviewed-MH-e2>',
           tag_ws        : 'tag<reviewed-WS>',
           tag_mh        : 'tag<reviewed-MH>',
           tag_          : 'tag<reviewed>',
           tag_p1c       : 'tag<p1c>',
           tag_p1n       : 'tag<p1n>',
           tag_p2c       : 'tag<p2c>',
           tag_p2n       : 'tag<p2n>',
           tag_wsmc      : 'tag<reviewed-WS-mc>',
           malesong      : 'val<Agelaius tricolor/Common Song>',
           altsong1      : 'val<Agelaius tricolor/Alternative Song>',
           altsong2      : 'val<Agelaius tricolor/Alternative Song 2>',
           courtsong     : 'val<Agelaius tricolor/Courtship Song>'}

songs = [malesong, courtsong, altsong2, altsong1]
song_columns = [columns[malesong], columns[courtsong], columns[altsong2], columns[altsong1]]
tags = [tag_wse, tag_wsm, tag_wsh, tag_mhe, tag_mhm, tag_mhh, tag_mhe2, tag_ws, tag_mh, tag_, tag_p1c, tag_p1n, tag_p2c, tag_p2n, tag_wsmc]
manual_tags = [columns[tag_mh], columns[tag_ws], columns[tag_]]
mini_manual_tags = [columns[tag_mhh], columns[tag_wsh], columns[tag_mhm], columns[tag_wsm]]
edge_tags = [columns[tag_p1c], columns[tag_p1n], columns[tag_p2c], columns[tag_p2n]]
edge_c_tags = [columns[tag_p1c], columns[tag_p2c]]
edge_n_tags = [columns[tag_p1n], columns[tag_p2n]]

data_foldername = 'Data/'
data_dir = Path(__file__).parents[0] / data_foldername
data_file = 'data.csv'
site_info_file = 'sites.csv'
data_fullfilename = data_dir / data_file
site_info_fullfilename = data_dir / site_info_file
file_types = ["Young Nestling", "Mid Nestling", "Old Nestling", "Female", "Male"]

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
            if f != data_file and f != site_info_file: 
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
    data_csv = Path(__file__).parents[0] / data_fullfilename

    #Validate the data file format
    headers = pd.read_csv(data_fullfilename, nrows=0).columns.tolist()
    if len(headers) != len(columns):
        show_error('Data file {} has an unexpected number of columns, {} instead of {}'.
                   format(data_fullfilename, len(headers), len(columns)))
    for col in columns:
        if not columns[col] in headers:
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

        #Sort descending, find first two consecutive items and drop everything after. CHECK THIS: Next line gives this error: A value is trying to be set on a copy of a slice from a DataFrame
        df_site.sort_index(inplace=True, ascending=False)
        dates = df_site.index.unique()
        for x,y in pairwise(dates):
            if abs((x-y).days) == 1:
                #found a match, need to keep only what's after this
                df_site = df_site.query("date <= '{}'".format(x.strftime('%Y-%m-%d')))
                break

        #Sort ascending, find first two consecutive items and drop everything before
        df_site.sort_index(inplace=True, ascending=True)
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

#
#
# UI and other setup
# 
#  
def get_site_to_analyze(site_list:list) -> str:
    return st.sidebar.selectbox('Site to summarize', site_list)

def get_date_range(df:pd.DataFrame) -> dict:
    df.sort_index(inplace=True)
    #Set the default date range to the first and last dates that we have data
    date_range_dict = {start_str : df.index[0].strftime("%m-%d-%Y"), end_str : df.index[len(df)-1].strftime("%m-%d-%Y")}
    
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
def set_global_theme():
    #https://matplotlib.org/stable/tutorials/introductory/customizing.html#matplotlib-rcparams
    custom_params = {'figure.dpi':'1200',
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

def get_days_per_month(df:pd.DataFrame) -> dict:
    date_list = df.columns.tolist()
    #Make a list of all the values, but only use the month name. Then, count how many of each month names there are, to get the number of days/mo
    months = [pd.to_datetime(date).strftime('%B') for date in date_list]
    return Counter(months)

#The axis already has all the dates in it, but they need to be formatted. 
def format_xdateticks(date_axis:plt.Axes):
    #Make a list of all the values
    date_values = [value for value in date_axis.xaxis.get_major_formatter().func.args[0].values()]

    #Make a list of all the ticks where they have the day number only.
    ticks = [pd.to_datetime(value).strftime('%d') for value in date_values]

    #Actually set the ticks and then format them as needed
    date_axis.xaxis.set_ticklabels(ticks)
    date_axis.tick_params(axis = 'x',labelrotation = 0)
    return

#Take the list of month length counts we got from the function above, and draw lines at those positions. 
#Skip the last one so we don't draw over the border
def draw_axis_labels(month_lengths:dict, axs:np.ndarray, gap:float):
    max = len(month_lengths)
    n = 0
    x = 0
    for month in month_lengths:
        mid = x + int(month_lengths[month]/2)
        axs[len(axs)-1].text(x=mid, y=gap*2.5, s=month, size='x-large')
        x += month_lengths[month]
        if n<max:
            for ax in axs:
                ax.axvline(x=x, color='black', lw=0.5)
            

# Create a graph, given a dataframe, list of row names, color map, and friendly names for the rows
def create_graph(df: pd.DataFrame, items:list, cmap:dict, draw_connectors=False, raw_data=pd.DataFrame, 
                 draw_vert_rects=False, draw_horiz_rects=False,title='') -> plt.figure:
    max = len(items)
    # Set figure size, values in inches
    w = 16
    h = 3
    top_gap = 0.8 if title != '' else 1
    tick_spacing = 7

    # Create the base figure for the graphs
    fig, axs = plt.subplots(nrows = max, 
                            ncols = 1,
                            sharex = 'col', 
                            gridspec_kw={'height_ratios': np.repeat(1,max), 
                                         'left':0, 'right':1, 'bottom':0, 'top':top_gap,
                                         'hspace':0},  #hspace is row spacing (gap between rows)
                            figsize=(w,h))
    # Draw the title https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html#matplotlib.pyplot.suptitle
    if len(title)>0:
        plt.suptitle(title, fontsize=36, fontweight='bold')


    #Set a mask on the zero values so that we can force them to display as white. Keep the original data as we
    #need it for drawing later. Use '<=0' because -100 is use to differentiate no data from data with zero value
    df_clean = pd.DataFrame()
    for col in df:
        df_clean[col] = df[col].mask(df[col] <= 0)

    i=0
    for item in items:
        # plotting the heatmap
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

        # hide the axis if there's nothing in the graph. but, we need to draw the graph so we have data for the tick labels
        if max_count == 0:
            axs[i].set_visible(False)
        
        # clear the ticks on the top graphs, only show them on the bottom one
        if i < max-1:
            axs[i].set_xticks([])
            axs[i].tick_params(bottom = False)
        
        #Add a rectangle around the regions of consective tags, and a line between non-consectutive if it's a N tag
        if draw_horiz_rects and len(raw_data)>0:
            df_col_nonzero = df.loc[item].to_frame()  #pull out the row we want, it turns into a column as above
            df_col_nonzero = df_col_nonzero.reset_index()   #index by ints for easy graphing
            df_col_nonzero = df_col_nonzero.query('`{}` != 0'.format(item))  #get only the nonzero values. 

            if len(df_col_nonzero):
                c = cm.get_cmap(cmap[item] if len(cmap) > 1 else cmap[0], 1)(1)
                if item in edge_c_tags: #these tags get a box around the whole block
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

                    # We now have a list of pairs of coordinates where we need a rect. For each pair, draw one.
                    for x in range(0,len(borders),2):
                        axs[i].add_patch(patches.Rectangle((borders[x],0), borders[x+1]-borders[x], 0.99, ec=c, fc=c, fill=False))
                    # For each pair of rects, draw a line between them.
                    for x in range(1,len(borders)-1,2):
                        # The +1/-1 are because we don't want to draw on top of the days, just between the days
                        axs[i].add_patch(patches.Rectangle((borders[x]+1,0.48), borders[x+1]-borders[x]-1, 0.04, ec=c, fc=c, fill=True)) 
        i += 1
    
        
    # add a rect around each day that has some data
    if draw_vert_rects and len(raw_data)>0:
        tagged_rows = filter_site(raw_data, mini_manual_tags)
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

    # Set the ticks on the axis we're going to use
    format_xdateticks(axs[max-1])
    month_counts = get_days_per_month(df_to_graph)
    draw_axis_labels(month_counts, axs, top_gap)

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
    plt.savefig(figure_path, dpi='figure', bbox_inches='tight')



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
site = get_site_to_analyze(site_list[site_str])
df = clean_data(df_original, site_list[site_str])

#nuke the original data, hopefully this frees up memory
df_original = ''

# Select the site matching the one of interest
site_df = df[df[columns[site_str]] == site]

#Using the site of interest, get the first & last dates and give the user the option to customize the range
date_range_dict = get_date_range(site_df)

#Decide if we're going to save the graphs as pics or not
save_files = st.sidebar.checkbox('Save as picture', value=False)

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
df_manual = filter_site(site_df, manual_tags)
manual_pt = make_pivot_table(df_manual, song_columns, date_range_dict)

# 
# COMMENT
#  
df_mini_manual = filter_site(site_df, mini_manual_tags)
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
for tag in edge_tags:
    df_for_tag = filter_site(site_df, [tag])
    if tag in edge_c_tags:
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




# ------------------------------------------------------------------------------------------------
# DISPLAY
#
# See here for color options: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
# Set format shared by all graphs
set_global_theme()
st.header(site)
st.subheader('Manual Analysis')
cmap = {columns[malesong]:'Greens', columns[courtsong]:'Oranges', columns[altsong2]:'Purples', columns[altsong1]:'Blues', 'bad':'Black'}
graph = create_graph(df = manual_pt, 
                     items = song_columns, 
                     cmap = cmap, 
                     title = site + ' Manual Analysis')
st.write(graph)
if save_files:
    save_figure(site, 'Manual')

st.subheader('Computer Assisted Analysis')
graph = create_graph(df = mini_manual_pt, 
                     items = song_columns, 
                     cmap = cmap, 
                     raw_data = site_df,
                     draw_vert_rects = True,
                     title = site + ' Mini Manual Analysis')
st.write(graph)
if save_files:
    save_figure(site, 'Mini_Manual')

cmap2 = {columns[tag_p1c]:'Oranges',columns[tag_p1n]:'Blues',columns[tag_p2c]:'Oranges',columns[tag_p2n]:'Blues'} 
graph = create_graph(df = edge_pt, 
                     items = edge_tags,
                     cmap = cmap2, 
                     raw_data = site_df,
                     draw_horiz_rects = True,
                     title = site + ' Mini Manual Analysis')
st.write(graph)
if save_files:
    save_figure(site, 'Edge')


if len(site_list[bad_files]) > 0:
    with st.expander("See possibly bad filenames"):  
        st.write(site_list[bad_files])

