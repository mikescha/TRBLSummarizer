import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from pathlib import Path
import os
import calendar

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
sites = 'sites'
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

columns = {data_filename : 'filename', 
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

#
#Graphing-related constants
#
tick_spacing = 14


data_foldername = 'Data/'
data_filename = 'data.csv'
site_info_filename = 'sites.csv'
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
def site_name(file_name:str, sites:list) -> str:
    site = ''
    for s in sites:
        if file_name[0:len(s)] == s:
            site = s
            break
    return site

def get_target_sites() -> dict:
    file_summary = {}
    for t in file_types:
        file_summary[t] = []
    file_summary[bad_files] = []
    file_summary[sites] = set()

    #Load the list of unique site names, keep just the 'Name' column, and then convert that to a list
    data_dir = Path(__file__).parents[0] / data_foldername
    site_info_csv = data_dir / site_info_filename
    site_list = pd.read_csv(site_info_csv, usecols = ['Name'])
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
                            file_summary[sites].add(s)
                            found = True
                            break
        if not found:
            if f != data_filename and f != site_info_filename: 
                file_summary[bad_files].append(f)
    
    #Confirm that there are the same set of files for each type
    if len(file_summary[sites]) > 0:
        for t in file_types:
            if len(file_summary[sites]) != len(file_summary[t]):
                if len(file_summary[t]) == 0:
                    show_error('Missing all files of type ' + t)
                else:
                    show_error('Wrong number of files of type ' + t)
    else:
        show_error('No site files found')

    return file_summary

def get_site_to_analyze(site_list:list) -> str:
    return st.sidebar.selectbox('Site to summarize', site_list)

#
#
#Main
#
#
st.title('TRBL Summary')

#Get the list of sites that we're going to do reports for
site_list = get_target_sites()
site = get_site_to_analyze(site_list[sites])
st.subheader(site)





if len(site_list[bad_files]) == 0:
    with st.expander("See possibly bad filenames"):  
        st.write(site_list[bad_files])

