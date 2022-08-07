
from this import d
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
    st.error(msg + " This may not work correctly.")


#
#
#File handling and setup
#
#
def get_target_sites() -> dict:
    file_summary = {}
    for t in file_types:
        file_summary[t] = []
    file_summary['bad'] = []

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
                if s.lower() == f[0:len(s)].lower(): # If the first part of the filename matches a site
                    f_type = f[len(s)+1:len(f)] # Cut off the site name
                    for t in file_types:
                        #CONVERT both to lower case before comparing
                        if t.lower() == f_type[0:len(t)].lower():
                            file_summary[t].append(f)
                            found = True
        if not found:
            if f != data_filename and f != site_info_filename: 
                file_summary['bad'].append(f)



    return file_summary

#
#
#Main
#
#
st.title('TRBL Summary')

#Get the list of sites that we're going to do reports for

get_target_sites()