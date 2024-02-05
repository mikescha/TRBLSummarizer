What To Do At The Start Of A New Season
---------------------------------------

Step 1: Get the latest list of sites and pattern matching jobs from Wendy

1. Open the tracking spreadsheet here:
https://docs.google.com/spreadsheets/d/1NQVKtxVv7zmODNuvn45TOYf-j-nU_u3Q1W-6Y_nCh-Y/edit?usp=sharing

2. Download sites.csv and copy it three places:
    - Weather project
    - PMJ Downloader project, to the PMJ Source folder
    - This, the Summarizer project

3. Download pmj.csv and copy it to PMJ Downloader Project


Step 2: Download the pattern matching jobs

1. In the PMJ Downloader project, make sure the following flag is set correctly:
    resume_if_possible = True #Set to false if we want to just nuke all the old stuff and start fresh

2. 
Step 3: Download the weather
(set flags in the code appropriately)

Step 4: Run the summarizer locally
(set flags in the code appropriately)

Step 5 (optional): Upload to the Streamlit site
1. Go to https://share.streamlit.io/
2. Click Sign In, and sign in with GitHub credentials
3. Click the moon icon indicating the app is in Sleep Mode, and follow the prompts
4. Go here to view the app:
https://mikescha-trblsummarizer-trblsummarizertrblsummarizer-l93kzs.streamlit.app/