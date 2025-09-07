#!/usr/bin/env python3
"""
CSV File Validator Script
Processes CSV files matching "data *.csv" pattern and checks for specific conditions.
"""

import os
import glob
import pandas as pd
from pathlib import Path

def process_csv_files():
    # Define the columns to check and validate
    columns_to_check = [
        "tag<reviewed-WS-m>",
        "tag<reviewed-MH-m>",
        "tag<reviewed-MH-h>"
    ]
    
    columns_to_validate = [
        "val<Agelaius tricolor/Alternative Song>",
        "val<Agelaius tricolor/Alternative Song 2>",
        "val<Agelaius tricolor/Common Song>",
        "val<Agelaius tricolor/Courtship Song>"
    ]
    
    # Find all CSV files matching the pattern "data *.csv"
    csv_files = glob.glob("data *.csv")
    
    if not csv_files:
        print("No CSV files matching 'data *.csv' pattern found in the current directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # List to store filenames that meet the criteria
    matching_filenames = []
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            missing_check_cols = [col for col in columns_to_check if col not in df.columns]
            missing_validate_cols = [col for col in columns_to_validate if col not in df.columns]
            
            if missing_check_cols:
                print(f"  Warning: Missing check columns in {csv_file}: {missing_check_cols}")
            
            if missing_validate_cols:
                print(f"  Warning: Missing validation columns in {csv_file}: {missing_validate_cols}")
            
            if 'filename' not in df.columns:
                print(f"  Warning: 'filename' column not found in {csv_file}")
                continue
            
            # Process each row
            for index, row in df.iterrows():
                # Check if any of the check columns contain "1"
                check_condition_met = False
                for col in columns_to_check:
                    if col in df.columns and str(row[col]).strip() == "1":
                        check_condition_met = True
                        break
                
                # If check condition is met, validate the validation columns
                if check_condition_met:
                    for col in columns_to_validate:
                        if col in df.columns and str(row[col]).strip() == "---":
                            filename_value = str(row['filename']).strip()
                            if filename_value and filename_value != 'nan':
                                matching_filenames.append(filename_value)
                                print(f"  Found match in row {index + 1}: {filename_value}")
                            break  # Only need to find one "---" in validation columns
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {str(e)}")
            continue
    
    # Save the results to a file
    output_file = "invalid_filenames.txt"
    
    if matching_filenames:
        # Remove duplicates while preserving order
        unique_filenames = list(dict.fromkeys(matching_filenames))
        
        with open(output_file, 'w') as f:
            for filename in unique_filenames:
                f.write(f"{filename}\n")
        
        print(f"\nResults saved to '{output_file}'")
        print(f"Total matching filenames found: {len(unique_filenames)}")
        print("Matching filenames:")
        for filename in unique_filenames:
            print(f"  - {filename}")
    else:
        print(f"\nNo matching filenames found.")
        # Create empty file to indicate completion
        with open(output_file, 'w') as f:
            f.write("# No matching filenames found\n")

def main():
    print("CSV File Validator Script")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Process the CSV files
    process_csv_files()
    
    print("\nScript completed.")

if __name__ == "__main__":
    main()
