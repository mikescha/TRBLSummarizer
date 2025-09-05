import pandas as pd
import glob
import os
import re
from pathlib import Path

def clean_column_names(df):
    """
    Clean column names by removing single quotes around content within angle brackets.
    
    Examples:
    - tag<'p1n'> -> tag<p1n>
    - column<'test'> -> column<test>
    - normal_column -> normal_column (unchanged)
    """
    # Pattern to match single quotes within angle brackets
    pattern = r"<'([^']+)'>"
    
    cleaned_columns = []
    for col in df.columns:
        # Replace <'content'> with <content>
        cleaned_col = re.sub(pattern, r'<\1>', col)
        cleaned_columns.append(cleaned_col)
    
    df.columns = cleaned_columns
    return df

def process_csv_files(file_pattern, backup=True, in_place=True):
    """
    Process CSV files to clean column headers.
    
    Args:
        file_pattern (str): Glob pattern for CSV files (e.g., "*.csv" or "data/*.csv")
        backup (bool): Create backup files before modifying
        in_place (bool): Modify files in place, otherwise create new files with _cleaned suffix
    """
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(files)} files to process:")
    
    for file_path in files:
        try:
            print(f"\nProcessing: {file_path}")
            
            # Read the CSV
            df = pd.read_csv(file_path)
            
            # Store original columns for comparison
            original_columns = list(df.columns)
            
            # Clean the column names
            df = clean_column_names(df)
            
            # Check if any changes were made
            changes_made = original_columns != list(df.columns)
            
            if changes_made:
                # Show what changed
                for orig, new in zip(original_columns, df.columns):
                    if orig != new:
                        print(f"  Changed: '{orig}' -> '{new}'")
                
                # Create backup if requested
                if backup:
                    backup_path = f"{file_path}.backup"
                    if not os.path.exists(backup_path):
                        pd.read_csv(file_path).to_csv(backup_path, index=False)
                        print(f"  Backup created: {backup_path}")
                
                # Save the cleaned file
                if in_place:
                    df.to_csv(file_path, index=False)
                    print(f"  Updated: {file_path}")
                else:
                    # Create new file with _cleaned suffix
                    path_obj = Path(file_path)
                    new_path = path_obj.parent / f"{path_obj.stem}_cleaned{path_obj.suffix}"
                    df.to_csv(new_path, index=False)
                    print(f"  Created cleaned file: {new_path}")
                    
            else:
                print("  No changes needed - headers already clean")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}")

def preview_changes(file_pattern):
    """
    Preview what changes would be made without actually modifying files.
    """
    files = glob.glob(file_pattern)
    
    print(f"Preview mode - found {len(files)} files:")
    
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            original_columns = list(df.columns)
            df_cleaned = clean_column_names(df.copy())
            
            changes_made = original_columns != list(df_cleaned.columns)
            
            print(f"\n{file_path}:")
            if changes_made:
                for orig, new in zip(original_columns, df_cleaned.columns):
                    if orig != new:
                        print(f"  Would change: '{orig}' -> '{new}'")
            else:
                print("  No changes needed")
                
        except Exception as e:
            print(f"  Error reading {file_path}: {str(e)}")

# Simple integration function for existing workflows
def load_csv_with_clean_headers(file_path):
    """
    Load a CSV file and automatically clean the headers.
    Drop-in replacement for pd.read_csv() with header cleaning.
    """
    df = pd.read_csv(file_path)
    return clean_column_names(df)

# Example usage:
if __name__ == "__main__":
    # Preview changes first
    #print("=== PREVIEW MODE ===")
    #preview_changes("*.csv")  # Adjust pattern as needed
    
    # Uncomment to actually process files:
    print("\n=== PROCESSING FILES ===")
    #process_csv_files("*.csv", backup=True, in_place=True)
    
    # Alternative: process specific files
    process_csv_files("data*.csv", backup=True, in_place=True)
    
    # Or use in your existing code:
    # df = load_csv_with_clean_headers("client_data.csv")