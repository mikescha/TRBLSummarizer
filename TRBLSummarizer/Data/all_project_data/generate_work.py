import pandas as pd
import json
import glob
import os
from pathlib import Path

def load_invalid_filenames(filename="invalid_filenames.txt"):
    """Load the list of invalid filenames."""
    try:
        with open(filename, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(filenames)} invalid filenames from {filename}")
        return filenames
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

def load_all_csv_files():
    """Load all recordings CSV files into memory."""
    print("Loading CSV files...")
    
    # Find all recordings files
    pattern = "recordings.[0-9][0-9][0-9][0-9].csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found matching pattern '{pattern}'")
        return None
    
    print(f"Found {len(csv_files)} CSV files matching pattern")
    
    # Load and combine all dataframes
    all_dfs = []
    for i, csv_file in enumerate(sorted(csv_files)):
        try:
            print(f"Loading {csv_file} ({i+1}/{len(csv_files)})")
            df = pd.read_csv(csv_file)
            
            # Ensure recording_id is string for consistency
            df['recording_id'] = df['recording_id'].astype(str)
            
            all_dfs.append(df[['meta', 'recording_id']])
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_dfs:
        print("Error: No CSV files could be loaded")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset has {len(combined_df)} total rows")
    
    return combined_df

def parse_meta_column(meta_str):
    """Parse the meta column JSON string and extract filename."""
    try:
        meta_dict = json.loads(meta_str)
        return meta_dict.get('filename', None)
    except (json.JSONDecodeError, TypeError):
        return None

def find_recording_ids(df, invalid_filenames):
    """Find recording IDs for each invalid filename."""
    print(f"\nSearching for recording IDs...")
    
    filename_to_id = {}
    errors = []
    
    # Parse all meta columns once to create a lookup
    print("Parsing meta columns...")
    df['parsed_filename'] = df['meta'].apply(parse_meta_column)
    
    # Create a mapping from filename to recording_id
    filename_lookup = df.set_index('parsed_filename')['recording_id'].to_dict()
    
    # Remove None keys (failed parsing)
    filename_lookup = {k: v for k, v in filename_lookup.items() if k is not None}
    
    print(f"Created lookup table with {len(filename_lookup)} parsed filenames")
    
    # Find each invalid filename
    for i, filename in enumerate(invalid_filenames):
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(invalid_filenames)} processed")
        
        if filename in filename_lookup:
            recording_id = filename_lookup[filename]
            filename_to_id[filename] = recording_id
        else:
            error_msg = f"Filename not found: {filename}"
            print(f"  Warning: {error_msg}")
            errors.append(error_msg)
    
    print(f"\nFound {len(filename_to_id)} matching recording IDs")
    print(f"Errors: {len(errors)}")
    
    return filename_to_id, errors

def create_html_file(filename_to_id, output_file="validations_needed.html"):
    """Create the HTML file with checkboxes and links."""
    print(f"\nCreating HTML file: {output_file}")
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validations Needed</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        .progress-counter {
            background-color: #f0f8ff;
            border: 1px solid #ccc;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-weight: bold;
            color: #333;
        }
        .item {
            margin: 8px 0;
            padding: 5px;
        }
        .item.completed {
            background-color: #f0f8f0;
            border-left: 3px solid #28a745;
        }
        .separator {
            border-top: 1px solid #ccc;
            margin: 20px 0;
            height: 1px;
        }
        label {
            cursor: pointer;
            margin-left: 8px;
        }
        label:hover {
            color: #0066cc;
            text-decoration: underline;
        }
        a {
            color: inherit;
            text-decoration: none;
        }
        a:hover {
            color: #0066cc;
            text-decoration: underline;
        }
        .counter {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>Work To Do</h1>
    <div class="progress-counter" id="progress-counter">
        Loading progress...
    </div>
    <div class="counter">Total items: """ + str(len(filename_to_id)) + """</div>
    
    <script>
        // Local storage functions
        function saveCheckboxState(id, checked) {
            localStorage.setItem('checkbox_' + id, checked.toString());
            updateProgressCounter();
        }
        
        function loadCheckboxState(id) {
            const saved = localStorage.getItem('checkbox_' + id);
            return saved === 'true';
        }
        
        function updateProgressCounter() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]');
            const checkedCount = document.querySelectorAll('input[type="checkbox"]:checked').length;
            const totalCount = checkboxes.length;
            
            document.getElementById('progress-counter').textContent = 
                `Progress: ${checkedCount}/${totalCount} completed (${Math.round(checkedCount/totalCount*100)}%)`;
        }
        
        function initializeCheckboxes() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]');
            
            checkboxes.forEach(checkbox => {
                // Load saved state
                const isChecked = loadCheckboxState(checkbox.id);
                checkbox.checked = isChecked;
                
                // Update visual state
                updateItemVisual(checkbox);
                
                // Add event listener for future changes
                checkbox.addEventListener('change', function() {
                    saveCheckboxState(this.id, this.checked);
                    updateItemVisual(this);
                });
            });
            
            updateProgressCounter();
        }
        
        function updateItemVisual(checkbox) {
            const item = checkbox.closest('.item');
            if (checkbox.checked) {
                item.classList.add('completed');
            } else {
                item.classList.remove('completed');
            }
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializeCheckboxes);
    </script>
"""
    
    # Sort by filename for consistent ordering
    sorted_items = sorted(filename_to_id.items())
    
    for i, (filename, recording_id) in enumerate(sorted_items):
        # Add separator every 10 items
        if i > 0 and i % 10 == 0:
            html_content += '    <div class="separator"></div>\n'
        
        url = f"https://arbimon.org/project/tricolored-blackbird-breeding-phenology/visualizer/rec/{recording_id}"
        
        html_content += f"""    <div class="item">
        <input type="checkbox" id="item_{i}" name="validation_{i}">
        <label for="item_{i}">
            <a href="{url}" target="_blank">{filename}</a>
        </label>
    </div>
"""
    
    html_content += """</body>
</html>"""
    
    # Write the HTML file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully created {output_file} with {len(filename_to_id)} items")
    except Exception as e:
        print(f"Error writing HTML file: {e}")

def write_error_log(errors, log_file="errors.log"):
    """Write errors to a log file."""
    if errors:
        try:
            with open(log_file, 'w') as f:
                f.write(f"Processing errors ({len(errors)} total):\n\n")
                for error in errors:
                    f.write(f"{error}\n")
            print(f"Errors logged to {log_file}")
        except Exception as e:
            print(f"Error writing log file: {e}")

def main():
    """Main function to orchestrate the entire process."""
    print("=== Validation HTML Generator ===\n")
    
    # Step 1: Load invalid filenames
    invalid_filenames = load_invalid_filenames()
    if not invalid_filenames:
        return
    
    # Step 2: Load all CSV files
    df = load_all_csv_files()
    if df is None:
        return
    
    # Step 3: Find recording IDs
    filename_to_id, errors = find_recording_ids(df, invalid_filenames)
    
    # Step 4: Write error log if there are errors
    if errors:
        write_error_log(errors)
    
    # Step 5: Create HTML file
    if filename_to_id:
        create_html_file(filename_to_id)
    else:
        print("No valid filename/recording_id pairs found. HTML file not created.")
    
    print(f"\n=== Summary ===")
    print(f"Invalid filenames loaded: {len(invalid_filenames)}")
    print(f"Recording IDs found: {len(filename_to_id)}")
    print(f"Errors encountered: {len(errors)}")
    print(f"Success rate: {len(filename_to_id)/len(invalid_filenames)*100:.1f}%" if invalid_filenames else "N/A")

if __name__ == "__main__":
    main()
