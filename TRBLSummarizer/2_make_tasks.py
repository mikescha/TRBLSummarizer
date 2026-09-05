import csv
import glob
import json
import os
import re

# Configuration
LOG_FILE = r".\TRBLSummarizer\error_log.txt"
CSV_DIR = r"C:\Users\mikes\GitHub\TRBL-Extractor-Data\export"
OUTPUT_CSV = r".\TRBLSummarizer\error_fixes_needed.csv"
BASE_URL = "https://arbimon.org/p/tricolored-blackbird-breeding-phenology/visualizer/rec/"

def load_target_metadata(log_path: str) -> dict:
    """
    Extracts unique target filenames along with their 'Site' and 'Date'
    from tab-delimited error_log.txt.
    """
    targets_info = {}
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found. Run the validation script first.")
        return targets_info

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rec_name = row.get("Recording_Name")
            if rec_name and not rec_name.startswith("Row_"):
                rec_name = rec_name.strip()
                if rec_name not in targets_info:
                    # Flexibly check for either 'Site'/'site' or 'Date'/'date'
                    site = row.get("Site") or row.get("site") or ""
                    date = row.get("Date") or row.get("date") or ""
                    targets_info[rec_name] = {
                        "site": site.strip(),
                        "date": date.strip()
                    }

    return targets_info

def extract_filename_from_meta(meta_str: str) -> str:
    """Fast extraction of filename from JSON meta column with regex fallback."""
    try:
        data = json.loads(meta_str)
        return data.get("filename", "")
    except Exception:
        match = re.search(r'"filename"\s*:\s*"([^"]+)"', meta_str)
        return match.group(1) if match else ""

def main():
    targets_info = load_target_metadata(LOG_FILE)
    if not targets_info:
        print("No target filenames found to lookup.")
        return

    print(f"Loaded {len(targets_info)} unique target filenames from {LOG_FILE}.")

    recordings_files = glob.glob(os.path.join(CSV_DIR, "recordings*.csv"))
    if not recordings_files:
        print(f"No CSV files found matching 'recordings*.csv' in {CSV_DIR}")
        return

    found_matches = {}  # filename -> recording_id
    remaining_targets = set(targets_info.keys())

    # Single-pass streaming through all CSV files
    for csv_file in recordings_files:
        if not remaining_targets:
            break  # Exit early if all targets have been matched

        print(f"Scanning {os.path.basename(csv_file)}...")
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta = row.get("meta", "")
                
                # Fast string filter check before parsing JSON
                if any(target in meta for target in remaining_targets):
                    filename = extract_filename_from_meta(meta)
                    if filename in remaining_targets:
                        rec_id = row.get("recording_id")
                        found_matches[filename] = rec_id
                        remaining_targets.remove(filename)
                        if not remaining_targets:
                            break

    # Write results to output CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["Filename", "Site", "Date", "Recording_ID", "Visualizer_URL", "Google_Sheets_Link"])

        for filename, info in targets_info.items():
            site = info["site"]
            date = info["date"]
            rec_id = found_matches.get(filename)
            
            if rec_id:
                raw_url = f"{BASE_URL}{rec_id}"
                sheets_link = f'=HYPERLINK("{raw_url}", "Open Visualizer")'
                writer.writerow([filename, site, date, rec_id, raw_url, sheets_link])
            else:
                writer.writerow([filename, site, date, "NOT_FOUND", "", ""])

    print("\n" + "=" * 50)
    print("Process Complete!")
    print(f"Target Filenames: {len(targets_info)}")
    print(f"Matches Found:   {len(found_matches)}")
    print(f"Missing:         {len(remaining_targets)}")
    print(f"Output saved to: {OUTPUT_CSV}")
    print("=" * 50)

if __name__ == "__main__":
    main()