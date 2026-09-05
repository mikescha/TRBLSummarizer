import glob
import os

import pandas as pd

DATA_DIR = os.path.join(".", "Data")
LOG_FILE = "error_log.txt"

# Column groups
COND1_TAGS = ["tag<reviewed-MH>", "tag<reviewed-WS>", "tag<reviewed>"]
COND1_VALS = [
    "val<Agelaius tricolor/Alternative Song 2>",
    "val<Agelaius tricolor/Alternative Song>",
    "val<Agelaius tricolor/Common Song>"
]

COND2_TAGS = ["tag<reviewed-WS-m>", "tag<reviewed-MH-m>"]
COND2_VALS = [
    "val<Agelaius tricolor/Alternative Song 2>",
    "val<Agelaius tricolor/Alternative Song>",
    "val<Agelaius tricolor/Common Song>",
    "val<Agelaius tricolor/Courtship Song>"
]

COND3_TAG = "tag<reviewed-MH-h>"

def is_numeric(val) -> bool:
    """Returns True if the value can be converted to float, False otherwise."""
    if pd.isna(val):
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def main():
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {DATA_DIR}")
        return

    total_errors = 0
    processed_files = 0

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("Site\tDate\tRecording_Name\tRow_Index\tError_Type\tColumn\tInvalid_Value\n")

        for file_path in parquet_files:
            file_name = os.path.basename(file_path)
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue

            processed_files += 1

            # Determine recording name column (defaults to index if not found)
            rec_col = next((col for col in ['recording_name', 'recording', 'filename', 'file_name'] if col in df.columns), None)

            for idx, row in df.iterrows():
                rec_name = str(row[rec_col]) if rec_col else f"Row_{idx}"
                site = str(row.get("site", "Unknown"))
                date = str(row.get("dt", "Unknown"))
                # --- Condition 1 ---
                c1_triggered = any(
                    tag in df.columns and is_numeric(row[tag]) and float(row[tag]) > 0
                    for tag in COND1_TAGS
                )

                if c1_triggered:
                    for val_col in COND1_VALS:
                        if val_col in df.columns:
                            val = row[val_col]
                            if not is_numeric(val):
                                log.write(f"{site}\t{date}\t{rec_name}\t{idx}\tNon-numeric in C1\t{val_col}\t{val}\n")
                                total_errors += 1

                # --- Condition 2 ---
                c2_triggered = any(
                    tag in df.columns and is_numeric(row[tag]) and float(row[tag]) > 0
                    for tag in COND2_TAGS
                )

                if c2_triggered:
                    for val_col in COND2_VALS:
                        if val_col in df.columns:
                            val = row[val_col]
                            if not is_numeric(val):
                                log.write(f"{site}\t{date}\t{rec_name}\t{idx}\tNon-numeric in C2\t{val_col}\t{val}\n")
                                total_errors += 1

                # --- Condition 3 ---
                if COND3_TAG in df.columns:
                    mh_val = row[COND3_TAG]
                    if pd.notna(mh_val):
                        # Flag any value that isn't numeric zero
                        if is_numeric(mh_val):
                            if float(mh_val) != 0:
                                log.write(f"{site}\t{date}\t{rec_name}\t{idx}\tNon-zero MH-h tag\t{COND3_TAG}\t{mh_val}\n")
                                total_errors += 1
                        else:
                            # String value or symbol non-zero
                            log.write(f"{site}\t{date}\t{rec_name}\t{idx}\tNon-zero MH-h tag\t{COND3_TAG}\t{mh_val}\n")
                            total_errors += 1

    print("=" * 50)
    print("Scan Complete!")
    print(f"Files Processed: {processed_files}")
    print(f"Total Errors Found: {total_errors}")
    print(f"Detailed log written to: {LOG_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    main()