#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PMJ data extraction pipeline (Python 3.13)

Implements the user's specification (2025-09-07):
- Reads CSVs with encoding="utf-8-sig"
- Treats numeric-looking IDs as strings (preserves leading zeros)
- Builds and caches a global recordings index (recording_id, site_id, datetime)
- For each site & each mapped pattern, gathers ROI hits, joins to recordings, and writes per-pattern CSVs
- If a pattern has zero ROI rows, still create the CSV with headers only
- Logs all user-facing messages to results.txt (with timestamp) and prints them
- Backs up existing 'PMJ Data' directory and 'results.txt' per naming rules
- Logs always include both pattern_matching_name and pattern_matching_id wherever an ID is logged
"""

from __future__ import annotations

import os
import sys
import glob
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
ENC_READ = "utf-8-sig"   # for reading all input CSVs
ENC_WRITE = "utf-8"      # for writing output CSVs (no BOM)
RESULTS_FILENAME = "results.txt"
RECORDINGS_INDEX_PKL = "recordings_index.pkl"
PMJ_FOLDER_NAME = "PMJ Data"
DATESTAMP_FMT = "%Y-%m-%d_%H-%M"

# Column map for pmj.csv -> output filename token
PMJ_TO_OUTPUT_MAP: Dict[str, str] = {
    "Male Song": "Male Song",
    "Male Chorus": "Male Chorus",
    "Female Song": "Female",
    "HTCH": "Hatchling",
    "NEST": "Nestling",
    "FLDG": "Fledgling",
    "Insect sp30": "Insect 30",
    "Insect sp31": "Insect 31",
    "Insect sp32": "Insect 32",
    "Insect sp33": "Insect 33",
    "Pacific Tree Frog": "Pacific Tree Frog",
    "Red-legged Frog": "Red-legged Frog",
    "Bull Frog": "Bull Frog",
}

# -----------------------------
# Utilities
# -----------------------------
def now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_for_name() -> str:
    return datetime.now().strftime(DATESTAMP_FMT)


class Logger:
    """
    Logger that prints to stdout and writes to results.txt.
    If results.txt already exists at start, it gets backed up to:
    'results backup {YYYY-MM-DD_hh-mm}.txt' (or '... (2).txt' if needed).
    """

    def __init__(self, fname: str = RESULTS_FILENAME) -> None:
        self.fname = Path(fname)
        self._prepare_results_file()

    def _prepare_results_file(self) -> None:
        if self.fname.exists():
            dt = timestamp_for_name()
            backup_base = f"results backup {dt}.txt"
            backup_path = Path(backup_base)
            if backup_path.exists():
                backup_path = Path(f"results backup {dt} (2).txt")
            try:
                self.fname.rename(backup_path)
            except Exception:
                shutil.copy2(self.fname, backup_path)
                self.fname.unlink()
        self.fname.write_text("", encoding=ENC_WRITE)

    def log(self, message: str) -> None:
        line = f"[{now_local_str()}] {message}"
        print(line)
        with self.fname.open("a", encoding=ENC_WRITE) as f:
            f.write(line + "\n")


def backup_or_create_pmj_folder(logger: Logger) -> Path:
    """
    Create 'PMJ Data' in CWD. If already exists, rename it to:
    'PMJ Data backup {YYYY-MM-DD_hh-mm}' (or add ' (2)' if already taken).
    Then create a fresh 'PMJ Data'.
    """
    pmj_path = Path(PMJ_FOLDER_NAME)
    if pmj_path.exists():
        dt = timestamp_for_name()
        backup_name = f"{PMJ_FOLDER_NAME} backup {dt}"
        backup_path = Path(backup_name)
        if backup_path.exists():
            backup_path = Path(f"{backup_name} (2)")
        try:
            pmj_path.rename(backup_path)
            logger.log(f'note: renamed existing "{PMJ_FOLDER_NAME}" to "{backup_path.name}"')
        except Exception as e:
            logger.log(f'error: failed to rename existing "{PMJ_FOLDER_NAME}" -> "{backup_path.name}": {e}')
            try:
                shutil.copytree(pmj_path, backup_path)
                shutil.rmtree(pmj_path)
                logger.log(f'note: copied and removed original "{PMJ_FOLDER_NAME}" as fallback')
            except Exception as e2:
                logger.log(f'error: could not backup "{PMJ_FOLDER_NAME}": {e2}')
                raise
    pmj_path.mkdir(parents=True, exist_ok=False)
    logger.log(f'note: created folder "{PMJ_FOLDER_NAME}"')
    return pmj_path


def read_csv_strict(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read CSV with utf-8-sig and dtype=str for all columns to preserve leading zeros.
    """
    return pd.read_csv(
        path,
        encoding=ENC_READ,
        dtype=str,
        usecols=usecols,
        keep_default_na=False,  # keep empty strings as empty, not NaN
    )


def load_pmj(logger: Logger) -> pd.DataFrame:
    path = Path("pmj.csv")
    if not path.exists():
        logger.log('error: "pmj.csv" not found in current directory')
        raise FileNotFoundError("pmj.csv")
    df = read_csv_strict(path)
    expected_cols = ["Site ID", "Site Name"] + list(PMJ_TO_OUTPUT_MAP.keys())
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logger.log(f'error: pmj.csv missing expected columns: {missing}')
        raise ValueError("pmj.csv missing expected columns")
    logger.log(f"note: loaded pmj.csv with {len(df)} rows")
    return df


def load_pattern_matchings(logger: Logger) -> pd.DataFrame:
    path = Path("pattern_matchings.0001.csv")
    if not path.exists():
        logger.log('error: "pattern_matchings.0001.csv" not found')
        raise FileNotFoundError("pattern_matchings.0001.csv")
    df = read_csv_strict(path)
    for col in ["pattern_matching_id", "name"]:
        if col not in df.columns:
            logger.log(f'error: pattern_matchings.0001.csv missing column "{col}"')
            raise ValueError("pattern_matchings.0001.csv missing required columns")
    logger.log(f"note: loaded pattern_matchings.0001.csv with {len(df)} rows")
    return df


def load_rois(logger: Logger) -> pd.DataFrame:
    """
    Load and concatenate the three ROI files in file order,
    using the corrected filenames: pattern_matching_rois.0001/0002/0003.csv
    """
    roi_files = [
        "pattern_matching_rois.0001.csv",
        "pattern_matching_rois.0002.csv",
        "pattern_matching_rois.0003.csv",
    ]
    dfs = []
    for fname in roi_files:
        path = Path(fname)
        if not path.exists():
            logger.log(f'error: ROI file missing: "{fname}"')
            raise FileNotFoundError(fname)
        df = read_csv_strict(path)
        for col in ["pattern_matching_id", "recording_id", "validated"]:
            if col not in df.columns:
                logger.log(f'error: {fname} missing required column "{col}"')
                raise ValueError(f"{fname} missing column {col}")
        dfs.append(df)
        logger.log(f"note: loaded {fname} with {len(df)} rows")
    conc = pd.concat(dfs, ignore_index=True)
    logger.log(f"note: concatenated ROI rows: {len(conc)} total")
    return conc


def build_or_load_recordings_index(logger: Logger) -> pd.DataFrame:
    """
    Load cached recordings index if present; otherwise build from recordings.*.csv files.
    Keep only: recording_id, site_id, datetime (all as str).
    """
    pkl_path = Path(RECORDINGS_INDEX_PKL)
    if pkl_path.exists():
        try:
            df = pd.read_pickle(pkl_path)
            for col in ["recording_id", "site_id", "datetime"]:
                if col not in df.columns:
                    raise ValueError("cached recordings_index.pkl missing columns")
            logger.log(f"note: loaded cached recordings index from {RECORDINGS_INDEX_PKL} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.log(f"error: failed to load {RECORDINGS_INDEX_PKL} ({e}); will rebuild")

    files = sorted(glob.glob("recordings.*.csv"))
    if not files:
        logger.log('error: no files matched pattern "recordings.*.csv"')
        raise FileNotFoundError("recordings.*.csv not found")

    usecols = ["recording_id", "site_id", "datetime"]
    dfs = []
    total_rows = 0
    for f in files:
        df = read_csv_strict(Path(f), usecols=usecols)
        dfs.append(df)
        total_rows += len(df)
        logger.log(f"note: loaded {f} with {len(df)} rows")

    all_df = pd.concat(dfs, ignore_index=True)
    logger.log(f"note: built recordings index with {len(all_df)} rows (from {len(files)} files, ~{total_rows} raw rows)")
    try:
        all_df.to_pickle(pkl_path)
        logger.log(f"note: cached recordings index to {RECORDINGS_INDEX_PKL}")
    except Exception as e:
        logger.log(f"error: failed to cache recordings index to {RECORDINGS_INDEX_PKL}: {e}")
    return all_df


def is_nd_or_empty(val: str) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if s == "":
        return True
    return s.lower() == "nd"


def validated_label(v: str) -> str:
    s = (v or "").strip()
    if s == "1":
        return "present"
    if s == "0":
        return "not present"
    return "(not validated)"


def parse_mdy_datetime(dt_str: str) -> Optional[datetime]:
    """
    Parse strings like '4/24/2017 7:40 AM' robustly.
    No infer_datetime_format (deprecated). Use pandas strict parser.
    """
    if dt_str is None:
        return None
    s = str(dt_str).strip()
    if not s:
        return None
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def resolve_pattern_id(pattern_df: pd.DataFrame, pattern_name: str, logger: Logger) -> Optional[str]:
    """
    Return the first pattern_matching_id (by file order) for a given name.
    Logs per spec, including both name and id wherever an id is logged.
    """
    matches = pattern_df.index[pattern_df["name"] == pattern_name].tolist()
    if not matches:
        logger.log(f"error: No pattern matching id found for {pattern_name}, id (not found)")
        return None
    if len(matches) > 1:
        chosen_idx = matches[0]
        chosen_id = pattern_df.at[chosen_idx, "pattern_matching_id"]
        logger.log(
            f"error: multiple pattern matching ids found for {pattern_name}; "
            f"using the first in file order. chosen id {chosen_id}"
        )
        return chosen_id
    first_idx = matches[0]
    pmid = pattern_df.at[first_idx, "pattern_matching_id"]
    # Not an error; but we may still log a debug note if desired:
    # logger.log(f"note: resolved pattern_matching_id {pmid} for {pattern_name}")
    return pmid


def ensure_site_folder(base: Path, site_name: str, logger: Logger) -> Optional[Path]:
    """
    Create PMJ Data/{Site Name}. If it already exists, log error and return None.
    """
    site_path = base / site_name
    if site_path.exists():
        logger.log(f'error: site folder already exists: "{site_path.as_posix()}" — skipping this site')
        return None
    try:
        site_path.mkdir(parents=True, exist_ok=False)
        logger.log(f'note: created site folder "{site_path.as_posix()}"')
        return site_path
    except Exception as e:
        logger.log(f'error: failed to create site folder "{site_path.as_posix()}": {e}')
        return None


def write_headers_only_csv(path: Path, logger: Logger) -> None:
    """
    Create a CSV with headers only, per required column order.
    """
    headers = ["site_id", "site", "recording_id", "year", "month", "day", "validated"]
    pd.DataFrame(columns=headers).to_csv(path, index=False, encoding=ENC_WRITE)
    logger.log(f'note: wrote headers-only file "{path.as_posix()}"')


def main() -> None:
    logger = Logger()

    # 1) Prepare PMJ Data folder
    pmj_root = backup_or_create_pmj_folder(logger)

    # Load inputs
    pmj_df = load_pmj(logger)
    pattern_df = load_pattern_matchings(logger)
    roi_df = load_rois(logger)
    recordings_index = build_or_load_recordings_index(logger)

    # Iterate pmj rows
    for row_idx, row in pmj_df.iterrows():
        site_id = (row.get("Site ID") or "").strip()
        site_name = (row.get("Site Name") or "").strip()

        if site_id == "" or site_name == "":
            logger.log(f"error: pmj row {row_idx} missing Site ID or Site Name — skipping row")
            continue

        logger.log(f'note: processing site "{site_name}" (Site ID {site_id})')

        # 3.2) Site folder
        site_folder = ensure_site_folder(pmj_root, site_name, logger)
        if site_folder is None:
            continue

        # 3.3) recordings_df for this site
        recordings_df = recordings_index[recordings_index["site_id"] == site_id][["recording_id", "site_id", "datetime"]].copy()
        logger.log(f"note: site {site_name}: found {len(recordings_df)} recordings for site_id={site_id}")

        # 3.4) For each key in the map
        for key, output_name in PMJ_TO_OUTPUT_MAP.items():
            pattern_matching_name = (row.get(key) or "").strip()

            if is_nd_or_empty(pattern_matching_name):
                logger.log(f"note: {site_name} and {key} skipped due to ND or empty")
                # Per spec: do NOT create an empty CSV in this case
                continue

            # 3.4.2) Lookup pattern_matching_id by name (first in file order)
            pm_id = resolve_pattern_id(pattern_df, pattern_matching_name, logger)
            if pm_id is None:
                # already logged with name+id
                continue

            # 3.4.3) Filter ROI rows for this pattern_matching_id
            roi_hits = roi_df[roi_df["pattern_matching_id"] == pm_id].copy()
            if roi_hits.empty:
                logger.log(
                    f"note: no recordings found for pattern matching job {pattern_matching_name}, id {pm_id}"
                )
                # Create headers-only CSV
                out_filename = f"{site_name} {output_name}.csv"
                out_path = site_folder / out_filename
                write_headers_only_csv(out_path, logger)
                continue

            # 3.4.4) Prepare output table columns
            out_rows: List[Dict[str, str]] = []
            seen_recording_ids: set[str] = set()

            # 3.4.5) For each ROI row (only if there were rows)
            for _, roi_row in roi_hits.iterrows():
                rec_id = (roi_row.get("recording_id") or "").strip()
                if rec_id == "":
                    logger.log(
                        f"error: ROI row missing recording_id (pattern {pattern_matching_name}, id {pm_id}) — skipping ROI row"
                    )
                    continue

                # Lookup datetime in recordings_df
                rec_match = recordings_df[recordings_df["recording_id"] == rec_id]
                if rec_match.empty:
                    logger.log(
                        f"error: recording_id {rec_id} not found in site recordings for site_id={site_id} "
                        f"(pattern {pattern_matching_name}, id {pm_id}) — skipping"
                    )
                    continue

                dt_str = (rec_match.iloc[0]["datetime"] or "").strip()
                dt = parse_mdy_datetime(dt_str)
                if dt is None:
                    logger.log(
                        f"error: could not parse datetime '{dt_str}' for recording_id {rec_id} "
                        f"(pattern {pattern_matching_name}, id {pm_id}) — skipping"
                    )
                    continue

                year = str(dt.year)
                month = str(dt.month)
                day = str(dt.day)

                val_label = validated_label(str(roi_row.get("validated", "")).strip())

                if rec_id in seen_recording_ids:
                    logger.log(
                        f"error: duplicate recording_id {rec_id} encountered "
                        f"(pattern {pattern_matching_name}, id {pm_id}) at site {site_name}; skipping duplicate row"
                    )
                    continue

                	
                seen_recording_ids.add(rec_id)                    

                out_rows.append({
                    "site_id": site_id,
                    "site": site_name,
                    "recording_id": rec_id,
                    "year": year,
                    "month": month,
                    "day": day,
                    "validated": val_label,
                })

            # If all ROI rows were skipped due to errors, we keep prior behavior:
            if not out_rows:
                logger.log(
                    f"note: no valid rows remained after processing for pattern {pattern_matching_name}, id {pm_id} "
                    f"at site {site_name} — no file created"
                )
                continue

            out_df = pd.DataFrame(out_rows, columns=["site_id", "site", "recording_id", "year", "month", "day", "validated"])

            # 3.4.6) Sort by date (year, month, day), then recording_id
            try:
                out_df["year_i"] = out_df["year"].astype(int)
                out_df["month_i"] = out_df["month"].astype(int)
                out_df["day_i"] = out_df["day"].astype(int)
                out_df.sort_values(by=["year_i", "month_i", "day_i", "recording_id"], inplace=True)
                out_df.drop(columns=["year_i", "month_i", "day_i"], inplace=True)
            except Exception:
                out_df.sort_values(by=["year", "month", "day", "recording_id"], inplace=True)

            # 3.4.7) Write file PMJ Data/{Site Name}/{Site Name} {Output File Name}.csv
            out_filename = f"{site_name} {output_name}.csv"
            out_path = site_folder / out_filename
            try:
                out_df.to_csv(out_path, index=False, encoding=ENC_WRITE)
                logger.log(
                    f'note: wrote "{out_path.as_posix()}" with {len(out_df)} rows '
                    f"(pattern {pattern_matching_name}, id {pm_id})"
                )
            except Exception as e:
                logger.log(f'error: failed to write "{out_path.as_posix()}": {e}')

    logger.log("note: processing complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # In case logger hasn't been created yet, print a basic error.
        print(f"[{now_local_str()}] error: unhandled exception: {e}", file=sys.stderr)
        sys.exit(1)
