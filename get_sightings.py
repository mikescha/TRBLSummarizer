import csv
import time
from datetime import date, timedelta

import requests

# === CONFIGURATION ===
API_KEY = "jvrdn0c915eh"
HOTSPOT_LOC_ID = "L2776216"  # eBird locId for the hotspot, e.g. "L99381"
SPECIES_CODE = "tribla"      # Tricolored Blackbird
OUTFILE = "tricolored_blackbird_checklists_2021_2024.csv"

HEADERS = {
    "X-eBirdApiToken": API_KEY
}

BASE_URL = "https://api.ebird.org/v2"


def get_historic_obs_for_date(loc_id: str, d: date):
    """
    Call the historic observations endpoint for a single location and date.
    Returns a list of observation dicts (possibly empty).
    """
    url = f"{BASE_URL}/data/obs/{loc_id}/historic/{d.year}/{d.month}/{d.day}"
    params = {
        "detail": "full",
        "maxResults": 10000,  # single hotspot, this should be safe
    }
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_breeding_code_for_checklist(sub_id: str, species_code: str) -> str:
    """
    Call View Checklist and extract breeding code(s) for the given species.
    Returns a semicolon-separated string of breeding codes, or "" if none.
    """
    url = f"{BASE_URL}/product/checklist/view/{sub_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Some implementations may return a list; normalize to a single dict.
    if isinstance(data, list):
        if not data:
            return ""
        data = data[0]

    breeding_codes = []

    for obs in data.get("obs", []):
        if obs.get("speciesCode") != species_code:
            continue

        # obsAux holds breeding_code info, if present
        for aux in obs.get("obsAux", []):
            if aux.get("fieldName") == "breeding_code":
                # value is the breeding code (e.g. "C4"), auxCode is a short code like "UN"
                code = aux.get("value") or aux.get("auxCode")
                if code:
                    breeding_codes.append(code)

    # Deduplicate and join
    breeding_codes = sorted(set(breeding_codes))
    return ";".join(breeding_codes) if breeding_codes else ""


def main():
    # subId -> aggregated info
    checklists = {}

    # 1. Collect all TRBL observations from historic endpoint
    for year in range(2021, 2025):  # inclusive 2021–2024
        start = date(year, 4, 1)
        end = date(year, 8, 30)
        total_days = (end - start).days + 1

        current = start
        while current <= end:
            # --- Progress Bar ---
            days_done = (current - start).days
            pct = (days_done / total_days) * 100

            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)

            print(
                f"\r{year}: {current} |{bar}| {pct:6.2f}%",
                end="",
                flush=True
            )

            # --- Call Ebird --- 
            try:
                obs_list = get_historic_obs_for_date(HOTSPOT_LOC_ID, current)
            except Exception as e:
                print(f"Error fetching {current}: {e}")
                current += timedelta(days=1)
                # small delay to be nice to the API
                time.sleep(0.2)
                continue

            for obs in obs_list:
                if obs.get("speciesCode") != SPECIES_CODE:
                    continue

                sub_id = obs.get("subId")
                if not sub_id:
                    continue

                entry = checklists.setdefault(
                    sub_id,
                    {
                        "date":obs.get("obsDt"),
                        "checklist":obs.get("checklistId"),
                        "raw_counts": [],
                        "comments": [],
                        "has_rich_media":bool(obs.get("hasRichMedia")),
                    },
                )

                how_many = obs.get("howMany")
                if how_many is not None:
                    entry["raw_counts"].append(str(how_many))
                elif obs.get("howManyStr"):
                    entry["raw_counts"].append(obs["howManyStr"])
    
                if obs.get("hasComments"):
                    oc = obs.get("obsComments")
                    if oc:
                        entry["comments"].append(oc)

            current += timedelta(days=1)
            # politeness delay – keep it small but nonzero
            time.sleep(0.2)

    print()
    print(f"Found {len(checklists)} checklists with {SPECIES_CODE} in the date range.")

    # 2. For each checklist, resolve numeric count and fetch breeding code
    rows = []
    for idx, (sub_id, info) in enumerate(checklists.items(), start=1):
        # Convert counts to numeric if possible and sum them
        numeric_counts = []
        for s in info["raw_counts"]:
            try:
                numeric_counts.append(int(s))
            except (TypeError, ValueError):
                # Non-numeric value like 'X' – ignore for numeric sum
                pass

        total_count = sum(numeric_counts) if numeric_counts else ""

        # Combine unique comments
        unique_comments = list(dict.fromkeys(info["comments"]))  # preserve order
        combined_comments = " | ".join(unique_comments)

        # Get breeding code(s) for this checklist/species
        try:
            breeding_code = get_breeding_code_for_checklist(sub_id, SPECIES_CODE)
        except Exception as e:
            print(f"Error fetching breeding code for {sub_id}: {e}")
            breeding_code = ""

        rows.append(
            {
                "checklist_id": sub_id,             # eBird submission ID (S123456789)
                "date": info["date"],
                "checklist": info["checklist"],
                "has_rich_media": info["has_rich_media"],
                "trbl_count": total_count,          # best-effort numeric count
                "trbl_raw_counts": ",".join(info["raw_counts"]),
                "trbl_comments": combined_comments,
                "trbl_breeding_code": breeding_code,
            }
        )

        # Again, a short delay between checklist calls
        time.sleep(0.2)

    # 3. Write output CSV
    fieldnames = [
        "checklist_id",
        "date",
        "checklist",
        "has_rich_media",
        "trbl_count",
        "trbl_raw_counts",
        "trbl_comments",
        "trbl_breeding_code",
    ]

    with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {OUTFILE}")


if __name__ == "__main__":
    main()
