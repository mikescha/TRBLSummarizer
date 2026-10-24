from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1] / "TRBLSummarizer"
sys.path.insert(0, str(APP_DIR))
import trbl_summarizer as app


class WeatherSiteIdentityTests(unittest.TestCase):
    def load_weather(self, rows, sites):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weather.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with (
                patch.object(app, "WEATHER_FILE", path),
                patch.object(app, "load_all_file", return_value=pd.DataFrame(sites)),
            ):
                return app.load_weather_data_from_file.__wrapped__()

    def weather_row(self, site_id, name):
        return {
            "Site Id": site_id, "Name": name, "date": "2018-05-01",
            "tmax_F": 72.5, "tmin_F": 40.0, "precip_in": 0.1,
            "wind_speed_mean_m_s": 2.5,
        }

    def test_old_weather_labels_use_stable_ids_and_preserve_values(self):
        data = self.load_weather(
            [self.weather_row(7, "2018 Source Marsh")],
            [{"Site ID": "7", "Name": "2018 Canonical Marsh"}],
        )
        self.assertEqual(set(data.index), {"2018 Canonical Marsh"})
        self.assertEqual(set(data["date"]), {pd.Timestamp("2018-05-01")})
        self.assertEqual(data.set_index("datatype")["value"].to_dict(), {
            "tmax": 72.5, "tmin": 40.0, "prcp": 0.1, "wspd": 2.5,
        })

    def test_unknown_id_cannot_borrow_weather_from_a_matching_name(self):
        data = self.load_weather(
            [self.weather_row(7, "old label"), self.weather_row(999, "2018 Canonical Marsh")],
            [{"Site ID": 7, "Name": "2018 Canonical Marsh"}],
        )
        self.assertEqual(len(data), 4)
        self.assertEqual(set(data.index), {"2018 Canonical Marsh"})


if __name__ == "__main__":
    unittest.main()
