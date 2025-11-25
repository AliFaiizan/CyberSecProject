import os
import pandas as pd
from haien_mapping import haien_mapping_dict

# ---------------------------------------------------------
# Verified continuous (physical) sensors per dataset
# ---------------------------------------------------------
PHYSICAL_2103 = [
    "P1_FCV01Z","P1_FCV02Z","P1_FCV03Z",
    "P1_FT01","P1_FT01Z","P1_FT02","P1_FT02Z","P1_FT03","P1_FT03Z",
    "P1_LCV01Z","P1_LIT01","P1_PCV01Z","P1_PCV02Z",
    "P1_PIT01","P1_PIT02","P1_TIT01","P1_TIT02",
    "P2_24Vdc","P2_SIT01","P2_VT01",
    "P3_FIT01","P3_LIT01","P3_PIT01",
    "P4_HT_FD","P4_HT_PO","P4_ST_FD","P4_ST_GOV","P4_ST_PO","P4_ST_PT01","P4_ST_TT01"
]

PHYSICAL_2204 = [
    "P1_FCV01Z","P1_FCV02Z","P1_FCV03Z",
    "P1_FT01","P1_FT01Z","P1_FT02","P1_FT02Z","P1_FT03","P1_FT03Z",
    "P1_LCV01Z","P1_LIT01","P1_PCV01Z","P1_PIT01","P1_PIT02",
    "P1_TIT01","P1_TIT02","P1_TIT03",
    "P2_24Vdc","P2_SCST","P2_VIBTR01","P2_VIBTR02","P2_VIBTR03","P2_VIBTR04","P2_VT01",
    "P3_FIT01","P3_LIT01","P3_PIT01",
    "P4_HT_FD","P4_HT_PO","P4_ST_FD","P4_ST_GOV","P4_ST_PO","P4_ST_PT01","P4_ST_TT01"
]

PHYSICAL_HAIEND = [
    "DM-FCV01-Z","DM-FCV02-Z","DM-FCV03-Z",
    "DM-FT01","DM-FT01Z","DM-FT02","DM-FT02Z","DM-FT03","DM-FT03Z",
    "DM-LCV01-Z","DM-LIT01","DM-PCV01-Z","DM-PIT01","DM-PIT02",
    "DM-TIT01","DM-TIT02","GATEOPEN","DM-PP04-AO"
]

PHYSICAL_HAIEND_MAPPED = [haien_mapping_dict.get(s, s) for s in PHYSICAL_HAIEND]

# ---------------------------------------------------------
# Utility to save a list into continuous_sensors.txt
# ---------------------------------------------------------
def save_list(folder, sensor_list):
    out_path = os.path.join(folder, "continuous_sensors.txt")
    os.makedirs(folder, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(sensor_list))
    print(f"  Saved {len(sensor_list)} sensors → {out_path}")

# ---------------------------------------------------------
# Auto-detect base path (Windows / Linux)
# ---------------------------------------------------------
if os.name == "nt":  # Windows
    ROOT = r"C:\Users\Aser\Documents\Study_Project\datasets"
else:  # Linux (Tahiti)
    ROOT = os.path.expanduser("~/Study_Project/datasets")

# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
def main():
    datasets = {
        "hai-21.03": PHYSICAL_2103,
        "hai-22.04": PHYSICAL_2204,
        "haiend-23.05": PHYSICAL_HAIEND_MAPPED
    }

    print("=== UNIVERSAL PHYSICAL SENSOR EXPORTER ===\n")

    for name, sensors in datasets.items():
        folder = os.path.join(ROOT, name)
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue
        save_list(folder, sensors)

    print("\n=== Completed saving continuous physical sensors for all datasets ===")

if __name__ == "__main__":
    main()
