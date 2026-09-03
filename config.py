"""
Central configuration for the texture_force_paper analysis code.

All scripts read their input/output locations from here instead of hardcoding
paths. Override without editing this file by setting environment variables:

    TEXTURE_FORCE_DATA    -> DATA_DIR   (inputs: binned CSVs, raw sessions, friction table)
    TEXTURE_FORCE_OUTPUT  -> OUTPUT_DIR (generated figures, tables, stats)

Defaults point at ./data and ./output relative to wherever you run the script.
Set DATA_DIR to the folder that holds the two binned master CSVs, the raw
per-session folders (SubjectN_SessionM/), and dynamic_friction_long_table.csv.
"""
import os

# Root of all input data (raw + processed). Empty string or "./data" by default.
DATA_DIR = os.environ.get("TEXTURE_FORCE_DATA", "./data")

# Root for all generated figures, tables, and statistics.
OUTPUT_DIR = os.environ.get("TEXTURE_FORCE_OUTPUT", "./output")

# --- Convenience paths derived from DATA_DIR -------------------------------
# The two bin-level master tables.
BINNED_FORCE_CSV = os.path.join(
    DATA_DIR, "Subjects1_to_17_Binned_complete_added_columns.csv")
BINNED_IMAGE_CSV = os.path.join(
    DATA_DIR, "Subjects1_to_17_Binned_Images_Added_Columns.csv")

# Raw per-session folders (SubjectN_SessionM/{Force,Images,reports.csv}) live
# directly under DATA_DIR; used by Figures 4 and 5 and the friction extraction.
SESSION_DIR = DATA_DIR

# Trial-level dynamic-friction table produced by friction/Multiprocessing.py.
FRICTION_LONG_TABLE = os.path.join(DATA_DIR, "dynamic_friction_long_table.csv")
