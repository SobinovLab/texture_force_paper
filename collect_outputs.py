"""
collect_outputs.py

Run this LAST, after you've run the Data_Generation_*.py and
Figure_Generation_*.py / Table_Generation_*.py scripts for whichever
figures/tables you want checked. It walks the output directory and zips
every CSV, PNG, and SVG it finds into a single archive, preserving the
Figure_N / Table_N folder structure, so it can be uploaded back for
comparison against the manuscript's reported numbers.

Usage:
    python collect_outputs.py --base_path ./output

This does NOT re-run any analysis -- it only collects whatever output
files already exist under base_path.
"""

import os
import argparse
import zipfile
from datetime import datetime


INCLUDE_EXTENSIONS = ('.csv', '.png', '.svg', '.pdf')


def collect_outputs(base_path, zip_name=None):
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"base_path does not exist: {base_path}")

    if zip_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"Manuscript_Outputs_{timestamp}.zip"

    zip_path = os.path.join(base_path, zip_name)

    n_files = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_path):
            # Don't zip up a zip we're currently writing, or previous collection zips
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for fname in files:
                if fname.lower().endswith(INCLUDE_EXTENSIONS):
                    full_path = os.path.join(root, fname)
                    if os.path.abspath(full_path) == os.path.abspath(zip_path):
                        continue
                    rel_path = os.path.relpath(full_path, base_path)
                    zf.write(full_path, rel_path)
                    n_files += 1

    print(f"Zipped {n_files} files into: {zip_path}")
    return zip_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect all figure/table outputs into one zip.")
    parser.add_argument('--base_path', required=True,
                         help='Directory containing Figure_1/, Figure_2/, ..., Table_1/, Table_2/ subfolders')
    parser.add_argument('--zip_name', default=None,
                         help='Optional output zip filename (default: timestamped)')
    args = parser.parse_args()

    collect_outputs(args.base_path, zip_name=args.zip_name)
