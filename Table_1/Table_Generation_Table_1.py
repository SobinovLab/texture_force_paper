"""
Table_Generation_Table_1.py

Formats the long-form stats from Data_Generation_Table_1.py into the wide
display layout used in the manuscript: rows = Texture, columns grouped by
Task x Force Type x {Median +/- IQR, 99th Percentile}, values rounded to
2 significant figures.
"""

import os
import numpy as np
import pandas as pd

TASK_ORDER = ['Hardness', 'Roughness', 'Slipperiness']
FORCE_ORDER = ['Normal', 'Tangential']


def _fmt2g(x):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.2g}"


def _fmt_median_iqr(med, iqr):
    if np.isnan(med) or np.isnan(iqr):
        return "NA"
    return f"{float(med):.2g} \u00b1 {float(iqr):.2g}"


def build_table1_wide(base_path):
    long_path = os.path.join(base_path, "Table_1", "Data_Table_1_long.csv")
    agg = pd.read_csv(long_path)

    agg['Median \u00b1 IQR'] = [_fmt_median_iqr(m, i) for m, i in zip(agg['median'], agg['iqr'])]
    agg['P99_str'] = [_fmt2g(v) for v in agg['p99']]

    wide_parts = []
    for task in TASK_ORDER:
        for force in FORCE_ORDER:
            sub = agg[(agg['Task'] == task) & (agg['Force Type'] == force)]
            part = sub.set_index('Texture')[['Median \u00b1 IQR', 'P99_str']]
            part = part.rename(columns={
                'Median \u00b1 IQR': f"{task} | {force} | Median \u00b1 IQR",
                'P99_str': f"{task} | {force} | 99th Percentile"
            })
            wide_parts.append(part)

    wide_df = pd.concat(wide_parts, axis=1)

    try:
        wide_df = wide_df.sort_index(key=lambda x: x.map(lambda v: float(v)))
    except Exception:
        wide_df = wide_df.sort_index()

    out_path = os.path.join(base_path, "Table_1", "Data_Table_1_wide.csv")
    wide_df.to_csv(out_path)
    print(f"Saved Table 1 (formatted) to: {out_path}")

    return wide_df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import OUTPUT_DIR

    build_table1_wide(OUTPUT_DIR)
