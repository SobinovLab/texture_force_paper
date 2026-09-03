"""
Table_Generation_Table_2.py

Formats Data_Table_2_long.csv into the manuscript's wide layout:
one row per Regressor, columns grouped by Task, each showing
R2 and p-value (with a 'Significant' flag for p < 0.05, matching
"Significant relationships (p<0.05) were highlighted in bold" in the
Table 2 caption).
"""

import os
import numpy as np
import pandas as pd

BLOCK_LABELS = {'H': 'Hardness', 'S': 'Slipperiness', 'R': 'Roughness'}
BLOCK_ORDER = ['H', 'S', 'R']

REGRESSOR_DISPLAY_ORDER = [
    'Mean_Speed_Keypoint_8',
    'Max_Speed_Keypoint_8',
    'Median_Force_Fz',
    'Max_Force_Fz',
    'Median_Force_Rate_Fz',
    'Median_Force_2D_Tangential_Vector',
    'Max_Force_2D_Tangential_Vector',
    'Median_Force_Rate_2D_Tangential_Vector',
]
REGRESSOR_DISPLAY_NAME = {
    'Mean_Speed_Keypoint_8': 'Index Fingertip Speed, Mean',
    'Max_Speed_Keypoint_8': 'Index Fingertip Speed, Max',
    'Median_Force_Fz': 'Normal Force, Median',
    'Max_Force_Fz': 'Normal Force, Max',
    'Median_Force_Rate_Fz': 'Normal Force Rate, Median',
    'Median_Force_2D_Tangential_Vector': 'Tan. Force, Median',
    'Max_Force_2D_Tangential_Vector': 'Tan. Force, Max',
    'Median_Force_Rate_2D_Tangential_Vector': 'Tan. Force Rate, Median',
}


def build_table2_wide(base_path, alpha=0.05):
    long_path = os.path.join(base_path, "Table_2", "Data_Table_2_long.csv")
    df = pd.read_csv(long_path)

    rows = []
    for predictor in REGRESSOR_DISPLAY_ORDER:
        row = {'Regressor': REGRESSOR_DISPLAY_NAME[predictor]}
        for block in BLOCK_ORDER:
            sub = df[(df['Regressor'] == predictor) & (df['Block'] == block)]
            if sub.empty:
                r2, p = np.nan, np.nan
            else:
                r2 = sub['cv_r2'].iloc[0]
                p = sub['p_value'].iloc[0]
            sig = (not np.isnan(p)) and p < alpha
            r2_str = "NA" if np.isnan(r2) else f"{r2:.2f}"
            p_str = "NA" if np.isnan(p) else (f"<{alpha}" if sig else f"{p:.3f}")
            label = BLOCK_LABELS[block]
            row[f"{label} R2"] = f"**{r2_str}**" if sig else r2_str
            row[f"{label} p-value"] = f"**{p_str}**" if sig else p_str
        rows.append(row)

    wide_df = pd.DataFrame(rows)

    out_path = os.path.join(base_path, "Table_2", "Data_Table_2_wide.csv")
    wide_df.to_csv(out_path, index=False)
    print(f"Saved Table 2 (formatted) to: {out_path}")

    return wide_df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import OUTPUT_DIR

    build_table2_wide(OUTPUT_DIR)
