import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from utils import load_and_clean_data, normalize 
from utils import compare_datasets , compute_common_states

def z_score_normalize(df):
    """Z-score normalization (standardization)"""
    return (df - df.mean()) / df.std()

def min_max_normalize(df):
    """Min-Max normalization to [0, 1] range"""
    return (df - df.min()) / (df.max() - df.min())

hai_21_sensor_cols = [
   'P1_FT01', #0–2,500 mmH₂O — measured flowrate of the return water tank. 
   'P1_FT02', #0–2,500 mmH₂O — measured flowrate of the heating water tank. 
   'P1_FT03', #0–2,500 mmH₂O — measured outflow rate of the return water tank. 
   'P1_LIT01', #0–720 mm — water level of the return water tank.
   'P1_PIT01',
   'P1_PIT02',
   'P1_TIT01',
   'P1_TIT02',
   'P2_SIT01',
   'P2_VT01',
   'P3_FIT01',
   'P3_LIT01',
   'P3_PIT01'
]
actuators = ['P1_PCV01Z', 'P1_PCV02Z', 'P1_LCV01Z', 'P1_FCV03Z', 'P1_FCV01Z', 'P1_FCV02Z', ]

hai_21_train_files = sorted(glob("../hai-21.03/train*.csv"))
hai_21_test_files = sorted(glob("../hai-21.03/test*.csv"))

hai_21_attack_cols = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']



# Process hai-21.03 data
# Process hai-21.03 data
train_df, test_df = load_and_clean_data(hai_21_train_files, hai_21_test_files, hai_21_attack_cols)



train_normalized, test_normalized = normalize(train_df[hai_21_sensor_cols], test_df[hai_21_sensor_cols])



ks_Static = compare_datasets(train_normalized, test_normalized)



# plt.bar(range(len(ks_Static)), list(ks_Static.values()))
# plt.xticks(range(len(ks_Static)), list(ks_Static.keys()), rotation=90)
# plt.ylabel("K–S statistic (21.03_train vs 21.03_test)")
# plt.tight_layout()
# plt.show()

train_s, test_s, common_states = compute_common_states(train_df.round(2), test_df.round(2), actuators)

train_coverage_by_test = len(common_states) / len(train_s) * 100
test_coverage_by_train = len(common_states) / len(test_s) * 100


# Convert common_states to a set of tuples for fast matching
common_states_set = set(tuple(row) for row in common_states.values)

# Create a boolean mask for rows in train_df where actuator values match any common state
mask = train_df[actuators].apply(tuple, axis=1).isin(common_states_set)

# Filter train_df to only those rows
matched_train_df = train_df[mask]

# Read sensor values for matched rows
matched_sensor_values = matched_train_df[hai_21_sensor_cols] 