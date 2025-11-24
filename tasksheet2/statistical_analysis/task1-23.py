import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from utils import load_and_clean_data,normalize 
from utils import compare_datasets , compute_common_states , quantize_valves, compute_ccdf

sensor_cols = [
    # Flow transmitters + converted rates
    "DM-FT01", "DM-FT01Z",
    "DM-FT02", "DM-FT02Z",
    "DM-FT03", "DM-FT03Z",
    # Level
    "DM-LIT01",
    # Pressure
    "DM-PIT01", "DM-PIT02",
    # Temperatures (main/heat tanks)
    "DM-TIT01", "DM-TIT02",
    # Tank temperatures & extra pressure
    "DM-TWIT-03", "DM-TWIT-04", "DM-TWIT-05",
    "DM-PWIT-03",
    #"DM-LSH-03",
    #"DM-LSH-04",
    #"DM-LSH01",
    #"DM-LSH02",
    #"DM-LSL-04", 
    #"DM-LSL01", 
    #"DM-LSL02"
]
actuators =  [
    "DM-FCV01-Z", "DM-FCV02-Z", "DM-FCV03-Z",
    "DM-LCV01-Z",
    "DM-PCV01-Z", "DM-PCV02-Z",

    "DM-PP01A-R",  # main pump A running
    "DM-PP01B-R",  # main pump B running
    "DM-PP02-R",    # heating-water pump running

    "DM-PP04-AO"   # cooling water pump speed (Hz)
]

train_files = sorted(glob("../datasets/haiend-23.05/end-train*.csv"))
test_files = sorted(glob("../datasets/haiend-23.05/end-test*.csv"))
label_files = sorted(glob("../datasets/haiend-23.05/label-test*.csv"))


train_df, test_df = load_and_clean_data(train_files, test_files, attack_cols=None, label_files=label_files)

# return normalized sensors only
train_normalized, test_normalized = normalize(train_df[sensor_cols], test_df[sensor_cols])

# 1a Computing K-S statistics of sensor values
ks_statistics_without_states = compare_datasets(train_normalized, test_normalized)


plt.bar(range(len(ks_statistics_without_states)), list(ks_statistics_without_states.values()))
plt.xticks(range(len(ks_statistics_without_states)), list(ks_statistics_without_states.keys()), rotation=90)
plt.ylabel("K–S statistic (end23_train vs end23_test)")
plt.tight_layout()
plt.savefig('23_ks_statistics_without_states.png', dpi=300, bbox_inches='tight')

# 1b Extending code to calculate system states
a_train_df = train_df.copy()
a_test_df = test_df.copy()

a_train_df = quantize_valves(a_train_df, actuators, ignore=[], step=5) # mutating df
print(a_train_df[actuators])
a_test_df = quantize_valves(a_test_df, actuators, ignore=[], step=5) # mutating df

train_s, test_s, common_states = compute_common_states(a_train_df, a_test_df, actuators)

#calculating the percentage of system states
train_coverage_by_test = len(common_states) / len(train_s)
test_coverage_by_train = len(common_states) / len(test_s)

print("Train coverage by Test: {:.2f}%".format(train_coverage_by_test * 100))
print("Test coverage by Train: {:.2f}%".format(test_coverage_by_train * 100))
print("Number of common states: {}".format(len(common_states)))


a_train_df_normalized, a_test_df_normalized = normalize(a_train_df[sensor_cols], a_test_df[sensor_cols], Q=None)

#1c For each common state, get all matching sensor values
ks_statistics_per_state = {}

for state in common_states:
    train_mask = (a_train_df[actuators] == pd.Series(state, index=actuators)).all(axis=1)
    train_matching_rows = a_train_df_normalized[train_mask]
    #train_matching_rows[sensor_cols]
    test_mask = (a_test_df[actuators] == pd.Series(state, index=actuators)).all(axis=1)
    test_matching_rows = a_test_df_normalized[test_mask]

    ks_statistics_per_state[state] = compare_datasets(train_matching_rows[sensor_cols], test_matching_rows[sensor_cols])


ks_df = pd.DataFrame(ks_statistics_per_state).T

# Compute mean KS per sensor
avg_ks_per_sensor = ks_df.mean(axis=0).sort_values(ascending=False)

print("Average KS per sensor:")
print(avg_ks_per_sensor)

# 1c.2 Combine all sensor data for common states
# Collect all sensor rows for common states from both train and test
all_train_sensor_rows = []
all_test_sensor_rows = []

for state in common_states:
    # Find all rows in a_train_df that match this state
    train_mask = (a_train_df[actuators] == pd.Series(state, index=actuators)).all(axis=1)
    train_matching_rows = a_train_df.loc[train_mask, sensor_cols]
    
    # Add state identifier to each row
    train_matching_rows = train_matching_rows.copy()
    train_matching_rows['state'] = str(state)
    
    all_train_sensor_rows.append(train_matching_rows)
    
    # Find all rows in a_test_df that match this state
    test_mask = (a_test_df[actuators] == pd.Series(state, index=actuators)).all(axis=1)
    test_matching_rows = a_test_df.loc[test_mask, sensor_cols]
    
    # Add state identifier to each row
    test_matching_rows = test_matching_rows.copy()
    test_matching_rows['state'] = str(state)
    
    all_test_sensor_rows.append(test_matching_rows)

# Combine all sensor data into DataFrames
train_sensor_data_df = pd.concat(all_train_sensor_rows, ignore_index=True)
test_sensor_data_df = pd.concat(all_test_sensor_rows, ignore_index=True)

print(f"Train shape: {train_sensor_data_df.shape}")
print(f"Test shape: {test_sensor_data_df.shape}")


# Compute K–S statistics on combined sensor data for common states
ks_statistic_with_states = compare_datasets(train_sensor_data_df[sensor_cols], test_sensor_data_df[sensor_cols])


plt.bar(range(len(ks_statistic_with_states)), list(ks_statistic_with_states.values()))
plt.xticks(range(len(ks_statistic_with_states)), list(ks_statistic_with_states.keys()), rotation=90)
plt.ylabel("K–S statistic (end23_train vs end23_test)")
plt.tight_layout()
plt.savefig('ks_statistics_with_states.png', dpi=300, bbox_inches='tight')

#1d Plot CCDFs of K–S statistics
x1, y1 = compute_ccdf(list(ks_statistic_with_states.values()))
x2, y2 = compute_ccdf(list(ks_statistics_without_states.values()))

plt.figure(figsize=(7,5))
plt.plot(x1, y1, label='With system states', color='red')
plt.plot(x2, y2, label='Without system states', color='blue')
plt.xlabel('K–S statistic')
plt.ylabel('P(KS ≥ x)')
plt.title('CCDFs of K–S statistics (end23 Train vs Test)')
plt.legend()
plt.grid(True)
plt.savefig('ccdf-end23.png', dpi=300, bbox_inches='tight')