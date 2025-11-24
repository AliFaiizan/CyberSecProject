import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from utils import load_and_clean_data, normalize 
from utils import compare_datasets , compute_common_states , quantize_valves, compute_ccdf


sensor_cols = [
    # Boiler Process (P1)
    'P1_FT01',  # Measured flowrate of the return water tank
    'P1_FT02',  # Measured flowrate of the heating water tank
    'P1_FT03',  # Measured flowrate of the return water tank
    'P1_PIT01', # Heat-exchanger outlet pressure
    'P1_PIT02', # Water supply pressure of the heating water pump
    'P1_LIT01', # Water level of the return water tank
    'P1_TIT01', # Heat-exchanger outlet temperature
    'P1_TIT02', # Temperature of the heating water tank

    # Turbine Process (P2)
    'P2_SIT01',    # Current turbine RPM measured by speed probe
    'P2_VT01',     # Phase lag signal of key phasor probe

    # Water Treatment Process (P3)
    'P3_FIT01',    # Flow rate of water into upper water tank
    'P3_LIT01',    # Water level of upper water tank
    'P3_PIT01',    # Pressure of water into upper water tank

    # HIL Simulation / Power Model (P4)
    'P4_HT_FD',    # Frequency deviation of hydro turbine model (HTM)
    'P4_HT_LD',    # Electrical load demand of HTM
    'P4_HT_PO',    # Output power of HTM
    'P4_HT_PS',    # Scheduled power demand of HTM
    'P4_ST_FD',    # Frequency deviation of steam turbine model (STM)
    'P4_ST_LD',    # Electrical load demand of STM
    'P4_ST_PO',    # Output power of STM
    'P4_ST_PS',    # Scheduled power demand of STM
    'P4_ST_PT01',  # Digital value of steam pressure of STM
    'P4_ST_TT01',  # Digital value of steam temperature of STM
]
actuators =  [
    'P1_FCV01Z',  # Feedback position of Flow Control Valve 01 (0–100 %)
    'P1_FCV02Z',  # Feedback position of Flow Control Valve 02 (0–100 %)
    'P1_FCV03Z',  # Feedback position of Flow Control Valve 03 (0–100 %)
    'P1_LCV01Z',  # Feedback position of Level Control Valve 01 (0–100 %)
    'P1_PCV01Z',  # Feedback position of Pressure Control Valve 01 (0–100 %)
    'P1_PCV02Z',  # Feedback position of Pressure Control Valve 02 (0–100 %)
    'P1_PP01AR',  # Running status of main pump PP01A (0/1)
    'P1_PP01BR',  # Running status of standby pump PP01B (0/1)
    'P1_PP02R'    # Running status of heating-water pump PP02 (0/1)
]

train_files = sorted(glob("../datasets/hai-21.03/train*.csv"))
test_files = sorted(glob("../datasets/hai-21.03/test*.csv"))

attack_cols = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']


train_df, test_df = load_and_clean_data(train_files, test_files, attack_cols)

# return normalized sensors only
train_normalized, test_normalized = normalize(train_df[sensor_cols], test_df[sensor_cols])

# 1a Computing K-S statistics of sensor values
ks_statistics_without_states = compare_datasets(train_normalized, test_normalized)


plt.bar(range(len(ks_statistics_without_states)), list(ks_statistics_without_states.values()))
plt.xticks(range(len(ks_statistics_without_states)), list(ks_statistics_without_states.keys()), rotation=90)
plt.ylabel("K–S statistic (21.03_train vs 21.03_test)")
plt.tight_layout()
plt.savefig('21_ks_statistics_without_states.png', dpi=300, bbox_inches='tight')

# 1b Extending code to calculate system states
a_train_df = train_df.copy()
a_test_df = test_df.copy()

a_train_df = quantize_valves(a_train_df, actuators, ignore=['P1_PP01BR','P1_PP02R'], step=5) # mutating df
print(a_train_df[actuators])
a_test_df = quantize_valves(a_test_df, actuators, ignore=['P1_PP01BR','P1_PP02R'], step=5) # mutating df

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
ks_statistics_with_states = compare_datasets(train_sensor_data_df[sensor_cols], test_sensor_data_df[sensor_cols])


plt.bar(range(len(ks_statistics_with_states)), list(ks_statistics_with_states.values()))
plt.xticks(range(len(ks_statistics_with_states)), list(ks_statistics_with_states.keys()), rotation=90)
plt.ylabel("K–S statistic score")
plt.title("(21.03_train vs 21.03_test) with states")
plt.tight_layout()
plt.savefig('21_ks_statistics_with_states.png', dpi=300, bbox_inches='tight')

#1d Plot CCDFs of K–S statistics
x1, y1 = compute_ccdf(list(ks_statistics_with_states.values()))
x2, y2 = compute_ccdf(list(ks_statistics_without_states.values()))

plt.figure(figsize=(7,5))
plt.plot(x1, y1, label='With system states', color='red')
plt.plot(x2, y2, label='Without system states', color='blue')
plt.xlabel('K–S statistic')
plt.ylabel('P(KS ≥ x)')
plt.title('CCDFs of K–S statistics (hai-21.03 Train vs Test)')
plt.legend()
plt.grid(True)
plt.savefig('ccdf-21.png', dpi=300, bbox_inches='tight')