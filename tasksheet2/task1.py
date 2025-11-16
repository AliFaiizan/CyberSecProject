from utils import load_and_clean_data
from glob import glob


train_files = sorted(glob("../haiend-23.05/end-train1.csv"))
test_files = sorted(glob("../haiend-23.05/end-test1.csv"))
label_files = sorted(glob("../haiend-23.05/label-test1.csv"))

haiEnd_df = load_and_clean_data(train_files, test_files, attack_cols=None, label_files=label_files) # merge train and test data

X = haiEnd_df.drop(columns=['label'], errors='ignore') # label here refers to attack label 0 or 1
y = haiEnd_df['label']