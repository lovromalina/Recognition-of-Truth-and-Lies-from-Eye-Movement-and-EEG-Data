import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

anno = pd.read_csv("Annotations.csv")

original_lengths = []
list_of_eeg_df = []
processed_indices = []  
min_row = 1000
target_rows = 1000

# Process each EEG CSV file path from the 'eeg' column
for idx, eeg_path in anno['eeg'].dropna().items():

    if not isinstance(eeg_path, str):
        continue

    eeg_dataframe = pd.read_csv(eeg_path)

    # Drop unwanted columns if they exist
    eeg_dataframe.drop("Z Value", axis=1, inplace=True, errors='ignore')

    cols_to_convert = [col for col in eeg_dataframe.columns if col != 'Timestamp']
    eeg_dataframe[cols_to_convert] = eeg_dataframe[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    original_lengths.append(eeg_dataframe.shape[0])

    eeg_dataframe['Timestamp'] = pd.to_datetime(eeg_dataframe['Timestamp'], errors='coerce')
    invalid_ts = eeg_dataframe[eeg_dataframe['Timestamp'].isna()]
    if not invalid_ts.empty:
        print(f"Warning: Found {len(invalid_ts)} invalid Timestamp entries, dropping them. File {eeg_path}")
        eeg_dataframe = eeg_dataframe.dropna(subset=['Timestamp'])

    eeg_dataframe.set_index('Timestamp', inplace=True)

    # Resample / interpolate to fixed length
    try:
        new_index = pd.date_range(start=eeg_dataframe.index.min(), end=eeg_dataframe.index.max(), periods=target_rows)
        eeg_dataframe = eeg_dataframe.reindex(new_index).interpolate(method='time')

        # Check if we got the right shape, otherwise skip this file
        if eeg_dataframe.shape[0] != target_rows:
            print(f"Warning: Resampled EEG length is {eeg_dataframe.shape[0]} instead of {target_rows}. Skipping.")
            continue

    except Exception as e:
        print(f"Error during resampling/interpolation for file {eeg_path}: {e}")
        continue

    np_eeg_dataframe = eeg_dataframe.to_numpy().flatten()

    if eeg_dataframe.shape[0] < min_row:
        min_row = eeg_dataframe.shape[0]

    list_of_eeg_df.append(np_eeg_dataframe)
    processed_indices.append(idx)

column_names = eeg_dataframe.columns  # last file's columns

# Build final DataFrame
final_dataframe = pd.DataFrame(data=list_of_eeg_df)

# Align truth labels only for successfully processed EEG files
final_dataframe['truth'] = anno.loc[processed_indices, 'truth'].reset_index(drop=True)

median_original_length = np.median(original_lengths)
