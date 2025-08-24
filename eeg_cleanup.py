import pandas as pd
import numpy as np

# Load annotations
anno = pd.read_csv("Annotations.csv")

row_index = 1  
value = anno['eeg'].iloc[row_index]
print(f"Row {row_index}: Value = {value} | Type = {type(value)}")

# ---- Load and process only that EEG file ----
if isinstance(value, str):
    eeg_dataframe = pd.read_csv(value)

    # Show column names and dtypes
    print("\nColumn dtypes before datetime conversion:")
    print(eeg_dataframe.dtypes)

    # Ensure Timestamp is parsed properly
    eeg_dataframe['Timestamp'] = pd.to_datetime(eeg_dataframe['Timestamp'].astype(str).str.strip(), errors='coerce')

    # Print a few values to verify conversion
    print("\nTimestamps after conversion:")
    print(eeg_dataframe['Timestamp'].head())

    # Check for any parsing issues
    if eeg_dataframe['Timestamp'].isna().any():
        print("\n⚠️ Warning: Some timestamps couldn't be parsed!")

    # Set index
    eeg_dataframe.set_index('Timestamp', inplace=True)

    print("\nIndex type:", type(eeg_dataframe.index))

else:
    print("❌ EEG path is not a valid string.")
