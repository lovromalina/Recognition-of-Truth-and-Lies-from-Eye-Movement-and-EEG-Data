import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load annotations
anno = pd.read_csv("Annotations.csv")

original_lengths = []
list_of_eeg_df = []
processed_indices = []  # to keep track of successful eeg file indices
min_row = 1000
target_rows = 1000

for idx, eeg_path in anno['eeg'].dropna().items():

    if not isinstance(eeg_path, str):
        continue

    eeg_dataframe = pd.read_csv(eeg_path)

    # Drop unwanted columns if they exist
    eeg_dataframe.drop("Z Value", axis=1, inplace=True, errors='ignore')

    # Convert all columns (except Timestamp) to numeric, coercing errors to NaN
    cols_to_convert = [col for col in eeg_dataframe.columns if col != 'Timestamp']
    eeg_dataframe[cols_to_convert] = eeg_dataframe[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    original_lengths.append(eeg_dataframe.shape[0])

    # Parse timestamps
    eeg_dataframe['Timestamp'] = pd.to_datetime(eeg_dataframe['Timestamp'], errors='coerce')
    invalid_ts = eeg_dataframe[eeg_dataframe['Timestamp'].isna()]
    if not invalid_ts.empty:
        print(f"Warning: Found {len(invalid_ts)} invalid Timestamp entries, dropping them. File {eeg_path}")
        eeg_dataframe = eeg_dataframe.dropna(subset=['Timestamp'])

    eeg_dataframe.set_index('Timestamp', inplace=True)

    # Drop rows with any NaNs (optional: or fill them with 0)
    eeg_dataframe = eeg_dataframe.dropna()

    # Keep only the numerical columns
    eeg_values = eeg_dataframe.values

    # If too short, pad with zeros
    if eeg_values.shape[0] < target_rows:
        padding = np.zeros((target_rows - eeg_values.shape[0], eeg_values.shape[1]))
        eeg_values_padded = np.vstack((eeg_values, padding))  # Pad at the end
    elif eeg_values.shape[0] > target_rows:
        eeg_values_padded = eeg_values[:target_rows]  # Truncate if too long
    else:
        eeg_values_padded = eeg_values

    # Flatten and store
    np_eeg_dataframe = eeg_values_padded.flatten()

    if eeg_values.shape[0] < min_row:
        min_row = eeg_values.shape[0]

    list_of_eeg_df.append(np_eeg_dataframe)
    processed_indices.append(idx)

# Save the column names from the last file
column_names = eeg_dataframe.columns

# Build final DataFrame
final_dataframe = pd.DataFrame(data=list_of_eeg_df)

# Align truth labels only for successfully processed EEG files
final_dataframe['truth'] = anno.loc[processed_indices, 'truth'].reset_index(drop=True)

median_original_length = np.median(original_lengths)

# Split into features and target
X = final_dataframe.drop('truth', axis=1)
y = final_dataframe['truth']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed  = imputer.fit_transform(X)

# Normalize features
# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X_imputed)

# Train-test split with fixed random seed for reproducibility
random_int = 42
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=random_int)

# Train classifier
rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=random_int)
rf_classifier.fit(X_train, y_train)

# Evaluate
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("____________________")
print("Median original EEG length:", median_original_length)

# Optional: save final dataframe
final_dataframe.to_csv("eeg_df_padded.csv", index=False)
