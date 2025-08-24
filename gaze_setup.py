import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score

anno = pd.read_csv("Annotations.csv")

target_rows = 540
list_of_gaze_df = []

# Process each gaze file
for gaze_path in anno['gaze']:
    gaze_dataframe = pd.read_csv(gaze_path)

    # Drop unnecessary columns
    gaze_dataframe.drop(["USER", "CS", "CY", "CX", "TIMETICK"], axis=1, inplace=True)

    # Convert timestamp and set index
    gaze_dataframe['CNT'] = pd.to_datetime(gaze_dataframe['CNT'], unit='s')
    gaze_dataframe.set_index('CNT', inplace=True)

    # Interpolate to a fixed number of time steps
    new_index = pd.date_range(start=gaze_dataframe.index.min(), end=gaze_dataframe.index.max(), periods=target_rows)
    gaze_dataframe = gaze_dataframe.reindex(new_index).interpolate(method='time').ffill().head(target_rows)

    # Flatten the dataframe into a 1D feature vector
    np_gaze_dataframe = gaze_dataframe.to_numpy().flatten()
    list_of_gaze_df.append(np_gaze_dataframe)

# Construct the final DataFrame
final_dataframe = pd.DataFrame(data=list_of_gaze_df)
final_dataframe['truth'] = anno['truth']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(final_dataframe.drop(columns=["truth"]))
y = final_dataframe['truth']

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42)

# Evaluate model with cross-validation
scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
print(f"Cross-validated accuracy: {scores.mean():.4f} ± {scores.std():.4f}")



final_dataframe.to_csv("gaze_df_2.csv", index=False)
