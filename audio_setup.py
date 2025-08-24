import os
import librosa
import numpy as np
import pandas as pd

anno = pd.read_csv("Annotations.csv")

anno['video'] = anno['video'].apply(lambda p: os.path.normpath(p))

# Directory with extracted audio
audio_dir = "ExtractedAudio"

feature_list = []

for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)
            audio_path = os.path.normpath(audio_path)

            # Infer original video path from audio file
            # e.g., ExtractedAudio/User_0/run_0/video_audio.wav → ./Finalised/User_0/run_0/video.mp4
            relative_parts = audio_path.replace("ExtractedAudio" + os.sep, "").split(os.sep)
            user, run, base = relative_parts
            video_base = base.replace("_audio.wav", ".mp4")
            original_video_path = os.path.normpath(os.path.join(".", "Finalised", user, run, video_base))

            # Match with annotation
            match = anno[anno['video'].apply(lambda x: os.path.normpath(x)) == original_video_path]
            if match.empty:
                print(f"Warning: No match found for {original_video_path}")
                continue

            truth = int(match.iloc[0]['truth']) 

            # Load audio
            y, sr = librosa.load(audio_path)

            # Extract features
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

            # Combine features
            features = [audio_path, truth, zcr, centroid, bandwidth, rolloff] + chroma.tolist() + mfccs.tolist()
            feature_list.append(features)

# Create DataFrame
columns = ["path", "truth", "zcr", "centroid", "bandwidth", "rolloff"] + \
          [f"chroma_{i}" for i in range(12)] + \
          [f"mfcc_{i}" for i in range(13)]

df = pd.DataFrame(feature_list, columns=columns)

# Save to CSV
df.to_csv("audio_features_with_truth.csv", index=False)
