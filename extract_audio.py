import pandas as pd
import os
from moviepy import VideoFileClip


anno = pd.read_csv("Annotations.csv")

# Directory to save extracted audio
output_audio_dir = "ExtractedAudio"

# Loop through all video paths
for i, video_path in enumerate(anno['video']):
    if pd.isna(video_path) or not isinstance(video_path, str):
        continue

    try:
        # Remove leading './Finalised/'
        rel_path = video_path
        prefix = './Finalised/'
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix):]
        
        # Remove extension
        base_name = os.path.splitext(rel_path)[0]

        # Create full output path inside ExtractedAudio folder
        output_audio_path = os.path.join(output_audio_dir, f"{base_name}_audio.wav")

        print(output_audio_path)

        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        # Load and extract audio
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio:
            audio.write_audiofile(output_audio_path)
            print(f"[{i}] Audio saved: {output_audio_path}")
        else:
            print(f"[{i}] No audio track found in: {video_path}")
    except Exception as e:
        print(f"[{i}] Error extracting from {video_path}: {e}")
