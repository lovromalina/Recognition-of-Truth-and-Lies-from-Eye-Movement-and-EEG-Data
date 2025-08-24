import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


anno = pd.read_csv("Annotations.csv")

# LBP settings
radius = 1
n_points = 8 * radius
METHOD = 'uniform'

def extract_lbp_features_from_video(video_path, num_frames=20):
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            times = np.linspace(0, duration, num_frames + 2)[1:-1]

            features = []
            for t in times:
                frame = clip.get_frame(t)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                lbp = local_binary_pattern(gray, n_points, radius, METHOD)
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-6)
                features.extend(hist)

            return np.array(features)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def process_video(row):
    video_path = row['video']
    truth = row['truth']
    if not isinstance(video_path, str) or not os.path.exists(video_path):
        return None

    lbp_features = extract_lbp_features_from_video(video_path)
    if lbp_features is not None:
        return {
            **{f'feature_{j}': val for j, val in enumerate(lbp_features)},
            'truth': truth,
            'path': video_path
        }
    return None

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support() 

    anno = pd.read_csv("Annotations.csv")

    # Parallel processing
    records = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_video, row) for _, row in anno.iterrows()]
        for i, future in enumerate(tqdm(futures, desc="Processing Videos")):
            result = future.result()
            if result:
                records.append(result)

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv("video_features.csv", index=False)
