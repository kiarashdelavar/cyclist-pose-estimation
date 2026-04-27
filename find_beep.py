import os
import tempfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
from moviepy import VideoFileClip


VIDEO_PATH = "data/IMG_0089.MOV"


def find_start_frame(video_path):
    video = VideoFileClip(video_path)
    fps = float(video.fps or 30.0)

    if video.audio is None:
        video.close()
        print("No audio was found in this video.")
        return 0

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.close()
    video.audio.write_audiofile(temp_audio.name, logger=None)
    video.close()

    y, sr = librosa.load(temp_audio.name, sr=None, mono=True)
    os.remove(temp_audio.name)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    if len(onset_env) == 0:
        print("No clear sound peak was found.")
        return 0

    threshold = np.percentile(onset_env, 95)
    candidates = np.where(onset_env >= threshold)[0]
    best_index = int(candidates[0]) if len(candidates) > 0 else int(np.argmax(onset_env))

    beep_time = float(times[best_index])
    start_frame = int(round(beep_time * fps))

    print("-" * 40)
    print(f"Start sound time: {beep_time:.3f} seconds")
    print(f"Start frame: {start_frame}")
    print("Check this frame in the video before using it.")
    print("-" * 40)

    plt.figure(figsize=(10, 4))
    plt.plot(times, onset_env, label="Audio onset")
    plt.axvline(x=beep_time, linestyle="--", linewidth=2, label=f"Start frame {start_frame}")
    plt.title("Start sound check")
    plt.xlabel("Time in seconds")
    plt.ylabel("Sound change")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return start_frame


if __name__ == "__main__":
    find_start_frame(VIDEO_PATH)
