import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
import os

def find_start_frame(video_path):
    print(f"Extracting audio from {video_path}...")
    
    # 1. Load the video to find out its Frames Per Second (fps)
    video = VideoFileClip(video_path)
    fps = video.fps
    
    # 2. Rip the audio from the video and save it temporarily
    temp_audio_path = "temp_audio.wav"
    video.audio.write_audiofile(temp_audio_path, logger=None)
    
    # 3. Load the audio into the Librosa analysis tool
    print("Analyzing audio track for the starting beep...")
    y, sr = librosa.load(temp_audio_path, sr=None)
    
    # 4. Calculate the "Energy" (volume spikes) in the audio
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # 5. Find the time of the absolute loudest spike
    beep_time = librosa.frames_to_time(onset_env.argmax(), sr=sr)
    
    # 6. Convert that time into a video frame number!
    start_frame = int(beep_time * fps)
    
    print("-" * 30)
    print(f" Beep detected at: {beep_time:.2f} seconds")
    print(f" EXACT START FRAME (T=0): {start_frame}")
    print("-" * 30)

    video.close()
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    plt.figure(figsize=(10, 4))
    times = librosa.times_like(onset_env, sr=sr)
    plt.plot(times, onset_env, label='Audio Volume', color='#1f77b4')
    plt.axvline(x=beep_time, color='red', linestyle='--', linewidth=2, label=f'Start Beep Detected (Frame {start_frame})')
    plt.title('Starting Signal Detection (T=0)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Volume Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

find_start_frame('data/IMG_0089.MOV')