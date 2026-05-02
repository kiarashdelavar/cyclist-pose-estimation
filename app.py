import os
import tempfile
from datetime import datetime

import cv2
import librosa
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from moviepy import VideoFileClip


st.set_page_config(page_title="Track Cycling Start Analysis", layout="wide")


st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        h1 {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
        }

        h2, h3 {
            font-weight: 750 !important;
            letter-spacing: -0.02em;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1rem;
            border-radius: 16px;
        }

        [data-testid="stInfo"] {
            border-radius: 12px;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }

        .small-muted {
            color: rgba(255, 255, 255, 0.68);
            font-size: 0.92rem;
        }

        .setup-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.75rem;
            margin-bottom: 1rem;
        }

        .setup-item {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 0.85rem;
        }

        .setup-label {
            color: rgba(255, 255, 255, 0.62);
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }

        .setup-value {
            font-weight: 700;
            font-size: 1rem;
        }

        @media (max-width: 900px) {
            .setup-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@st.cache_resource
def load_mmpose():
    from mmpose.apis import MMPoseInferencer

    return MMPoseInferencer(pose2d="human")


def clean_name(name):
    if not name:
        return "unknown_athlete"
    return "_".join(name.strip().lower().split())


def normalize_signal(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    min_value = np.min(values)
    max_value = np.max(values)

    if max_value - min_value == 0:
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def clamp_frame(frame_number, total_frames):
    max_frame = max(total_frames - 1, 0)
    return max(0, min(int(frame_number), max_frame))


def reset_video_state_if_new_video(uploaded_video, total_frames, fps):
    video_id = f"{uploaded_video.name}_{uploaded_video.size}_{total_frames}_{fps}"

    if st.session_state.get("current_video_id") != video_id:
        keys_to_clear = [
            "start_frame",
            "detected_gun_time",
            "audio_candidates_df",
            "audio_df",
            "wheel_preview",
            "wheel_message",
            "pixels_per_meter",
            "gate_preview",
            "gate_message",
        ]

        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state["current_video_id"] = video_id


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps is None or fps <= 1:
        fps = 30.0

    return fps, total_frames, width, height


def get_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        return None

    return frame


def make_roi(width, height, roi_mode, custom_roi):
    if roi_mode == "Full video":
        return 0, 0, width, height
    if roi_mode == "Left side":
        return 0, 0, int(width * 0.55), height
    if roi_mode == "Middle":
        return int(width * 0.20), 0, int(width * 0.60), height
    if roi_mode == "Right side":
        return int(width * 0.45), 0, int(width * 0.55), height

    x1 = int(width * custom_roi[0] / 100)
    x2 = int(width * custom_roi[1] / 100)
    y1 = int(height * custom_roi[2] / 100)
    y2 = int(height * custom_roi[3] / 100)

    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    return x1, y1, x2 - x1, y2 - y1


def crop_frame(frame, roi):
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def point_in_roi(point, roi):
    x, y = point
    rx, ry, rw, rh = roi
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def safe_point(keypoints, index):
    if keypoints is None or len(keypoints) <= index:
        return None

    point = keypoints[index]
    if point is None or len(point) < 2:
        return None

    x, y = float(point[0]), float(point[1])

    if np.isnan(x) or np.isnan(y):
        return None

    return [x, y]


def build_measurement(frame_number, fps, keypoints, side, pixels_per_meter, start_frame):
    if side == "Right body side":
        hip = safe_point(keypoints, KEYPOINTS["right_hip"])
        knee = safe_point(keypoints, KEYPOINTS["right_knee"])
        ankle = safe_point(keypoints, KEYPOINTS["right_ankle"])
    else:
        hip = safe_point(keypoints, KEYPOINTS["left_hip"])
        knee = safe_point(keypoints, KEYPOINTS["left_knee"])
        ankle = safe_point(keypoints, KEYPOINTS["left_ankle"])

    left_shoulder = safe_point(keypoints, KEYPOINTS["left_shoulder"])
    right_shoulder = safe_point(keypoints, KEYPOINTS["right_shoulder"])
    left_hip = safe_point(keypoints, KEYPOINTS["left_hip"])
    right_hip = safe_point(keypoints, KEYPOINTS["right_hip"])

    if hip is None or knee is None or ankle is None:
        return None

    body_points = [p for p in [left_shoulder, right_shoulder, left_hip, right_hip] if p is not None]

    if len(body_points) >= 2:
        com_x_px = float(np.mean([p[0] for p in body_points]))
        com_y_px = float(np.mean([p[1] for p in body_points]))
    else:
        com_x_px = hip[0]
        com_y_px = hip[1]

    return {
        "absolute_frame": int(frame_number),
        "time_s": (frame_number - start_frame) / fps,
        "knee_angle_deg": calculate_angle(hip, knee, ankle),
        "hip_x_px": hip[0],
        "hip_y_px": hip[1],
        "com_x_px": com_x_px,
        "com_y_px": com_y_px,
        "hip_x_m": hip[0] / pixels_per_meter,
        "hip_y_m": hip[1] / pixels_per_meter,
        "com_x_m": com_x_px / pixels_per_meter,
        "com_y_m": com_y_px / pixels_per_meter,
    }


def draw_keypoints(frame, keypoints, side):
    if frame is None or keypoints is None:
        return None

    output = frame.copy()

    pairs = [
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    for a, b in pairs:
        pa = safe_point(keypoints, a)
        pb = safe_point(keypoints, b)

        if pa is not None and pb is not None:
            cv2.line(output, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 220, 0), 2)

    for point in keypoints:
        if point is not None and len(point) >= 2:
            cv2.circle(output, (int(point[0]), int(point[1])), 4, (0, 220, 0), -1)

    if side == "Right body side":
        hip = safe_point(keypoints, KEYPOINTS["right_hip"])
        knee = safe_point(keypoints, KEYPOINTS["right_knee"])
        ankle = safe_point(keypoints, KEYPOINTS["right_ankle"])
    else:
        hip = safe_point(keypoints, KEYPOINTS["left_hip"])
        knee = safe_point(keypoints, KEYPOINTS["left_knee"])
        ankle = safe_point(keypoints, KEYPOINTS["left_ankle"])

    if hip is not None and knee is not None and ankle is not None:
        angle = calculate_angle(hip, knee, ankle)
        cv2.putText(
            output,
            f"{angle:.0f} deg",
            (int(knee[0]), int(knee[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
        )

    return output


def build_preview_targets(start_frame, fps):
    offsets = [-0.50, -0.25, 0.00, 0.25, 0.50]

    targets = []
    for offset in offsets:
        frame_number = int(round(start_frame + offset * fps))
        targets.append(
            {
                "label": f"T = {offset:+.2f} s",
                "frame": frame_number,
                "offset": offset,
            }
        )

    return targets


@st.cache_data(show_spinner=False)
def detect_start_gun_frame(video_path, search_start_s=0.0, search_end_s=None, selection_mode="Last strong peak"):
    try:
        video = VideoFileClip(video_path)
        fps = float(video.fps or 30.0)

        if video.audio is None:
            video.close()
            return 0, 0.0, "No audio was found.", pd.DataFrame(), pd.DataFrame()

        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.close()

        video.audio.write_audiofile(temp_audio.name, logger=None)
        video.close()

        y, sr = librosa.load(temp_audio.name, sr=None, mono=True)

        if os.path.exists(temp_audio.name):
            os.remove(temp_audio.name)

        if y.size == 0:
            return 0, 0.0, "The audio is empty.", pd.DataFrame(), pd.DataFrame()

        hop_length = 256
        frame_length = 1024

        volume = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        onset = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )

        volume_change = np.abs(np.diff(volume, prepend=volume[0]))

        times = librosa.frames_to_time(
            np.arange(len(volume)),
            sr=sr,
            hop_length=hop_length,
        )

        min_length = min(len(times), len(volume), len(onset), len(volume_change))
        times = times[:min_length]
        volume = volume[:min_length]
        onset = onset[:min_length]
        volume_change = volume_change[:min_length]

        if len(times) == 0:
            return 0, 0.0, "No clear sound was found.", pd.DataFrame(), pd.DataFrame()

        video_duration = float(times[-1])

        if search_end_s is None or search_end_s <= search_start_s:
            search_end_s = video_duration

        search_start_s = max(0.0, float(search_start_s))
        search_end_s = min(float(search_end_s), video_duration)

        volume_norm = normalize_signal(volume)
        onset_norm = normalize_signal(onset)
        volume_change_norm = normalize_signal(volume_change)

        gun_score = 0.50 * volume_norm + 0.35 * onset_norm + 0.15 * volume_change_norm
        gun_score_norm = normalize_signal(gun_score)

        full_audio_df = pd.DataFrame(
            {
                "time_s": times,
                "volume": volume_norm,
                "onset": onset_norm,
                "volume_change": volume_change_norm,
                "gun_score": gun_score_norm,
            }
        )

        mask = (times >= search_start_s) & (times <= search_end_s)

        if not np.any(mask):
            return 0, 0.0, "No audio was found in this search window.", pd.DataFrame(), full_audio_df

        window_times = times[mask]
        window_scores = gun_score_norm[mask]
        window_volume = volume_norm[mask]
        window_onset = onset_norm[mask]
        window_change = volume_change_norm[mask]

        if len(window_scores) == 0:
            return 0, 0.0, "No sound was found in this search window.", pd.DataFrame(), full_audio_df

        strong_threshold = max(0.45, float(np.percentile(window_scores, 90)))

        peak_rows = []

        for index in range(1, len(window_scores) - 1):
            is_local_peak = (
                window_scores[index] >= window_scores[index - 1]
                and window_scores[index] >= window_scores[index + 1]
            )

            is_strong = window_scores[index] >= strong_threshold

            if is_local_peak and is_strong:
                peak_rows.append(
                    {
                        "time_s": float(window_times[index]),
                        "frame": int(round(float(window_times[index]) * fps)),
                        "gun_score": float(window_scores[index]),
                        "volume": float(window_volume[index]),
                        "onset": float(window_onset[index]),
                        "volume_change": float(window_change[index]),
                    }
                )

        candidates_df = pd.DataFrame(peak_rows)

        if candidates_df.empty:
            best_index = int(np.argmax(window_scores))

            candidates_df = pd.DataFrame(
                [
                    {
                        "time_s": float(window_times[best_index]),
                        "frame": int(round(float(window_times[best_index]) * fps)),
                        "gun_score": float(window_scores[best_index]),
                        "volume": float(window_volume[best_index]),
                        "onset": float(window_onset[best_index]),
                        "volume_change": float(window_change[best_index]),
                    }
                ]
            )

        candidates_df = candidates_df.sort_values("time_s").reset_index(drop=True)

        if selection_mode == "First strong peak":
            selected_row = candidates_df.iloc[0]
        elif selection_mode == "Strongest peak":
            selected_row = candidates_df.sort_values("gun_score", ascending=False).iloc[0]
        else:
            selected_row = candidates_df.iloc[-1]

        best_time = float(selected_row["time_s"])
        best_frame = int(selected_row["frame"])

        candidates_df = candidates_df.sort_values("gun_score", ascending=False).head(10).reset_index(drop=True)

        message = "Start gun found. Check the graph and candidates."

        return best_frame, best_time, message, candidates_df, full_audio_df

    except Exception as error:
        return 0, 0.0, f"Audio check failed: {error}", pd.DataFrame(), pd.DataFrame()


def detect_start_gate_red_area(frame, roi=None):
    if frame is None:
        return None, "No frame was found."

    preview = frame.copy()

    if roi is not None:
        x, y, w, h = roi
        search_area = frame[y : y + h, x : x + w]
        offset_x = x
        offset_y = y
    else:
        search_area = frame
        offset_x = 0
        offset_y = 0

    hsv = cv2.cvtColor(search_area, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 70, 50])
    upper_red_2 = np.array([180, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask_1 + mask_2

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return preview, "No clear red start gate area was found."

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 20 or h < 20:
            continue

        valid_contours.append((area, x, y, w, h))

    if not valid_contours:
        return preview, "Red areas were found, but they were too small."

    valid_contours.sort(reverse=True, key=lambda item: item[0])
    area, x, y, w, h = valid_contours[0]

    x1 = x + offset_x
    y1 = y + offset_y
    x2 = x1 + w
    y2 = y1 + h

    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 3)
    cv2.putText(
        preview,
        "Possible start gate",
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0),
        2,
    )

    message = f"Possible start gate found. Box: x={x1}, y={y1}, width={w}, height={h}"
    return preview, message


def detect_wheel_pixels(frame, roi, wheel_diameter_m):
    if frame is None:
        return None, None, "No frame was found."

    area = crop_frame(frame, roi)
    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    min_radius = max(20, int(min(area.shape[:2]) * 0.08))
    max_radius = max(min_radius + 5, int(min(area.shape[:2]) * 0.45))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(50, min_radius * 2),
        param1=80,
        param2=28,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    preview = frame.copy()

    if circles is None:
        return None, preview, "No clear wheel circle was found. Use manual calibration."

    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda c: c[2], reverse=True)

    x, y, r = circles[0]
    rx, ry, _, _ = roi
    center = (int(x + rx), int(y + ry))
    radius = int(r)

    cv2.circle(preview, center, radius, (0, 220, 0), 3)
    cv2.circle(preview, center, 4, (0, 220, 0), -1)

    pixels_per_meter = (2 * radius) / wheel_diameter_m

    return pixels_per_meter, preview, "Wheel circle found. Please check if the circle is on the wheel."


def calculate_pixels_per_meter_from_points(point_a, point_b, real_distance_m):
    if real_distance_m <= 0:
        return None

    pixel_distance = float(np.linalg.norm(np.array(point_a) - np.array(point_b)))

    if pixel_distance <= 0:
        return None

    return pixel_distance / real_distance_m


def draw_manual_calibration_preview(frame, point_a, point_b):
    if frame is None:
        return None

    preview = frame.copy()

    point_a = (int(point_a[0]), int(point_a[1]))
    point_b = (int(point_b[0]), int(point_b[1]))

    cv2.circle(preview, point_a, 8, (0, 220, 0), -1)
    cv2.circle(preview, point_b, 8, (0, 220, 0), -1)
    cv2.line(preview, point_a, point_b, (0, 220, 0), 3)

    pixel_distance = float(np.linalg.norm(np.array(point_a) - np.array(point_b)))

    label_x = int((point_a[0] + point_b[0]) / 2)
    label_y = int((point_a[1] + point_b[1]) / 2)

    cv2.putText(
        preview,
        f"{pixel_distance:.1f} px",
        (label_x, label_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0),
        2,
    )

    return preview


def choose_mmpose_person(predictions, roi, target_x):
    best_keypoints = None
    best_score = -1_000_000

    for person in predictions:
        keypoints = person.get("keypoints")

        if keypoints is None:
            continue

        left_hip = safe_point(keypoints, KEYPOINTS["left_hip"])
        right_hip = safe_point(keypoints, KEYPOINTS["right_hip"])
        hips = [p for p in [left_hip, right_hip] if p is not None]

        if not hips:
            continue

        hip_x = float(np.mean([p[0] for p in hips]))
        hip_y = float(np.mean([p[1] for p in hips]))

        if not point_in_roi((hip_x, hip_y), roi):
            continue

        xs = [float(p[0]) for p in keypoints if len(p) >= 2]
        ys = [float(p[1]) for p in keypoints if len(p) >= 2]

        body_area = (max(xs) - min(xs)) * (max(ys) - min(ys)) if xs and ys else 0
        distance_penalty = abs(hip_x - target_x)
        score = body_area - distance_penalty * 5

        if score > best_score:
            best_score = score
            best_keypoints = keypoints

    return best_keypoints


def process_video(video_path, model_name, roi, side, pixels_per_meter, start_frame, max_frames, target_x):
    fps, total_frames, _, _ = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)

    if model_name == "MediaPipe":
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )
    else:
        inferencer = load_mmpose()

    rows = []
    preview_frames = []
    preview_targets = build_preview_targets(start_frame, fps)
    captured_preview_frames = set()

    frame_number = -1
    processed = 0
    frames_processed = 0
    frames_with_keypoints = 0

    progress = st.progress(0)
    status = st.empty()
    live_preview = st.empty()

    while cap.isOpened():
        success, frame = cap.read()

        if not success or frame is None:
            break

        frame_number += 1

        if max_frames > 0 and processed >= max_frames:
            break

        processed += 1
        frames_processed += 1

        progress_total = min(total_frames, max_frames if max_frames > 0 else total_frames)

        if progress_total > 0:
            progress.progress(min(processed / progress_total, 1.0))

        status.write(f"Processing frame {frame_number} of {total_frames}")

        keypoints = None

        if model_name == "MediaPipe":
            x, y, w, h = roi
            crop = frame[y : y + h, x : x + w]
            image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                keypoints = []

                for lm in result.pose_landmarks.landmark:
                    keypoints.append([lm.x * w + x, lm.y * h + y])
        else:
            result_generator = inferencer(frame, return_vis=False, show=False)
            result = next(result_generator)
            predictions = result.get("predictions", [[]])[0]
            keypoints = choose_mmpose_person(predictions, roi, target_x)

        if keypoints is not None:
            frames_with_keypoints += 1

            row = build_measurement(
                frame_number,
                fps,
                keypoints,
                side,
                pixels_per_meter,
                start_frame,
            )

            if row is not None:
                rows.append(row)
                live_frame = draw_keypoints(frame, keypoints, side)

                if live_frame is not None and processed % 5 == 0:
                    live_preview.image(
                        cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Live analysis - frame {frame_number} - knee angle {row['knee_angle_deg']:.1f} deg",
                        use_container_width=True,
                    )

                if live_frame is not None:
                    for target in preview_targets:
                        target_frame = target["frame"]

                        if target_frame < 0 or target_frame >= total_frames:
                            continue

                        if target["label"] in captured_preview_frames:
                            continue

                        if abs(frame_number - target_frame) <= 1:
                            preview_frames.append(
                                {
                                    "label": target["label"],
                                    "frame": frame_number,
                                    "image": cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB),
                                }
                            )
                            captured_preview_frames.add(target["label"])

    cap.release()
    progress.empty()
    status.empty()

    quality = {
        "frames_processed": frames_processed,
        "frames_with_keypoints": frames_with_keypoints,
    }

    if not rows:
        return pd.DataFrame(), preview_frames, fps, quality

    df = pd.DataFrame(rows)
    df = add_metrics(df, fps)

    return df, preview_frames, fps, quality


def add_metrics(df, fps):
    df = df.sort_values("absolute_frame").reset_index(drop=True)

    dt = df["absolute_frame"].diff() / fps
    dt = dt.replace(0, np.nan)

    start_rows = df[df["time_s"] >= 0]

    if len(start_rows) > 0:
        start_com_x = float(start_rows.iloc[0]["com_x_m"])
        start_hip_x = float(start_rows.iloc[0]["hip_x_m"])
    else:
        start_com_x = float(df.iloc[0]["com_x_m"])
        start_hip_x = float(df.iloc[0]["hip_x_m"])

    after_start = df[(df["time_s"] >= 0) & (df["time_s"] <= 0.8)]

    if len(after_start) > 3:
        forward_sign = np.sign(after_start["com_x_m"].median() - start_com_x)

        if forward_sign == 0:
            forward_sign = 1
    else:
        forward_sign = 1

    df["com_forward_m"] = (df["com_x_m"] - start_com_x) * forward_sign
    df["hip_forward_m"] = (df["hip_x_m"] - start_hip_x) * forward_sign

    df["hip_dx_m"] = df["hip_x_m"].diff()
    df["hip_dy_m"] = df["hip_y_m"].diff()
    df["hip_speed_m_s"] = np.sqrt(df["hip_dx_m"] ** 2 + df["hip_dy_m"] ** 2) / dt
    df["hip_forward_velocity_m_s"] = df["hip_forward_m"].diff() / dt

    df["hip_speed_smooth_m_s"] = df["hip_speed_m_s"].rolling(window=5, min_periods=1, center=True).mean()
    df["hip_forward_velocity_smooth_m_s"] = df["hip_forward_velocity_m_s"].rolling(window=5, min_periods=1, center=True).mean()
    df["knee_angle_smooth_deg"] = df["knee_angle_deg"].rolling(window=5, min_periods=1, center=True).mean()

    return df

def get_pose_quality_summary(df, quality):
    usable_frames = int(len(df))

    if quality["frames_processed"] == 0:
        pose_found_rate = 0.0
        usable_pose_rate = 0.0
    else:
        pose_found_rate = (quality["frames_with_keypoints"] / quality["frames_processed"]) * 100
        usable_pose_rate = (usable_frames / quality["frames_processed"]) * 100

    if df.empty:
        average_knee_angle = np.nan
        average_hip_velocity = np.nan
        average_body_movement = np.nan
    else:
        average_knee_angle = float(df["knee_angle_deg"].mean())
        average_hip_velocity = float(df["hip_forward_velocity_smooth_m_s"].dropna().mean())
        average_body_movement = float(df["com_forward_m"].max())

    return {
        "frames_processed": quality["frames_processed"],
        "frames_with_keypoints": quality["frames_with_keypoints"],
        "usable_frames": usable_frames,
        "pose_found_rate": pose_found_rate,
        "usable_pose_rate": usable_pose_rate,
        "average_knee_angle": average_knee_angle,
        "average_hip_velocity": average_hip_velocity,
        "average_body_movement": average_body_movement,
    }

def get_metric_summary(df):
    after = df[df["time_s"] >= 0]
    before = df[(df["time_s"] >= -0.6) & (df["time_s"] <= 0)]
    window = df[(df["time_s"] >= -0.6) & (df["time_s"] <= 0.8)]

    if after.empty:
        return {
            "v_start": np.nan,
            "t_30cm": np.nan,
            "peak_before": np.nan,
            "peak_time_before": np.nan,
            "max_forward": np.nan,
        }

    start_index = after.index[0]
    v_start = df.loc[start_index, "hip_forward_velocity_smooth_m_s"]

    t_30cm = np.nan
    reached = after[after["com_forward_m"] >= 0.30]

    if len(reached) > 0:
        t_30cm = float(reached.iloc[0]["time_s"])

    peak_before = np.nan
    peak_time_before = np.nan

    if len(before) > 0:
        peak_index = before["hip_forward_velocity_smooth_m_s"].idxmax()
        peak_before = float(df.loc[peak_index, "hip_forward_velocity_smooth_m_s"])
        peak_time_before = float(df.loc[peak_index, "time_s"])

    max_forward = float(window["com_forward_m"].max()) if len(window) > 0 else np.nan

    return {
        "v_start": v_start,
        "t_30cm": t_30cm,
        "peak_before": peak_before,
        "peak_time_before": peak_time_before,
        "max_forward": max_forward,
    }


def style_plot(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(size=13),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


def add_start_line(fig):
    fig.add_vline(x=0, line_dash="dash", annotation_text="Start")
    return fig


st.title("Track cycling start analysis")
st.markdown(
    """
    <div class="small-muted">
        Upload a cycling start video, detect the gun moment, calibrate the wheel scale,
        and inspect hip velocity, body movement, and knee angle around the start.
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_video = st.sidebar.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is None:
    st.info("Upload a video to start.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as temp_video:
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

fps, total_frames, width, height = get_video_info(video_path)
video_duration_s = total_frames / fps if fps > 0 else 0.0
reset_video_state_if_new_video(uploaded_video, total_frames, fps)

st.sidebar.header("Video data")
st.sidebar.info("This section shows the basic video information used for frame and timing calculations.")
st.sidebar.write(f"Frames: {total_frames}")
st.sidebar.write(f"FPS: {fps:.2f}")
st.sidebar.write(f"Size: {width} x {height}")
st.sidebar.write(f"Duration: {video_duration_s:.2f} s")

st.sidebar.header("Rider selection")
st.sidebar.info(
    "Select the pose model and rider area. Use Custom if there are other riders or people in the video."
)

model_name = st.sidebar.selectbox("Pose model", ["MediaPipe", "MMPose"])
body_side = st.sidebar.selectbox("Body side for knee angle", ["Right body side", "Left body side"])
roi_mode = st.sidebar.selectbox("Rider area", ["Full video", "Left side", "Middle", "Right side", "Custom"])

custom_roi = (0, 100, 0, 100)

if roi_mode == "Custom":
    x_range = st.sidebar.slider("Area left and right percent", 0, 100, (0, 100))
    y_range = st.sidebar.slider("Area top and bottom percent", 0, 100, (0, 100))
    custom_roi = (x_range[0], x_range[1], y_range[0], y_range[1])

roi = make_roi(width, height, roi_mode, custom_roi)
target_x = roi[0] + roi[2] / 2

st.sidebar.header("Start time")
st.sidebar.info(
    "Use this to find the gun sound. If the full video gives a wrong result, search only around the expected start moment."
)

audio_search_start = st.sidebar.number_input(
    "Search gun after seconds",
    min_value=0.0,
    max_value=float(video_duration_s),
    value=0.0,
    step=1.0,
)

audio_search_end = st.sidebar.number_input(
    "Search gun before seconds",
    min_value=0.0,
    max_value=float(video_duration_s),
    value=float(video_duration_s),
    step=1.0,
)

gun_selection_mode = st.sidebar.selectbox(
    "Gun detection mode",
    ["Last strong peak", "Strongest peak", "First strong peak"],
)

if st.sidebar.button("Find start gun"):
    detected_frame, detected_time, message, candidates_df, audio_df = detect_start_gun_frame(
        video_path,
        audio_search_start,
        audio_search_end,
        gun_selection_mode,
    )

    detected_frame = clamp_frame(detected_frame, total_frames)

    st.session_state["start_frame"] = detected_frame
    st.session_state["detected_gun_time"] = detected_time
    st.session_state["audio_candidates_df"] = candidates_df
    st.session_state["audio_df"] = audio_df

    st.sidebar.write(message)
    st.sidebar.write(f"Frame: {detected_frame}")
    st.sidebar.write(f"Time: {detected_time:.3f} s")

max_start_frame = max(total_frames - 1, 0)
saved_start_frame = int(st.session_state.get("start_frame", 0))
saved_start_frame = clamp_frame(saved_start_frame, total_frames)
st.session_state["start_frame"] = saved_start_frame

start_frame = st.sidebar.number_input(
    "Start frame",
    min_value=0,
    max_value=max_start_frame,
    value=saved_start_frame,
    step=1,
)

if "audio_df" in st.session_state and not st.session_state["audio_df"].empty:
    st.subheader("Start gun check")

    audio_df = st.session_state["audio_df"]
    detected_gun_time = st.session_state.get("detected_gun_time", 0.0)

    audio_plot_df = audio_df[
        (audio_df["time_s"] >= audio_search_start)
        & (audio_df["time_s"] <= audio_search_end)
    ].copy()

    audio_fig = px.line(
        audio_plot_df,
        x="time_s",
        y="gun_score",
        labels={
            "time_s": "Video time (s)",
            "gun_score": "Start gun score",
        },
    )

    audio_fig.add_vline(
        x=detected_gun_time,
        line_dash="dash",
        annotation_text="Detected start gun",
    )

    style_plot(audio_fig, "Audio gun detection score")
    st.plotly_chart(audio_fig, use_container_width=True)

if "audio_candidates_df" in st.session_state and not st.session_state["audio_candidates_df"].empty:
    st.write("Best start gun candidates")
    st.dataframe(st.session_state["audio_candidates_df"], use_container_width=True)

st.sidebar.header("Scale")
st.sidebar.info(
    "Use the front wheel to set the scale. Place two points on the left and right edge of the wheel, then click Use manual scale."
)

wheel_diameter_m = st.sidebar.number_input(
    "Wheel diameter in meters",
    min_value=0.10,
    max_value=2.00,
    value=0.67,
    step=0.01,
)

manual_pixels_per_meter = st.sidebar.number_input(
    "Fallback pixels per meter",
    min_value=1.0,
    value=400.0,
    step=1.0,
)

safe_calibration_frame = clamp_frame(start_frame, total_frames)

calibration_frame = st.sidebar.number_input(
    "Calibration frame",
    min_value=0,
    max_value=max_start_frame,
    value=safe_calibration_frame,
    step=1,
)

first_frame = get_frame(video_path, calibration_frame)

if first_frame is not None:
    roi_preview = first_frame.copy()
    x, y, w, h = roi
    cv2.rectangle(roi_preview, (x, y), (x + w, y + h), (0, 220, 0), 3)

    st.image(
        cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB),
        caption="Selected rider area",
        use_container_width=True,
    )

scale_method = st.sidebar.selectbox(
    "Scale method",
    ["Manual wheel diameter", "Automatic wheel circle"],
)

if scale_method == "Manual wheel diameter" and first_frame is not None:
    st.sidebar.write("Set two points on the wheel edge.")

    default_y = int(height * 0.62)
    default_x_left = int(width * 0.28)
    default_x_right = int(width * 0.40)

    point_a_x = st.sidebar.slider("Point A x", 0, max(width - 1, 0), default_x_left, step=1)
    point_a_y = st.sidebar.slider("Point A y", 0, max(height - 1, 0), default_y, step=1)
    point_b_x = st.sidebar.slider("Point B x", 0, max(width - 1, 0), default_x_right, step=1)
    point_b_y = st.sidebar.slider("Point B y", 0, max(height - 1, 0), default_y, step=1)

    point_a = (point_a_x, point_a_y)
    point_b = (point_b_x, point_b_y)

    manual_preview = draw_manual_calibration_preview(first_frame, point_a, point_b)

    if manual_preview is not None:
        st.image(
            cv2.cvtColor(manual_preview, cv2.COLOR_BGR2RGB),
            caption="Manual wheel calibration",
            use_container_width=True,
        )

    manual_ppm = calculate_pixels_per_meter_from_points(point_a, point_b, wheel_diameter_m)

    if manual_ppm is not None:
        st.sidebar.write(f"Measured wheel diameter: {manual_ppm * wheel_diameter_m:.1f} px")
        st.sidebar.write(f"Calculated pixels per meter: {manual_ppm:.2f}")

    if st.sidebar.button("Use manual scale") and manual_ppm is not None:
        st.session_state["pixels_per_meter"] = float(manual_ppm)
        st.session_state["wheel_message"] = "Manual wheel scale was applied."

elif scale_method == "Automatic wheel circle" and first_frame is not None:
    if st.sidebar.button("Try automatic wheel scale"):
        found_ppm, wheel_preview, wheel_message = detect_wheel_pixels(first_frame, roi, wheel_diameter_m)
        st.session_state["wheel_message"] = wheel_message

        if wheel_preview is not None:
            st.session_state["wheel_preview"] = cv2.cvtColor(wheel_preview, cv2.COLOR_BGR2RGB)

        if found_ppm is not None:
            st.session_state["pixels_per_meter"] = float(found_ppm)

st.sidebar.header("Vision tools")
st.sidebar.info(
    "This tool tries to find the start gate or timing machine in the frame. Use it as a visual check, not as a final measurement."
)

if st.sidebar.button("Find start gate") and first_frame is not None:
    gate_preview, gate_message = detect_start_gate_red_area(first_frame, roi=None)

    st.session_state["gate_message"] = gate_message

    if gate_preview is not None:
        st.session_state["gate_preview"] = cv2.cvtColor(gate_preview, cv2.COLOR_BGR2RGB)

if "gate_message" in st.session_state:
    st.sidebar.write(st.session_state["gate_message"])

if "gate_preview" in st.session_state:
    st.image(
        st.session_state["gate_preview"],
        caption="Start gate check",
        use_container_width=True,
    )

if "wheel_preview" in st.session_state:
    st.image(
        st.session_state["wheel_preview"],
        caption=st.session_state.get("wheel_message", "Wheel check"),
        use_container_width=True,
    )

if "wheel_message" in st.session_state:
    st.sidebar.write(st.session_state["wheel_message"])

pixels_per_meter = st.sidebar.number_input(
    "Final pixels per meter",
    min_value=1.0,
    value=float(st.session_state.get("pixels_per_meter", manual_pixels_per_meter)),
    step=1.0,
)

st.sidebar.header("Run")
st.sidebar.info(
    "Add the athlete name before running. Use 0 for maximum frames if you want to process the full video."
)

athlete_name = st.sidebar.text_input("Athlete name", value="")
max_frames = st.sidebar.number_input("Maximum frames to process. Use 0 for full video.", min_value=0, value=0, step=100)
plot_start = st.sidebar.number_input("Plot start time", value=-0.6, step=0.1)
plot_end = st.sidebar.number_input("Plot end time", value=0.8, step=0.1)

run_analysis = st.button("Run analysis", type="primary")

if run_analysis:
    df, preview_frames, real_fps, quality = process_video(
        video_path=video_path,
        model_name=model_name,
        roi=roi,
        side=body_side,
        pixels_per_meter=pixels_per_meter,
        start_frame=start_frame,
        max_frames=int(max_frames),
        target_x=target_x,
    )

    if df.empty:
        st.warning("No pose was found. Try a smaller rider area or use the other model.")
        st.stop()

    athlete = athlete_name.strip() if athlete_name.strip() else "Unknown athlete"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df.insert(0, "date_time", now)
    df.insert(0, "athlete", athlete)
    df.insert(2, "model", model_name)
    df.insert(3, "fps", real_fps)
    df.insert(4, "pixels_per_meter", pixels_per_meter)

    summary = get_metric_summary(df)
    pose_quality = get_pose_quality_summary(df, quality)
    st.subheader("Analysis setup")
    
    st.subheader("Pose quality summary")

    quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)

    quality_col1.metric(
    "Frames processed",
    pose_quality["frames_processed"],
    )

    quality_col2.metric(
    "Frames with pose",
    pose_quality["frames_with_keypoints"],
    )

    quality_col3.metric(
    "Usable pose frames",
    f"{pose_quality['usable_pose_rate']:.1f}%",
    )

    quality_col4.metric(
    "Average knee angle",
    "n/a" if np.isnan(pose_quality["average_knee_angle"]) else f"{pose_quality['average_knee_angle']:.1f}°",
    )

    quality_col5, quality_col6, quality_col7, quality_col8 = st.columns(4)

    quality_col5.metric(
    "Pose found rate",
    f"{pose_quality['pose_found_rate']:.1f}%",
    )

    quality_col6.metric(
    "Average hip velocity",
    "n/a" if np.isnan(pose_quality["average_hip_velocity"]) else f"{pose_quality['average_hip_velocity']:.2f} m/s",
    )

    quality_col7.metric(
    "Max body movement",
    "n/a" if np.isnan(pose_quality["average_body_movement"]) else f"{pose_quality['average_body_movement']:.2f} m",
    )

    quality_col8.metric(
    "Model checked",
    model_name,
    )

    st.write(
    "Pose quality helps you check if the selected model and rider area worked well. "
    "A low usable pose percentage means the ROI, model, or video quality should be checked."
    )

    detected_gun_time = st.session_state.get("detected_gun_time", start_frame / real_fps if real_fps else 0.0)

    st.markdown(
        f"""
        <div class="setup-grid">
            <div class="setup-item">
                <div class="setup-label">Athlete</div>
                <div class="setup-value">{athlete}</div>
            </div>
            <div class="setup-item">
                <div class="setup-label">Pose model</div>
                <div class="setup-value">{model_name}</div>
            </div>
            <div class="setup-item">
                <div class="setup-label">Start frame</div>
                <div class="setup-value">{start_frame}</div>
            </div>
            <div class="setup-item">
                <div class="setup-label">Start time</div>
                <div class="setup-value">{detected_gun_time:.3f} s</div>
            </div>
            <div class="setup-item">
                <div class="setup-label">FPS</div>
                <div class="setup-value">{real_fps:.2f}</div>
            </div>
            <div class="setup-item">
                <div class="setup-label">Scale</div>
                <div class="setup-value">{pixels_per_meter:.2f} px/m</div>
            </div>
            <div class="setup-item">
                <div class="setup-label">Rider area</div>
                <div class="setup-value">{roi_mode}</div>
            </div>
            <div class="setup-item">
                <div class="setup-label">Body side</div>
                <div class="setup-value">{body_side}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Main results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Velocity at start", "n/a" if np.isnan(summary["v_start"]) else f"{summary['v_start']:.2f} m/s")
    col2.metric("Time to 30 cm", "not reached" if np.isnan(summary["t_30cm"]) else f"{summary['t_30cm']:.3f} s")
    col3.metric("Peak velocity before start", "n/a" if np.isnan(summary["peak_before"]) else f"{summary['peak_before']:.2f} m/s")
    col4.metric("Time of peak", "n/a" if np.isnan(summary["peak_time_before"]) else f"{summary['peak_time_before']:.3f} s")

    st.write("The numbers are estimates from video. Check the start frame and scale before using them.")

    if preview_frames:
        st.subheader("Pose check")
        cols = st.columns(min(len(preview_frames), 5))

        for index, preview_item in enumerate(preview_frames[:5]):
            label = preview_item["label"]
            frame = preview_item["frame"]
            image = preview_item["image"]

            cols[index % len(cols)].image(
                image,
                caption=f"{label} | frame {frame}",
                use_container_width=True,
            )

    plot_df = df[(df["time_s"] >= plot_start) & (df["time_s"] <= plot_end)].copy()

    tab1, tab2, tab3, tab4 = st.tabs(["Hip velocity", "Body movement", "Knee angle", "Data"])

    with tab1:
        fig = px.line(
            plot_df,
            x="time_s",
            y="hip_forward_velocity_smooth_m_s",
            labels={
                "time_s": "Time from start (s)",
                "hip_forward_velocity_smooth_m_s": "Hip forward velocity (m/s)",
            },
        )
        add_start_line(fig)
        style_plot(fig, "Hip forward velocity around start")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.line(
            plot_df,
            x="time_s",
            y="com_forward_m",
            labels={
                "time_s": "Time from start (s)",
                "com_forward_m": "Body forward movement (m)",
            },
        )
        add_start_line(fig)
        fig.add_hline(y=0.30, line_dash="dash", annotation_text="30 cm")
        style_plot(fig, "Body forward movement around start")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            plot_df,
            x="com_x_m",
            y="com_y_m",
            labels={
                "com_x_m": "Body X (m)",
                "com_y_m": "Body Y (m)",
            },
        )
        style_plot(fig2, "Body movement path")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig = px.line(
            plot_df,
            x="time_s",
            y="knee_angle_smooth_deg",
            labels={
                "time_s": "Time from start (s)",
                "knee_angle_smooth_deg": "Knee angle (degree)",
            },
        )
        add_start_line(fig)
        style_plot(fig, "Knee angle around start")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"{clean_name(athlete)}_{model_name.lower()}_start_analysis.csv",
        mime="text/csv",
    )