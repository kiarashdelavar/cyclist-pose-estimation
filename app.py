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


@st.cache_data(show_spinner=False)
def auto_detect_start_frame(video_path):
    try:
        video = VideoFileClip(video_path)
        fps = float(video.fps or 30.0)

        if video.audio is None:
            video.close()
            return 0, 0.0, "No audio was found."

        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.close()
        video.audio.write_audiofile(temp_audio.name, logger=None)
        video.close()

        y, sr = librosa.load(temp_audio.name, sr=None, mono=True)
        os.remove(temp_audio.name)

        if y.size == 0:
            return 0, 0.0, "The audio is empty."

        onset = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.frames_to_time(np.arange(len(onset)), sr=sr)

        if len(onset) == 0:
            return 0, 0.0, "No clear sound peak was found."

        limit = max(1, int(len(onset) * 0.98))
        search = onset[:limit]
        threshold = np.percentile(search, 95)
        candidates = np.where(search >= threshold)[0]
        best_index = int(candidates[0]) if len(candidates) > 0 else int(np.argmax(search))

        beep_time = float(times[best_index])
        start_frame = int(round(beep_time * fps))

        return start_frame, beep_time, "Start sound found. Please check it in the video."
    except Exception as error:
        return 0, 0.0, f"Audio check failed: {error}"


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
    frame_number = -1
    processed = 0
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

                if live_frame is not None and (
                    len(preview_frames) < 5
                    or frame_number
                    in [
                        start_frame,
                        start_frame + int(fps * 0.25),
                        start_frame + int(fps * 0.5),
                    ]
                ):
                    preview_frames.append(cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB))

    cap.release()
    progress.empty()
    status.empty()

    if not rows:
        return pd.DataFrame(), preview_frames, fps

    df = pd.DataFrame(rows)
    df = add_metrics(df, fps)
    return df, preview_frames, fps


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


def add_start_line(fig):
    fig.add_vline(x=0, line_dash="dash", annotation_text="Start")
    return fig


st.title("Track cycling start analysis")
st.write("Upload a video. Pick the rider area. Then run the analysis.")

uploaded_video = st.sidebar.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is None:
    st.info("Upload a video to start.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as temp_video:
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

fps, total_frames, width, height = get_video_info(video_path)

st.sidebar.header("Video data")
st.sidebar.write(f"Frames: {total_frames}")
st.sidebar.write(f"FPS: {fps:.2f}")
st.sidebar.write(f"Size: {width} x {height}")

st.sidebar.header("Rider selection")
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
if st.sidebar.button("Find start sound"):
    detected_frame, detected_time, message = auto_detect_start_frame(video_path)
    st.session_state["start_frame"] = detected_frame
    st.sidebar.write(message)
    st.sidebar.write(f"Frame: {detected_frame}")
    st.sidebar.write(f"Time: {detected_time:.3f} s")

start_frame = st.sidebar.number_input(
    "Start frame",
    min_value=0,
    max_value=max(total_frames - 1, 0),
    value=int(st.session_state.get("start_frame", 0)),
    step=1,
)

st.sidebar.header("Scale")
wheel_diameter_m = st.sidebar.number_input("Wheel diameter in meters", min_value=0.10, max_value=2.00, value=0.67, step=0.01)
manual_pixels_per_meter = st.sidebar.number_input("Pixels per meter", min_value=1.0, value=400.0, step=1.0)
calibration_frame = st.sidebar.number_input("Calibration frame", min_value=0, max_value=max(total_frames - 1, 0), value=int(start_frame), step=1)

first_frame = get_frame(video_path, calibration_frame)
if first_frame is not None:
    preview = first_frame.copy()
    x, y, w, h = roi
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 220, 0), 3)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Selected rider area", use_container_width=True)

if st.sidebar.button("Try wheel scale") and first_frame is not None:
    found_ppm, wheel_preview, wheel_message = detect_wheel_pixels(first_frame, roi, wheel_diameter_m)
    st.session_state["wheel_message"] = wheel_message

    if wheel_preview is not None:
        st.session_state["wheel_preview"] = cv2.cvtColor(wheel_preview, cv2.COLOR_BGR2RGB)

    if found_ppm is not None:
        st.session_state["pixels_per_meter"] = float(found_ppm)

if "wheel_preview" in st.session_state:
    st.image(st.session_state["wheel_preview"], caption=st.session_state.get("wheel_message", "Wheel check"), use_container_width=True)

pixels_per_meter = st.sidebar.number_input(
    "Final pixels per meter",
    min_value=1.0,
    value=float(st.session_state.get("pixels_per_meter", manual_pixels_per_meter)),
    step=1.0,
)

st.sidebar.header("Run")
athlete_name = st.sidebar.text_input("Athlete name", value="")
max_frames = st.sidebar.number_input("Maximum frames to process. Use 0 for full video.", min_value=0, value=0, step=100)
plot_start = st.sidebar.number_input("Plot start time", value=-0.6, step=0.1)
plot_end = st.sidebar.number_input("Plot end time", value=0.8, step=0.1)

run_analysis = st.button("Run analysis", type="primary")

if run_analysis:
    df, preview_frames, real_fps = process_video(
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

    st.subheader("Main results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Velocity at start", "n/a" if np.isnan(summary["v_start"]) else f"{summary['v_start']:.2f} m/s")
    col2.metric("Time to 30 cm", "not reached" if np.isnan(summary["t_30cm"]) else f"{summary['t_30cm']:.3f} s")
    col3.metric("Peak velocity before start", "n/a" if np.isnan(summary["peak_before"]) else f"{summary['peak_before']:.2f} m/s")
    col4.metric("Time of peak", "n/a" if np.isnan(summary["peak_time_before"]) else f"{summary['peak_time_before']:.3f} s")

    st.write("The numbers are estimates from video. Check the start frame and scale before using them.")

    if preview_frames:
        st.subheader("Pose check")
        cols = st.columns(min(len(preview_frames), 4))
        for index, image in enumerate(preview_frames[:4]):
            cols[index % len(cols)].image(image, use_container_width=True)

    plot_df = df[(df["time_s"] >= plot_start) & (df["time_s"] <= plot_end)].copy()

    tab1, tab2, tab3, tab4 = st.tabs(["Hip velocity", "Body movement", "Knee angle", "Data"])

    with tab1:
        fig = px.line(
            plot_df,
            x="time_s",
            y="hip_forward_velocity_smooth_m_s",
            labels={"time_s": "Time from start (s)", "hip_forward_velocity_smooth_m_s": "Hip forward velocity (m/s)"},
        )
        add_start_line(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.line(
            plot_df,
            x="time_s",
            y="com_forward_m",
            labels={"time_s": "Time from start (s)", "com_forward_m": "Body forward movement (m)"},
        )
        add_start_line(fig)
        fig.add_hline(y=0.30, line_dash="dash", annotation_text="30 cm")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(plot_df, x="com_x_m", y="com_y_m", labels={"com_x_m": "Body X (m)", "com_y_m": "Body Y (m)"})
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig = px.line(
            plot_df,
            x="time_s",
            y="knee_angle_smooth_deg",
            labels={"time_s": "Time from start (s)", "knee_angle_smooth_deg": "Knee angle (degree)"},
        )
        add_start_line(fig)
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
