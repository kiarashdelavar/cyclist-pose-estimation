import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import plotly.express as px
import pandas as pd
from datetime import datetime
import librosa
from moviepy import VideoFileClip
import os

@st.cache_resource
def load_mmpose():
    from mmpose.apis import MMPoseInferencer
    return MMPoseInferencer(pose2d='human')

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# --- NEW: AUTO AUDIO DETECTOR ---
@st.cache_data
def auto_detect_beep(video_path):
    try:
        video = VideoFileClip(video_path)
        fps = video.fps
        temp_audio_path = "temp_dashboard_audio.wav"
        video.audio.write_audiofile(temp_audio_path, logger=None)
        
        y, sr = librosa.load(temp_audio_path, sr=None)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beep_time = librosa.frames_to_time(onset_env.argmax(), sr=sr)
        start_frame = int(beep_time * fps)
        
        video.close()
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        return start_frame
    except Exception as e:
        return 0

st.set_page_config(page_title="Track Cycling Kinematics", layout="wide", page_icon="🚴‍♂️")
st.title("🚴‍♂️ Track Cycling Kinematics MVP")
st.markdown("**NOC*NSF Ambient Intelligence Project** - Universal Analysis Dashboard")

st.sidebar.header("⚙️ Analysis Settings")
athlete_name = st.sidebar.text_input("👤 Athlete Name", placeholder="e.g., Harrie Lavreysen")
selected_model = st.sidebar.selectbox("🤖 Select Pose Model", ["MediaPipe (Complex=2, Conf=0.8)", "MMPose (RTMPose)"])

st.sidebar.markdown("---")
st.sidebar.header("🎯 Target Tracking Filter")
tracking_limit = st.sidebar.slider("Ignore background people past X:", min_value=0.0, max_value=1.0, value=0.80, step=0.05)
st.sidebar.info("💡 Hint: If the AI draws on the coach, lower this slider to ignore them.")

# --- AUTO SYNC UI ---
st.sidebar.markdown("---")
st.sidebar.header("⏱️ Sync & Calibration")

# File uploader moved up so we can use the video for auto-sync before running analysis!
uploaded_video = st.file_uploader("📂 Upload Video FIRST", type=["mp4", "mov"])
video_path = None

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

# Auto-Beep Button
detected_frame = 0
if video_path and st.sidebar.button("🔊 Auto-Detect Beep Frame"):
    with st.sidebar.status("Analyzing audio track..."):
        detected_frame = auto_detect_beep(video_path)
    st.sidebar.success(f"Beep found at frame: {detected_frame}")

start_frame = st.sidebar.number_input("🏁 Starting Signal Frame (T=0)", min_value=0, value=detected_frame, step=1)
pixels_per_meter = st.sidebar.number_input("📏 Pixels per Meter", min_value=1.0, value=392.56, step=10.0)

# --- MAIN ANALYSIS LOOP ---
if video_path is not None:
    if st.button("🚀 Run Analysis", type="primary"):
        col_vid, col_data = st.columns([1.2, 1])
        
        with col_vid:
            st.markdown(f"#### Processing Video with {selected_model.split()[0]}...")
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            
        if selected_model == "MediaPipe (Complex=2, Conf=0.8)":
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            pose = mp_pose.Pose(
                static_image_mode=False, 
                model_complexity=2, 
                min_detection_confidence=0.8, 
                min_tracking_confidence=0.8
            )
        else:
            inferencer = load_mmpose()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        
        knee_angles_over_time = []
        com_x_over_time = []
        hip_x_pixels = []   
        hip_y_pixels = []  

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            current_frame += 1
            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))

            if selected_model == "MediaPipe (Complex=2, Conf=0.8)":
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True

                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        h, w, _ = frame.shape
                        hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
                        
                        # STRICT FILTER: Only do math and drawing IF it's the cyclist
                        if hip_x < tracking_limit:
                            hip = [hip_x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                            
                            angle = calculate_angle(hip, knee, ankle)
                            knee_angles_over_time.append(angle)
                            
                            hip_x_pixels.append(hip_x * w)
                            hip_y_pixels.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
                            
                            l_shoulder = landmarks[11].x * w
                            r_shoulder = landmarks[12].x * w
                            l_hip = landmarks[23].x * w
                            r_hip = landmarks[24].x * w
                            com_x = (l_shoulder + r_shoulder + l_hip + r_hip) / 4.0
                            com_x_over_time.append(com_x)
                            
                            # ONLY DRAW ON THE CYCLIST
                            knee_pixel_coords = tuple(np.multiply(knee, [w, h]).astype(int))
                            cv2.putText(image_rgb, f"{int(angle)} deg", knee_pixel_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                except:
                    pass

            else:
                result_generator = inferencer(frame, return_vis=True, show=False)
                result = next(result_generator)
                vis_image = result['visualization'][0]
                image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                
                try:
                    h, w, _ = image_rgb.shape
                    keypoints = result['predictions'][0][0]['keypoints']
                    hip_x_mmpose = keypoints[12][0] / w
                    
                    # STRICT FILTER
                    if hip_x_mmpose < tracking_limit:
                        hip = keypoints[12]
                        knee = keypoints[14]
                        ankle = keypoints[16]
                        angle = calculate_angle(hip, knee, ankle)
                        knee_angles_over_time.append(angle)

                        hip_x_pixels.append(keypoints[12][0])
                        hip_y_pixels.append(keypoints[12][1])

                        l_shoulder = keypoints[5][0]
                        r_shoulder = keypoints[6][0]
                        l_hip = keypoints[11][0]
                        r_hip = keypoints[12][0]
                        com_x = (l_shoulder + r_shoulder + l_hip + r_hip) / 4.0
                        com_x_over_time.append(com_x)
                        
                        # ONLY DRAW ON THE CYCLIST
                        cv2.putText(image_rgb, f"{int(angle)} deg", (int(knee[0]), int(knee[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                except:
                    pass

            video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)

        cap.release()
        
        with col_vid:
            st.success("✅ Video Analysis Complete!")
        
        if len(knee_angles_over_time) > 0:
            fps = 30.0 
            frames_array = np.array(range(len(knee_angles_over_time)))
            relative_time = (frames_array - start_frame) / fps
            
            com_x_meters = [x / pixels_per_meter for x in com_x_over_time]
            hip_x_meters = [x / pixels_per_meter for x in hip_x_pixels]
            hip_y_meters = [y / pixels_per_meter for y in hip_y_pixels]

            df = pd.DataFrame({
                'Absolute Frame': frames_array,
                'Time (Seconds)': relative_time,
                'Knee Angle (Degrees)': knee_angles_over_time,
                'Relative Displacement (Meters)': com_x_meters,
                'Hip X (m)': hip_x_meters, 
                'Hip Y (m)': hip_y_meters 
            })
   
            df['dx'] = df['Hip X (m)'].diff()
            df['dy'] = df['Hip Y (m)'].diff()
            dt = 1.0 / fps 
            df['Raw Velocity (m/s)'] = np.sqrt(df['dx']**2 + df['dy']**2) / dt
            df['Smoothed Hip Velocity (m/s)'] = df['Raw Velocity (m/s)'].rolling(window=5, min_periods=1).mean()

            df_display = df[df['Time (Seconds)'] >= 0]
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            safe_athlete_name = athlete_name if athlete_name else "Unknown_Athlete"
            
            df.insert(0, 'Date & Time', current_time)
            df.insert(0, 'Athlete Name', safe_athlete_name)
            
            line_color = '#1f77b4' if "MediaPipe" in selected_model else '#d62728'
            
            with col_data:
                st.markdown("### 🏆 Performance Metrics (NOC*NSF)")
                
                start_row = df_display.iloc[0]
                start_pos = start_row['Relative Displacement (Meters)']
                
                if len(df_display) > 1:
                    next_row = df_display.iloc[1]
                    delta_dist = abs(next_row['Relative Displacement (Meters)'] - start_pos)
                    delta_time = next_row['Time (Seconds)'] - start_row['Time (Seconds)']
                    v_start = delta_dist / delta_time
                else:
                    v_start = 0.0

                t_30cm = None
                for index, row in df_display.iterrows():
                    moved_distance = abs(row['Relative Displacement (Meters)'] - start_pos)
                    if moved_distance >= 0.30:  
                        t_30cm = row['Time (Seconds)']
                        break

                m1, m2 = st.columns(2)
                m1.metric(label="V_start (Velocity at Beep)", value=f"{v_start:.2f} m/s")
                if t_30cm is not None:
                    m2.metric(label="T_30cm (Time to cover 30cm)", value=f"{t_30cm:.2f} s")
                else:
                    m2.metric(label="T_30cm", value="Distance not reached")
                    
                st.markdown("---")
                
                tab1, tab2, tab3 = st.tabs(["💨 Hip Velocity", "📍 CoM Trajectory", "📐 Knee Angle"])
                
                with tab1:
                    fig_vel = px.line(df_display, x='Time (Seconds)', y='Smoothed Hip Velocity (m/s)', color_discrete_sequence=['#ff7f0e'])
                    fig_vel.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Start Signal")
                    st.plotly_chart(fig_vel, use_container_width=True)
                    
                with tab2:
                    fig_com = px.line(df_display, x='Time (Seconds)', y='Relative Displacement (Meters)', color_discrete_sequence=['#2ca02c'])
                    fig_com.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Start Signal")
                    st.plotly_chart(fig_com, use_container_width=True)
                    
                with tab3:
                    fig_knee = px.line(df_display, x='Time (Seconds)', y='Knee Angle (Degrees)', color_discrete_sequence=[line_color])
                    fig_knee.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Start Signal")
                    st.plotly_chart(fig_knee, use_container_width=True)

            st.markdown("---")
            st.markdown("### 💾 Export Data")
            
            csv = df.to_csv(index=False).encode('utf-8')
            clean_filename = safe_athlete_name.replace(" ", "_").lower()
            
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{clean_filename}_kinematics_{selected_model.split()[0].lower()}.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.warning("⚠️ No pose data detected! Try adjusting the 'Target Tracking Filter' slider in the sidebar to make sure the AI isn't ignoring the cyclist.")