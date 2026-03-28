import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import plotly.express as px
import pandas as pd

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


st.set_page_config(page_title="AMI Pose Estimation MVP", layout="wide")
st.title(" Track Cycling Kinematics MVP")
st.markdown("**NOC*NSF Ambient Intelligence Project** - MediaPipe vs MMPose Comparison")

st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox("Select Pose Model", ["MediaPipe (Baseline)", "MMPose (RTMPose)"])

# File uploader
uploaded_video = st.file_uploader("Upload a video of a cyclist (.mp4, .mov)", type=["mp4", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    if st.button("Run Analysis"):
        st.write(f"Running analysis using **{selected_model}**...")
        
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        if selected_model == "MediaPipe (Baseline)":
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            inferencer = load_mmpose()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        
    
        knee_angles_over_time = []
        com_x_over_time = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            current_frame += 1
            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))

            if selected_model == "MediaPipe (Baseline)":
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True

                try:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = frame.shape
                    
                    # Knee Angle Math
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    angle = calculate_angle(hip, knee, ankle)
                    knee_angles_over_time.append(angle)
                    
                    # Center of Mass Math (X-axis)
                    l_shoulder = landmarks[11].x * w
                    r_shoulder = landmarks[12].x * w
                    l_hip = landmarks[23].x * w
                    r_hip = landmarks[24].x * w
                    com_x = (l_shoulder + r_shoulder + l_hip + r_hip) / 4.0
                    com_x_over_time.append(com_x)
                    
                    knee_pixel_coords = tuple(np.multiply(knee, [w, h]).astype(int))
                    cv2.putText(image_rgb, str(int(angle)), knee_pixel_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            else:
                result_generator = inferencer(frame, return_vis=True, show=False)
                result = next(result_generator)
                vis_image = result['visualization'][0]
                image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                
                try:
                    keypoints = result['predictions'][0][0]['keypoints']
                    
                    # Knee Angle Math
                    hip = keypoints[12]
                    knee = keypoints[14]
                    ankle = keypoints[16]
                    angle = calculate_angle(hip, knee, ankle)
                    knee_angles_over_time.append(angle)
                    
                    # Center of Mass Math (X-axis)
                    l_shoulder = keypoints[5][0]
                    r_shoulder = keypoints[6][0]
                    l_hip = keypoints[11][0]
                    r_hip = keypoints[12][0]
                    com_x = (l_shoulder + r_shoulder + l_hip + r_hip) / 4.0
                    com_x_over_time.append(com_x)
                    
                    cv2.putText(image_rgb, str(int(angle)), (int(knee[0]), int(knee[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

            video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)

        cap.release()
        

        st.success("Analysis Complete!")
        
        if len(knee_angles_over_time) > 0:
            df = pd.DataFrame({
                'Frame': range(1, len(knee_angles_over_time) + 1),
                'Knee Angle (Degrees)': knee_angles_over_time,
                'CoM X-Position (Pixels)': com_x_over_time
            })
            
            line_color = '#1f77b4' if selected_model == "MediaPipe (Baseline)" else '#d62728'
            
            st.markdown("###  Joint Kinematics")
            fig_knee = px.line(df, x='Frame', y='Knee Angle (Degrees)', color_discrete_sequence=[line_color])
            fig_knee.update_layout(hovermode="x unified")
            st.plotly_chart(fig_knee, use_container_width=True)
            
            st.markdown("###  Forward Propulsion (Center of Mass)")
            fig_com = px.line(df, x='Frame', y='CoM X-Position (Pixels)', color_discrete_sequence=['#2ca02c'])
            fig_com.update_layout(hovermode="x unified")
            st.plotly_chart(fig_com, use_container_width=True)
            
            st.markdown("### Export Data")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"cyclist_kinematics_{selected_model.split()[0].lower()}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No pose data detected in this video.")