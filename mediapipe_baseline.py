import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def calculate_angle(a, b, c):
    a = np.array(a) # First point (Hip)
    b = np.array(b) # Mid point (Knee)
    c = np.array(c) # End point (Ankle)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_path = 'data/test_video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

print("Processing video... Press 'q' to stop.")

knee_angles_over_time = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    
    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # get the X coordinate of the hip
            hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
            
    
            if hip_x < 0.65: 
                
                hip = [hip_x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)
                knee_angles_over_time.append(angle)

                knee_pixel_coords = tuple(np.multiply(knee, [w, h]).astype(int))
                cv2.putText(frame, str(int(angle)), 
                            knee_pixel_coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
    except Exception as e:
        print("Error:", e)
        pass

    cv2.imshow('MediaPipe Pose Baseline - With Angles', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(knee_angles_over_time) > 0:
    print("Generating graph...")
    plt.figure(figsize=(10, 5))
    plt.plot(knee_angles_over_time, label='Right Knee Angle', color='blue')
    plt.title('Cyclist Right Knee Angle Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Angle (Degrees)')
    plt.legend()
    plt.grid(True)
    
    plt.show() 
else:
    print("No angles were recorded.")