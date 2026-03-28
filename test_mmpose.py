from mmpose.apis import MMPoseInferencer
import cv2

print("Initializing MMPose... (This will download model weights the first time)")
inferencer = MMPoseInferencer(pose2d='human')

video_path = 'data/test_video.mp4'

print(f"Processing {video_path}...")

result_generator = inferencer(video_path, show=True, radius=3, thickness=2)

try:
    for result in result_generator:
        pass
except KeyboardInterrupt:
    print("Stopped by user.")
except Exception as e:
    print(f"An error occurred: {e}")

cv2.destroyAllWindows()
print("Test complete.")