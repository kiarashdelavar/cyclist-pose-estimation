import cv2
import math

WHEEL_SIZE_METERS = 0.67 

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibration", frame)

        if len(points) == 2:

            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)
            
            pixel_distance = math.dist(points[0], points[1])
            pixels_per_meter = pixel_distance / WHEEL_SIZE_METERS
            
            print("\n" + "-" * 45)
            print(f" Wheel measured on screen: {pixel_distance:.2f} pixels")
            print(f" YOUR CALIBRATION RATIO: {pixels_per_meter:.2f}")
            print("-" * 45 + "\n")
            print("You can close the window now!")

video_path = 'data/IMG_0089.MOV' 
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_FRAMES, 1402)

success, frame = cap.read()
if not success:
    print("Error: Could not read video.")
    exit()

print("INSTRUCTIONS:")
print("1. A picture of the video will open.")
print("2. Click exactly on the LEFT edge of the front wheel.")
print("3. Click exactly on the RIGHT edge of the front wheel.")

cv2.imshow("Calibration", frame)
cv2.setMouseCallback("Calibration", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()