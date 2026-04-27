import cv2
import math

VIDEO_PATH = "data/IMG_0089.MOV"
WHEEL_DIAMETER_METERS = 0.67
FRAME_NUMBER = 0

points = []
frame = None


def click_event(event, x, y, flags, params):
    global frame
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    points.append((x, y))
    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow("Calibration", frame)
    if len(points) == 2:
        cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        pixel_distance = math.dist(points[0], points[1])
        pixels_per_meter = pixel_distance / WHEEL_DIAMETER_METERS
        print("-" * 40)
        print(f"Wheel size on screen: {pixel_distance:.2f} pixels")
        print(f"Pixels per meter: {pixels_per_meter:.2f}")
        print("Copy this value into the app.")
        print("-" * 40)


def main():
    global frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
    success, frame = cap.read()
    cap.release()
    if not success:
        print("Could not read the video. Check the path and frame number.")
        return
    print("Click two points on the wheel diameter.")
    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
