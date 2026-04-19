import cv2
from ultralytics import YOLO

# Load YOLOv8 Medium model
model = YOLO("yolov8m.pt")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLOv8 Medium Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()