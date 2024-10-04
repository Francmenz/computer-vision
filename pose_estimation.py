import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_styles as dr_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing

# Initialize MediaPipe pose solution

pose = mp_pose.Pose()


# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"Ignoring camera frame.")
        continue

    # Convert the image from BGR (OpenCV default) to RGB (MediaPipe uses RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find pose landmarks
    result = pose.process(image_rgb)

    # Draw pose landmarks on the original image
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # Display the result
    cv2.imshow('Pose Estimation', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
