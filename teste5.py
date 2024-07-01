import cv2
import mediapipe as mp

# Function to draw a line on an image between two points
def draw_line(image, point1, point2, color, thickness):
    cv2.line(image, point1, point2, color, thickness)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB for processing with Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with Mediapipe Hands to find hand landmarks
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Assume we are tracking the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmark coordinates (you may need to adjust the landmarks based on your needs)
        landmarks = hand_landmarks.landmark
        # For example, get the coordinates of index finger tip and thumb tip
        if len(landmarks) >= 8:  # Assuming at least 8 landmarks are detected
            x1 = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y1 = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            x2 = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
            y2 = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
            
            # Draw the line between index finger tip and thumb tip
            draw_line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the frame with the drawn line
    cv2.imshow('Hand Tracking', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
