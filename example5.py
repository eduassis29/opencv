import cv2

# Function to detect the hand
def detect_hand(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply a threshold to create a binary image
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) == 0:
        return None

    # Find the largest contour (the hand)
    hand_contour = max(contours, key=cv2.contourArea)

    # Calculate the convex hull of the hand contour
    hull = cv2.convexHull(hand_contour)

    # Calculate the area of the hull and the hand contour
    hull_area = cv2.contourArea(hull)
    hand_area = cv2.contourArea(hand_contour)

    # Calculate the solidity (ratio of hull area to hand contour area)
    solidity = float(hull_area) / hand_area

    # Determine if the hand is open or closed based on the solidity
    if solidity < 0.3:
        return "Closed"
    else:
        return "Open"

# Load the pre-trained hand cascade classifier
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')

# Initialize the video capture
capture = cv2.VideoCapture(0)

while True:
    # Read the next frame from the video
    ret, frame = capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected hand
    for (x, y, w, h) in hands:
        # Draw a rectangle around the hand
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the hand region of interest (ROI)
        hand_roi = frame[y:y+h, x:x+w]

        # Detect if the hand is open or closed
        result = detect_hand(hand_roi)

        # Display the result on the frame
        cv2.putText(frame, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Hand Detection', frame)

    # Check if the 'q' key was pressed to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
capture.release()
cv2.destroyAllWindows()