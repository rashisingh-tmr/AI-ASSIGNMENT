import cv2
import numpy as np
import time

# Define the quadrant boundaries (full frame coverage)
quadrant_boundaries = [(0, 0, 640, 480), (0, 0, 320, 240), (320, 0, 640, 240), (0, 240, 320, 480), (320, 240, 640, 480)]

# Define the ball colors
ball_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow

# Define the event log file
event_log_file = "event_log.txt"

# Load the video
cap = cv2.VideoCapture("input_video.mp4")

if not cap.isOpened():
    print("Error: Unable to open video file")
    exit(1)

# Get the video frame rate and duration
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

if fps <= 0 or frame_count <= 0:
    print("Error: Unable to retrieve video properties")
    exit(1)

video_duration = frame_count / fps

# Initialize the event log file
with open(event_log_file, "w") as f:
    f.write("Time, Quadrant Number, Ball Colour, Event Type (Entry or Exit)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Track the balls
    for i, ball_color in enumerate(ball_colors):
        lower_bound = np.array([ball_color[0] - 10, 100, 100])
        upper_bound = np.array([ball_color[0] + 10, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                ball_center = (x + w // 2, y + h // 2)

                # Initialize quadrant_number to -1 (not in any quadrant)
                quadrant_number = -1

                # Check which quadrant the ball is in
                for j, (x1, y1, x2, y2) in enumerate(quadrant_boundaries):
                    if x1 < ball_center[0] < x2 and y1 < ball_center[1] < y2:
                        quadrant_number = j + 1
                        break

                # Check if the ball has entered or exited a quadrant
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                event_type = "Entry" if current_time > 0 else "Exit"
                with open(event_log_file, "a") as f:
                    f.write(f"{current_time:.2f}, {quadrant_number}, {i}, {event_type}\n")

                # Draw the ball tracking and event text on the frame
                cv2.circle(frame, ball_center, 5, ball_color, -1)
                cv2.putText(frame, f"Quadrant {quadrant_number} - {event_type} - {current_time:.2f} sec", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the processed frame
    cv2.imshow("Processed Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()