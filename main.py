import cv2
import mediapipe as mp
import tkinter as tk
import pyautogui

# Initialize tkinter to get screen dimensions
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.withdraw()  # Hide the tkinter root window

# Set the desired proportion for the window size
width_proportion = 0.8  # Example: 80% of the screen width
height_proportion = 0.8  # Example: 80% of the screen height

# Calculate the window dimensions
window_width = int(screen_width * width_proportion)
window_height = int(screen_height * height_proportion)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


# Initialize OpenCV capture
cap = cv2.VideoCapture(0)

# Create a named window and set its size
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', window_width, window_height)

def detect_gesture(hand_landmarks):
    if hand_landmarks:
            for landmarks in hand_landmarks:
                # Extract landmarks for specific fingers
                thumb_tip = landmarks.landmark[4]
                index_tip = landmarks.landmark[8]
                middle_tip = landmarks.landmark[12]
                ring_tip = landmarks.landmark[16]
                pinky_tip = landmarks.landmark[20]
                
                # Map landmarks to window size
                thumb_x = int(thumb_tip.x * window_width)
                thumb_y = int(thumb_tip.y * window_height)
                index_x = int(index_tip.x * window_width)
                index_y = int(index_tip.y * window_height)
                middle_x = int(middle_tip.x * window_width)
                middle_y = int(middle_tip.y * window_height)
                ring_x = int(ring_tip.x * window_width)
                ring_y = int(ring_tip.y * window_height)
                pinky_x = int(pinky_tip.x * window_width)
                pinky_y = int(pinky_tip.y * window_height)
                
                # Define thresholds for gestures
                click_threshold = 20  # Distance threshold for clicking
                gesture_threshold = 20  # Threshold for finger proximity
                move_threshold = 150  # Distance threshold to consider index as raised
    
                # Calculate distances between fingers and thumb
                thumb_index_dist = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
                thumb_middle_dist = ((thumb_x - middle_x) ** 2 + (thumb_y - middle_y) ** 2) ** 0.5
                thumb_ring_dist = ((thumb_x - ring_x) ** 2 + (thumb_y - ring_y) ** 2) ** 0.5
                thumb_pinky_dist = ((thumb_x - pinky_x) ** 2 + (thumb_y - pinky_y) ** 2) ** 0.5
                
                # Distance between index finger and other fingers to determine if it is raised
                index_middle_dist = ((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2) ** 0.5
                index_ring_dist = ((index_x - ring_x) ** 2 + (index_y - ring_y) ** 2) ** 0.5
                index_pinky_dist = ((index_x - pinky_x) ** 2 + (index_y - pinky_y) ** 2) ** 0.5
    
                # Check if the index finger and thumb are close together for a left-click
                if thumb_index_dist < click_threshold:
                    pyautogui.leftClick()  # Perform a left-click
                    return  # Exit after clicking to avoid conflicting actions
    
                # Check if the pinky finger and thumb are close together for a right-click
                if thumb_pinky_dist < click_threshold:
                    pyautogui.rightClick()  # Perform a right-click
                    return  # Exit after clicking to avoid conflicting actions
    
                # Check if the index finger is far away (raised) to move the mouse
                if (index_middle_dist > move_threshold and
                    index_ring_dist > move_threshold and
                    index_pinky_dist > move_threshold):
                    # Move the mouse according to the index finger position
                    pyautogui.moveTo(index_x, index_y)
                else:
                    # Detect scrolling down (middle and thumb together)
                    if thumb_middle_dist < gesture_threshold:
                        pyautogui.scroll(-1)  # Scroll down
    
                    # Detect scrolling up (ring and thumb together)
                    if thumb_ring_dist < gesture_threshold:
                        pyautogui.scroll(1)  # Scroll up
            
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the window size
    frame_resized = cv2.resize(frame, (window_width, window_height))

    # Flip the frame horizontally for selfie view
    flipped = cv2.flip(frame_resized, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            detect_gesture(results.multi_hand_landmarks)

    # Display the image
    cv2.imshow('Hand Gesture Recognition', flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

