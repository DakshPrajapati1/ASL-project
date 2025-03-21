import cv2
import mediapipe as mp
import time
 
# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
 
# Variables for stabilization
last_sign = ""
last_sign_time = 0
debounce_duration = 1.0  # Time in seconds to wait before confirming a new sign
 
# Buffer to track last N signs
sign_buffer = []
buffer_size = 5  # Number of consecutive frames to check for consistency
sign_threshold = 3  # Number of times the same sign should appear in the buffer before being accepted
 
# Function to detect "Good Morning"
def is_sign_good_morning(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y < index_tip.y
 
# Function to detect "How are you?"
def is_sign_how_are_you(hand_landmarks1, hand_landmarks2):
    thumb1_tip = hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb2_tip = hand_landmarks2.landmark[mp_hands.HandLandmark.THUMB_TIP]
    return abs(thumb1_tip.x - thumb2_tip.x) < 0.1 and abs(thumb1_tip.y - thumb2_tip.y) < 0.1
 
# Function to detect "Nice to meet"
def is_sign_nice_to_meet(hand_landmarks1, hand_landmarks2):
    index1_tip = hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index2_tip = hand_landmarks2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return abs(index1_tip.x - index2_tip.x) < 0.1 and abs(index1_tip.y - index2_tip.y) < 0.1
 
# Function to detect "Thank you"
def is_sign_thank_you(hand_landmarks):
    chin_y = 0.6  # Approximate chin height in normalized coordinates
    hand_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    return hand_tip.y < chin_y and palm_base.y > hand_tip.y
 
# Function to detect "Goodbye"
previous_x = None
direction_changes = 0
def is_sign_goodbye(hand_landmarks):
    global previous_x, direction_changes
    palm_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    if previous_x is not None:
        if (palm_tip.x - previous_x) > 0.05 or (previous_x - palm_tip.x) > 0.05:
            direction_changes += 1
    previous_x = palm_tip.x
    return direction_changes >= 2  # Detects at least two side-to-side movements
 
# Function to determine the detected sign
def sign_to_text(frame, hands):
    if hands.multi_hand_landmarks:
        if len(hands.multi_hand_landmarks) == 1:
            hand_landmarks = hands.multi_hand_landmarks[0]
            if is_sign_good_morning(hand_landmarks):
                return "Good Morning!"
            elif is_sign_thank_you(hand_landmarks):
                return "Thank you!"
            elif is_sign_goodbye(hand_landmarks):
                return "Goodbye!"
        elif len(hands.multi_hand_landmarks) == 2:
            hand1, hand2 = hands.multi_hand_landmarks
            if is_sign_how_are_you(hand1, hand2):
                return "How are you?"
            elif is_sign_nice_to_meet(hand1, hand2):
                return "Nice to meet"
    return ""
 
def main():
    global last_sign, last_sign_time, sign_buffer  # Access global variables for stabilization
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        # Get the detected sign text
        detected_sign = sign_to_text(frame, results)
        if detected_sign:
            # Add detected sign to the buffer
            sign_buffer.append(detected_sign)
            # If buffer exceeds its size, remove the oldest sign
            if len(sign_buffer) > buffer_size:
                sign_buffer.pop(0)
            # Check if the most common sign in the buffer exceeds the threshold
            common_sign = max(set(sign_buffer), key=sign_buffer.count)
            if sign_buffer.count(common_sign) >= sign_threshold:
                # Only update the sign if it's consistent across multiple frames
                last_sign = common_sign
                last_sign_time = time.time()
 
        # Display the recognized sign on the frame if it's stable
        if last_sign:
            cv2.putText(frame, last_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("ASL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()