import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1  # Only detect one hand for better accuracy
)

def detect_distress_signal(hand_landmarks):
    """
    Detects the universal distress signal with improved accuracy.
    Returns: (signal_type, confidence)
    """
    # Get key landmark positions
    landmarks = {}
    for landmark_id, landmark in enumerate(hand_landmarks.landmark):
        landmarks[landmark_id] = landmark

    # Check if hand is raised (wrist below palm center)
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    palm_center = np.mean([
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    ])
    
    if wrist.y < palm_center:
        return False, 0.0

    # Detect signal stages with confidence scores
    thumb_tucked = (landmarks[mp_hands.HandLandmark.THUMB_TIP].x > 
                   landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x)
    
    fingers_closed = all([
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > wrist.y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > wrist.y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > wrist.y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > wrist.y
    ])

    # Calculate confidence scores
    if thumb_tucked and fingers_closed:
        return "Distress Signal", 0.95
    elif thumb_tucked:
        return "Thumb Tucked", 0.7
    elif all([l.y < wrist.y for l in [
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.PINKY_TIP]]]):
        return "Palm Open", 0.6
    
    return False, 0.0

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set improved camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_alert_time = 0
    alert_cooldown = 3  # Seconds between alerts

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # Check for hand landmarks
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                    )
                    
                    signal, confidence = detect_distress_signal(hand_landmarks)
                    
                    if signal:
                        # Display signal with confidence
                        cv2.putText(
                            frame, 
                            f"{signal} ({confidence:.2f})", 
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255), 
                            2
                        )

                        # Alert on distress signal with cooldown
                        if signal == "Distress Signal" and time.time() - last_alert_time > alert_cooldown:
                            print("🚨 ALERT: Distress Signal Detected! 🚨")
                            last_alert_time = time.time()

            # Add help text
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )

            # Show the video feed
            cv2.imshow("AI Surveillance - Distress Signal Detector", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()