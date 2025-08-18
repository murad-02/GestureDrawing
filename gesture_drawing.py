import cv2
import numpy as np
import mediapipe as mp
import math
import time

# Initialize Mediapipe Hand
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Create canvas
canvas = None

# Store previous finger position
prev_x, prev_y = None, None

# Colors for drawing
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
color_index = 0
brush_color = colors[color_index]

# Brush settings
brush_thickness = 8
eraser_thickness = 50

# Flag for color change debounce
color_change_cooldown = False

# Start webcam
cap = cv2.VideoCapture(0)


def fingers_up(hand_landmarks, w, h):
    """Check which fingers are up"""
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    # Index to Pinky
    for id in tip_ids[1:]:
        tip_y = hand_landmarks.landmark[id].y * h
        base_y = hand_landmarks.landmark[id - 2].y * h
        fingers.append(1 if tip_y < base_y else 0)

    return fingers


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get fingertip (index finger = 8)
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # Detect gestures
            finger_status = fingers_up(handLms, w, h)

            # === Gesture 1: Draw (only index finger up) ===
            if finger_status == [1, 0, 0, 0]:
                if prev_x is None:
                    prev_x, prev_y = x, y

                # Movement threshold (avoid jumps)
                dist = math.hypot(x - prev_x, y - prev_y)
                if dist < 40:  # only draw if movement is small
                    # Smooth line drawing
                    mid_x = (prev_x + x) // 2
                    mid_y = (prev_y + y) // 2
                    cv2.line(canvas, (prev_x, prev_y), (mid_x, mid_y), brush_color, brush_thickness)
                    prev_x, prev_y = mid_x, mid_y
                else:
                    prev_x, prev_y = x, y

            # === Gesture 2: Erase (3 fingers up: index+middle+ring) ===
            elif finger_status == [1, 1, 1, 0]:
                cv2.circle(canvas, (x, y), eraser_thickness, (0, 0, 0), -1)
                prev_x, prev_y = None, None
                cv2.putText(frame, "Eraser", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # === Gesture 3: Change color (Thumb + Index pinch) ===
            else:
                # Thumb tip (4) and Index tip (8)
                thumb_x = int(handLms.landmark[4].x * w)
                thumb_y = int(handLms.landmark[4].y * h)
                pinch_dist = math.hypot(thumb_x - x, thumb_y - y)

                if pinch_dist < 40 and not color_change_cooldown:
                    color_index = (color_index + 1) % len(colors)
                    brush_color = colors[color_index]
                    color_change_cooldown = True
                    cv2.putText(frame, "Color Changed", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, brush_color, 3)

                if pinch_dist > 60:
                    color_change_cooldown = False

                prev_x, prev_y = None, None

    # Merge drawing with camera
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Gesture Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('s'):  # Press 's' to save
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"✅ Drawing saved as {filename}")

cap.release()
cv2.destroyAllWindows()
