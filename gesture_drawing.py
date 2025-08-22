import cv2
import numpy as np
import mediapipe as mp
import math
import time

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Drawing canvas
canvas = None

# For smoothing strokes
prev_points = {}

# Colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
color_index = 0
brush_color = colors[color_index]

# Brush settings
brush_thickness = 8
eraser_thickness = 50

# Cooldown for color change
last_color_change = 0
cooldown_time = 1.0  # seconds

# Webcam
cap = cv2.VideoCapture(0)


def fingers_up(hand_landmarks, w, h):
    """Returns list: [index, middle, ring, pinky] → 1 if up, 0 if down"""
    fingers = []
    tip_ids = [8, 12, 16, 20]

    for id in tip_ids:
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
        for hand_idx, handLms in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            finger_status = fingers_up(handLms, w, h)

            # Hand ID for smoothing (track separately for each hand)
            hand_id = f"hand_{hand_idx}"

            # === Gesture 1: Draw (only index up) ===
            if finger_status == [1, 0, 0, 0]:
                if hand_id not in prev_points:
                    prev_points[hand_id] = (x, y)

                prev_x, prev_y = prev_points[hand_id]

                dist = math.hypot(x - prev_x, y - prev_y)
                if dist < 50:  # prevent big jumps
                    mid_x = (prev_x + x) // 2
                    mid_y = (prev_y + y) // 2
                    cv2.line(canvas, (prev_x, prev_y), (mid_x, mid_y), brush_color, brush_thickness)
                    prev_points[hand_id] = (mid_x, mid_y)
                else:
                    prev_points[hand_id] = (x, y)

            # === Gesture 2: Erase (index+middle+ring up) ===
            elif finger_status == [1, 1, 1, 0]:
                cv2.circle(canvas, (x, y), eraser_thickness, (0, 0, 0), -1)
                prev_points[hand_id] = None
                cv2.putText(frame, "Eraser", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # === Gesture 3: Change Color (V sign → index+middle up only) ===
            elif finger_status == [1, 1, 0, 0]:
                current_time = time.time()
                if current_time - last_color_change > cooldown_time:
                    color_index = (color_index + 1) % len(colors)
                    brush_color = colors[color_index]
                    last_color_change = current_time
                    cv2.putText(frame, "Color Changed", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, brush_color, 3)
                prev_points[hand_id] = None

            else:
                prev_points[hand_id] = None

    # Merge drawing with live feed
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Gesture Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"✅ Drawing saved as {filename}")

cap.release()
cv2.destroyAllWindows()
