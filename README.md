# Gesture Drawing

A Python application that lets you draw on a virtual canvas using hand gestures detected via your webcam. Powered by OpenCV and MediaPipe.

## Features

- **Draw**: Use your index finger to draw on the screen.
- **Erase**: Use three fingers (index, middle, ring) to erase.
- **Change Color**: Pinch your thumb and index finger to cycle through brush colors.
- **Save Drawing**: Press `s` to save your artwork as a PNG file.
- **Exit**: Press `ESC` to quit.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy

Install dependencies with:

```sh
pip install opencv-python mediapipe numpy
```

## Usage
<!-- Header -->
<h1 align="center">🖌️ Hand Gesture Drawing with Mediapipe</h1>
<p align="center">
  Draw, Erase, and Change Colors using just your <b>Hand Gestures</b> captured via Webcam! <br>
  Built with <code>OpenCV</code>, <code>Mediapipe</code>, and <code>Python</code>.
</p>

---

## ✨ Features
- ✍️ **Draw** on screen using your **index finger**.
- 🧽 **Erase** with **index + middle + ring** fingers.
- 🎨 **Change brush color** with a **V-sign (index + middle)**.
- 🖼️ Save your artwork anytime by pressing **`S`**.
- 🎥 Real-time hand tracking using **Mediapipe Hands**.

---

## 🎮 Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ (Index Finger Up) | Draw |
| ✌️ (Index + Middle Up) | Change Brush Color |
| 🤟 (Index + Middle + Ring Up) | Erase |
| 💾 Press `S` | Save Drawing |
| ❌ Press `ESC` | Exit |

---
## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/gesture-drawing.git
cd gesture-drawing

Run the script:

```sh
python gesture_drawing.py
```

- Make sure your webcam is connected.
- The drawing window will open.
- Use gestures to draw, erase, and change colors as described above.

## Controls

| Gesture/Key         | Action                |
|---------------------|-----------------------|
| Index finger up     | Draw                  |
| 3 fingers up        | Erase                 |
| Thumb+Index pinch   | Change brush color    |
| `s` key             | Save drawing as PNG   |
| `ESC` key           | Exit application      |

## File

- [gesture_drawing.py](gesture_drawing.py)

---
2️⃣ Install Dependencies
pip install opencv-python mediapipe numpy
3️⃣ Run the Application
python gesture_drawing.py

💡 Future Improvements

* Add shape drawing (circle, rectangle, line) with gestures.

* Add gesture to change brush size.

* Add undo/redo functionality.

Inspired by gesture-based drawing applications using computer vision.
