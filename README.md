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

Inspired by gesture-based drawing applications using computer vision.