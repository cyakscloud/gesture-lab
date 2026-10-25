# PALM//PILOT — Gesture Lab

**A webcam, a hand, and a tiny bit of sci-fi.** PALM//PILOT reads hand poses live, keeps the camera feed on your laptop, and gives you a clean little control-room view of what it thinks it sees.

I wanted the first launch to feel like powering up a prototype, not opening a notebook and praying. So: one command, a real camera window, steady labels instead of frame-to-frame chaos, and numbers you can actually measure.

## Spin it up on Windows

You need Python 3.11 or newer and a webcam. On first launch, the project creates its own `.venv` and installs the two pinned packages. Google's 8.4 MB gesture model is included, so the demo can start without a model download. Windows auto-selects Media Foundation first, then falls back to DirectShow.

```powershell
cd C:\Users\Siya\gesture-lab
.\run.ps1
```

If PowerShell blocks the script in this window, run this once in that window, then launch it again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run.ps1
```

To pick another camera:

```powershell
.\run.ps1 --camera 1
```

The camera view opens mirrored, like a selfie preview. Turn that off with `--no-mirror`. You can put a ceiling on a quick run with `--duration 15`, or run a saved clip without a webcam:

```powershell
.\run.ps1 --duration 15
.\run.ps1 --source .\demo.mp4
.\run.ps1 --max-hands 2 --width 1280 --height 720
```

The app recognizes the gestures bundled with MediaPipe: **open palm, fist, point, thumbs up, thumbs down, peace,** and **I love you**. It follows one hand by default to keep the camera loop light; `--max-hands 2` turns on a second. It reads poses; it is not a sign-language translator, and moving gestures like a wave are outside this model's job.

## A real run, on this laptop

I ran the camera app locally on Windows with Python 3.14.6, a 640×480 webcam feed, and one-hand tracking. One 13.23-second session processed **162 frames at 12.25 camera FPS**. Mean model response was **57.72 ms**, with a **65.49 ms p95**. The session recorded no stable gesture events, so this is a runtime measurement, not a gesture-accuracy score. More detail and the camera-backend comparison live in [`BENCHMARK.md`](BENCHMARK.md).

## The controls

| Key | Action |
| --- | --- |
| `Q` or `Esc` | Shut the camera down cleanly |
| `S` | Save the current event list and session metrics |
| `R` | Clear the current gesture counters |

Every session writes a small CSV event log and a JSON recap into `sessions/`. The recap reports frames processed, average throughput, mean inference time, and p95 inference time. Those are measurements from the machine running the app, not benchmark numbers pasted into a README. The raw camera frames are never saved.

## What is happening under the hood

```text
Webcam frame → MediaPipe hand landmarks → gesture category + confidence
             → short stability window → OpenCV HUD + session event log
```

The recognizer runs in MediaPipe's live-stream mode, on its own callback path, so inference doesn't freeze the camera preview. MediaPipe can skip frames when it needs to keep latency down. PALM//PILOT adds a short voting window so one noisy prediction does not instantly become a counted event. The overlay shows camera FPS and the latest model response time separately, along with tracked hands and the session's most common gestures.

The MIT license in this repo covers the project source. The bundled model is Google's upstream asset; its source and checksum are documented in [`models/README.md`](models/README.md).

## Project map

```text
gesture-lab/
├── gesture_lab.py       # camera loop, recognition, HUD, and session recap
├── models/              # downloaded gesture_recognizer.task bundle
├── sessions/            # local CSV/JSON session output; never committed
├── requirements.txt     # pinned runtime dependencies
└── run.ps1              # Windows setup + launch
```

## A couple of honest constraints

- This uses MediaPipe's published, pre-trained gesture model; it is not trained on a custom dataset.
- A bright, uncluttered view of your hand helps. Low light, motion blur, and hands partly outside the frame can throw it off.
- The app performs inference locally. Once the model is downloaded, the camera loop does not need an internet connection.
- Session files contain gesture names, confidence values, event times, and runtime measurements. They do not contain images or video.

## Give the rig a different setup

```powershell
.\run.ps1 --help
```

The score threshold, camera index, hand count, frame size, and mirroring behavior are command-line options. The recognition model and its listed categories come from [MediaPipe Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer); the camera window is powered by [OpenCV](https://pypi.org/project/opencv-python/).
