# Gesture model

`gesture_recognizer.task` is MediaPipe's published Gesture Recognizer bundle. It is included so a fresh clone can start without downloading a model at runtime. If the file is removed, `gesture_lab.py` fetches the latest bundle from Google's model storage.

- Model guide: [MediaPipe Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)
- Source: [Google-hosted model bundle](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task)
- SHA-256: `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`

The recognizer's own model card describes the included pose classes and their limits. The bundle reads hand landmarks; it is not a sign-language translation model.
