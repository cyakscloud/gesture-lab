# Local run notes

These numbers came from actual runs on the laptop this project was built on. They are a small sanity check, not a lab-grade performance claim; camera exposure, lighting, background load, and webcam drivers can change the result.

## Gesture app run

Command: `.\run.ps1 --duration 12`

| Measurement | Result |
| --- | ---: |
| OS | Windows |
| Python | 3.14.6, 64-bit |
| Camera backend | Media Foundation |
| Requested frame size | 640 × 480 |
| Hand limit | 1 |
| Run duration | 13.23 s |
| Frames processed | 162 |
| Camera loop throughput | 12.25 FPS |
| Recognition results received | 162 |
| Mean model response | 57.72 ms |
| p95 model response | 65.49 ms |
| Runtime | CPU / XNNPACK |
| Stable gesture events | 0 |
| Video written | No |

No gesture-accuracy result is claimed because this short run did not include a labeled gesture sequence.

## Camera-driver check

I also measured the webcam without running the recognizer: 4 seconds at a requested 640 × 480 and 30 FPS.

| Windows backend | Frames | Measured capture rate |
| --- | ---: | ---: |
| DirectShow | 27 | 6.69 FPS |
| Media Foundation | 48 | 11.87 FPS |

That check is why the Windows auto mode tries Media Foundation first. It keeps DirectShow as a fallback in case another camera behaves differently.
