# Chakra-Based Multi-Modal Gesture Interaction System

A real-time webcam system that combines **hand gestures**, **facial expression / eye
state** and **temporal patterns** with an **internal state machine** (chakra,
cooldowns, modes) to trigger Naruto-style AR effects.

```
pip install -r requirements.txt
python main.py            # real cooldowns (7 min / 20 min)
python main.py --demo     # 8 s / 30 s cooldowns for testing
python main.py --debug    # skeleton, palm/back, gesture confidences, expression values
python -m pytest tests -q # 20 logic tests, no camera needed
```

Keys: `q`/`Esc` quit · `d` debug overlay · `c` clear cooldowns · `r` refill chakra

## Jutsu

| Jutsu | Trigger | Type | Cost | Cooldown |
|---|---|---|---|---|
| Rasengan | both hands cupped close together, held 0.5 s | continuous: sphere grows while held | 5 + 12/s | 7 min (from release) |
| Chidori | claw hand, held 0.6 s | instant, 3 s effect | 40 | 7 min |
| Lion Fist | open palm facing the camera, held 0.5 s | instant, 2 s effect | 10 | 7 min |
| Dojutsu | prayer sign (index fingers touching and aligned), held 0.8 s; classified from the face | mode, up to 45 s; repeat the sign to release | 5 + 2/s | 7 min (from release) |
| Byakugou Seal | head **stable** for 1.5 s **and** a forehead tap | mode, 90 s, **infinite chakra** | – | 20 min (from activation) |

The Dojutsu type is chosen from your expression while you hold the sign:
**Sharingan** if your brows are furrowed or your eyes are focused (squinting),
**Byakugan** if your face is neutral.

All numbers live in [config/settings.py](config/settings.py).

## Architecture

```
ThreadedCamera ──► InferenceWorker thread ─────────────────────────► main thread
 (newest frame)    mirror ─► HandTracker ┐                            HandFeatures
                            FaceTracker ─┴ (parallel) ──► queue(1) ─► GestureClassifier
                                                                      TemporalLogic
                                                                      JutsuEngine ◄─► StateManager
                                                                      Renderer ─► imshow
```

| Module | File | Input → Output |
|---|---|---|
| Hand tracking | [modules/hand_tracker.py](modules/hand_tracker.py) | RGB frame, t → `Hand[]` (21 pixel landmarks, handedness), One-Euro filtered |
| Face + expression | [modules/face_tracker.py](modules/face_tracker.py) | RGB frame, t → `Face` (478 landmarks incl. iris, 52 blendshapes, forehead/eyes/roll) |
| Feature extraction | [modules/feature_extractor.py](modules/feature_extractor.py) | `Hand` → `HandFeatures` (joint bends, normalized tip distances, spread, palm-facing, index direction) |
| Gesture classification | [modules/gesture_classifier.py](modules/gesture_classifier.py) | features + face → raw per-frame booleans (`claw:Right`, `rasengan_cup`, …) |
| Temporal logic | [modules/temporal.py](modules/temporal.py) | raw booleans, face, t → `TemporalState` (held gestures, edges, tap, stability, expression) |
| State manager | [modules/state_manager.py](modules/state_manager.py) | spend / drain / cooldown requests → chakra, cooldowns, Byakugou mode |
| Jutsu engine | [engine/jutsu_engine.py](engine/jutsu_engine.py) | `TemporalState` + state → active `Jutsu` list + on-screen messages |
| AR renderer | [engine/renderer.py](engine/renderer.py) | frame + jutsu + landmarks → frame with effects and HUD |

### Hand tracking
The tracker uses the MediaPipe Tasks `HandLandmarker` in `VIDEO` mode. In this mode,
landmarks from one frame seed the next, so the palm detector only runs when a hand is lost.
Landmarks are converted to pixel space (`z` is scaled by the width, so 3-D angles make
sense). A One-Euro filter per hand removes jitter at rest without adding lag during fast
motion. If both hands get the same label, which happens briefly when tracking flips,
one of them is relabelled.

### Face + expression
The tracker uses `FaceLandmarker` with `output_face_blendshapes=True`. Expressions come
from the model's blendshape scores (`browDownLeft/Right`, `eyeSquint*`, `eyeWide*`,
`mouthSmile*`) instead of hand-made landmark ratios, which are fragile under head
rotation. It also provides:
- the forehead point (landmarks 151 and 9),
- iris centre and radius (468–477),
- eye contours, used so the eyelids hide the overlays,
- face width (the scale for everything face-related),
- roll angle and the face's up vector.

### Feature extraction
Every distance is divided by the **palm size** (wrist → middle MCP), so the rules work
at any distance from the camera.

- **Bend** of each finger = bend at the PIP joint + bend at the DIP joint, in degrees (0 = straight).
- **MCP bend** is measured separately, because it separates a cupped hand from a flat one.
- **Palm facing**: the 2-D cross product of (index MCP − wrist) × (pinky MCP − wrist)
  gives the hand's chirality (which way round it appears on screen). Combined with the
  handedness label, this tells palm from back. It doesn't change when the hand rotates
  in the image plane, and it gives the same answer on mirrored images: mirroring flips
  both the sign and the label. The sign was checked on real images.

### Gesture classification (rule-based)
These rules are geometric rather than learned. They need no dataset, every threshold
corresponds to one physical property, and they are fast (microseconds).

- **Open palm**: all 4 fingers straight (< 45°), thumb out, fingers spread, palm facing the camera.
- **Claw**: at least 3 fingers bent 60–175° *while the fingertips stay away from the palm*. This rules out a fist.
- **Rasengan cup**: 2 hands, palm centres 0.5–2.6 palm sizes apart, both moderately curled (mean MCP+PIP+DIP 40–220°).
- **Prayer**: both index fingers straight, index tips < 0.45 palm sizes apart, wrists close, index directions parallel (cos > 0.85).
- **Forehead touch**: the index or middle fingertip is within 0.22 face widths of the forehead point.

**Spatial conflict resolution.** Gestures are evaluated in priority order
`forehead_tap > prayer > rasengan_cup > claw > open_palm`. A hand that matches a
higher-priority gesture is *consumed* and is not checked for lower ones. For example,
two cupped hands also look like two claws, so they would otherwise fire Chidori. A
forehead touch counts only when **exactly one** hand is at the forehead; two index
fingers there are a prayer held in front of the face.

### Temporal logic
Everything is measured in seconds, not frames, so behaviour is identical at 15 or 60 FPS.
A test checks this at 15, 30 and 60 FPS.

1. **Smoothing.** Each gesture's raw boolean feeds an exponential moving average (EMA).
   Its α comes from a time constant: `α = 1 − e^(−dt/τ)`, with τ = 0.12 s. The result is
   a 0..1 confidence.
2. **Hysteresis.** A gesture becomes *active* when confidence rises above 0.65 and
   inactive only when it falls below 0.35. A flickering 50/50 detection never triggers.
3. **Hold detection.** A gesture is *held* once it has been active for its hold time
   (0.5–0.8 s). The engine reacts to the rising edge (`just_held`), so a single hold
   fires exactly once.
4. **Grace period.** A dropout shorter than 0.25 s (missed detections, motion blur) does
   not reset a hold or end a Rasengan.
5. **Tap detection.** A contact lasting 0.06–1.2 s, then released, counts as a tap. A
   one-frame graze does not. A long press (≥ 0.8 s) also counts.
6. **Stability.** The system averages the speed of the nose-bridge anchor over 1.5 s,
   in face widths per second, so the measure doesn't depend on distance or resolution.
   Byakugou needs a speed ≤ 0.25 and a window that is at least 80 % filled.
7. **Expression.** Expression scores are averaged over the last 0.6 s. They are then
   measured *relative to the user's neutral face*: a slow EMA (τ = 8 s), frozen while the
   prayer sign is held. Resting blendshape values vary a lot between people. A smile
   raises both squint and brow-down scores, so a smile vetoes Sharingan.
8. **Temporal conflicts** are handled in the engine. Only **one hand jutsu**
   (Rasengan, Chidori or Lion Fist) can run at a time. Modes (Dojutsu, Byakugou) stack
   on top of hand jutsu.

The renderer draws a **progress ring** around a gesture while it is being held, so the
user can see it was recognized.

### Chakra system and cooldowns
- `chakra = 100`. Instant jutsu pay up front with `spend()`, which refuses the jutsu if
  you can't afford it. Continuous jutsu call `drain(rate, dt)` every frame.
- **Chakra 0 blocks every jutsu and ends running ones.**
- Passive regeneration is 1.5/s, but only when no jutsu is running. Without it the
  system would stay locked once chakra ran out. Set `CHAKRA_REGEN_PER_S = 0` to disable it.
- **Byakugou** refills chakra and turns `infinite_chakra` on. While it is active,
  `spend` and `drain` cost nothing.
- **Cooldowns.** Each jutsu stores its own `ready_at` time. The standard cooldown is
  7 min and starts when the jutsu ends; Byakugou's is 20 min and starts at activation.
  Trying a jutsu during its cooldown shows the remaining time and does nothing.
- `--demo` shortens cooldowns so you can test without waiting.

### AR effects
Every effect draws into a small layer the size of its region of interest (ROI), then
blends into the frame.

- **Rasengan**: placed at the midpoint between the palm centres, with a radius of
  0.3 → 1.1 palm sizes that eases in as it charges. It is drawn as a dark-blue body,
  12 rotating elliptical arcs with random tilt and speed fixed per activation, a white
  core, and orbiting wind particles. A bloom pass and additive blending make it look
  emissive.
- **Chidori**: jagged lightning bolts generated fresh every frame by midpoint
  displacement (4 levels). They radiate from the palm, crackle out to the fingertips
  and branch. Each bolt is a thick blue stroke with a thin white core, plus bloom and
  random flicker.
- **Lion Fist**: an aura made from the hand's convex hull, dilated and blurred into a
  ring at ¼ resolution. Flame tongues lick outward from the hull and an opening
  shockwave ring expands.
- **Sharingan / Byakugan**: a high-resolution (96 px) iris sprite, downscaled with
  `INTER_AREA` for antialiasing and placed at the iris centre at the iris radius.
  - Its alpha is masked by the **eye contour**, so the eyelids cover it naturally.
  - Sharingan has 3 rotating tomoe and a faint red bloom.
  - Byakugan has a pale iris and bulging veins that grow from the eye corners toward the temples.
- **Byakugou Seal**: a violet diamond at the forehead, rotated with head roll.
  Markings follow chains of face-mesh landmarks: forehead → temples → jaw → cheek
  branches, then extended down the neck and body along the face's down vector. Each
  chain is smoothed with Chaikin's algorithm and revealed by arc length over 2.5 s.
  The seal pulses with glow, and a burst plays on activation.

## Real-time performance

What's implemented, and why:

1. **Threaded capture.** `ThreadedCamera` keeps only the newest frame
   (`CAP_PROP_BUFFERSIZE=1`), so old frames are dropped rather than queued. Queued
   frames are what causes lag.
2. **Pipelined inference.** `InferenceWorker` infers frame N while the main thread
   renders frame N−1, so a frame costs the slower stage instead of the sum of both.
   Each result carries its own frame, so overlays stay aligned with the image they
   were computed on.
3. **Hand and face models run in parallel.** MediaPipe releases the GIL during
   inference, so the two models can run on separate threads.
4. **VIDEO running mode** tracks landmarks between frames instead of detecting every frame.
5. **Cheap rendering.** Effects are drawn only inside their ROI. Large blurs run on a
   ⅛–½ scale copy, which looks the same and is up to 64× cheaper. Random parameters
   are cached per activation. There are no per-pixel Python loops.
6. **OpenCV threads** are capped at 1 (`OPENCV_THREADS`) so OpenCV's thread pool
   doesn't compete with MediaPipe's XNNPACK threads.
7. Optional knobs: `FACE_EVERY_N_FRAMES = 2` and `INFERENCE_SCALE`. Measured, the
   scale makes no difference: the models resize their input internally.

Measured on this development machine (640×480, CPU only):

| Stage | Time |
|---|---|
| hand landmarker, 2 hands | ~23 ms |
| face landmarker | ~12 ms (runs in parallel with hands) |
| effects | 4–10 ms each (they were 30–58 ms before the downscaled-blur and ROI changes) |
| end-to-end worst case: 2 hands + face + Lion Fist + Sharingan at once | **~21–24 FPS** pipelined |

With one hand or a single effect it runs faster. The hand model is the floor
(~11 ms per hand).

## Tuning
- Run with `--debug` to see each hand's `palm`/`back` state, live gesture confidences,
  head speed and the baseline-relative expression scores.
- If a gesture triggers too easily, raise its hold time in `HOLD_S` or tighten its threshold.
- If Sharingan/Byakugan picks the wrong type, adjust `SHARINGAN_MIN_BROW_DOWN` and `SHARINGAN_MIN_SCORE`.
