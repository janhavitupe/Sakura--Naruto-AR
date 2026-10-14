"""Central configuration. Every threshold, cost and timing lives here so the
system can be tuned without touching module code."""

# ---------------------------------------------------------------- camera / IO
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
WINDOW_NAME = "Chakra System"

HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"

# Inference is run on a downscaled copy of the frame when this is < 1.0.
# Landmarks are normalized, so they map back onto the full-res frame for free.
INFERENCE_SCALE = 1.0
# Run the face model every N frames (1 = every frame). Face motion is slow and
# the landmark filter hides the gap, so 2 is a cheap FPS win on weak CPUs.
FACE_EVERY_N_FRAMES = 1
# OpenCV's own thread pool competes with MediaPipe's XNNPACK threads; one
# OpenCV thread measured as fastest and most consistent for this pipeline.
OPENCV_THREADS = 1

# ---------------------------------------------------------------- detectors
HAND_DETECTION_CONF = 0.6
HAND_PRESENCE_CONF = 0.6
HAND_TRACKING_CONF = 0.5
FACE_DETECTION_CONF = 0.5
FACE_PRESENCE_CONF = 0.5
FACE_TRACKING_CONF = 0.5

# One-Euro landmark filter (jitter removal for rendering + features)
ONE_EURO_MIN_CUTOFF = 1.5   # lower = smoother when still
ONE_EURO_BETA = 0.02        # higher = less lag when moving fast
ONE_EURO_D_CUTOFF = 1.0

# ---------------------------------------------------------------- gestures
# Finger joint bend (degrees, 0 = straight) thresholds
FINGER_EXTENDED_MAX_BEND = 45.0     # total PIP+DIP bend below this = straight
                                    # (relaxed real fingers measure 25-40)
FINGER_CURLED_MIN_BEND = 150.0      # above this = fully folded (fist)

# Claw: fingers bent at PIP/DIP but tips still away from the palm, spread out
CLAW_MIN_BEND = 60.0
CLAW_MAX_BEND = 175.0
CLAW_MIN_TIP_DIST = 0.75            # tip-to-palm-centre / palm size
CLAW_MIN_FINGERS = 3

# Open palm
PALM_MIN_SPREAD = 0.55              # index-tip to pinky-tip / palm size
PALM_MIN_THUMB_OUT = 0.35           # thumb-tip to index-MCP / palm size

# Rasengan cup: two cupped hands close together
CUP_MIN_BEND = 40.0                 # mean MCP+PIP+DIP curl: flat ~<30, fist ~>230
CUP_MAX_BEND = 220.0
CUP_MAX_CENTER_DIST = 2.6           # palm-centre distance / mean palm size
CUP_MIN_CENTER_DIST = 0.5

# Prayer: index fingers extended, tips touching, pointing the same way
PRAYER_MAX_INDEX_TIP_DIST = 0.45    # / palm size
PRAYER_MAX_WRIST_DIST = 1.2         # / palm size
PRAYER_MIN_INDEX_COS = 0.85         # cosine between the two index directions

# Forehead tap
FOREHEAD_TOUCH_RADIUS = 0.22        # fingertip-to-forehead / face width
TAP_MIN_CONTACT_S = 0.06
TAP_MAX_CONTACT_S = 1.2
TAP_HOLD_COUNTS_AS_TAP_S = 0.8      # a long press also counts

# Stability (low body movement) for Byakugou
STABILITY_WINDOW_S = 1.5
STABILITY_MAX_SPEED = 0.25          # face-widths per second (mean head speed)

# ---------------------------------------------------------------- temporal
SMOOTHING_TAU_S = 0.12      # EMA time constant for gesture confidence
CONF_ON = 0.65              # hysteresis: become active above this
CONF_OFF = 0.35             # ...and inactive below this
DEFAULT_HOLD_S = 0.6        # stability required before a gesture "fires"
HOLD_S = {
    "rasengan_cup": 0.5,
    "claw": 0.6,
    "prayer": 0.8,
    "open_palm": 0.5,
}
RELEASE_GRACE_S = 0.25      # tolerated dropout before a held gesture ends

# Highest priority first. A hand consumed by a higher gesture can't feed a
# lower one in the same frame.
GESTURE_PRIORITY = ["forehead_tap", "prayer", "rasengan_cup", "claw", "open_palm"]

# ---------------------------------------------------------------- chakra
CHAKRA_MAX = 100.0
CHAKRA_REGEN_PER_S = 1.5        # passive regen while no jutsu is active
COST = {
    "rasengan_per_s": 12.0,
    "rasengan_start": 5.0,
    "chidori": 40.0,
    "lion_fist": 10.0,
    "dojutsu_start": 5.0,
    "dojutsu_per_s": 2.0,
}

# ---------------------------------------------------------------- cooldowns
STANDARD_COOLDOWN_S = 7 * 60
BYAKUGOU_COOLDOWN_S = 20 * 60
DEMO_STANDARD_COOLDOWN_S = 8
DEMO_BYAKUGOU_COOLDOWN_S = 30

# ---------------------------------------------------------------- durations
CHIDORI_DURATION_S = 3.0
LION_FIST_DURATION_S = 2.0
DOJUTSU_MAX_DURATION_S = 45.0
BYAKUGOU_DURATION_S = 90.0
BYAKUGOU_SPREAD_S = 2.5          # time for the markings to cover the face
BYAKUGOU_BODY_MARKS = False      # True: also continue markings down the neck (not in reference)
RASENGAN_MIN_RADIUS = 0.30       # x palm size, so the sphere scales with distance
RASENGAN_MAX_RADIUS = 1.10
RASENGAN_FULL_CHARGE_S = 3.0

# Dojutsu expression classification (MediaPipe blendshape scores, 0..1)
# Scores are *relative to the user's neutral face* (see ExpressionBaseline).
DOJUTSU_SAMPLE_S = 0.6           # expression averaged over the last N seconds
EXPRESSION_BASELINE_TAU_S = 8.0  # how fast the neutral-face estimate adapts
SHARINGAN_MIN_BROW_DOWN = 0.20
SHARINGAN_MIN_SCORE = 0.22       # 0.6*browDown + 0.4*eyeSquint
SHARINGAN_MAX_SMILE = 0.40       # smiling raises squint/brow scores: not "focused"
