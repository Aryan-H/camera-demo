import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import "./style.css";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const SAMPLE_INTERVAL_MS = 100;
const AUXILIARY_SAMPLE_INTERVAL_MS = 200;
const GAZE_SMOOTHING_ALPHA = 0.35;
const EYE_CONTACT_ENTER_FRAMES = 3;
const EYE_CONTACT_EXIT_FRAMES = 2;
const BLINK_HOLD_FRAMES = 3;
const CONFIDENCE_MIN_VALID_SAMPLES = 10;
const CONFIDENCE_HISTORY_GAP_MS = 750;
const SMILE_TIME_CONSTANT_SECONDS = 0.45;
const CONFIDENCE_SCORE_TIME_CONSTANT_SECONDS = 1;
const MOTION_SAMPLE_WIDTH = 96;
const MOTION_SAMPLE_HEIGHT = 72;
const MOTION_HISTORY_GAP_MS = 750;
const MOTION_TIME_CONSTANT_SECONDS = 0.3;
const MOTION_PIXEL_DELTA_THRESHOLD = 18;
const MOTION_ENTER_FRACTION = 0.045;
const MOTION_EXIT_FRACTION = 0.018;
const MOTION_ENTER_FRAMES = 3;
const MOTION_EXIT_FRAMES = 4;
const MOTION_WARMUP_SAMPLES = 5;
const FACE_TOUCH_RELEASE_FRAMES = 4;
const FACE_TOUCH_COOLDOWN_MS = 800;
const GESTURE_TIME_CONSTANT_SECONDS = 0.35;
const POSTURE_BASELINE_SAMPLES = 15;
const POSTURE_MIN_HIP_BASELINE_SAMPLES = 10;
const POSTURE_TIME_CONSTANT_SECONDS = 0.4;

type FaceLandmarkerResult = ReturnType<FaceLandmarker["detectForVideo"]>;
type HandLandmarkerResult = ReturnType<HandLandmarker["detectForVideo"]>;
type PoseLandmarkerResult = ReturnType<PoseLandmarker["detectForVideo"]>;
type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;
type FacialTransformationMatrix = FaceLandmarkerResult["facialTransformationMatrixes"][number];
type HandPoint = HandLandmarkerResult["landmarks"][number][number];
type PosePoint = PoseLandmarkerResult["landmarks"][number][number];

interface CameraGazeEstimate {
  eyeDirectionScore: number;
  headYawDegrees: number;
  headPitchDegrees: number;
}

interface FaceBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  edgeMargin: number;
}

interface SimulatedConfidenceDiagnostics {
  available: boolean;
  unavailableReason: string | null;
  displayedScore: number | null;
  rawScore: number | null;
  components: {
    cameraFacing: number;
    eyeCentered: number;
    yawForward: number;
    pitchForward: number;
    positiveExpression: number;
    smileEvidence: number;
  };
  inputs: {
    eyeDirectionScore: number;
    headYawDegrees: number;
    headPitchDegrees: number;
    faceWidthNormalized: number;
    faceHeightNormalized: number;
    edgeMarginNormalized: number;
    rawSmileSignal: number;
    smoothedSmileSignal: number;
  };
  weights: {
    cameraEngagement: number;
    positiveExpression: number;
  };
  signalQuality: {
    level: "low" | "fair" | "good";
    framing: number;
    frameCentering: number;
    faceSize: number;
    edgeMargin: number;
  };
  excludedSignals: string[];
}

interface MovementDiagnostics {
  available: boolean;
  unavailableReason: string | null;
  detected: boolean | null;
  intensityPercent: number | null;
  activePixelRatio: number | null;
  smoothedActivePixelRatio: number | null;
  meanResidualLuma: number | null;
  compensatedBrightnessShift: number | null;
  thresholds: {
    enterFraction: number;
    exitFraction: number;
    activePixelDelta: number;
  };
  sampleSize: { width: number; height: number };
}

interface PoseSample {
  shoulderMidScreenX: number;
  shoulderSpan: number;
  torsoAngleDegrees: number | null;
  shoulderHipDepthMeters: number | null;
  shoulderVisibility: number;
  hipVisibility: number | null;
}

interface PostureBaseline {
  shoulderMidScreenX: number;
  shoulderSpan: number;
  torsoAngleDegrees: number | null;
  shoulderHipDepthMeters: number | null;
  fullTorso: boolean;
}

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing page element: #${id}`);
  return element as T;
}

const video = byId<HTMLVideoElement>("video");
const canvas = byId<HTMLCanvasElement>("overlay");
const context = canvas.getContext("2d") as CanvasRenderingContext2D | null;
if (!context) throw new Error("Could not create a 2D canvas context.");
const drawingContext: CanvasRenderingContext2D = context;
const motionCanvas = document.createElement("canvas");
motionCanvas.width = MOTION_SAMPLE_WIDTH;
motionCanvas.height = MOTION_SAMPLE_HEIGHT;
const motionContext = motionCanvas.getContext("2d", { willReadFrequently: true });
if (!motionContext) throw new Error("Could not create the movement-analysis canvas context.");
const motionDrawingContext: CanvasRenderingContext2D = motionContext;

const ui = {
  start: byId<HTMLButtonElement>("startBtn"),
  stop: byId<HTMLButtonElement>("stopBtn"),
  reset: byId<HTMLButtonElement>("resetBtn"),
  placeholder: byId<HTMLDivElement>("placeholder"),
  status: byId<HTMLSpanElement>("status"),
  tracking: byId<HTMLDivElement>("tracking"),
  trackingBar: byId<HTMLSpanElement>("trackingBar"),
  framing: byId<HTMLDivElement>("framing"),
  framingBar: byId<HTMLSpanElement>("framingBar"),
  blinks: byId<HTMLDivElement>("blinks"),
  smile: byId<HTMLDivElement>("smile"),
  eyeContact: byId<HTMLDivElement>("eyeContact"),
  movement: byId<HTMLDivElement>("movementDetected"),
  movementHint: byId<HTMLDivElement>("movementHint"),
  movementBar: byId<HTMLSpanElement>("movementBar"),
  faceTouches: byId<HTMLDivElement>("faceTouches"),
  faceTouchHint: byId<HTMLDivElement>("faceTouchHint"),
  handActivity: byId<HTMLDivElement>("handActivity"),
  handActivityHint: byId<HTMLDivElement>("handActivityHint"),
  handActivityBar: byId<HTMLSpanElement>("handActivityBar"),
  posture: byId<HTMLDivElement>("postureState"),
  postureHint: byId<HTMLDivElement>("postureHint"),
  postureBar: byId<HTMLSpanElement>("postureBar"),
  confidence: byId<HTMLDivElement>("confidenceScore"),
  confidenceHint: byId<HTMLDivElement>("confidenceHint"),
  confidenceBar: byId<HTMLSpanElement>("confidenceBar"),
  detected: byId<HTMLElement>("detected"),
  position: byId<HTMLElement>("position"),
  eyes: byId<HTMLElement>("eyes"),
  latency: byId<HTMLElement>("latency"),
  raw: byId<HTMLPreElement>("raw"),
};

let landmarker: FaceLandmarker | null = null;
let handLandmarker: HandLandmarker | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let handModelDelegate: "GPU" | "CPU" | null = null;
let poseModelDelegate: "GPU" | "CPU" | null = null;
let handModelError: string | null = null;
let poseModelError: string | null = null;
let auxiliaryModelsAttempted = false;
let stream: MediaStream | null = null;
let animationId = 0;
let lastInferenceAt = 0;
let lastHandInferenceAt = 0;
let lastPoseInferenceAt = 0;
let nextAuxiliaryModel: "hand" | "pose" = "hand";
let processedFrames = 0;
let detectedFrames = 0;
let blinkCount = 0;
let wasBlinking = false;
let eyeContact = false;
let eyeContactPassFrames = 0;
let eyeContactFailFrames = 0;
let closedEyeFrames = 0;
let smoothedCameraGaze: CameraGazeEstimate | null = null;
let lastConfidenceSampleAt = 0;
let smoothedSmileSignal: number | null = null;
let confidenceValidSamples = 0;
let displayedConfidenceScore: number | null = null;
let lastConfidenceScoreAt = 0;
let previousMotionLuma: Uint8Array | null = null;
let lastMotionSampleAt = 0;
let smoothedActivePixelRatio: number | null = null;
let motionWarmupSamples = 0;
let movementDetected = false;
let movementEnterFrames = 0;
let movementExitFrames = 0;
let faceTouchCount = 0;
let faceTouchActive = false;
let faceTouchRecentCandidates: boolean[] = [];
let faceTouchReleaseFrames = 0;
let lastFaceTouchEventAt = -Infinity;
let previousHandTracks: Array<{
  x: number;
  y: number;
  fingertipOffsets: Array<{ x: number; y: number }>;
  handedness: string | null;
}> = [];
let lastHandSampleAt = 0;
let smoothedGestureSpeed: number | null = null;
let handGestureReferenceMode: "face-relative" | "frame-relative" | null = null;
let postureCalibrationSamples: PoseSample[] = [];
let postureBaseline: PostureBaseline | null = null;
let lastPostureSampleAt = 0;
let smoothedPostureLateral: number | null = null;
let smoothedPostureDepth: number | null = null;
let postureMode: "full-torso" | "shoulders-only" | null = null;
let postureLateralState: "centered" | "screen-left" | "screen-right" = "centered";
let postureDepthState: "baseline" | "toward" | "away" = "baseline";
let latestHandGestureDiagnostics: Record<string, unknown> = {
  available: false,
  unavailableReason: "camera not started",
};
let latestFaceTouchDiagnostics: Record<string, unknown> = {
  available: false,
  unavailableReason: "camera not started",
  eventCount: 0,
};
let latestPostureDiagnostics: Record<string, unknown> = {
  available: false,
  unavailableReason: "camera not started",
};

function setStatus(message: string): void {
  ui.status.textContent = message;
}

function blendshapeMap(result: ReturnType<FaceLandmarker["detectForVideo"]>): Record<string, number> {
  const map: Record<string, number> = {};
  for (const category of result.faceBlendshapes?.[0]?.categories ?? []) {
    map[category.categoryName] = category.score;
  }
  return map;
}

function setEyeContact(value: boolean): void {
  eyeContact = value;
  ui.eyeContact.textContent = value ? "True" : "False";
  ui.eyeContact.classList.toggle("is-true", value);
  ui.eyeContact.classList.toggle("is-false", !value);
}

function resetEyeContactEstimate(): void {
  eyeContactPassFrames = 0;
  eyeContactFailFrames = 0;
  closedEyeFrames = 0;
  smoothedCameraGaze = null;
  setEyeContact(false);
}

function clearConfidenceHistory(): void {
  lastConfidenceSampleAt = 0;
  smoothedSmileSignal = null;
  confidenceValidSamples = 0;
  displayedConfidenceScore = null;
  lastConfidenceScoreAt = 0;
}

function setConfidenceUnavailable(
  message = "appearance-based practice heuristic, not a psychological measurement",
): void {
  ui.confidence.textContent = "—";
  ui.confidenceHint.textContent = message;
  ui.confidenceBar.style.width = "0%";
}

function resetConfidenceProxy(message?: string): void {
  clearConfidenceHistory();
  setConfidenceUnavailable(message);
}

function setConfidenceCalibrating(): void {
  ui.confidence.textContent = "…";
  ui.confidenceHint.textContent = "calibrating the appearance-based simulation";
  ui.confidenceBar.style.width = "0%";
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function smoothstep(low: number, high: number, value: number): number {
  const progress = clamp01((value - low) / (high - low));
  return progress * progress * (3 - 2 * progress);
}

function rounded(value: number, digits = 3): number {
  return Number(value.toFixed(digits));
}

function median(values: number[]): number | null {
  const finiteValues = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finiteValues.length) return null;
  const middle = Math.floor(finiteValues.length / 2);
  return finiteValues.length % 2
    ? finiteValues[middle]
    : (finiteValues[middle - 1] + finiteValues[middle]) / 2;
}

function calculateFaceBounds(
  points: FaceLandmarkerResult["faceLandmarks"][number],
): FaceBounds {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }
  return {
    minX,
    minY,
    maxX,
    maxY,
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    width: maxX - minX,
    height: maxY - minY,
    edgeMargin: Math.max(0, Math.min(minX, minY, 1 - maxX, 1 - maxY)),
  };
}

function resetFaceTouchEstimate(
  resetCount: boolean,
  hint = "2D hand/face proximity events",
): void {
  if (resetCount) faceTouchCount = 0;
  faceTouchActive = false;
  faceTouchRecentCandidates = [];
  faceTouchReleaseFrames = 0;
  lastFaceTouchEventAt = resetCount ? -Infinity : lastFaceTouchEventAt;
  ui.faceTouches.textContent = String(faceTouchCount);
  ui.faceTouchHint.textContent = hint;
  latestFaceTouchDiagnostics = {
    available: false,
    unavailableReason: hint,
    active: false,
    eventCount: faceTouchCount,
  };
}

function updateFaceTouchEstimate(
  result: HandLandmarkerResult,
  faceBounds: FaceBounds | null,
  timestampMs: number,
): void {
  if (!faceBounds || faceBounds.width <= 0 || faceBounds.height <= 0) {
    faceTouchRecentCandidates.push(false);
    faceTouchRecentCandidates = faceTouchRecentCandidates.slice(-3);
    if (faceTouchActive && ++faceTouchReleaseFrames >= FACE_TOUCH_RELEASE_FRAMES) {
      faceTouchActive = false;
      faceTouchReleaseFrames = 0;
    }
    ui.faceTouches.textContent = String(faceTouchCount);
    ui.faceTouchHint.textContent = "face unavailable; event detection paused";
    latestFaceTouchDiagnostics = {
      available: false,
      unavailableReason: "face unavailable",
      active: faceTouchActive,
      eventCount: faceTouchCount,
    };
    return;
  }

  const radiusX = Math.max(faceBounds.width * 0.6, 0.001);
  const radiusY = Math.max(faceBounds.height * 0.58, 0.001);
  const ellipseDistance = (point: HandPoint): number =>
    Math.hypot(
      (point.x - faceBounds.centerX) / radiusX,
      (point.y - faceBounds.centerY) / radiusY,
    );
  const fingertipPairs = [
    [4, 3],
    [8, 7],
    [12, 11],
    [16, 15],
    [20, 19],
  ] as const;

  let closestEllipseDistance = Infinity;
  let contactLandmarkCount = 0;
  let pairedFingerContact = false;
  for (const hand of result.landmarks) {
    let handContactLandmarks = 0;
    for (const point of hand) {
      const distance = ellipseDistance(point);
      closestEllipseDistance = Math.min(closestEllipseDistance, distance);
      if (distance <= 1) handContactLandmarks += 1;
    }
    contactLandmarkCount = Math.max(contactLandmarkCount, handContactLandmarks);
    pairedFingerContact ||= fingertipPairs.some(
      ([tip, joint]) => ellipseDistance(hand[tip]) <= 1 && ellipseDistance(hand[joint]) <= 1.2,
    );
  }

  const candidate = pairedFingerContact || contactLandmarkCount >= 4;
  faceTouchRecentCandidates.push(candidate);
  faceTouchRecentCandidates = faceTouchRecentCandidates.slice(-3);
  const stableCandidate = faceTouchRecentCandidates.filter(Boolean).length >= 2;

  if (!faceTouchActive && stableCandidate) {
    faceTouchActive = true;
    faceTouchReleaseFrames = 0;
    if (timestampMs - lastFaceTouchEventAt >= FACE_TOUCH_COOLDOWN_MS) {
      faceTouchCount += 1;
      lastFaceTouchEventAt = timestampMs;
    }
  } else if (faceTouchActive) {
    faceTouchReleaseFrames = candidate ? 0 : faceTouchReleaseFrames + 1;
    if (faceTouchReleaseFrames >= FACE_TOUCH_RELEASE_FRAMES) {
      faceTouchActive = false;
      faceTouchReleaseFrames = 0;
      faceTouchRecentCandidates = [];
    }
  }

  ui.faceTouches.textContent = String(faceTouchCount);
  ui.faceTouchHint.textContent = faceTouchActive
    ? "possible contact now · 2D proximity estimate"
    : "2D overlap estimate; near-face gestures may also count";
  latestFaceTouchDiagnostics = {
    available: true,
    unavailableReason: null,
    active: faceTouchActive,
    eventCount: faceTouchCount,
    candidate,
    recentCandidateFrames: faceTouchRecentCandidates,
    releaseFrames: faceTouchReleaseFrames,
    closestEllipseDistance: Number.isFinite(closestEllipseDistance)
      ? rounded(closestEllipseDistance)
      : null,
    contactLandmarkCount,
    thresholds: {
      ellipseRadiusFaceWidth: 0.6,
      ellipseRadiusFaceHeight: 0.58,
      stableFrames: "2 of 3",
      releaseFrames: FACE_TOUCH_RELEASE_FRAMES,
      cooldownMs: FACE_TOUCH_COOLDOWN_MS,
    },
    limitation: "2D proximity only; it cannot verify physical contact",
  };
}

function resetHandGestureEstimate(message = "movement amount, not gesture quality"): void {
  previousHandTracks = [];
  lastHandSampleAt = 0;
  smoothedGestureSpeed = null;
  handGestureReferenceMode = null;
  ui.handActivity.textContent = "—";
  ui.handActivityHint.textContent = message;
  ui.handActivityBar.style.width = "0%";
  latestHandGestureDiagnostics = {
    available: false,
    unavailableReason: message,
  };
}

function handPalmCenter(hand: HandPoint[]): { x: number; y: number } {
  const palmIndices = [0, 5, 9, 13, 17];
  return palmIndices.reduce(
    (sum, index) => ({ x: sum.x + hand[index].x / 5, y: sum.y + hand[index].y / 5 }),
    { x: 0, y: 0 },
  );
}

function updateHandGestureEstimate(
  result: HandLandmarkerResult,
  faceBounds: FaceBounds | null,
  timestampMs: number,
): void {
  if (lastHandSampleAt && timestampMs - lastHandSampleAt > CONFIDENCE_HISTORY_GAP_MS) {
    previousHandTracks = [];
    smoothedGestureSpeed = null;
  }

  if (!result.landmarks.length) {
    previousHandTracks = [];
    smoothedGestureSpeed = null;
    lastHandSampleAt = timestampMs;
    ui.handActivity.textContent = "Not visible";
    ui.handActivityHint.textContent = "hands may be outside the webcam frame";
    ui.handActivityBar.style.width = "0%";
    latestHandGestureDiagnostics = {
      available: true,
      unavailableReason: null,
      handsVisible: 0,
      category: "not-visible",
    };
    return;
  }

  const anchorX = faceBounds?.centerX ?? 0.5;
  const anchorY = faceBounds?.centerY ?? 0.5;
  const scaleX = Math.max(faceBounds?.width ?? 0.3, 0.1);
  const scaleY = Math.max(faceBounds?.height ?? 0.4, 0.12);
  const referenceMode = faceBounds ? "face-relative" : "frame-relative";
  if (handGestureReferenceMode && handGestureReferenceMode !== referenceMode) {
    previousHandTracks = [];
    smoothedGestureSpeed = null;
    lastHandSampleAt = 0;
  }
  handGestureReferenceMode = referenceMode;
  const fingertipIndices = [4, 8, 12, 16, 20];
  const currentTracks = result.landmarks.map((hand, handIndex) => {
    const palm = handPalmCenter(hand);
    const normalizedPalm = {
      x: (palm.x - anchorX) / scaleX,
      y: (palm.y - anchorY) / scaleY,
    };
    return {
      ...normalizedPalm,
      fingertipOffsets: fingertipIndices.map((index) => ({
        x: (hand[index].x - palm.x) / scaleX,
        y: (hand[index].y - palm.y) / scaleY,
      })),
      handedness: result.handedness[handIndex]?.[0]?.categoryName ?? null,
    };
  });

  const elapsedSeconds = lastHandSampleAt ? (timestampMs - lastHandSampleAt) / 1000 : 0;
  const unusedPrevious = new Set(previousHandTracks.map((_, index) => index));
  const handSpeeds: number[] = [];
  for (const current of currentTracks) {
    let bestIndex = -1;
    let bestCost = Infinity;
    for (const previousIndex of unusedPrevious) {
      const previous = previousHandTracks[previousIndex];
      const handednessPenalty =
        current.handedness && previous.handedness && current.handedness !== previous.handedness
          ? 0.35
          : 0;
      const cost = Math.hypot(current.x - previous.x, current.y - previous.y) + handednessPenalty;
      if (cost < bestCost) {
        bestCost = cost;
        bestIndex = previousIndex;
      }
    }
    if (bestIndex < 0 || elapsedSeconds <= 0) continue;
    const previous = previousHandTracks[bestIndex];
    unusedPrevious.delete(bestIndex);
    const palmSpeed = Math.hypot(current.x - previous.x, current.y - previous.y) / elapsedSeconds;
    const fingertipSpeeds = current.fingertipOffsets.map((offset, index) =>
      Math.hypot(
        offset.x - previous.fingertipOffsets[index].x,
        offset.y - previous.fingertipOffsets[index].y,
      ) / elapsedSeconds,
    );
    const fingerSpeed = median(fingertipSpeeds) ?? 0;
    handSpeeds.push(Math.min(4, 0.75 * palmSpeed + 0.25 * fingerSpeed));
  }

  previousHandTracks = currentTracks;
  lastHandSampleAt = timestampMs;
  if (!handSpeeds.length) {
    ui.handActivity.textContent = "Calibrating…";
    ui.handActivityHint.textContent = `${currentTracks.length} hand${currentTracks.length === 1 ? "" : "s"} visible · matching movement`;
    ui.handActivityBar.style.width = "0%";
    latestHandGestureDiagnostics = {
      available: false,
      unavailableReason: "calibrating",
      handsVisible: currentTracks.length,
    };
    return;
  }

  const rawSpeed = Math.max(...handSpeeds);
  const alpha = 1 - Math.exp(-elapsedSeconds / GESTURE_TIME_CONSTANT_SECONDS);
  smoothedGestureSpeed =
    smoothedGestureSpeed === null
      ? rawSpeed
      : smoothedGestureSpeed + alpha * (rawSpeed - smoothedGestureSpeed);
  const category =
    smoothedGestureSpeed < 0.22
      ? "Still"
      : smoothedGestureSpeed < 0.55
        ? "Low"
        : smoothedGestureSpeed < 1.1
          ? "Moderate"
          : "High";
  const activityPercent = 100 * smoothstep(0.15, 1.5, smoothedGestureSpeed);
  ui.handActivity.textContent = category;
  ui.handActivityHint.textContent = `${currentTracks.length} hand${currentTracks.length === 1 ? "" : "s"} visible · movement amount only`;
  ui.handActivityBar.style.width = `${activityPercent}%`;
  latestHandGestureDiagnostics = {
    available: true,
    unavailableReason: null,
    handsVisible: currentTracks.length,
    category: category.toLowerCase(),
    activityPercent: rounded(activityPercent, 1),
    rawSpeedFaceUnitsPerSecond: rounded(rawSpeed),
    smoothedSpeedFaceUnitsPerSecond: rounded(smoothedGestureSpeed),
    referenceFrame: faceBounds ? "relative to current face" : "normalized webcam frame fallback",
    thresholdsFaceUnitsPerSecond: { still: 0.22, low: 0.55, moderate: 1.1 },
  };
}

function resetPostureEstimate(message = "relative to your natural starting position"): void {
  postureCalibrationSamples = [];
  postureBaseline = null;
  lastPostureSampleAt = 0;
  smoothedPostureLateral = null;
  smoothedPostureDepth = null;
  postureMode = null;
  postureLateralState = "centered";
  postureDepthState = "baseline";
  ui.posture.textContent = "—";
  ui.postureHint.textContent = message;
  ui.postureBar.style.width = "0%";
  latestPostureDiagnostics = {
    available: false,
    unavailableReason: message,
  };
}

function posePointVisible(point: PosePoint | undefined, threshold: number): point is PosePoint {
  return Boolean(point && (point.visibility ?? 1) >= threshold);
}

function extractPoseSample(result: PoseLandmarkerResult): PoseSample | null {
  const points = result.landmarks[0];
  if (!points) return null;
  const leftShoulder = points[11];
  const rightShoulder = points[12];
  if (!posePointVisible(leftShoulder, 0.6) || !posePointVisible(rightShoulder, 0.6)) return null;

  const shoulderMidScreenX = 1 - (leftShoulder.x + rightShoulder.x) / 2;
  const shoulderSpan = Math.abs(leftShoulder.x - rightShoulder.x);
  if (shoulderSpan < 0.02) return null;
  const shoulderVisibility = Math.min(leftShoulder.visibility ?? 1, rightShoulder.visibility ?? 1);
  const leftHip = points[23];
  const rightHip = points[24];
  const hipsVisible = posePointVisible(leftHip, 0.55) && posePointVisible(rightHip, 0.55);
  let torsoAngleDegrees: number | null = null;
  let shoulderHipDepthMeters: number | null = null;
  let hipVisibility: number | null = null;

  if (hipsVisible) {
    const hipMidScreenX = 1 - (leftHip.x + rightHip.x) / 2;
    const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
    const hipMidY = (leftHip.y + rightHip.y) / 2;
    torsoAngleDegrees =
      Math.atan2(
        (shoulderMidScreenX - hipMidScreenX) * (video.videoWidth || 640),
        (hipMidY - shoulderMidY) * (video.videoHeight || 480),
      ) *
      (180 / Math.PI);
    hipVisibility = Math.min(leftHip.visibility ?? 1, rightHip.visibility ?? 1);

    const world = result.worldLandmarks[0];
    if (world?.[11] && world?.[12] && world?.[23] && world?.[24]) {
      const shoulderDepth = (world[11].z + world[12].z) / 2;
      const hipDepth = (world[23].z + world[24].z) / 2;
      shoulderHipDepthMeters = shoulderDepth - hipDepth;
    }
  }

  return {
    shoulderMidScreenX,
    shoulderSpan,
    torsoAngleDegrees,
    shoulderHipDepthMeters,
    shoulderVisibility,
    hipVisibility,
  };
}

function updateLateralPostureState(value: number, enter: number, exit: number): void {
  if (postureLateralState === "screen-left" && value > -exit) postureLateralState = "centered";
  if (postureLateralState === "screen-right" && value < exit) postureLateralState = "centered";
  if (postureLateralState === "centered") {
    if (value <= -enter) postureLateralState = "screen-left";
    if (value >= enter) postureLateralState = "screen-right";
  }
}

function updateDepthPostureState(value: number, enter: number, exit: number): void {
  if (postureDepthState === "toward" && value < exit) postureDepthState = "baseline";
  if (postureDepthState === "away" && value > -exit) postureDepthState = "baseline";
  if (postureDepthState === "baseline") {
    if (value >= enter) postureDepthState = "toward";
    if (value <= -enter) postureDepthState = "away";
  }
}

function updatePostureEstimate(result: PoseLandmarkerResult, timestampMs: number): void {
  const sample = extractPoseSample(result);
  if (!sample) {
    ui.posture.textContent = "Not visible";
    ui.postureHint.textContent = "keep both shoulders in the webcam frame";
    ui.postureBar.style.width = "0%";
    latestPostureDiagnostics = {
      available: false,
      unavailableReason: "shoulders not reliably visible",
      baselineReady: Boolean(postureBaseline),
    };
    return;
  }

  if (!postureBaseline) {
    postureCalibrationSamples.push(sample);
    if (postureCalibrationSamples.length < POSTURE_BASELINE_SAMPLES) {
      ui.posture.textContent = "Calibrating…";
      ui.postureHint.textContent = `sit naturally · ${postureCalibrationSamples.length}/${POSTURE_BASELINE_SAMPLES}`;
      ui.postureBar.style.width = "0%";
      latestPostureDiagnostics = {
        available: false,
        unavailableReason: "calibrating natural starting position",
        calibrationSamples: postureCalibrationSamples.length,
        requiredSamples: POSTURE_BASELINE_SAMPLES,
      };
      return;
    }

    const fullTorsoSamples = postureCalibrationSamples.filter(
      (value) => value.torsoAngleDegrees !== null && value.shoulderHipDepthMeters !== null,
    );
    const fullTorso = fullTorsoSamples.length >= POSTURE_MIN_HIP_BASELINE_SAMPLES;
    postureBaseline = {
      shoulderMidScreenX: median(postureCalibrationSamples.map((value) => value.shoulderMidScreenX)) ?? 0.5,
      shoulderSpan: median(postureCalibrationSamples.map((value) => value.shoulderSpan)) ?? sample.shoulderSpan,
      torsoAngleDegrees: fullTorso
        ? median(fullTorsoSamples.map((value) => value.torsoAngleDegrees as number))
        : null,
      shoulderHipDepthMeters: fullTorso
        ? median(fullTorsoSamples.map((value) => value.shoulderHipDepthMeters as number))
        : null,
      fullTorso,
    };
  }

  const canUseFullTorso = Boolean(
    postureBaseline.fullTorso &&
      postureBaseline.torsoAngleDegrees !== null &&
      postureBaseline.shoulderHipDepthMeters !== null &&
      sample.torsoAngleDegrees !== null &&
      sample.shoulderHipDepthMeters !== null,
  );
  const currentMode = canUseFullTorso ? "full-torso" : "shoulders-only";
  let lateralSignal: number;
  let depthSignal: number;
  let lateralEnter: number;
  let lateralExit: number;
  let depthEnter: number;
  let depthExit: number;

  if (canUseFullTorso) {
    lateralSignal =
      (sample.torsoAngleDegrees as number) - (postureBaseline.torsoAngleDegrees as number);
    // Positive means the shoulders moved closer to the camera than the hips.
    depthSignal =
      (postureBaseline.shoulderHipDepthMeters as number) -
      (sample.shoulderHipDepthMeters as number);
    lateralEnter = 8;
    lateralExit = 4;
    depthEnter = 0.055;
    depthExit = 0.03;
  } else {
    lateralSignal =
      (sample.shoulderMidScreenX - postureBaseline.shoulderMidScreenX) /
      Math.max(postureBaseline.shoulderSpan, 0.02);
    depthSignal = sample.shoulderSpan / Math.max(postureBaseline.shoulderSpan, 0.02) - 1;
    lateralEnter = 0.2;
    lateralExit = 0.1;
    depthEnter = 0.12;
    depthExit = 0.06;
  }

  if (postureMode !== currentMode || !lastPostureSampleAt || timestampMs - lastPostureSampleAt > 750) {
    postureMode = currentMode;
    smoothedPostureLateral = lateralSignal;
    smoothedPostureDepth = depthSignal;
    postureLateralState = "centered";
    postureDepthState = "baseline";
  } else {
    const elapsedSeconds = Math.max(0, (timestampMs - lastPostureSampleAt) / 1000);
    const alpha = 1 - Math.exp(-elapsedSeconds / POSTURE_TIME_CONSTANT_SECONDS);
    const previousLateral = smoothedPostureLateral ?? lateralSignal;
    const previousDepth = smoothedPostureDepth ?? depthSignal;
    smoothedPostureLateral = previousLateral + alpha * (lateralSignal - previousLateral);
    smoothedPostureDepth = previousDepth + alpha * (depthSignal - previousDepth);
  }
  lastPostureSampleAt = timestampMs;
  const smoothedLateral = smoothedPostureLateral ?? lateralSignal;
  const smoothedDepth = smoothedPostureDepth ?? depthSignal;
  updateLateralPostureState(smoothedLateral, lateralEnter, lateralExit);
  updateDepthPostureState(smoothedDepth, depthEnter, depthExit);

  const labels: string[] = [];
  if (postureLateralState === "screen-left") labels.push("Screen left");
  if (postureLateralState === "screen-right") labels.push("Screen right");
  if (postureDepthState === "toward") labels.push("Toward camera");
  if (postureDepthState === "away") labels.push("Away from camera");
  const label = labels.length ? labels.join(" + ") : "Centered";
  const intensity = 100 * Math.max(
    smoothstep(lateralExit, lateralEnter * 2, Math.abs(smoothedLateral)),
    smoothstep(depthExit, depthEnter * 2, Math.abs(smoothedDepth)),
  );
  ui.posture.textContent = label;
  ui.postureHint.textContent =
    currentMode === "full-torso"
      ? "torso angle/depth relative to your starting posture"
      : "shoulder-only position/scale estimate";
  ui.postureBar.style.width = `${intensity}%`;
  latestPostureDiagnostics = {
    available: true,
    unavailableReason: null,
    state: label,
    mode: currentMode,
    intensityPercent: rounded(intensity, 1),
    measurements: {
      lateralSignal: rounded(smoothedLateral),
      towardCameraSignal: rounded(smoothedDepth),
      shoulderVisibility: rounded(sample.shoulderVisibility),
      hipVisibility: sample.hipVisibility === null ? null : rounded(sample.hipVisibility),
    },
    baseline: postureBaseline,
    thresholds: {
      lateralEnter,
      lateralExit,
      depthEnter,
      depthExit,
    },
    limitation: "relative visual posture estimate, not an ergonomic or psychological assessment",
  };
}

function setAuxiliaryModelsUnavailable(): void {
  if (!handLandmarker) {
    const reason = handModelError ?? "hand model unavailable";
    resetHandGestureEstimate(reason);
    resetFaceTouchEstimate(false, reason);
  }
  if (!poseLandmarker) resetPostureEstimate(poseModelError ?? "pose model unavailable");
}

function updateAuxiliaryMetrics(faceBounds: FaceBounds | null, timestampMs: number): void {
  const handDue = Boolean(
    handLandmarker &&
      (!lastHandInferenceAt || timestampMs - lastHandInferenceAt >= AUXILIARY_SAMPLE_INTERVAL_MS),
  );
  const poseDue = Boolean(
    poseLandmarker &&
      (!lastPoseInferenceAt || timestampMs - lastPoseInferenceAt >= AUXILIARY_SAMPLE_INTERVAL_MS),
  );
  let task: "hand" | "pose" | null = null;
  if (handDue && poseDue) task = nextAuxiliaryModel;
  else if (handDue) task = "hand";
  else if (poseDue) task = "pose";

  if (task === "hand" && handLandmarker) {
    lastHandInferenceAt = timestampMs;
    nextAuxiliaryModel = "pose";
    try {
      const result = handLandmarker.detectForVideo(video, timestampMs);
      updateHandGestureEstimate(result, faceBounds, timestampMs);
      updateFaceTouchEstimate(result, faceBounds, timestampMs);
    } catch (error) {
      const reason = `hand analysis unavailable: ${error instanceof Error ? error.message : String(error)}`;
      resetHandGestureEstimate(reason);
      resetFaceTouchEstimate(false, reason);
    }
  } else if (task === "pose" && poseLandmarker) {
    lastPoseInferenceAt = timestampMs;
    nextAuxiliaryModel = "hand";
    try {
      updatePostureEstimate(poseLandmarker.detectForVideo(video, timestampMs), timestampMs);
    } catch (error) {
      resetPostureEstimate(
        `pose analysis unavailable: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
}

function setMovementDisplay(detected: boolean | null, intensityPercent = 0, hint?: string): void {
  ui.movement.classList.toggle("is-true", detected === true);
  ui.movement.classList.toggle("is-false", detected === false);
  ui.movement.textContent = detected === null ? "—" : detected ? "True" : "False";
  ui.movementBar.style.width = `${Math.max(0, Math.min(100, intensityPercent))}%`;
  ui.movementHint.textContent =
    hint ??
    `visual frame motion ${Math.round(intensityPercent)}% · background movement can also count`;
}

function clearMovementHistory(): void {
  previousMotionLuma = null;
  lastMotionSampleAt = 0;
  smoothedActivePixelRatio = null;
  motionWarmupSamples = 0;
  movementDetected = false;
  movementEnterFrames = 0;
  movementExitFrames = 0;
}

function resetMovementEstimate(message = "sustained visual motion in the webcam frame"): void {
  clearMovementHistory();
  setMovementDisplay(null, 0, message);
}

function unavailableMovementDiagnostics(reason: string): MovementDiagnostics {
  return {
    available: false,
    unavailableReason: reason,
    detected: null,
    intensityPercent: null,
    activePixelRatio: null,
    smoothedActivePixelRatio: null,
    meanResidualLuma: null,
    compensatedBrightnessShift: null,
    thresholds: {
      enterFraction: MOTION_ENTER_FRACTION,
      exitFraction: MOTION_EXIT_FRACTION,
      activePixelDelta: MOTION_PIXEL_DELTA_THRESHOLD,
    },
    sampleSize: { width: MOTION_SAMPLE_WIDTH, height: MOTION_SAMPLE_HEIGHT },
  };
}

function updateMovementEstimate(hasFace: boolean, timestampMs: number): MovementDiagnostics {
  if (!hasFace) {
    resetMovementEstimate("face unavailable; movement estimate paused");
    return unavailableMovementDiagnostics("face unavailable");
  }
  if (
    lastMotionSampleAt &&
    timestampMs - lastMotionSampleAt > MOTION_HISTORY_GAP_MS
  ) {
    clearMovementHistory();
  }

  try {
    motionDrawingContext.drawImage(video, 0, 0, MOTION_SAMPLE_WIDTH, MOTION_SAMPLE_HEIGHT);
    const rgba = motionDrawingContext.getImageData(
      0,
      0,
      MOTION_SAMPLE_WIDTH,
      MOTION_SAMPLE_HEIGHT,
    ).data;
    const currentLuma = new Uint8Array(MOTION_SAMPLE_WIDTH * MOTION_SAMPLE_HEIGHT);
    for (let pixel = 0, channel = 0; pixel < currentLuma.length; pixel += 1, channel += 4) {
      currentLuma[pixel] =
        (54 * rgba[channel] + 183 * rgba[channel + 1] + 19 * rgba[channel + 2]) >> 8;
    }

    if (!previousMotionLuma || !lastMotionSampleAt) {
      previousMotionLuma = currentLuma;
      lastMotionSampleAt = timestampMs;
      setMovementDisplay(null, 0, "calibrating visual movement");
      return unavailableMovementDiagnostics("calibrating");
    }

    const elapsedSeconds = (timestampMs - lastMotionSampleAt) / 1000;
    const borderX = Math.round(MOTION_SAMPLE_WIDTH * 0.05);
    const borderY = Math.round(MOTION_SAMPLE_HEIGHT * 0.05);
    let signedDifferenceTotal = 0;
    let sampledPixels = 0;
    for (let y = borderY; y < MOTION_SAMPLE_HEIGHT - borderY; y += 1) {
      for (let x = borderX; x < MOTION_SAMPLE_WIDTH - borderX; x += 1) {
        const index = y * MOTION_SAMPLE_WIDTH + x;
        signedDifferenceTotal += currentLuma[index] - previousMotionLuma[index];
        sampledPixels += 1;
      }
    }
    const brightnessShift = sampledPixels ? signedDifferenceTotal / sampledPixels : 0;
    let residualTotal = 0;
    let activePixels = 0;
    for (let y = borderY; y < MOTION_SAMPLE_HEIGHT - borderY; y += 1) {
      for (let x = borderX; x < MOTION_SAMPLE_WIDTH - borderX; x += 1) {
        const index = y * MOTION_SAMPLE_WIDTH + x;
        const residual = Math.abs(
          currentLuma[index] - previousMotionLuma[index] - brightnessShift,
        );
        residualTotal += Math.min(255, residual);
        if (residual >= MOTION_PIXEL_DELTA_THRESHOLD) activePixels += 1;
      }
    }

    const activePixelRatio = sampledPixels ? activePixels / sampledPixels : 0;
    const meanResidualLuma = sampledPixels ? residualTotal / sampledPixels : 0;
    const alpha =
      elapsedSeconds > 0
        ? 1 - Math.exp(-elapsedSeconds / MOTION_TIME_CONSTANT_SECONDS)
        : 1;
    smoothedActivePixelRatio =
      smoothedActivePixelRatio === null
        ? activePixelRatio
        : smoothedActivePixelRatio + alpha * (activePixelRatio - smoothedActivePixelRatio);
    const intensityPercent =
      100 * clamp01((smoothedActivePixelRatio - 0.005) / (0.12 - 0.005));

    previousMotionLuma = currentLuma;
    lastMotionSampleAt = timestampMs;
    motionWarmupSamples += 1;
    if (motionWarmupSamples < MOTION_WARMUP_SAMPLES) {
      setMovementDisplay(
        null,
        intensityPercent,
        `calibrating visual movement · ${motionWarmupSamples}/${MOTION_WARMUP_SAMPLES}`,
      );
      return {
        available: false,
        unavailableReason: "calibrating",
        detected: null,
        intensityPercent: rounded(intensityPercent, 1),
        activePixelRatio: rounded(activePixelRatio),
        smoothedActivePixelRatio: rounded(smoothedActivePixelRatio),
        meanResidualLuma: rounded(meanResidualLuma, 1),
        compensatedBrightnessShift: rounded(brightnessShift, 1),
        thresholds: {
          enterFraction: MOTION_ENTER_FRACTION,
          exitFraction: MOTION_EXIT_FRACTION,
          activePixelDelta: MOTION_PIXEL_DELTA_THRESHOLD,
        },
        sampleSize: { width: MOTION_SAMPLE_WIDTH, height: MOTION_SAMPLE_HEIGHT },
      };
    }

    if (!movementDetected) {
      movementEnterFrames =
        smoothedActivePixelRatio >= MOTION_ENTER_FRACTION ? movementEnterFrames + 1 : 0;
      movementExitFrames = 0;
      if (movementEnterFrames >= MOTION_ENTER_FRAMES) movementDetected = true;
    } else {
      movementExitFrames =
        smoothedActivePixelRatio <= MOTION_EXIT_FRACTION ? movementExitFrames + 1 : 0;
      movementEnterFrames = 0;
      if (movementExitFrames >= MOTION_EXIT_FRAMES) movementDetected = false;
    }

    setMovementDisplay(movementDetected, intensityPercent);

    return {
      available: true,
      unavailableReason: null,
      detected: movementDetected,
      intensityPercent: rounded(intensityPercent, 1),
      activePixelRatio: rounded(activePixelRatio),
      smoothedActivePixelRatio: rounded(smoothedActivePixelRatio),
      meanResidualLuma: rounded(meanResidualLuma, 1),
      compensatedBrightnessShift: rounded(brightnessShift, 1),
      thresholds: {
        enterFraction: MOTION_ENTER_FRACTION,
        exitFraction: MOTION_EXIT_FRACTION,
        activePixelDelta: MOTION_PIXEL_DELTA_THRESHOLD,
      },
      sampleSize: { width: MOTION_SAMPLE_WIDTH, height: MOTION_SAMPLE_HEIGHT },
    };
  } catch {
    resetMovementEstimate("movement sampling unavailable");
    return unavailableMovementDiagnostics("frame sampling unavailable");
  }
}

function estimateCameraGaze(
  blendshapes: Record<string, number>,
  matrix?: FacialTransformationMatrix,
): CameraGazeEstimate | null {
  const eyeDirectionNames = [
    "eyeLookOutLeft",
    "eyeLookInRight",
    "eyeLookInLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
  ] as const;
  if (!eyeDirectionNames.every((name) => Number.isFinite(blendshapes[name]))) return null;
  if (!matrix || matrix.rows !== 4 || matrix.columns !== 4 || matrix.data.length !== 16) return null;

  const lookLeft = (blendshapes.eyeLookOutLeft + blendshapes.eyeLookInRight) / 2;
  const lookRight = (blendshapes.eyeLookInLeft + blendshapes.eyeLookOutRight) / 2;
  const lookUp = (blendshapes.eyeLookUpLeft + blendshapes.eyeLookUpRight) / 2;
  const lookDown = (blendshapes.eyeLookDownLeft + blendshapes.eyeLookDownRight) / 2;
  const eyeDirectionScore = Math.max(lookLeft, lookRight, lookUp, lookDown);

  // MediaPipe returns a column-major 4x4 transform. Its normalized third
  // column is the face's forward direction; scale is removed first.
  const [forwardXRaw, forwardYRaw, forwardZRaw] = [matrix.data[8], matrix.data[9], matrix.data[10]];
  const scale = Math.hypot(forwardXRaw, forwardYRaw, forwardZRaw);
  if (![eyeDirectionScore, scale].every(Number.isFinite) || scale < Number.EPSILON) return null;

  const forwardX = forwardXRaw / scale;
  const forwardY = forwardYRaw / scale;
  const forwardZ = forwardZRaw / scale;
  const radiansToDegrees = 180 / Math.PI;

  return {
    eyeDirectionScore,
    headYawDegrees: Math.atan2(forwardX, forwardZ) * radiansToDegrees,
    headPitchDegrees: Math.atan2(-forwardY, Math.hypot(forwardX, forwardZ)) * radiansToDegrees,
  };
}

function updateEyeContactEstimate(estimate: CameraGazeEstimate | null, blinkSignal: number): boolean {
  if (!estimate) {
    resetEyeContactEstimate();
    return false;
  }

  if (blinkSignal >= 0.55) {
    closedEyeFrames += 1;
    eyeContactPassFrames = 0;
    if (closedEyeFrames >= BLINK_HOLD_FRAMES) setEyeContact(false);
    return eyeContact;
  }
  closedEyeFrames = 0;

  smoothedCameraGaze = smoothedCameraGaze
    ? {
        eyeDirectionScore:
          smoothedCameraGaze.eyeDirectionScore * (1 - GAZE_SMOOTHING_ALPHA) +
          estimate.eyeDirectionScore * GAZE_SMOOTHING_ALPHA,
        headYawDegrees:
          smoothedCameraGaze.headYawDegrees * (1 - GAZE_SMOOTHING_ALPHA) +
          estimate.headYawDegrees * GAZE_SMOOTHING_ALPHA,
        headPitchDegrees:
          smoothedCameraGaze.headPitchDegrees * (1 - GAZE_SMOOTHING_ALPHA) +
          estimate.headPitchDegrees * GAZE_SMOOTHING_ALPHA,
      }
    : estimate;

  const gaze = smoothedCameraGaze;
  const meetsEntryThreshold =
    gaze.eyeDirectionScore < 0.35 &&
    Math.abs(gaze.headYawDegrees) < 18 &&
    Math.abs(gaze.headPitchDegrees) < 15;
  const remainsWithinExitThreshold =
    gaze.eyeDirectionScore < 0.45 &&
    Math.abs(gaze.headYawDegrees) < 22 &&
    Math.abs(gaze.headPitchDegrees) < 18;

  if (!eyeContact) {
    eyeContactPassFrames = meetsEntryThreshold ? eyeContactPassFrames + 1 : 0;
    eyeContactFailFrames = 0;
    if (eyeContactPassFrames >= EYE_CONTACT_ENTER_FRAMES) setEyeContact(true);
  } else {
    eyeContactFailFrames = remainsWithinExitThreshold ? 0 : eyeContactFailFrames + 1;
    eyeContactPassFrames = 0;
    if (eyeContactFailFrames >= EYE_CONTACT_EXIT_FRAMES) setEyeContact(false);
  }

  return eyeContact;
}

function updateConfidenceProxy(
  estimate: CameraGazeEstimate | null,
  smileSignal: number | null,
  centerFramingScore: number,
  bounds: FaceBounds,
  timestampMs: number,
): SimulatedConfidenceDiagnostics | null {
  if (!estimate || !smoothedCameraGaze || smileSignal === null) {
    resetConfidenceProxy("required face cues unavailable; estimate paused");
    return null;
  }

  if (lastConfidenceSampleAt && timestampMs - lastConfidenceSampleAt > CONFIDENCE_HISTORY_GAP_MS) {
    clearConfidenceHistory();
  }

  confidenceValidSamples += 1;
  const sampleIntervalSeconds = lastConfidenceSampleAt
    ? (timestampMs - lastConfidenceSampleAt) / 1000
    : null;
  if (smoothedSmileSignal === null) {
    smoothedSmileSignal = smileSignal;
  } else if (
    sampleIntervalSeconds !== null &&
    sampleIntervalSeconds > 0 &&
    sampleIntervalSeconds <= CONFIDENCE_HISTORY_GAP_MS / 1000
  ) {
    const smileAlpha = 1 - Math.exp(-sampleIntervalSeconds / SMILE_TIME_CONSTANT_SECONDS);
    smoothedSmileSignal += smileAlpha * (smileSignal - smoothedSmileSignal);
  }
  lastConfidenceSampleAt = timestampMs;

  const gaze = smoothedCameraGaze;
  const eyeCentered = 1 - smoothstep(0.18, 0.5, gaze.eyeDirectionScore);
  const yawForward = 1 - smoothstep(10, 30, Math.abs(gaze.headYawDegrees));
  const pitchForward = 1 - smoothstep(8, 25, Math.abs(gaze.headPitchDegrees));
  const cameraFacing = eyeCentered ** 0.5 * yawForward ** 0.3 * pitchForward ** 0.2;

  // Framing describes how reliable the measurement is, not the person's confidence.
  const frameCentering = clamp01(centerFramingScore / 100);
  const faceSize =
    smoothstep(0.12, 0.22, bounds.width) * (1 - smoothstep(0.62, 0.84, bounds.width));
  const edgeMargin = smoothstep(0.005, 0.05, bounds.edgeMargin);
  const framing = frameCentering ** 0.55 * faceSize ** 0.3 * edgeMargin ** 0.15;
  const signalQualityLevel = framing >= 0.75 ? "good" : framing >= 0.45 ? "fair" : "low";

  const smileEvidence = smoothstep(0.08, 0.5, smoothedSmileSignal);
  // A neutral expression starts at 0.5 so "not smiling" is never equated with
  // "no confidence". A visible smile can add at most 17.5 points overall.
  const positiveExpression = 0.5 + 0.5 * smileEvidence;

  const calibrated = confidenceValidSamples >= CONFIDENCE_MIN_VALID_SAMPLES;
  const rawScore = 100 * (0.65 * cameraFacing + 0.35 * positiveExpression);

  if (!calibrated) {
    setConfidenceCalibrating();
  } else {
    if (displayedConfidenceScore === null || lastConfidenceScoreAt === 0) {
      displayedConfidenceScore = rawScore;
    } else {
      const elapsedSeconds = Math.max(0, (timestampMs - lastConfidenceScoreAt) / 1000);
      const alpha = 1 - Math.exp(-elapsedSeconds / CONFIDENCE_SCORE_TIME_CONSTANT_SECONDS);
      displayedConfidenceScore += alpha * (rawScore - displayedConfidenceScore);
    }
    lastConfidenceScoreAt = timestampMs;
    setMeter(ui.confidence, ui.confidenceBar, displayedConfidenceScore);
    ui.confidenceHint.textContent = `appearance-based simulation · ${signalQualityLevel} signal quality`;
  }

  return {
    available: calibrated,
    unavailableReason: calibrated ? null : "calibrating",
    displayedScore: calibrated && displayedConfidenceScore !== null ? rounded(displayedConfidenceScore, 1) : null,
    rawScore: rounded(rawScore, 1),
    components: {
      cameraFacing: rounded(cameraFacing),
      eyeCentered: rounded(eyeCentered),
      yawForward: rounded(yawForward),
      pitchForward: rounded(pitchForward),
      positiveExpression: rounded(positiveExpression),
      smileEvidence: rounded(smileEvidence),
    },
    inputs: {
      eyeDirectionScore: rounded(gaze.eyeDirectionScore),
      headYawDegrees: rounded(gaze.headYawDegrees, 1),
      headPitchDegrees: rounded(gaze.headPitchDegrees, 1),
      faceWidthNormalized: rounded(bounds.width),
      faceHeightNormalized: rounded(bounds.height),
      edgeMarginNormalized: rounded(bounds.edgeMargin),
      rawSmileSignal: rounded(smileSignal),
      smoothedSmileSignal: rounded(smoothedSmileSignal),
    },
    weights: { cameraEngagement: 0.65, positiveExpression: 0.35 },
    signalQuality: {
      level: signalQualityLevel,
      framing: rounded(framing),
      frameCentering: rounded(frameCentering),
      faceSize: rounded(faceSize),
      edgeMargin: rounded(edgeMargin),
    },
    excludedSignals: [
      "blink frequency",
      "facial geometry",
      "smile symmetry",
      "movement speed or gesture amount",
      "demographic characteristics",
    ],
  };
}

function setMeter(value: HTMLElement, bar: HTMLElement, score: number): void {
  const bounded = Math.max(0, Math.min(100, score));
  value.textContent = `${Math.round(bounded)}%`;
  (bar as HTMLSpanElement).style.width = `${bounded}%`;
}

function resizeCanvas(): void {
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

function drawLandmarks(points?: ReturnType<FaceLandmarker["detectForVideo"]>["faceLandmarks"][number]): void {
  resizeCanvas();
  drawingContext.clearRect(0, 0, canvas.width, canvas.height);
  if (!points) return;

  drawingContext.fillStyle = "rgba(111, 245, 195, .82)";
  for (let index = 0; index < points.length; index += 3) {
    const point = points[index];
    drawingContext.beginPath();
    drawingContext.arc(
      point.x * canvas.width,
      point.y * canvas.height,
      Math.max(1, canvas.width / 520),
      0,
      Math.PI * 2,
    );
    drawingContext.fill();
  }
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function updateMetrics(
  result: ReturnType<FaceLandmarker["detectForVideo"]>,
  inferenceMs: number,
  timestampMs: number,
): void {
  processedFrames += 1;
  const points = result.faceLandmarks[0];
  const hasFace = Boolean(points?.length);
  const bounds = hasFace && points ? calculateFaceBounds(points) : null;
  const movementEstimate = updateMovementEstimate(hasFace, timestampMs);
  updateAuxiliaryMetrics(bounds, timestampMs);

  if (hasFace) detectedFrames += 1;
  ui.detected.textContent = hasFace ? "Yes" : "No";
  ui.latency.textContent = `${inferenceMs.toFixed(0)} ms`;
  setMeter(ui.tracking, ui.trackingBar, processedFrames ? (100 * detectedFrames) / processedFrames : 0);

  if (!hasFace || !points) {
    drawLandmarks();
    setMeter(ui.framing, ui.framingBar, 0);
    ui.position.textContent = "—";
    ui.eyes.textContent = "—";
    ui.smile.textContent = "—";
    resetEyeContactEstimate();
    resetConfidenceProxy("face unavailable; estimate paused");
    ui.raw.textContent = JSON.stringify(
      {
        timestampMs: Math.round(timestampMs),
        faceDetected: false,
        blinkEventCount: blinkCount,
        movementEstimate,
        possibleFaceTouchEvents: latestFaceTouchDiagnostics,
        handGestureAnalysis: latestHandGestureDiagnostics,
        postureLeanEstimate: latestPostureDiagnostics,
        simulatedConfidenceEstimate: { available: false, unavailableReason: "face unavailable" },
        modelStatus: {
          hand: { loaded: Boolean(handLandmarker), delegate: handModelDelegate, error: handModelError },
          pose: { loaded: Boolean(poseLandmarker), delegate: poseModelDelegate, error: poseModelError },
        },
      },
      null,
      2,
    );
    return;
  }

  drawLandmarks(points);
  if (!bounds) return;
  const { centerX, centerY } = bounds;
  // Demo-only heuristic: this measures how close the face center is to the frame center.
  const dx = (centerX - 0.5) / 0.5;
  const dy = (centerY - 0.5) / 0.5;
  const framingScore = 100 * Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy) / 1.15);
  setMeter(ui.framing, ui.framingBar, framingScore);
  ui.position.textContent = `${Math.round(centerX * 100)}%, ${Math.round(centerY * 100)}%`;

  const blendshapes = blendshapeMap(result);
  const blinkSignal = ((blendshapes.eyeBlinkLeft ?? 0) + (blendshapes.eyeBlinkRight ?? 0)) / 2;
  const smileSignal =
    Number.isFinite(blendshapes.mouthSmileLeft) && Number.isFinite(blendshapes.mouthSmileRight)
      ? (blendshapes.mouthSmileLeft + blendshapes.mouthSmileRight) / 2
      : null;
  const cameraGaze = estimateCameraGaze(blendshapes, result.facialTransformationMatrixes[0]);
  const hasEyeContact = updateEyeContactEstimate(cameraGaze, blinkSignal);
  const simulatedConfidenceEstimate = updateConfidenceProxy(
    cameraGaze,
    smileSignal,
    framingScore,
    bounds,
    timestampMs,
  );
  if (blinkSignal > 0.62 && !wasBlinking) blinkCount += 1;
  wasBlinking = blinkSignal > 0.42;

  ui.blinks.textContent = String(blinkCount);
  ui.eyes.textContent = percent(1 - blinkSignal);
  ui.smile.textContent = smileSignal === null ? "—" : percent(smileSignal);
  ui.raw.textContent = JSON.stringify(
    {
      timestampMs: Math.round(timestampMs),
      faceDetected: true,
      faceLandmarkCount: points.length,
      blinkEventCount: blinkCount,
      faceCenterNormalized: { x: Number(centerX.toFixed(3)), y: Number(centerY.toFixed(3)) },
      demoFramingHeuristic: Math.round(framingScore),
      cameraGazeHeuristic: {
        eyeContact: hasEyeContact,
        eyeDirectionScore: cameraGaze ? Number(cameraGaze.eyeDirectionScore.toFixed(3)) : null,
        smoothedEyeDirectionScore: smoothedCameraGaze
          ? Number(smoothedCameraGaze.eyeDirectionScore.toFixed(3))
          : null,
        headYawDegrees: cameraGaze ? Number(cameraGaze.headYawDegrees.toFixed(1)) : null,
        headPitchDegrees: cameraGaze ? Number(cameraGaze.headPitchDegrees.toFixed(1)) : null,
      },
      movementEstimate,
      possibleFaceTouchEvents: latestFaceTouchDiagnostics,
      handGestureAnalysis: latestHandGestureDiagnostics,
      postureLeanEstimate: latestPostureDiagnostics,
      simulatedConfidenceEstimate:
        simulatedConfidenceEstimate ?? {
          available: false,
          unavailableReason: "required face cues unavailable",
        },
      blendshapes,
      facialTransformationMatrix: result.facialTransformationMatrixes[0]?.data ?? null,
      modelStatus: {
        hand: { loaded: Boolean(handLandmarker), delegate: handModelDelegate, error: handModelError },
        pose: { loaded: Boolean(poseLandmarker), delegate: poseModelDelegate, error: poseModelError },
      },
    },
    null,
    2,
  );
}

function renderLoop(now: number): void {
  if (!stream || !landmarker) return;
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && now - lastInferenceAt >= SAMPLE_INTERVAL_MS) {
    lastInferenceAt = now;
    const startedAt = performance.now();
    try {
      const result = landmarker.detectForVideo(video, now);
      updateMetrics(result, performance.now() - startedAt, now);
    } catch (error) {
      setStatus(`Detection error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  animationId = requestAnimationFrame(renderLoop);
}

async function createFaceLandmarker(vision: VisionFileset): Promise<FaceLandmarker> {
  const options = {
    runningMode: "VIDEO" as const,
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  };

  try {
    return await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    });
  } catch {
    // Fall back for browsers or devices without a working WebGL delegate.
    return FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
    });
  }
}

async function createHandLandmarker(
  vision: VisionFileset,
): Promise<{ task: HandLandmarker; delegate: "GPU" | "CPU" }> {
  const options = {
    runningMode: "VIDEO" as const,
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.5,
  };
  try {
    return {
      task: await HandLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
      }),
      delegate: "GPU",
    };
  } catch {
    return {
      task: await HandLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "CPU" },
      }),
      delegate: "CPU",
    };
  }
}

async function createPoseLandmarker(
  vision: VisionFileset,
): Promise<{ task: PoseLandmarker; delegate: "GPU" | "CPU" }> {
  const options = {
    runningMode: "VIDEO" as const,
    numPoses: 1,
    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.55,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  };
  try {
    return {
      task: await PoseLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
      }),
      delegate: "GPU",
    };
  } catch {
    return {
      task: await PoseLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "CPU" },
      }),
      delegate: "CPU",
    };
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function initializeLandmarkers(): Promise<void> {
  if (landmarker && auxiliaryModelsAttempted) return;
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  landmarker ??= await createFaceLandmarker(vision);
  if (auxiliaryModelsAttempted) return;
  auxiliaryModelsAttempted = true;

  const [handResult, poseResult] = await Promise.allSettled([
    createHandLandmarker(vision),
    createPoseLandmarker(vision),
  ]);
  if (handResult.status === "fulfilled") {
    handLandmarker = handResult.value.task;
    handModelDelegate = handResult.value.delegate;
    handModelError = null;
  } else {
    handModelError = errorMessage(handResult.reason);
  }
  if (poseResult.status === "fulfilled") {
    poseLandmarker = poseResult.value.task;
    poseModelDelegate = poseResult.value.delegate;
    poseModelError = null;
  } else {
    poseModelError = errorMessage(poseResult.reason);
  }
}

async function startCamera(): Promise<void> {
  ui.start.disabled = true;
  setStatus("Loading face, hand, and pose models…");
  try {
    await initializeLandmarkers();
    setAuxiliaryModelsUnavailable();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    video.hidden = false;
    canvas.hidden = false;
    ui.placeholder.hidden = true;
    await video.play();
    ui.stop.disabled = false;
    const unavailableModels = [
      !handLandmarker ? "hands" : null,
      !poseLandmarker ? "posture" : null,
    ].filter(Boolean);
    setStatus(
      unavailableModels.length
        ? `Running · ${unavailableModels.join(" + ")} unavailable`
        : "Running · face 10 Hz · hands/posture 5 Hz",
    );
    animationId = requestAnimationFrame(renderLoop);
  } catch (error) {
    setStatus(`Could not start: ${errorMessage(error)}`);
    ui.start.disabled = false;
  }
}

function stopCamera(): void {
  cancelAnimationFrame(animationId);
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  video.hidden = true;
  canvas.hidden = true;
  ui.placeholder.hidden = false;
  ui.start.disabled = false;
  ui.stop.disabled = true;
  resetEyeContactEstimate();
  resetConfidenceProxy();
  resetMovementEstimate();
  lastHandInferenceAt = 0;
  lastPoseInferenceAt = 0;
  nextAuxiliaryModel = "hand";
  resetHandGestureEstimate("camera stopped");
  resetFaceTouchEstimate(false, "camera stopped · session count preserved");
  resetPostureEstimate("camera stopped · baseline will recalibrate next time");
  setStatus("Stopped");
}

function resetSession(): void {
  processedFrames = 0;
  detectedFrames = 0;
  blinkCount = 0;
  wasBlinking = false;
  ui.tracking.textContent = "—";
  ui.framing.textContent = "—";
  ui.trackingBar.style.width = "0%";
  ui.framingBar.style.width = "0%";
  ui.blinks.textContent = "0";
  ui.smile.textContent = "—";
  resetEyeContactEstimate();
  resetConfidenceProxy();
  resetMovementEstimate();
  lastHandInferenceAt = 0;
  lastPoseInferenceAt = 0;
  nextAuxiliaryModel = "hand";
  resetHandGestureEstimate();
  resetFaceTouchEstimate(true);
  resetPostureEstimate();
  ui.detected.textContent = "No";
  ui.position.textContent = "—";
  ui.eyes.textContent = "—";
  ui.latency.textContent = "—";
}

ui.start.addEventListener("click", () => void startCamera());
ui.stop.addEventListener("click", stopCamera);
ui.reset.addEventListener("click", resetSession);
