# Interview Video Signals Prototype

A small local webcam demo using MediaPipe Face, Hand, and Pose Landmarker. It displays a face-landmark overlay, face tracking coverage, a simple face-centering heuristic, blink events, a raw smile blendshape signal, a camera-facing eye-contact estimate, possible face-touch events, hand gesture activity, posture/leaning, a simulated confidence estimate, inference time, and the latest raw output.

## Run in VS Code

Requirements: Node.js and npm.

```bash
npm install
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`) in Chrome or another modern browser, then allow camera access.

## What the numbers mean

- **Face tracking**: the percentage of sampled frames where MediaPipe returned face landmarks.
- **Camera framing**: a demo heuristic based only on how close the detected face center is to the image center.
- **Blink events / eye openness**: rough signals derived from eye-blink blendshape scores.
- **Smile signal**: average of the left/right mouth-smile blendshape scores.
- **Eye contact estimate**: a rough camera-gaze heuristic combining eye-direction blendshapes with head orientation. It is not a measure of attention, honesty, or engagement.
- **Movement detected**: sustained visual motion measured by comparing small consecutive webcam frames. It is smoothed and uses separate on/off thresholds to reduce flicker. It can see hand and body movement in frame, but camera shake, moving backgrounds, and large lighting changes may also count.
- **Possible face touches**: counts debounced 2D overlap events between tracked hand points and a face-shaped region. A hand moving close to the face can count even without physical contact, so this is deliberately labeled “possible.”
- **Hand gesture activity**: reports `Still`, `Low`, `Moderate`, or `High` from smoothed palm/finger movement relative to the face. `Not visible` means the model could not see a hand; it does not mean the hands were still. This measures movement amount, not gesture quality.
- **Posture / leaning**: first observes roughly three seconds of the user's natural seated posture, then reports lateral and toward/away changes relative to that personal baseline. It uses shoulders and hips when visible and clearly falls back to a shoulder-only estimate when needed. It is not an ergonomic assessment.
- **Simulated confidence estimate**: an appearance-based practice heuristic made from 65% camera-facing gaze/head pose and 35% positive expression from a smoothed smile signal. A neutral expression receives half of the positive-expression component, so a person does not need to smile to score well. Movement speed, gesture amount, framing, and blink frequency do not affect the estimate; framing is reported separately as signal quality. The simulation calibrates briefly and reports unavailable when required face data is missing. Its component scores and inputs are shown under **Inspect latest raw output**.

These are observable video cues. The simulated confidence estimate approximates how confidence might appear on camera; it cannot measure a person's internal state, competence, or interview quality and should not be used to make hiring decisions. Camera placement, lighting, glasses, anatomy, disability, culture, and natural communication style can all affect the signals. The three new descriptive signals do not change the confidence score. The demo processes camera frames in the browser and does not send them to a project backend. The app downloads MediaPipe's package, WebAssembly files, and face/hand/pose models from public hosts.
