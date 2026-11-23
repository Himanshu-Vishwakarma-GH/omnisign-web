/**
 * Procedural motion generator in LexiSight's 225-dim Holistic feature space.
 *
 * Feature layout per frame (225 floats, MediaPipe Holistic order):
 *   0..98    pose landmarks 0..32  (x, y, z)
 *   99..161  left hand landmarks 0..20 (x, y, z)
 *   162..224 right hand landmarks 0..20 (x, y, z)
 * Coordinates are MediaPipe normalized image coords (x, y in [0,1], z
 * relative depth). Missing landmarks are zeros.
 *
 * This module is shared by:
 *   - the browser app (demo mode + lexicon keyframes), and
 *   - scripts/train-tfjs.mjs (synthetic training data for the bundled
 *     fallback TF.js model).
 *
 * IMPORTANT: these motions are AUTHOR-PROCEDURAL, not motion-captured ISL.
 * Each word is a distinct hand trajectory + hand shape, so the pipeline
 * (extract -> 30-frame window -> classifier -> speech) is real and the
 * format matches LexiSight's (30 x 225), but the signs themselves are
 * stylised. Real ISL accuracy requires the LexiSight weights (see README).
 */

export const FRAMES = 30
export const POSE_LANDMARKS = 33
export const HAND_LANDMARKS = 21
export const FEATURE_DIM = 225 // 75 landmarks x 3

// ---------------------------------------------------------------------------
// Seeded RNG (mulberry32) — deterministic sequences per (word, seed).
// ---------------------------------------------------------------------------
export function mulberry32(seed) {
  let t = seed >>> 0
  return function () {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(rng) {
  // Box-Muller
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

// ---------------------------------------------------------------------------
// Rest pose (MediaPipe Holistic 33 pose landmarks, normalized coords).
// Person facing the camera, arms relaxed at the sides.
// ---------------------------------------------------------------------------
const REST_POSE = [
  [0.5, 0.155, 0], // 0 nose
  [0.48, 0.14, 0], // 1 left_eye_inner
  [0.47, 0.14, 0], // 2 left_eye
  [0.46, 0.14, 0], // 3 left_eye_outer
  [0.52, 0.14, 0], // 4 right_eye_inner
  [0.53, 0.14, 0], // 5 right_eye
  [0.54, 0.14, 0], // 6 right_eye_outer
  [0.445, 0.155, 0], // 7 left_ear
  [0.555, 0.155, 0], // 8 right_ear
  [0.48, 0.185, 0], // 9 mouth_left
  [0.52, 0.185, 0], // 10 mouth_right
  [0.375, 0.3, 0], // 11 left_shoulder
  [0.625, 0.3, 0], // 12 right_shoulder
  [0.33, 0.46, 0], // 13 left_elbow
  [0.67, 0.46, 0], // 14 right_elbow
  [0.3, 0.61, -0.05], // 15 left_wrist
  [0.7, 0.61, -0.05], // 16 right_wrist
  [0.285, 0.6, -0.05], // 17 left_pinky
  [0.715, 0.6, -0.05], // 18 right_pinky
  [0.315, 0.595, -0.05], // 19 left_index
  [0.685, 0.595, -0.05], // 20 right_index
  [0.32, 0.615, -0.05], // 21 left_thumb
  [0.68, 0.615, -0.05], // 22 right_thumb
  [0.43, 0.64, 0], // 23 left_hip
  [0.57, 0.64, 0], // 24 right_hip
  [0.425, 0.8, 0], // 25 left_knee
  [0.575, 0.8, 0], // 26 right_knee
  [0.42, 0.95, 0], // 27 left_ankle
  [0.58, 0.95, 0], // 28 right_ankle
  [0.415, 0.965, 0], // 29 left_heel
  [0.585, 0.965, 0], // 30 right_heel
  [0.41, 0.98, 0], // 31 left_foot_index
  [0.59, 0.98, 0], // 32 right_foot_index
]

const REST_WRIST_L = [0.3, 0.61, -0.05]
const REST_WRIST_R = [0.7, 0.61, -0.05]

// ---------------------------------------------------------------------------
// Hand kinematics: 21 landmarks from wrist + per-finger curls.
// curls: [thumb, index, middle, ring, pinky], 0 = extended, 1 = fully curled.
// `side`: -1 for the right hand (thumb toward image-left), +1 for left hand.
// Fingers point "up" (-y) when extended; curl rotates them toward the palm.
// ---------------------------------------------------------------------------
const FINGER_LENGTHS = {
  index: [0.026, 0.019, 0.016],
  middle: [0.028, 0.021, 0.017],
  ring: [0.027, 0.02, 0.016],
  pinky: [0.021, 0.016, 0.013],
}
const FINGER_X = { index: -0.026, middle: -0.009, ring: 0.009, pinky: 0.026 }
const BEND_SHARE = [0.45, 0.35, 0.2]

function fingerChain(base, baseAngle, curl, lengths) {
  const pts = [[base[0], base[1], base[2]]]
  let a = baseAngle
  let p = base
  for (let s = 0; s < 3; s++) {
    a += curl * 1.9 * BEND_SHARE[s]
    p = [
      p[0] + Math.cos(a) * lengths[s],
      p[1] + Math.sin(a) * lengths[s],
      p[2] - 0.004 * s,
    ]
    pts.push(p)
  }
  return pts // [MCP, PIP, DIP, TIP]
}

/** 21 hand landmarks [x, y, z] for a wrist position + curls. */
export function handLandmarks(wrist, curls, side, spread = 0) {
  const [wx, wy, wz] = wrist
  const out = Array.from({ length: 21 }, () => [wx, wy, wz])
  const up = -Math.PI / 2
  // Thumb: base near wrist, points outward-up; curl brings it across the palm.
  const thumbBase = [wx + side * -0.028, wy - 0.012, wz]
  out[1] = [wx + side * -0.02, wy - 0.006, wz]
  const thumbChain = fingerChain(
    thumbBase,
    up + side * 1.05,
    curls[0],
    [0.02, 0.015, 0.014],
  )
  out[2] = thumbChain[1]
  out[3] = thumbChain[2]
  out[4] = thumbChain[3]
  // Four fingers.
  const fingers = ['index', 'middle', 'ring', 'pinky']
  const baseIdx = [5, 9, 13, 17]
  fingers.forEach((name, fi) => {
    const mcp = [wx + side * FINGER_X[name], wy - 0.042, wz - 0.002]
    const angle = up + side * (fi - 1.5) * 0.1 * (1 + spread)
    const chain = fingerChain(mcp, angle, curls[fi + 1], FINGER_LENGTHS[name])
    out[baseIdx[fi]] = chain[0]
    out[baseIdx[fi] + 1] = chain[1]
    out[baseIdx[fi] + 2] = chain[2]
    out[baseIdx[fi] + 3] = chain[3]
  })
  out[0] = [wx, wy, wz]
  return out
}

// Named hand shapes (curls: thumb, index, middle, ring, pinky).
export const SHAPES = {
  OPEN: [0, 0, 0, 0, 0],
  FLAT: [0.1, 0.1, 0.1, 0.1, 0.1],
  FIST: [1, 1, 1, 1, 1],
  POINT: [0.9, 0, 1, 1, 1],
  PEACE: [1, 0, 0, 1, 1],
  CUP: [0.5, 0.5, 0.5, 0.5, 0.5],
  OK: [0.2, 0.2, 0.9, 0.9, 0.9],
  HOOK: [0.3, 0.7, 0.7, 0.7, 0.7],
  CLAW: [0.6, 0.4, 0.4, 0.4, 0.4],
  PINCH: [0.2, 0.3, 0.9, 0.9, 0.9],
}

// ---------------------------------------------------------------------------
// Vocabulary. Conversation-first; the 20 emergency words are a subset
// (see EMERGENCY_WORDS) and double as ordinary conversation words.
// ---------------------------------------------------------------------------
export const EMERGENCY_WORDS = [
  'HELP', 'DOCTOR', 'PAIN', 'WATER', 'YES', 'NO', 'FAMILY', 'EMERGENCY',
  'HOSPITAL', 'MEDICINE', 'STOP', 'WAIT', 'CALL', 'POLICE', 'FIRE', 'HURT',
  'BREATHE', 'ALLERGY', 'THANKYOU', 'PLEASE',
]

export const VOCAB_WORDS = [
  // Greetings & politeness
  'HELLO', 'HI', 'GOOD', 'MORNING', 'THANKYOU', 'PLEASE', 'SORRY', 'WELCOME', 'BYE',
  // Pronouns
  'I', 'YOU', 'HE', 'SHE', 'WE', 'THEY', 'MY', 'YOUR', 'ME',
  // Questions
  'WHAT', 'WHERE', 'WHEN', 'HOW', 'WHO', 'WHY',
  // Verbs
  'GO', 'COME', 'WANT', 'NEED', 'LIKE', 'LOVE', 'HELP', 'EAT', 'DRINK',
  'SLEEP', 'WORK', 'STUDY', 'PLAY', 'SEE', 'KNOW', 'UNDERSTAND', 'GIVE',
  'TAKE', 'SPEAK', 'LISTEN',
  // Nouns
  'NAME', 'FRIEND', 'FAMILY', 'MOTHER', 'FATHER', 'WATER', 'FOOD', 'HOUSE',
  'SCHOOL', 'DOCTOR', 'HOSPITAL', 'MEDICINE', 'MONEY', 'TIME', 'DAY',
  'TODAY', 'TOMORROW', 'NIGHT', 'CHILD', 'TEACHER', 'BOOK', 'PHONE',
  // Adjectives / states
  'BAD', 'HAPPY', 'SAD', 'BIG', 'SMALL', 'HOT', 'COLD', 'NEW', 'FINE',
  'BEAUTIFUL',
  // Yes / no
  'YES', 'NO',
  // Emergency-only additions
  'EMERGENCY', 'PAIN', 'STOP', 'WAIT', 'CALL', 'POLICE', 'FIRE', 'HURT',
  'BREATHE', 'ALLERGY',
]

// ---------------------------------------------------------------------------
// Per-word motion parameters.
//   hand: 'R' | 'L' | 'BOTH' — dominant hand(s)
//   shape / shape2: hand curls (start -> end morph for two-phase signs)
//   path: dominant-wrist waypoints in normalized coords (x, y)
//   path2: waypoints for the other hand (BOTH or two-handed signs)
//   repeat: how many times the path is traversed in 30 frames
//   head: 'nod' | 'shake' | null — head motion cue (YES/NO)
// ---------------------------------------------------------------------------
const P = (hand, shape, path, opts = {}) => ({ hand, shape, path, ...opts })

const CHEST = [0.5, 0.45]
const FACE = [0.5, 0.2]
const REST_R = [0.7, 0.61]
const REST_L = [0.3, 0.61]

export const WORD_PARAMS = {
  HELLO: P('R', SHAPES.OPEN, [[0.62, 0.35], [0.72, 0.3], [0.62, 0.35], [0.72, 0.3]], { repeat: 2 }),
  HI: P('R', SHAPES.OPEN, [[0.62, 0.35], [0.72, 0.3], [0.62, 0.35]], { repeat: 2 }),
  GOOD: P('R', SHAPES.FLAT, [[0.55, 0.3], [0.5, 0.45]]),
  MORNING: P('BOTH', SHAPES.CUP, [[0.35, 0.5], [0.5, 0.25]], { path2: [[0.65, 0.5], [0.5, 0.25]] }),
  THANKYOU: P('R', SHAPES.FLAT, [[0.5, 0.28], [0.5, 0.45]], { shape2: SHAPES.OPEN }),
  PLEASE: P('R', SHAPES.FLAT, [[0.5, 0.42], [0.44, 0.42], [0.56, 0.42], [0.5, 0.42]], { repeat: 2 }),
  SORRY: P('R', SHAPES.FIST, [[0.5, 0.42], [0.44, 0.44], [0.56, 0.44], [0.5, 0.42]], { repeat: 2 }),
  WELCOME: P('BOTH', SHAPES.OPEN, [[0.42, 0.45], [0.3, 0.4]], { path2: [[0.58, 0.45], [0.7, 0.4]] }),
  BYE: P('R', SHAPES.OPEN, [[0.68, 0.32], [0.76, 0.28], [0.68, 0.32], [0.76, 0.28]], { repeat: 2 }),
  I: P('R', SHAPES.POINT, [[0.5, 0.42]]),
  YOU: P('R', SHAPES.POINT, [[0.62, 0.45]]),
  HE: P('R', SHAPES.POINT, [[0.68, 0.4]]),
  SHE: P('R', SHAPES.POINT, [[0.68, 0.4]], { shape2: SHAPES.POINT }),
  WE: P('R', SHAPES.POINT, [[0.45, 0.45], [0.58, 0.45]]),
  THEY: P('R', SHAPES.POINT, [[0.68, 0.4], [0.6, 0.45]]),
  MY: P('R', SHAPES.FLAT, [[0.5, 0.42]]),
  YOUR: P('R', SHAPES.FLAT, [[0.58, 0.45]]),
  ME: P('R', SHAPES.POINT, [[0.5, 0.42]]),
  WHAT: P('BOTH', SHAPES.OPEN, [[0.42, 0.4], [0.42, 0.48]], { path2: [[0.58, 0.4], [0.58, 0.48]], repeat: 2 }),
  WHERE: P('R', SHAPES.POINT, [[0.5, 0.4], [0.42, 0.4], [0.58, 0.4]]),
  WHEN: P('R', SHAPES.POINT, [[0.5, 0.42], [0.5, 0.3]]),
  HOW: P('BOTH', SHAPES.CUP, [[0.45, 0.45], [0.5, 0.3]], { path2: [[0.55, 0.45], [0.5, 0.3]] }),
  WHO: P('R', SHAPES.POINT, [[0.55, 0.4], [0.5, 0.4], [0.55, 0.4]], { repeat: 2 }),
  WHY: P('R', SHAPES.POINT, [[0.5, 0.25], [0.5, 0.35]]),
  GO: P('R', SHAPES.POINT, [[0.5, 0.45], [0.7, 0.4]]),
  COME: P('R', SHAPES.CUP, [[0.7, 0.45], [0.5, 0.45]], { repeat: 2 }),
  WANT: P('BOTH', SHAPES.CLAW, [[0.42, 0.45], [0.45, 0.35]], { path2: [[0.58, 0.45], [0.55, 0.35]] }),
  NEED: P('R', SHAPES.HOOK, [[0.5, 0.35], [0.5, 0.5]]),
  LIKE: P('R', SHAPES.FLAT, [[0.5, 0.42], [0.55, 0.32]]),
  LOVE: P('BOTH', SHAPES.FIST, [[0.45, 0.42], [0.5, 0.4]], { path2: [[0.55, 0.42], [0.5, 0.4]], repeat: 2 }),
  HELP: P('R', SHAPES.FLAT, [[0.55, 0.5], [0.55, 0.32]]),
  EAT: P('R', SHAPES.PINCH, [[0.55, 0.45], [0.5, 0.22]], { repeat: 2 }),
  DRINK: P('R', SHAPES.CUP, [[0.55, 0.45], [0.5, 0.22]], { repeat: 2 }),
  SLEEP: P('BOTH', SHAPES.FLAT, [[0.55, 0.3], [0.6, 0.42]], { path2: [[0.45, 0.3], [0.4, 0.42]] }),
  WORK: P('BOTH', SHAPES.FIST, [[0.42, 0.45], [0.46, 0.4]], { path2: [[0.58, 0.45], [0.54, 0.4]], repeat: 2 }),
  STUDY: P('BOTH', SHAPES.FLAT, [[0.4, 0.5], [0.42, 0.44]], { path2: [[0.6, 0.5], [0.58, 0.44]], repeat: 2 }),
  PLAY: P('BOTH', SHAPES.PEACE, [[0.4, 0.4], [0.34, 0.32]], { path2: [[0.6, 0.4], [0.66, 0.32]], repeat: 2 }),
  SEE: P('R', SHAPES.PEACE, [[0.55, 0.2], [0.62, 0.18]]),
  KNOW: P('R', SHAPES.FLAT, [[0.52, 0.22], [0.58, 0.3]]),
  UNDERSTAND: P('R', SHAPES.POINT, [[0.5, 0.22], [0.5, 0.3]]),
  GIVE: P('BOTH', SHAPES.OPEN, [[0.42, 0.5], [0.4, 0.4]], { path2: [[0.58, 0.5], [0.6, 0.4]] }),
  TAKE: P('R', SHAPES.CLAW, [[0.62, 0.4], [0.5, 0.45]]),
  SPEAK: P('R', SHAPES.FLAT, [[0.55, 0.28], [0.6, 0.22]], { repeat: 2 }),
  LISTEN: P('R', SHAPES.CUP, [[0.62, 0.2], [0.58, 0.16]]),
  NAME: P('BOTH', SHAPES.POINT, [[0.45, 0.35], [0.55, 0.35]], { path2: [[0.55, 0.35], [0.45, 0.35]] }),
  FRIEND: P('BOTH', SHAPES.HOOK, [[0.42, 0.42], [0.48, 0.42]], { path2: [[0.58, 0.42], [0.52, 0.42]], repeat: 2 }),
  FAMILY: P('BOTH', SHAPES.OPEN, [[0.4, 0.4], [0.55, 0.32]], { path2: [[0.6, 0.4], [0.45, 0.32]] }),
  MOTHER: P('R', SHAPES.FLAT, [[0.58, 0.2], [0.58, 0.28]]),
  FATHER: P('R', SHAPES.FLAT, [[0.58, 0.16], [0.58, 0.24]]),
  WATER: P('R', SHAPES.CUP, [[0.55, 0.4], [0.52, 0.24]], { repeat: 2 }),
  FOOD: P('R', SHAPES.PINCH, [[0.55, 0.42], [0.5, 0.24]], { repeat: 2 }),
  HOUSE: P('BOTH', SHAPES.FLAT, [[0.42, 0.3], [0.4, 0.45]], { path2: [[0.58, 0.3], [0.6, 0.45]] }),
  SCHOOL: P('BOTH', SHAPES.FLAT, [[0.42, 0.42], [0.46, 0.42]], { path2: [[0.58, 0.42], [0.54, 0.42]], repeat: 2 }),
  DOCTOR: P('R', SHAPES.FLAT, [[0.58, 0.3], [0.5, 0.42]]),
  HOSPITAL: P('R', SHAPES.FLAT, [[0.5, 0.42], [0.62, 0.3]]),
  MEDICINE: P('R', SHAPES.PINCH, [[0.55, 0.45], [0.5, 0.28]]),
  MONEY: P('R', SHAPES.OK, [[0.55, 0.45], [0.55, 0.38]], { repeat: 2 }),
  TIME: P('R', SHAPES.POINT, [[0.5, 0.45], [0.58, 0.45]]),
  DAY: P('R', SHAPES.FLAT, [[0.5, 0.35], [0.5, 0.5]]),
  TODAY: P('R', SHAPES.POINT, [[0.55, 0.4], [0.5, 0.5]]),
  TOMORROW: P('R', SHAPES.POINT, [[0.5, 0.45], [0.6, 0.38]]),
  NIGHT: P('BOTH', SHAPES.FLAT, [[0.45, 0.35], [0.5, 0.5]], { path2: [[0.55, 0.35], [0.5, 0.5]] }),
  CHILD: P('R', SHAPES.FLAT, [[0.55, 0.35], [0.55, 0.5]]),
  TEACHER: P('BOTH', SHAPES.FLAT, [[0.42, 0.3], [0.45, 0.45]], { path2: [[0.58, 0.3], [0.55, 0.45]] }),
  BOOK: P('BOTH', SHAPES.FLAT, [[0.42, 0.45], [0.45, 0.38]], { path2: [[0.58, 0.45], [0.55, 0.38]] }),
  PHONE: P('R', SHAPES.POINT, [[0.6, 0.28], [0.56, 0.2]]),
  BAD: P('R', SHAPES.FLAT, [[0.55, 0.3], [0.5, 0.5]]),
  HAPPY: P('BOTH', SHAPES.FLAT, [[0.42, 0.4], [0.4, 0.32]], { path2: [[0.58, 0.4], [0.6, 0.32]], repeat: 2 }),
  SAD: P('BOTH', SHAPES.FLAT, [[0.45, 0.3], [0.45, 0.45]], { path2: [[0.55, 0.3], [0.55, 0.45]] }),
  BIG: P('BOTH', SHAPES.OPEN, [[0.45, 0.4], [0.32, 0.38]], { path2: [[0.55, 0.4], [0.68, 0.38]] }),
  SMALL: P('BOTH', SHAPES.PINCH, [[0.4, 0.4], [0.47, 0.4]], { path2: [[0.6, 0.4], [0.53, 0.4]] }),
  HOT: P('R', SHAPES.CLAW, [[0.55, 0.35], [0.5, 0.45]]),
  COLD: P('BOTH', SHAPES.FIST, [[0.42, 0.42], [0.44, 0.38]], { path2: [[0.58, 0.42], [0.56, 0.38]], repeat: 2 }),
  NEW: P('R', SHAPES.FLAT, [[0.5, 0.45], [0.55, 0.3]]),
  FINE: P('R', SHAPES.FLAT, [[0.55, 0.42], [0.6, 0.3]]),
  BEAUTIFUL: P('R', SHAPES.OPEN, [[0.55, 0.25], [0.5, 0.4]]),
  YES: P('R', SHAPES.FIST, [[0.55, 0.42], [0.55, 0.36]], { repeat: 2, head: 'nod' }),
  NO: P('R', SHAPES.POINT, [[0.55, 0.4], [0.48, 0.4], [0.62, 0.4]], { repeat: 2, head: 'shake' }),
  EMERGENCY: P('BOTH', SHAPES.FIST, [[0.42, 0.35], [0.42, 0.5]], { path2: [[0.58, 0.35], [0.58, 0.5]], repeat: 2 }),
  PAIN: P('R', SHAPES.POINT, [[0.5, 0.42], [0.45, 0.42]], { repeat: 2 }),
  STOP: P('R', SHAPES.FLAT, [[0.55, 0.3], [0.55, 0.45]]),
  WAIT: P('R', SHAPES.FLAT, [[0.55, 0.42], [0.55, 0.34]]),
  CALL: P('R', SHAPES.POINT, [[0.6, 0.3], [0.58, 0.2]]),
  POLICE: P('R', SHAPES.POINT, [[0.58, 0.28], [0.5, 0.42]]),
  FIRE: P('BOTH', SHAPES.CLAW, [[0.42, 0.45], [0.44, 0.32]], { path2: [[0.58, 0.45], [0.56, 0.32]], repeat: 2 }),
  HURT: P('R', SHAPES.POINT, [[0.5, 0.42], [0.46, 0.44]], { repeat: 2 }),
  BREATHE: P('BOTH', SHAPES.OPEN, [[0.42, 0.42], [0.34, 0.4]], { path2: [[0.58, 0.42], [0.66, 0.4]], repeat: 2 }),
  ALLERGY: P('R', SHAPES.POINT, [[0.52, 0.25], [0.5, 0.35]]),
}

// ---------------------------------------------------------------------------
// Sequence generation.
// ---------------------------------------------------------------------------
function smoothstep(t) {
  return t * t * (3 - 2 * t)
}

/** Position along a waypoint path at progress t in [0,1] (eased per leg). */
function pathAt(path, t) {
  if (path.length === 1) return path[0]
  const legs = path.length - 1
  const scaled = Math.min(t, 0.999999) * legs
  const leg = Math.floor(scaled)
  const lt = smoothstep(scaled - leg)
  const a = path[leg]
  const b = path[leg + 1]
  return [a[0] + (b[0] - a[0]) * lt, a[1] + (b[1] - a[1]) * lt]
}

function lerpShape(a, b, t) {
  return a.map((v, i) => v + (b[i] - v) * t)
}

/** Build one 225-dim frame from wrist positions + hand curls + head offset. */
export function buildFrame(wristL, wristR, curlsL, curlsR, headDx = 0, headDy = 0) {
  const frame = new Float32Array(FEATURE_DIM)
  // Pose landmarks (33 x 3), starting from the rest pose.
  for (let i = 0; i < POSE_LANDMARKS; i++) {
    frame[i * 3] = REST_POSE[i][0]
    frame[i * 3 + 1] = REST_POSE[i][1]
    frame[i * 3 + 2] = REST_POSE[i][2]
  }
  // Head landmarks (nose..mouth, 0..10) shift with head motion.
  for (let i = 0; i <= 10; i++) {
    frame[i * 3] += headDx
    frame[i * 3 + 1] += headDy
  }
  // Arms follow the wrists: elbow = shoulder + 0.5*(wrist-shoulder) + outward.
  const setPose = (idx, x, y, z) => {
    frame[idx * 3] = x
    frame[idx * 3 + 1] = y
    frame[idx * 3 + 2] = z
  }
  const placeArm = (shoulderIdx, elbowIdx, wristIdx, wrist, side) => {
    const sx = REST_POSE[shoulderIdx][0]
    const sy = REST_POSE[shoulderIdx][1]
    setPose(wristIdx, wrist[0], wrist[1], wrist[2])
    setPose(
      elbowIdx,
      sx + (wrist[0] - sx) * 0.5 + side * 0.045,
      sy + (wrist[1] - sy) * 0.5 + 0.02,
      -0.02,
    )
  }
  placeArm(11, 13, 15, wristL, -1)
  placeArm(12, 14, 16, wristR, 1)
  // Pose hand-tip landmarks mirror the generated hand joints.
  const handL = handLandmarks(wristL, curlsL, 1)
  const handR = handLandmarks(wristR, curlsR, -1)
  setPose(17, handL[20][0], handL[20][1], handL[20][2]) // left pinky tip
  setPose(18, handR[20][0], handR[20][1], handR[20][2])
  setPose(19, handL[8][0], handL[8][1], handL[8][2]) // left index tip
  setPose(20, handR[8][0], handR[8][1], handR[8][2])
  setPose(21, handL[4][0], handL[4][1], handL[4][2]) // left thumb tip
  setPose(22, handR[4][0], handR[4][1], handR[4][2])
  // Hand landmarks: left hand 99..161, right hand 162..224.
  for (let i = 0; i < HAND_LANDMARKS; i++) {
    frame[99 + i * 3] = handL[i][0]
    frame[99 + i * 3 + 1] = handL[i][1]
    frame[99 + i * 3 + 2] = handL[i][2]
    frame[162 + i * 3] = handR[i][0]
    frame[162 + i * 3 + 1] = handR[i][1]
    frame[162 + i * 3 + 2] = handR[i][2]
  }
  return frame
}

/**
 * Full 30-frame sequence for a word. `seed` varies jitter/speed/phase so
 * training samples differ; seed 0 gives the clean canonical sequence.
 * Returns Float32Array of length 30 * 225.
 */
export function generateWordSequence(word, seed = 0) {
  const params = WORD_PARAMS[word]
  if (!params) throw new Error(`No motion params for word: ${word}`)
  const rng = mulberry32(seed * 2654435761 + word.length * 97 + 13)
  const out = new Float32Array(FRAMES * FEATURE_DIM)
  const repeat = params.repeat ?? 1
  // Per-sample variation: global shift, amplitude scale, phase offset.
  const dx = seed === 0 ? 0 : gaussian(rng) * 0.012
  const dy = seed === 0 ? 0 : gaussian(rng) * 0.012
  const amp = seed === 0 ? 1 : 0.92 + rng() * 0.16
  const phase = seed === 0 ? 0 : Math.floor(rng() * 4)
  const jitter = seed === 0 ? 0 : 0.0035
  const shape2 = params.shape2 ?? params.shape
  for (let f = 0; f < FRAMES; f++) {
    const t = (f + phase) / FRAMES
    const cycleT = (t * repeat) % 1
    const morph = params.shape2 ? Math.min(1, t * 2) : 0
    const shape = lerpShape(params.shape, shape2, morph)
    const wristFor = (isRight) => {
      const isDom = params.hand === 'BOTH' || (params.hand === 'R') === isRight
      const rest = isRight ? REST_WRIST_R : REST_WRIST_L
      let target = rest
      if (isDom) {
        const path = isRight ? params.path : params.path2 ?? params.path
        const p = pathAt(path, cycleT)
        target = [0.5 + (p[0] - 0.5) * amp + dx, 0.5 + (p[1] - 0.5) * amp + dy]
      }
      return [target[0], target[1], -0.05]
    }
    const wristL = wristFor(false)
    const wristR = wristFor(true)
    // Non-dominant hand mirrors the dominant shape (relaxed version).
    const curlsR = params.hand === 'L' ? SHAPES.FLAT : shape
    const curlsL = params.hand === 'R' ? SHAPES.FLAT : shape
    let headDx = 0
    let headDy = 0
    if (params.head === 'nod') headDy = Math.sin((f / FRAMES) * Math.PI * 4) * 0.008
    if (params.head === 'shake') headDx = Math.sin((f / FRAMES) * Math.PI * 4) * 0.008
    const frame = buildFrame(wristL, wristR, curlsL, curlsR, headDx, headDy)
    if (jitter > 0) {
      for (let i = 0; i < FEATURE_DIM; i++) frame[i] += gaussian(rng) * jitter
    }
    out.set(frame, f * FEATURE_DIM)
  }
  return out
}

/** 5 canonical keyframes (frames 0, 7, 15, 22, 29) for the lexicon. */
export function generateWordKeyframes(word) {
  const seq = generateWordSequence(word, 0)
  const picks = [0, 7, 15, 22, 29]
  return picks.map((f) => Array.from(seq.slice(f * FEATURE_DIM, (f + 1) * FEATURE_DIM)))
}

// ---------------------------------------------------------------------------
// Fingerspelling (single right hand, approximate ISL/ASL letter shapes).
// Curls: [thumb, index, middle, ring, pinky].
// ---------------------------------------------------------------------------
export const LETTER_SHAPES = {
  A: [0.3, 1, 1, 1, 1], B: [1, 0, 0, 0, 0], C: [0.5, 0.4, 0.4, 0.4, 0.4],
  D: [0.9, 0, 0.9, 0.9, 0.9], E: [1, 0.9, 0.9, 0.9, 0.9], F: [0.2, 0.2, 0, 0, 0],
  G: [0.8, 0.1, 1, 1, 1], H: [0.8, 0, 0, 1, 1], I: [0.8, 1, 1, 1, 0],
  J: [0.8, 1, 1, 1, 0], K: [0.3, 0, 0, 1, 1], L: [0, 0, 1, 1, 1],
  M: [1, 0.8, 0.8, 0.8, 0.8], N: [1, 0.8, 0.8, 0.8, 0.8], O: [0.4, 0.4, 0.4, 0.4, 0.4],
  P: [0.3, 0, 0, 1, 1], Q: [0.8, 0.1, 1, 1, 1], R: [0.8, 0, 0, 1, 1],
  S: [1, 1, 1, 1, 1], T: [0.7, 0.9, 0.9, 0.9, 0.9], U: [0.8, 0, 0, 1, 1],
  V: [1, 0, 0, 1, 1], W: [1, 0, 0, 0, 1], X: [0.8, 0.5, 1, 1, 1],
  Y: [0, 1, 1, 1, 0], Z: [0.8, 0.1, 1, 1, 1],
}

const LETTER_RAISE = [0.58, 0.38, -0.05]

/** 5 keyframes for one fingerspelled letter (hand rises, holds, lowers). */
export function generateLetterKeyframes(letter) {
  const curls = LETTER_SHAPES[letter]
  if (!curls) throw new Error(`No hand shape for letter: ${letter}`)
  const rest = buildFrame(REST_WRIST_L, REST_WRIST_R, SHAPES.FLAT, SHAPES.FLAT)
  const raised = buildFrame(REST_WRIST_L, LETTER_RAISE, SHAPES.FLAT, curls)
  const frame = (f) => Array.from(f)
  return [frame(rest), frame(raised), frame(raised), frame(raised), frame(rest)]
}
