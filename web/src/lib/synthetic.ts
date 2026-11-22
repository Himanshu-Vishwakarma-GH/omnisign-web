import { FRAMES, LANDMARKS, SIGNS, type SignWord } from './constants'

/**
 * Deterministic synthetic 30-frame landmark sequences.
 *
 * This generator is the SAME motion model used to train the ONNX BiLSTM
 * (see ml/model.py — constants mirrored exactly). Each class is a distinct
 * sinusoidal trajectory prototype:
 *   freq = 0.5 + 0.09 * classIndex   cycles over the 30-frame window
 *          (unique per class — 20 distinct frequencies)
 *   amp  = 0.05 + 0.02 * (class % 3)
 *   axis = class % 3                   0 = x, 1 = y, 2 = z
 *   hand = class % 3                   0 = right, 1 = left, 2 = both
 * plus a slow drift and small seeded Gaussian-ish noise.
 *
 * Base skeleton: two hands (21 joints each) at x = +/-0.25, y ~ 0.45,
 * plus 6 upper-body landmarks (shoulders, elbows, wrists).
 */

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Base (rest) position of landmark i, before motion. Returns [x, y, z]. */
export function basePosition(i: number): [number, number, number] {
  if (i < 42) {
    // Hand joints: hand 0 = right (x +0.25), hand 1 = left (x -0.25).
    const hand = i < 21 ? 0 : 1
    const j = i % 21
    const ox = hand === 0 ? 0.25 : -0.25
    if (j === 0) return [ox, 0.45, 0]
    const finger = Math.floor((j - 1) / 4) // 0 thumb .. 4 pinky
    const seg = (j - 1) % 4 // 0..3 along the finger
    const x = ox + (finger - 2) * 0.04 + (finger === 0 ? -0.03 : 0)
    const y = 0.45 + 0.03 + seg * 0.035
    const z = 0.01 * seg
    return [x, y, z]
  }
  // Upper body: 42 L shoulder, 43 R shoulder, 44 L elbow, 45 R elbow,
  // 46 L wrist, 47 R wrist.
  const body: [number, number, number][] = [
    [-0.2, 0.95, 0],
    [0.2, 0.95, 0],
    [-0.32, 0.68, 0],
    [0.32, 0.68, 0],
    [-0.25, 0.45, 0],
    [0.25, 0.45, 0],
  ]
  return body[i - 42]
}

export interface ClassMotion {
  freq: number
  amp: number
  axis: 0 | 1 | 2
  hand: 0 | 1 | 2 // 0 = right, 1 = left, 2 = both
}

export function classMotion(classIndex: number): ClassMotion {
  return {
    freq: 0.5 + 0.09 * classIndex,
    amp: 0.05 + 0.02 * (classIndex % 3),
    axis: (classIndex % 3) as 0 | 1 | 2,
    hand: (classIndex % 3) as 0 | 1 | 2,
  }
}

/**
 * Generate a synthetic [30, 48, 3] sequence for a sign.
 * `seed` makes the noise deterministic (same seed => same sequence).
 * Returns a flat Float32Array of length 30 * 144, frame-major.
 */
export function generateSyntheticSequence(
  word: SignWord,
  seed = 42,
): Float32Array {
  const classIndex = SIGNS.indexOf(word)
  if (classIndex < 0) throw new Error(`Unknown sign: ${word}`)
  const { freq, amp, axis, hand } = classMotion(classIndex)
  const rand = mulberry32(seed * 1000 + classIndex * 97 + 13)
  const out = new Float32Array(FRAMES * LANDMARKS * 3)

  for (let t = 0; t < FRAMES; t++) {
    const phase = (2 * Math.PI * freq * t) / FRAMES
    const drift = 0.02 * Math.sin((2 * Math.PI * 0.5 * t) / FRAMES + classIndex)
    for (let i = 0; i < LANDMARKS; i++) {
      const base = basePosition(i)
      let disp = drift
      const isRightHand = i < 21
      const isLeftHand = i >= 21 && i < 42
      const active =
        hand === 2 || (hand === 0 && isRightHand) || (hand === 1 && isLeftHand)
      if (active) {
        disp += amp * Math.sin(phase + 0.3 * i)
      } else if (i >= 42) {
        disp += amp * 0.3 * Math.sin(phase + 0.3 * i)
      }
      // Small seeded noise, approx Gaussian via sum of uniforms.
      const noise = (rand() + rand() + rand() - 1.5) * 0.01
      const p = [base[0], base[1], base[2]]
      p[axis] += disp
      p[0] += noise
      p[1] += noise * 0.5
      p[2] += noise * 0.5
      out[(t * LANDMARKS + i) * 3] = p[0]
      out[(t * LANDMARKS + i) * 3 + 1] = p[1]
      out[(t * LANDMARKS + i) * 3 + 2] = p[2]
    }
  }
  return out
}
