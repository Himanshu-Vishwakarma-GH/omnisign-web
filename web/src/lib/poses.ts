/**
 * Avatar pose definitions.
 *
 * A pose describes the avatar's upper body as a small set of joint angles
 * the Three.js rig interpolates between with quaternion SLERP:
 *   shoulder: [x, y, z] euler angles (radians) for each arm
 *   elbow:    bend angle (radians)
 *   hand:     openness 0 (fist) .. 1 (open palm)
 *   head:     [x, y, z] euler tilt
 *
 * poses.json (public/) carries the same data for all 20 emergency words
 * plus A–Z finger-spelling poses; this module provides the deterministic
 * fallback generator used when the JSON cannot be fetched.
 */

export interface ArmPose {
  shoulder: [number, number, number]
  elbow: number
  hand: number
}

export interface AvatarPose {
  left: ArmPose
  right: ArmPose
  head: [number, number, number]
}

const WORDS = [
  'HELP', 'DOCTOR', 'PAIN', 'WATER', 'YES', 'NO', 'FAMILY', 'EMERGENCY',
  'HOSPITAL', 'MEDICINE', 'STOP', 'WAIT', 'CALL', 'POLICE', 'FIRE', 'HURT',
  'BREATHE', 'ALLERGY', 'THANKYOU', 'PLEASE',
]

function hashStr(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return Math.abs(h)
}

/** Deterministic procedural pose for any word or letter. */
export function poseForToken(token: string): AvatarPose {
  const h = hashStr(token)
  const a = (n: number) => ((h >> n) % 100) / 100 // pseudo-random 0..1
  const arm = (off: number): ArmPose => ({
    shoulder: [
      -0.4 + a(off) * 1.2,
      -0.5 + a(off + 3) * 1.0,
      -0.6 + a(off + 6) * 1.2,
    ],
    elbow: 0.3 + a(off + 9) * 1.1,
    hand: 0.2 + a(off + 12) * 0.8,
  })
  return {
    left: arm(0),
    right: arm(16),
    head: [a(24) * 0.2 - 0.1, a(27) * 0.3 - 0.15, 0],
  }
}

export function defaultPoses(): Record<string, AvatarPose> {
  const out: Record<string, AvatarPose> = {}
  for (const w of WORDS) out[w] = poseForToken(w)
  for (let c = 65; c <= 90; c++) {
    const letter = String.fromCharCode(c)
    // Finger-spelling: right hand raised near the chest, left arm relaxed.
    const p = poseForToken(letter)
    p.right.shoulder = [0.5, -0.3, 0.9]
    p.right.elbow = 1.4
    p.left.shoulder = [0.1, 0.2, 0.15]
    p.left.elbow = 0.4
    out[letter] = p
  }
  return out
}
