/**
 * HandAvatar mapping: 225-dim lexicon frames -> rigged-hand bone poses.
 *
 * Pure logic (no WebGL) so it is unit-testable. The 3D side lives in
 * components/HandAvatar.tsx; this module only computes numbers.
 *
 * Frame layout (225 floats, MediaPipe Holistic order):
 *   0..98    pose landmarks 0..32  (x, y, z)
 *   99..161  left hand landmarks 0..20 (x, y, z)
 *   162..224 right hand landmarks 0..20 (x, y, z)
 * Per hand, landmark indices: wrist 0, thumb CMC 1 / MCP 2 / IP 3 / tip 4,
 * index MCP 5 / PIP 6 / DIP 7 / tip 8, middle 9/10/11/12, ring 13/14/15/16,
 * pinky 17/18/19/20.
 *
 * --- Per-finger curl -------------------------------------------------------
 * Curl is measured PER FINGER from that finger's tip-to-MCP distance
 * against its extended length. A two-point calibration is used because the
 * procedural finger chain curls into a hook: the tip-to-MCP distance only
 * shrinks ~9% from fully open to fully curled (measured from motionGen's
 * handLandmarks), so a naive `1 - d/extended` would cap out near 0.07.
 * Calibration is a PIECEWISE-LINEAR inverse of the measured
 * tip-distance curve (distances measured from motionGen's
 * handLandmarks at curls 0, 0.25, 0.5, 0.75, 1), because the
 * distance-vs-curl relation is strongly nonlinear: at curl 0.5 only
 * ~26% of the total tip-distance change has happened, so a two-point
 * linear map would read CUP (0.5) as ~0.26. The curve is monotonic
 * and exact at the endpoints (0 and 1), which is what the lexicon's
 * named shapes mostly use.
 *
 * *Thumb: the literal MCP (landmark 2) -> tip distance is nearly
 * insensitive to curl (0.0314 -> 0.0309, a 1.6% change), because the
 * thumb's first segment is short. The thumb curl is therefore measured
 * from the CMC (landmark 1) -> tip span. The thumb is also
 * side-aware: motionGen's thumb chain curls the left thumb's tip
 * toward the wrist (1->4: 0.0475 -> 0.0366) but swings the right
 * thumb's tip AWAY from the wrist (1->4: 0.0475 -> 0.0541), so the
 * right hand's curve rises instead of falling (and saturates above
 * curl 0.75 — the right thumb's 0.75 and 1.0 distances are identical).
 *
 * --- Wrist orientation -----------------------------------------------------
 * The hand model ("Rigged Hand" by J-Toastie, Poly Pizza) as loaded by
 * GLTFLoader has, in scene space (from its joint positions, computed
 * from the GLB node hierarchy):
 *   fingers point along MODEL_FINGER, palm faces MODEL_PALM, knuckle
 *   spread along MODEL_SPREAD. The model is a LEFT hand: its
 *   (spread, finger, palm) triad is left-handed, so the right-hand
 *   instance is a mirrored copy (scale.x = -1 on a wrapper group).
 * wristQuaternion() builds the rotation that maps the model's frame
 * onto the target frame derived from the hand landmarks of the
 * current lexicon frame (finger direction = wrist -> middle fingertip,
 * palm normal from the knuckle triangle). The palm normal is
 * chirality-aware: the two hands are mirror images, so a fixed cross
 * product flips sign between them (see palmNormal()).
 *
 * --- Curl -> bone angles ---------------------------------------------------
 * The model's bundled "Scene" animation articulates the finger bones
 * with local rotations dominated by their Z component (e.g. IndexF_lower
 * z: -0.004 -> 0.667), so curl is applied as a rotation about each
 * bone's LOCAL Z axis, positive = curling toward the palm. Total curl
 * angle is 1.9 rad (matching motionGen's total bend), distributed
 * [0.45, 0.35, 0.20] over proximal/middle/distal — the same shares
 * motionGen uses (BEND_SHARE). The finger Root bones (metacarpals) stay
 * at their bind rotation: they set the fan/spread, not the curl.
 * UNVERIFIED: the curl sign/axis is inferred from the bundled
 * animation, not from a visual check. If a browser test shows fingers
 * bending backwards, flip CURL_SIGN.
 */
import * as THREE from 'three'

export type FingerName = 'thumb' | 'index' | 'middle' | 'ring' | 'pinky'
export type FingerCurls = Record<FingerName, number>
export type HandSide = 'left' | 'right'

/** Model frame (as loaded by GLTFLoader), from the GLB joint positions. */
export const MODEL_FINGER = new THREE.Vector3(-0.997, -0.074, -0.017)
export const MODEL_PALM = new THREE.Vector3(-0.076, 0.98, 0.184)
export const MODEL_SPREAD = new THREE.Vector3(-0.003, -0.185, 0.983)

/** Sign of the local-Z curl rotation. +1 = curl toward the palm. */
export const CURL_SIGN = 1
/** Total curl angle (rad) at curl = 1, matching motionGen's 1.9 rad bend. */
export const TOTAL_CURL_ANGLE = 1.9

/** Bone names in the Rigged Hand model, per finger (curl bones only). */
export const FINGER_BONES: Record<FingerName, string[]> = {
  thumb: ['ThumbMiddle', 'ThumbTop'],
  index: ['IndexF_lower', 'IndexF_middle', 'IndexF_tip'],
  middle: ['MiddleF_lower', 'MiddleF_middle', 'MiddleF_tip'],
  ring: ['RingF_lower', 'RingF_middle', 'RingF_tip'],
  pinky: ['PinkyF_lower', 'PinkyF_middle', 'PinkyF_tip'],
}

/** Share of TOTAL_CURL_ANGLE per curl bone (proximal -> distal). */
export const BONE_CURL_SHARES: Record<FingerName, number[]> = {
  thumb: [0.55, 0.45],
  index: [0.45, 0.35, 0.2],
  middle: [0.45, 0.35, 0.2],
  ring: [0.45, 0.35, 0.2],
  pinky: [0.45, 0.35, 0.2],
}

/** Tip-distance curves per finger (measured from motionGen).
 * Each curve lists the tip-to-MCP distance at curls [0, .25, .5, .75, 1].
 * The thumb is side-aware (see the module docstring). */
const CURVE_CURLS = [0, 0.25, 0.5, 0.75, 1]
const CALIBRATION: Record<
  FingerName,
  { mcp: number; tip: number; curve: number[]; curveRight?: number[] }
> = {
  thumb: {
    mcp: 1,
    tip: 4,
    curve: [0.0475, 0.0442, 0.041, 0.0384, 0.0366],
    curveRight: [0.0475, 0.0505, 0.0528, 0.0541, 0.0541],
  },
  index: { mcp: 5, tip: 8, curve: [0.0622, 0.0618, 0.0607, 0.059, 0.0565] },
  middle: { mcp: 9, tip: 12, curve: [0.0671, 0.0667, 0.0655, 0.0636, 0.061] },
  ring: { mcp: 13, tip: 16, curve: [0.0641, 0.0638, 0.0627, 0.0609, 0.0584] },
  pinky: { mcp: 17, tip: 20, curve: [0.0514, 0.0511, 0.0503, 0.0488, 0.0469] },
}

/** Invert a monotonic distance curve: distance -> curl (0..1). */
function invertCurve(curve: number[], d: number): number {
  const rising = curve[curve.length - 1] > curve[0]
  if (!rising && d >= curve[0]) return 0
  if (!rising && d <= curve[curve.length - 1]) return 1
  if (rising && d <= curve[0]) return 0
  if (rising && d >= curve[curve.length - 1]) return 1
  for (let i = 0; i < curve.length - 1; i++) {
    const a = curve[i]
    const b = curve[i + 1]
    if (Math.abs(b - a) < 1e-9) continue
    if ((d - a) * (d - b) <= 0) {
      const t = (d - a) / (b - a)
      return clamp01(CURVE_CURLS[i] + t * (CURVE_CURLS[i + 1] - CURVE_CURLS[i]))
    }
  }
  return 0.5
}

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x))

function dist3(a: Float32Array, ai: number, b: Float32Array, bi: number): number {
  const dx = a[ai] - b[bi]
  const dy = a[ai + 1] - b[bi + 1]
  const dz = a[ai + 2] - b[bi + 2]
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

/**
 * Per-finger curls (0 = extended, 1 = fully curled) from one hand's 21
 * landmarks (flat array of 63 floats, landmark order 0..20).
 * The thumb calibration is side-aware (see CALIBRATION).
 */
export function fingerCurls(hand: Float32Array | number[], side: HandSide): FingerCurls {
  const out = {} as FingerCurls
  for (const name of Object.keys(CALIBRATION) as FingerName[]) {
    const c = CALIBRATION[name]
    const d = dist3(
      hand as Float32Array,
      c.mcp * 3,
      hand as Float32Array,
      c.tip * 3,
    )
    const curve = side === 'right' && c.curveRight !== undefined ? c.curveRight : c.curve
    out[name] = invertCurve(curve, d)
  }
  return out
}

/** Per-finger curls for one hand out of a 225-dim lexicon frame. */
export function handCurlsFromFrame(
  frame: Float32Array,
  handOffset: number,
  side: HandSide,
): FingerCurls {
  return fingerCurls(frame.subarray(handOffset * 3, (handOffset + 21) * 3), side)
}

/** Local-Z curl angles (rad) for each bone of a finger, from its curl. */
export function boneCurlAngles(finger: FingerName, curl: number): number[] {
  return BONE_CURL_SHARES[finger].map(
    (share) => CURL_SIGN * clamp01(curl) * TOTAL_CURL_ANGLE * share,
  )
}

/** Landmark -> scene vector, same mapping as Avatar.tsx. */
export function landmarkVec(frame: Float32Array, landmark: number): THREE.Vector3 {
  return new THREE.Vector3(
    (frame[landmark * 3] - 0.5) * 2.2,
    (0.5 - frame[landmark * 3 + 1]) * 2.2,
    frame[landmark * 3 + 2] * 2,
  )
}

/**
 * Palm normal (unit, scene space) from one hand's landmarks.
 * The two hands are mirror images, so the cross product flips sign
 * between them: for the right hand it is used as-is, for the left hand
 * negated. (Calibrated against motionGen's convention, where both hands
 * show their backs to the camera at rest: the right hand's palm is -Z,
 * the left hand's is also -Z, and cross(middle-wrist, index-pinky)
 * returns -Z for the right hand and +Z for the left.)
 */
export function palmNormal(
  frame: Float32Array,
  handOffset: number,
  side: HandSide,
): THREE.Vector3 {
  const wrist = landmarkVec(frame, handOffset)
  const indexMcp = landmarkVec(frame, handOffset + 5)
  const middleMcp = landmarkVec(frame, handOffset + 9)
  const pinkyMcp = landmarkVec(frame, handOffset + 17)
  const a = middleMcp.clone().sub(wrist)
  const b = indexMcp.clone().sub(pinkyMcp)
  const n = new THREE.Vector3().crossVectors(a, b)
  if (n.lengthSq() < 1e-12) return new THREE.Vector3(0, 0, side === 'right' ? -1 : 1)
  n.normalize()
  return side === 'right' ? n : n.negate()
}

export interface WristTarget {
  /** Scene-space wrist position. */
  position: THREE.Vector3
  /** Rotation mapping the model's frame onto the target frame. */
  quaternion: THREE.Quaternion
}

const _mModel = new THREE.Matrix4().makeBasis(
  MODEL_SPREAD.clone(),
  MODEL_FINGER.clone(),
  MODEL_PALM.clone(),
)
const _mModelInv = _mModel.clone().invert()
const _mirror = new THREE.Matrix4().makeScale(-1, 1, 1)

/**
 * Wrist position + orientation for one hand from a 225-dim frame.
 * The finger direction comes from the HAND landmarks themselves
 * (middle fingertip - wrist): the lexicon's rest pose holds the fingers
 * up while the forearm points down, so the forearm (elbow -> wrist)
 * is NOT the finger direction. The wrist position is the hand's wrist
 * landmark (identical to the pose wrist landmark).
 *
 * The returned quaternion already accounts for the mirror wrapper of
 * the right-hand instance: apply it to the hand assembly whose outer
 * group carries scale.x = -1 for the right hand (identity for the
 * left). Equivalently, Q = R_target * mirror, where R_target maps the
 * model basis onto the target basis.
 */
export function wristTarget(
  frame: Float32Array,
  side: HandSide,
  handOffset: number,
): WristTarget {
  const wrist = landmarkVec(frame, handOffset)
  const middleTip = landmarkVec(frame, handOffset + 12)

  // Target finger direction: wrist -> middle fingertip.
  const finger = middleTip.clone().sub(wrist)
  if (finger.lengthSq() < 1e-12) finger.set(0, 1, 0)
  finger.normalize()

  const palm = palmNormal(frame, handOffset, side)

  // Target spread, completing the orthonormal basis with the same
  // handedness as the model frame (left-handed for the left hand,
  // right-handed for the mirrored right hand).
  const spread =
    side === 'right'
      ? new THREE.Vector3().crossVectors(finger, palm)
      : new THREE.Vector3().crossVectors(palm, finger)
  if (spread.lengthSq() < 1e-12) spread.set(1, 0, 0)
  spread.normalize()
  // Re-orthogonalise the palm against finger/spread for safety.
  // The cross-product order is chirality-aware: for the left hand
  // finger x spread == palm, for the right hand spread x finger == palm
  // (because the right hand's spread is finger x palm, and
  // finger x (finger x palm) == -palm). The resulting target basis
  // [spread, finger, palm] is left-handed for the left hand (det -1,
  // matching the left-hand model) and right-handed for the right hand
  // (det +1, matching the mirrored instance).
  const palmOrtho =
    side === 'right'
      ? new THREE.Vector3().crossVectors(spread, finger)
      : new THREE.Vector3().crossVectors(finger, spread)
  palmOrtho.normalize()

  const mTarget = new THREE.Matrix4().makeBasis(spread, finger, palmOrtho)
  const mRot = mTarget.clone().multiply(_mModelInv)
  if (side === 'right') mRot.multiply(_mirror)
  return { position: wrist, quaternion: new THREE.Quaternion().setFromRotationMatrix(mRot) }
}
