/**
 * Tests for the HandAvatar mapping logic (pure math, no WebGL).
 * - Per-finger curls recovered from motionGen hand landmarks.
 * - Wrist quaternion maps the model frame onto the target frame.
 * - Bone-name mapping / curl-angle distribution.
 */
import * as THREE from 'three'
import { describe, expect, it } from 'vitest'
import {
  boneCurlAngles,
  CURL_SIGN,
  fingerCurls,
  FINGER_BONES,
  MODEL_FINGER,
  MODEL_PALM,
  MODEL_SPREAD,
  TOTAL_CURL_ANGLE,
  wristTarget,
  type FingerName,
} from './handMapping'
import { restFrame } from './lexicon'
import { handLandmarks, SHAPES } from './motionGen.mjs'

function flattenHand(lm: number[][]): Float32Array {
  return Float32Array.from(lm.flat())
}

describe('fingerCurls', () => {
  it('recovers OPEN (all extended)', () => {
    const curls = fingerCurls(flattenHand(handLandmarks([0.5, 0.5, 0], SHAPES.OPEN, 1)), 'left')
    for (const f of Object.keys(curls) as FingerName[]) {
      expect(curls[f]).toBeLessThan(0.15)
    }
  })

  it('recovers FIST (all curled)', () => {
    const curls = fingerCurls(flattenHand(handLandmarks([0.5, 0.5, 0], SHAPES.FIST, -1)), 'right')
    for (const f of Object.keys(curls) as FingerName[]) {
      expect(curls[f]).toBeGreaterThan(0.85)
    }
  })

  it('distinguishes fingers in POINT (index extended, rest curled)', () => {
    const curls = fingerCurls(flattenHand(handLandmarks([0.5, 0.5, 0], SHAPES.POINT, 1)), 'left')
    expect(curls.index).toBeLessThan(0.15)
    expect(curls.middle).toBeGreaterThan(0.85)
    expect(curls.ring).toBeGreaterThan(0.85)
    expect(curls.pinky).toBeGreaterThan(0.85)
    expect(curls.thumb).toBeGreaterThan(0.75)
  })

  it('distinguishes fingers in PEACE (index+middle extended)', () => {
    const curls = fingerCurls(flattenHand(handLandmarks([0.5, 0.5, 0], SHAPES.PEACE, 1)), 'left')
    expect(curls.index).toBeLessThan(0.15)
    expect(curls.middle).toBeLessThan(0.15)
    expect(curls.ring).toBeGreaterThan(0.85)
    expect(curls.pinky).toBeGreaterThan(0.85)
  })

  it('recovers CUP (half curl) near 0.5 for every finger', () => {
    const curls = fingerCurls(flattenHand(handLandmarks([0.5, 0.5, 0], SHAPES.CUP, 1)), 'left')
    for (const f of Object.keys(curls) as FingerName[]) {
      expect(curls[f]).toBeGreaterThan(0.3)
      expect(curls[f]).toBeLessThan(0.7)
    }
  })
})

describe('boneCurlAngles', () => {
  it('sums to the total curl angle at curl 1 and 0 at curl 0', () => {
    for (const f of Object.keys(FINGER_BONES) as FingerName[]) {
      const full = boneCurlAngles(f, 1)
      expect(full.length).toBe(FINGER_BONES[f].length)
      expect(full.reduce((a, b) => a + b, 0)).toBeCloseTo(
        CURL_SIGN * TOTAL_CURL_ANGLE,
        10,
      )
      expect(boneCurlAngles(f, 0).every((a) => a === 0)).toBe(true)
    }
  })

  it('maps every finger to named bones in the Rigged Hand model', () => {
    expect(Object.keys(FINGER_BONES).sort()).toEqual(
      ['index', 'middle', 'pinky', 'ring', 'thumb'].sort(),
    )
    expect(FINGER_BONES.thumb).toEqual(['ThumbMiddle', 'ThumbTop'])
    expect(FINGER_BONES.index[0]).toBe('IndexF_lower')
    expect(FINGER_BONES.middle[0]).toBe('MiddleF_lower')
    expect(FINGER_BONES.ring[0]).toBe('RingF_lower')
    expect(FINGER_BONES.pinky[0]).toBe('PinkyF_lower')
  })
})

describe('wristTarget', () => {
  // Expected directions, computed straight from the rest frame's own
  // landmarks (independent of wristTarget): finger = middle tip - wrist,
  // palm = -Z (both lexicon hands show their backs at rest).
  function expectedDirections(frame: Float32Array, handOffset: number) {
    const v = (lm: number) =>
      new THREE.Vector3(
        (frame[lm * 3] - 0.5) * 2.2,
        (0.5 - frame[lm * 3 + 1]) * 2.2,
        frame[lm * 3 + 2] * 2,
      )
    const finger = v(handOffset + 12).sub(v(handOffset)).normalize()
    return { finger }
  }

  it('left hand: maps model finger -> hand finger direction and model palm -> -Z at rest', () => {
    const frame = restFrame()
    const t = wristTarget(frame, 'left', 33)
    const { finger } = expectedDirections(frame, 33)
    const mappedFinger = MODEL_FINGER.clone().applyQuaternion(t.quaternion)
    expect(mappedFinger.dot(finger)).toBeGreaterThan(0.99)
    // Both lexicon hands show their backs to the camera at rest.
    const mappedPalm = MODEL_PALM.clone().applyQuaternion(t.quaternion)
    expect(mappedPalm.z).toBeLessThan(-0.9)
    // Position is the wrist landmark in scene coords.
    expect(t.position.x).toBeCloseTo(-0.44, 2)
    expect(t.position.y).toBeCloseTo(-0.242, 2)
  })

  it('right hand: mirror-composed rotation maps the model frame onto the target frame', () => {
    const frame = restFrame()
    const t = wristTarget(frame, 'right', 54)
    // The component applies Q inside a scale.x = -1 wrapper, so the
    // world-space map is Q * mirror (rotation matrix times mirror).
    const mirror = new THREE.Matrix4().makeScale(-1, 1, 1)
    const world = new THREE.Matrix4()
      .makeRotationFromQuaternion(t.quaternion)
      .multiply(mirror)
    const { finger } = expectedDirections(frame, 54)
    const mappedFinger = MODEL_FINGER.clone().applyMatrix4(world)
    expect(mappedFinger.dot(finger)).toBeGreaterThan(0.99)
    const mappedPalm = MODEL_PALM.clone().applyMatrix4(world)
    expect(mappedPalm.z).toBeLessThan(-0.9)
    // Spread maps to the anatomical index->pinky direction (right hand
    // at rest: index image-right, pinky image-left -> -X).
    const mappedSpread = MODEL_SPREAD.clone().applyMatrix4(world)
    expect(mappedSpread.x).toBeLessThan(-0.9)
  })

  it('model basis is orthonormal', () => {
    expect(MODEL_FINGER.length()).toBeCloseTo(1, 2)
    expect(MODEL_PALM.length()).toBeCloseTo(1, 2)
    expect(MODEL_SPREAD.length()).toBeCloseTo(1, 2)
    expect(Math.abs(MODEL_FINGER.dot(MODEL_PALM))).toBeLessThan(0.01)
    expect(Math.abs(MODEL_FINGER.dot(MODEL_SPREAD))).toBeLessThan(0.01)
    expect(Math.abs(MODEL_PALM.dot(MODEL_SPREAD))).toBeLessThan(0.01)
  })
})
