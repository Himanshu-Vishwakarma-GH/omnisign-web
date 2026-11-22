import { describe, expect, it } from 'vitest'
import { normalizeLandmarks } from './normalization'
import { LandmarkBuffer } from './buffer'
import { Debouncer } from './debounce'
import { tokenizeGloss } from './gloss'
import { generateSyntheticSequence } from './synthetic'
import { FRAMES, LANDMARKS } from './constants'

function makeFrame(scale = 1, offset: [number, number, number] = [0, 0, 0]): Float32Array {
  // Hand-crafted frame: right-hand wrist at (1,2,3), middle MCP at (1,3,3)
  // (unit distance in +y), left hand mirrored, shoulders 2 units apart.
  const f = new Float32Array(144)
  const set = (i: number, x: number, y: number, z: number) => {
    f[i * 3] = offset[0] + x * scale
    f[i * 3 + 1] = offset[1] + y * scale
    f[i * 3 + 2] = offset[2] + z * scale
  }
  // Fill every landmark with deterministic values first (no zeros).
  for (let i = 0; i < 48; i++) set(i, 0.5 + (i % 7) * 0.1, 1 + (i % 5) * 0.1, 0.2 + (i % 3) * 0.05)
  set(0, 1, 2, 3) // right wrist
  set(9, 1, 3, 3) // right middle MCP (distance 1 from wrist)
  set(5, 1.5, 2, 3) // another right-hand joint
  set(21, -1, 2, 3) // left wrist
  set(30, -1, 3, 3) // left middle MCP
  set(42, -1, 5, 3) // L shoulder
  set(43, 1, 5, 3) // R shoulder (width 2)
  set(44, -1.5, 4, 3)
  set(45, 1.5, 4, 3)
  set(46, -1, 2, 3)
  set(47, 1, 2, 3)
  return f
}

describe('wrist-centric normalization', () => {
  it('maps wrist to origin and middle MCP to unit distance', () => {
    const n = normalizeLandmarks(makeFrame())
    expect(n[0]).toBeCloseTo(0)
    expect(n[1]).toBeCloseTo(0)
    expect(n[2]).toBeCloseTo(0)
    // middle MCP: (0,1,0) after normalization
    expect(n[9 * 3]).toBeCloseTo(0)
    expect(n[9 * 3 + 1]).toBeCloseTo(1)
    expect(n[9 * 3 + 2]).toBeCloseTo(0)
    // joint 5 was (0.5, 0, 0) relative to wrist
    expect(n[5 * 3]).toBeCloseTo(0.5)
  })

  it('is translation and scale invariant', () => {
    const a = normalizeLandmarks(makeFrame(1, [0, 0, 0]))
    const b = normalizeLandmarks(makeFrame(3, [10, -7, 4]))
    for (let i = 0; i < 144; i++) expect(b[i]).toBeCloseTo(a[i], 5)
  })

  it('normalizes the left hand by its own wrist', () => {
    const n = normalizeLandmarks(makeFrame())
    expect(n[21 * 3]).toBeCloseTo(0)
    expect(n[21 * 3 + 1]).toBeCloseTo(0)
    expect(n[30 * 3 + 1]).toBeCloseTo(1)
  })

  it('normalizes upper body by shoulder midpoint/width', () => {
    const n = normalizeLandmarks(makeFrame())
    // shoulder midpoint was (0,5,3); width 2 -> L shoulder (-0.5, 0, 0)
    expect(n[42 * 3]).toBeCloseTo(-0.5)
    expect(n[42 * 3 + 1]).toBeCloseTo(0)
    expect(n[43 * 3]).toBeCloseTo(0.5)
  })

  it('stays finite on a degenerate (all-equal) frame', () => {
    const n = normalizeLandmarks(new Float32Array(144).fill(1))
    for (const v of n) expect(Number.isFinite(v)).toBe(true)
  })
})

describe('30-frame buffer', () => {
  it('produces a [1, 30, 48, 3] tensor (4320 floats)', () => {
    const buf = new LandmarkBuffer()
    for (let f = 0; f < FRAMES; f++) {
      const frame = new Float32Array(144).fill(f)
      buf.push(frame)
    }
    expect(buf.isFull).toBe(true)
    const t = buf.toTensor()
    expect(t.length).toBe(1 * FRAMES * LANDMARKS * 3)
    // frame-major: first frame all 0s, last frame all 29s
    expect(t[0]).toBe(0)
    expect(t[29 * 144]).toBe(29)
  })

  it('keeps only the latest 30 frames', () => {
    const buf = new LandmarkBuffer()
    for (let f = 0; f < 40; f++) buf.push(new Float32Array(144).fill(f))
    expect(buf.length).toBe(30)
    expect(buf.toTensor()[0]).toBe(10)
  })
})

describe('debounce filter', () => {
  it('commits only after >5 consecutive passes above 0.85', () => {
    const d = new Debouncer()
    for (let i = 0; i < 5; i++) expect(d.push('Help', 0.9)).toBeNull()
    expect(d.push('Help', 0.9)).toBe('Help')
  })

  it('does not commit below the confidence threshold', () => {
    const d = new Debouncer()
    for (let i = 0; i < 10; i++) expect(d.push('Help', 0.8)).toBeNull()
  })

  it('resets the streak when the label changes', () => {
    const d = new Debouncer()
    for (let i = 0; i < 4; i++) d.push('Help', 0.9)
    expect(d.push('Water', 0.95)).toBeNull()
    for (let i = 0; i < 4; i++) expect(d.push('Water', 0.95)).toBeNull()
    expect(d.push('Water', 0.95)).toBe('Water')
  })

  it('commits a continuous sign only once until the streak breaks', () => {
    const d = new Debouncer()
    for (let i = 0; i < 6; i++) d.push('Help', 0.9)
    expect(d.push('Help', 0.9)).toBeNull() // already committed, no repeat
    d.push('Water', 0.9) // break the streak
    for (let i = 0; i < 5; i++) expect(d.push('Help', 0.9)).toBeNull()
    expect(d.push('Help', 0.9)).toBe('Help')
  })
})

describe('gloss tokenizer', () => {
  it('"What is your name?" -> ["YOUR", "NAME", "WHAT"]', () => {
    expect(tokenizeGloss('What is your name?')).toEqual(['YOUR', 'NAME', 'WHAT'])
  })

  it('strips fillers and moves WH-words to the end', () => {
    expect(tokenizeGloss('Where is the hospital?')).toEqual(['HOSPITAL', 'WHERE'])
    expect(tokenizeGloss('I need a doctor')).toEqual(['I', 'NEED', 'DOCTOR'])
    expect(tokenizeGloss('Please help me')).toEqual(['PLEASE', 'HELP', 'ME'])
  })

  it('collapses "thank you" into THANKYOU', () => {
    expect(tokenizeGloss('Thank you!')).toEqual(['THANKYOU'])
  })

  it('handles empty input', () => {
    expect(tokenizeGloss('')).toEqual([])
  })
})

describe('synthetic sequence generator', () => {
  it('produces 30 frames x 48 landmarks x 3 coords', () => {
    const seq = generateSyntheticSequence('Help', 1)
    expect(seq.length).toBe(FRAMES * LANDMARKS * 3)
  })

  it('is deterministic for the same seed and differs per class', () => {
    const a = generateSyntheticSequence('Help', 7)
    const b = generateSyntheticSequence('Help', 7)
    const c = generateSyntheticSequence('Water', 7)
    expect(a).toEqual(b)
    expect(a).not.toEqual(c)
  })
})
