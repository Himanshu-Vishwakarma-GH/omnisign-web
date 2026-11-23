import { describe, expect, it } from 'vitest'
import { extractFeatures, HolisticBuffer, FEATURE_DIM, FRAMES } from './holistic'
import { LandmarkBuffer } from './buffer'
import { Debouncer } from './debounce'
import { tokenizeGloss } from './gloss'
import { expandKeyframes } from './lexicon'
import {
  FEATURE_DIM as GEN_DIM,
  generateLetterKeyframes,
  generateWordKeyframes,
  generateWordSequence,
  VOCAB_WORDS,
  EMERGENCY_WORDS,
} from './motionGen.mjs'
import { FRAMES as CONST_FRAMES } from './constants'

describe('LexiSight-format feature extraction (30 x 225)', () => {
  it('produces a 225-dim vector: pose 33, left hand 21, right hand 21', () => {
    const res = extractFeatures({
      poseLandmarks: Array.from({ length: 33 }, (_, i) => ({ x: i / 33, y: 0.5, z: 0 })),
      leftHandLandmarks: Array.from({ length: 21 }, () => ({ x: 0.2, y: 0.3, z: -0.05 })),
      rightHandLandmarks: Array.from({ length: 21 }, () => ({ x: 0.8, y: 0.3, z: -0.05 })),
    })
    expect(res.length).toBe(225)
    expect(res[0]).toBeCloseTo(0) // pose landmark 0 x
    expect(res[99]).toBeCloseTo(0.2) // left hand starts at dim 99
    expect(res[162]).toBeCloseTo(0.8) // right hand starts at dim 162
    expect(res[224]).toBeCloseTo(-0.05)
  })

  it('zero-fills a missing hand', () => {
    const res = extractFeatures({
      poseLandmarks: Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0 })),
      leftHandLandmarks: null,
      rightHandLandmarks: Array.from({ length: 21 }, () => ({ x: 0.8, y: 0.3, z: 0 })),
    })
    for (let i = 99; i < 162; i++) expect(res[i]).toBe(0)
    expect(res[162]).toBeCloseTo(0.8)
  })

  it('zero-fills everything when nothing is detected', () => {
    const res = extractFeatures({})
    expect(res.length).toBe(225)
    for (const v of res) expect(v).toBe(0)
  })
})

describe('30-frame buffer', () => {
  it('produces a [1, 30, 225] tensor (6750 floats)', () => {
    const buf = new HolisticBuffer()
    for (let f = 0; f < FRAMES; f++) {
      buf.push(new Float32Array(FEATURE_DIM).fill(f))
    }
    expect(buf.isFull).toBe(true)
    const t = buf.toTensor()
    expect(t.length).toBe(1 * FRAMES * FEATURE_DIM)
    expect(t[0]).toBe(0)
    expect(t[29 * FEATURE_DIM]).toBe(29)
  })

  it('keeps only the latest 30 frames', () => {
    const buf = new LandmarkBuffer()
    for (let f = 0; f < 40; f++) buf.push(new Float32Array(FEATURE_DIM).fill(f))
    expect(buf.length).toBe(30)
    expect(buf.toTensor()[0]).toBe(10)
  })

  it('constants agree: 30 frames, 225 dims', () => {
    expect(CONST_FRAMES).toBe(30)
    expect(FEATURE_DIM).toBe(225)
    expect(GEN_DIM).toBe(225)
  })
})

describe('debounce filter (LexiSight 5-frame consensus)', () => {
  it('commits only after >5 consecutive passes above 0.85', () => {
    const d = new Debouncer()
    for (let i = 0; i < 5; i++) expect(d.push('HELLO', 0.9)).toBeNull()
    expect(d.push('HELLO', 0.9)).toBe('HELLO')
  })

  it('does not commit below the confidence threshold', () => {
    const d = new Debouncer()
    for (let i = 0; i < 10; i++) expect(d.push('HELLO', 0.8)).toBeNull()
  })

  it('resets the streak when the label changes', () => {
    const d = new Debouncer()
    for (let i = 0; i < 4; i++) d.push('HELLO', 0.9)
    expect(d.push('WATER', 0.95)).toBeNull()
    for (let i = 0; i < 4; i++) expect(d.push('WATER', 0.95)).toBeNull()
    expect(d.push('WATER', 0.95)).toBe('WATER')
  })
})

describe('gloss tokenizer (conversation sentences)', () => {
  it('"What is your name?" -> ["YOUR", "NAME", "WHAT"]', () => {
    expect(tokenizeGloss('What is your name?')).toEqual(['YOUR', 'NAME', 'WHAT'])
  })

  it('"Hello, how are you today?" -> sensible gloss without fillers', () => {
    expect(tokenizeGloss('Hello, how are you today?')).toEqual([
      'HELLO', 'YOU', 'TODAY', 'HOW',
    ])
  })

  it('strips do/will and moves WH-words to the end', () => {
    expect(tokenizeGloss('Do you want water?')).toEqual(['YOU', 'WANT', 'WATER'])
    expect(tokenizeGloss('Where is the hospital?')).toEqual(['HOSPITAL', 'WHERE'])
  })

  it('collapses "thank you" into THANKYOU', () => {
    expect(tokenizeGloss('Thank you!')).toEqual(['THANKYOU'])
  })

  it('handles empty input', () => {
    expect(tokenizeGloss('')).toEqual([])
  })
})

describe('procedural motion generator (225-dim)', () => {
  it('produces 30 frames x 225 dims', () => {
    const seq = generateWordSequence('HELLO', 0)
    expect(seq.length).toBe(30 * 225)
  })

  it('is deterministic for the same seed and differs per word', () => {
    const a = generateWordSequence('HELLO', 7)
    const b = generateWordSequence('HELLO', 7)
    const c = generateWordSequence('WATER', 7)
    expect(a).toEqual(b)
    expect(a).not.toEqual(c)
  })

  it('covers the whole vocabulary including all 20 emergency words', () => {
    expect(VOCAB_WORDS.length).toBeGreaterThanOrEqual(60)
    for (const w of EMERGENCY_WORDS) {
      expect(VOCAB_WORDS).toContain(w)
      expect(generateWordSequence(w, 0).length).toBe(30 * 225)
    }
  })

  it('produces 5 keyframes per word for the lexicon', () => {
    const kf = generateWordKeyframes('HELLO')
    expect(kf.length).toBe(5)
    for (const f of kf) expect(f.length).toBe(225)
  })

  it('produces distinct fingerspelling keyframes per letter', () => {
    const a = generateLetterKeyframes('A')
    const b = generateLetterKeyframes('B')
    expect(a.length).toBe(5)
    expect(a).not.toEqual(b)
  })
})

describe('lexicon keyframe expansion', () => {
  it('interpolates keyframes to the requested frame count', () => {
    const kf = generateWordKeyframes('HELLO')
    const frames = expandKeyframes(kf, 24)
    expect(frames.length).toBe(24)
    for (const f of frames) expect(f.length).toBe(225)
    // endpoints match the first/last keyframes
    expect(frames[0][0]).toBeCloseTo(kf[0][0])
    expect(frames[23][0]).toBeCloseTo(kf[4][0])
  })
})
