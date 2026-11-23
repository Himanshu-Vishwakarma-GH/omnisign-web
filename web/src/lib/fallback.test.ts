import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'
import * as tf from '@tensorflow/tfjs-node'
import { EMERGENCY_WORDS } from './constants'
import { generateWordSequence } from './motionGen.mjs'

const FALLBACK_DIR = fileURLToPath(new URL('../../public/models/omnisign_tfjs/', import.meta.url))

/**
 * The procedural fallback model is what Simulate Sign uses (see
 * getFallbackModel in tfjsClient.ts). Live test 23 Nov 2026 showed the
 * real LexiSight weights return low-confidence garbage on procedural
 * motion (TOMORROW -> Afternoon 22.7%, THANKYOU -> Afternoon 29.3%),
 * so this test pins the fallback's behaviour on exactly those words:
 * the selected word must come back top-1 above the 85% speak
 * threshold, otherwise Simulate Sign cannot reach the transcript.
 */
describe('procedural fallback model (Simulate Sign path)', () => {
  it('recognises procedural sequences for the words that failed live', async () => {
    await tf.ready()
    const model = await tf.loadLayersModel(`file://${FALLBACK_DIR}model.json`)
    const labels: string[] = JSON.parse(fs.readFileSync(`${FALLBACK_DIR}labels.json`, 'utf8'))
    expect(labels.length).toBe(88)
    // All 20 emergency words must be in the fallback vocabulary, or the
    // Emergency-mode Simulate dropdown offers words it cannot recognise.
    for (const w of EMERGENCY_WORDS) expect(labels).toContain(w)

    for (const word of ['TOMORROW', 'THANKYOU', 'WATER', 'HELLO']) {
      const seq = generateWordSequence(word, 12345) as Float32Array
      const input = tf.tensor(seq, [1, 30, 225], 'float32')
      const out = model.predict(input) as tf.Tensor
      const probs = Array.from(await out.data()) as number[]
      expect(probs.length).toBe(88)
      expect(probs.every((p) => Number.isFinite(p))).toBe(true)
      expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
      const best = probs.indexOf(Math.max(...probs))
      expect(labels[best]).toBe(word)
      expect(probs[best]).toBeGreaterThan(0.85)
      input.dispose()
      out.dispose()
    }
    model.dispose()
  }, 60_000)
})
