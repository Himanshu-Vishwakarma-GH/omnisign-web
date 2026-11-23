import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'
import * as tf from '@tensorflow/tfjs'
import {
  LEXISIGHT_HAND_DIM,
  applyTemperature,
  labelsFromMap,
  patchTopology,
  sliceHands,
  topK,
} from './lexisight'
import './lexisightLayers'
import { loadLexisightFromArtifacts } from './tfjsClient'
import { generateLetterKeyframes, generateWordSequence } from './motionGen.mjs'

const LEXISIGHT_DIR = fileURLToPath(new URL('../../public/models/lexisight/', import.meta.url))

describe('patchTopology (Keras 3 -> TF.js)', () => {
  const keras3 = {
    modelTopology: {
      class_name: 'Functional',
      keras_version: '3.0.0',
      backend: 'tensorflow',
      config: {
        name: 'm',
        trainable: true,
        layers: [
          {
            module: 'keras.layers',
            class_name: 'InputLayer',
            config: { batch_shape: [null, 30, 126], dtype: 'float32', name: 'x' },
            name: 'x',
            inbound_nodes: [],
          },
          {
            module: 'keras.layers',
            class_name: 'Dense',
            config: {
              name: 'd',
              dtype: { module: 'keras', class_name: 'DTypePolicy', config: { name: 'float32' } },
              units: 4,
            },
            build_config: { input_shape: [null, 30, 126] },
            name: 'd',
            inbound_nodes: [
              {
                args: [
                  {
                    class_name: '__keras_tensor__',
                    config: { shape: [null, 30, 126], keras_history: ['x', 0, 0] },
                  },
                ],
                kwargs: {},
              },
            ],
          },
        ],
        input_layers: ['x', 0, 0],
        output_layers: ['d', 0, 0],
      },
    },
    weightsManifest: [],
  }

  it('rewrites batch_shape, dtype, inbound_nodes, input/output layers', () => {
    const patched = patchTopology(keras3) as any
    const [input, dense] = patched.config.layers
    expect(input.config.batch_input_shape).toEqual([null, 30, 126])
    expect(input.config.batch_shape).toBeUndefined()
    expect(dense.config.dtype).toBe('float32')
    expect(dense.build_config).toBeUndefined()
    expect(dense.inbound_nodes).toEqual([[['x', 0, 0, {}]]])
    expect(patched.config.input_layers).toEqual([['x', 0, 0]])
    expect(patched.config.output_layers).toEqual([['d', 0, 0]])
  })

  it('does not mutate the input', () => {
    const before = JSON.stringify(keras3)
    patchTopology(keras3)
    expect(JSON.stringify(keras3)).toBe(before)
  })
})

describe('sliceHands (225-dim -> 126-dim hand landmarks)', () => {
  it('takes dims 99..224 (left hand, then right hand) per frame', () => {
    const frames = new Float32Array(2 * 225)
    for (let i = 0; i < 450; i++) frames[i] = i
    const hands = sliceHands(frames, 2)
    expect(hands.length).toBe(2 * LEXISIGHT_HAND_DIM)
    expect(hands[0]).toBe(99) // frame 0, left hand dim 0
    expect(hands[62]).toBe(161) // frame 0, left hand last
    expect(hands[63]).toBe(162) // frame 0, right hand first
    expect(hands[125]).toBe(224)
    expect(hands[126]).toBe(225 + 99) // frame 1, left hand first
  })
})

describe('applyTemperature', () => {
  it('is the identity at T=1 and keeps a valid distribution', () => {
    expect(applyTemperature([0.7, 0.2, 0.1], 1)).toEqual([0.7, 0.2, 0.1])
    const out = applyTemperature([0.7, 0.2, 0.1], 1.05)
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
    expect(out[0]).toBeLessThan(0.7) // T>1 flattens
    expect(out[2]).toBeGreaterThan(0.1)
  })
})

describe('labelsFromMap', () => {
  it('orders by numeric key, not insertion order', () => {
    expect(labelsFromMap({ '2': 'C', '0': 'A', '10': 'K', '1': 'B' })).toEqual([
      'A', 'B', 'C', 'K',
    ])
  })
})

describe('topK', () => {
  it('ranks highest first', () => {
    expect(topK([0.1, 0.6, 0.3], ['a', 'b', 'c'], 2)).toEqual([
      { word: 'b', confidence: 0.6 },
      { word: 'c', confidence: 0.3 },
    ])
  })
})

describe('LexiSight model files (end-to-end load, browser TF.js API)', () => {
  it(
    'loads the temporal model: [null,30,126] -> 263, finite softmax outputs',
    async () => {
      await tf.ready()
      const modelJson = JSON.parse(
        fs.readFileSync(`${LEXISIGHT_DIR}/tfjs_model/model.json`, 'utf8'),
      )
      const wbin = fs.readFileSync(`${LEXISIGHT_DIR}/tfjs_model/group1-shard1of1.bin`)
      const weightData = wbin.buffer.slice(wbin.byteOffset, wbin.byteOffset + wbin.byteLength)
      const model = await loadLexisightFromArtifacts(tf, modelJson, weightData)
      expect(model.inputs[0].shape).toEqual([null, 30, 126])
      expect(model.outputs[0].shape).toEqual([null, 263])
      expect(model.weights.length).toBe(41)
      const zeros = tf.zeros([1, 30, 126])
      const out = model.predict(zeros) as tf.Tensor
      const probs = Array.from(await out.data()) as number[]
      expect(probs.length).toBe(263)
      expect(probs.every((p) => Number.isFinite(p))).toBe(true)
      expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
      // Procedural WATER sequence (hands slice of the flat 30x225 sequence).
      const seq = generateWordSequence('WATER', 7) as Float32Array
      const hands = sliceHands(seq, 30)
      const t = tf.tensor(hands, [1, 30, 126], 'float32')
      const out2 = model.predict(t) as tf.Tensor
      const p2 = Array.from(await out2.data()) as number[]
      expect(p2.every((p) => Number.isFinite(p))).toBe(true)
      expect(p2.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
      // NOTE: the top-1 label on procedural data is NOT expected to be
      // WATER — procedural motion is not real ISL. This only proves the
      // real model runs end-to-end on correctly-shaped input.
      zeros.dispose()
      out.dispose()
      t.dispose()
      out2.dispose()
    },
    60_000,
  )

  it(
    'loads the static fingerspelling model: [null,126] -> 36, finite outputs',
    async () => {
      await tf.ready()
      const modelJson = JSON.parse(
        fs.readFileSync(`${LEXISIGHT_DIR}/static_tfjs_model/model.json`, 'utf8'),
      )
      const wbin = fs.readFileSync(`${LEXISIGHT_DIR}/static_tfjs_model/group1-shard1of1.bin`)
      const weightData = wbin.buffer.slice(wbin.byteOffset, wbin.byteOffset + wbin.byteLength)
      const model = await loadLexisightFromArtifacts(tf, modelJson, weightData)
      expect(model.inputs[0].shape).toEqual([null, 126])
      expect(model.outputs[0].shape).toEqual([null, 36])
      expect(model.weights.length).toBe(16)
      const kf = generateLetterKeyframes('A') as unknown as number[][]
      const frame = new Float32Array(kf[2])
      const hands = sliceHands(frame, 1)
      const t = tf.tensor(hands, [1, 126], 'float32')
      const out = model.predict(t) as tf.Tensor
      const probs = Array.from(await out.data()) as number[]
      expect(probs.length).toBe(36)
      expect(probs.every((p) => Number.isFinite(p))).toBe(true)
      expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
      t.dispose()
      out.dispose()
    },
    60_000,
  )
})
