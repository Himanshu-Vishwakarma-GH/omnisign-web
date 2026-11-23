/**
 * TensorFlow.js model client.
 *
 * Model resolution order:
 *   1. Real LexiSight weights, bundled at `models/lexisight/`:
 *      - Temporal word model: `tfjs_model/model.json` +
 *        `group1-shard1of1.bin` + `label_map.json` (263 ISL words).
 *        Input [1, 30, 126] (hand landmarks only — the 225-dim capture
 *        frames are sliced to dims 99..224 first). Output is already a
 *        softmax; temperature T=1.05 is applied (see lexisight.ts).
 *        The model.json is a Keras 3 export, so it is loaded via
 *        `modelFromJSON` after `patchTopology`, with weights assigned
 *        by shape-group FIFO order (see `assignWeightsFromManifest`).
 *        Custom layers live in lexisightLayers.ts (imported for its
 *        registration side effect).
 *      - Static fingerspelling model: `static_tfjs_model/` + 
 *        `static_label_map.json` (0-9, A-Z), single-frame [1, 126].
 *        Exposed as `predictStatic` (best effort; undefined if it
 *        fails to load while the temporal model still works).
 *   2. Bundled fallback: `models/omnisign_tfjs/` — a small Conv1D model
 *      trained on procedural 225-dim sequences for the 88-word
 *      conversation vocabulary. It proves the full pipeline end-to-end;
 *      it is NOT trained on real signers.
 *
 * The LexiSight vocabulary covers 34 of our 88 conversation words
 * (plus THANK_YOU / HOW ARE YOU as phrases). Words outside it — HELP,
 * WATER, YES, NO, WHAT, WHERE, WHEN, etc. — cannot be recognised by
 * the LexiSight model; the transcript only ever shows labels the
 * loaded model can actually produce.
 *
 * Simulate Sign uses the fallback directly via getFallbackModel()
 * (procedural demo motion is the only input the fallback recognises);
 * the live camera path uses getModel() (LexiSight first).
 */
import './lexisightLayers'
import { FRAMES, FEATURE_DIM } from './holistic'
import {
  LEXISIGHT_HAND_DIM,
  LEXISIGHT_TEMPERATURE,
  applyTemperature,
  labelsFromMap,
  patchTopology,
  sliceHands,
  topK,
} from './lexisight'

export interface InferenceResult {
  word: string
  confidence: number
  latencyMs: number
  top3: { word: string; confidence: number }[]
}

export type ModelSource = 'lexisight' | 'synthetic' | 'none'

export interface LoadedModel {
  source: ModelSource
  labels: string[]
  numClasses: number
  /** Run one [1,30,225] inference (LexiSight slices the hands internally). */
  predict: (tensor: Float32Array) => Promise<InferenceResult>
  /** Single-frame fingerspelling inference (LexiSight static model only). */
  predictStatic?: (frame225: Float32Array) => Promise<InferenceResult>
}

type Tfjs = typeof import('@tensorflow/tfjs')
type LayersModel = import('@tensorflow/tfjs').LayersModel

let modelPromise: Promise<LoadedModel> | null = null
let fallbackPromise: Promise<LoadedModel> | null = null

function baseUrl(): string {
  return import.meta.env.BASE_URL
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url)
    if (!r.ok) return null
    return (await r.json()) as T
  } catch {
    return null
  }
}

async function fetchBytes(url: string): Promise<ArrayBuffer | null> {
  try {
    const r = await fetch(url)
    if (!r.ok) return null
    return await r.arrayBuffer()
  } catch {
    return null
  }
}

async function loadTfjs(): Promise<Tfjs> {
  const tf = await import('@tensorflow/tfjs')
  await tf.ready()
  return tf
}

interface WeightsManifestEntry {
  name: string
  shape: number[]
  dtype: string
}

interface ModelJsonFile {
  modelTopology: unknown
  weightsManifest: { paths: string[]; weights: WeightsManifestEntry[] }[]
}

/**
 * Assign .bin weights to a model built from a patched topology.
 * The LexiSight manifest uses bare duplicate names ('kernel', 'bias'),
 * so weights are matched by SHAPE-group FIFO order: TF.js orders
 * model.weights trainable-first (BatchNorm moving stats last), while
 * the manifest interleaves them per layer — within each shape group
 * the relative order is identical in both listings (verified in Node
 * for both models: 16/16 and 41/41 weights, all bytes consumed).
 */
export function assignWeightsFromManifest(
  tf: Tfjs,
  model: LayersModel,
  manifest: ModelJsonFile,
  weightData: ArrayBuffer,
): void {
  const specs = manifest.weightsManifest.flatMap((g) => g.weights)
  const f32 = new Float32Array(weightData)
  let off = 0
  const manifestTensors = specs.map((sp) => {
    const n = sp.shape.reduce((a, b) => a * b, 1)
    const t = tf.tensor(f32.slice(off, off + n), sp.shape, 'float32')
    off += n
    return t
  })
  if (off * 4 !== weightData.byteLength) {
    throw new Error(`weight bytes mismatch: consumed ${off * 4}, file has ${weightData.byteLength}`)
  }
  const byShape = new Map<string, import('@tensorflow/tfjs').Tensor[]>()
  specs.forEach((sp, i) => {
    const k = JSON.stringify(sp.shape)
    if (!byShape.has(k)) byShape.set(k, [])
    byShape.get(k)!.push(manifestTensors[i])
  })
  const tensors = model.weights.map((v) => {
    const k = JSON.stringify(v.shape)
    const q = byShape.get(k)
    if (!q || q.length === 0) {
      throw new Error(`no manifest tensor for variable ${v.name} shape ${k}`)
    }
    return q.shift()!
  })
  for (const [k, q] of byShape) {
    if (q.length > 0) throw new Error(`unconsumed manifest tensors for shape ${k}: ${q.length}`)
  }
  if (specs.length !== model.weights.length) {
    throw new Error(`weight count mismatch: manifest ${specs.length} vs model ${model.weights.length}`)
  }
  model.setWeights(tensors)
}

/**
 * Build a LayersModel from parsed LexiSight artifacts (no fetching).
 * Exported for tests: the browser path below just fetches the same files.
 */
export async function loadLexisightFromArtifacts(
  tf: Tfjs,
  modelJson: ModelJsonFile,
  weightData: ArrayBuffer,
): Promise<LayersModel> {
  const topology = patchTopology(modelJson)
  const model = await tf.models.modelFromJSON({ modelTopology: topology as never })
  assignWeightsFromManifest(tf, model, modelJson, weightData)
  return model
}

function rankProbs(probs: ArrayLike<number>, labels: string[]): InferenceResult {
  const top = topK(probs, labels, 3)
  return { word: top[0].word, confidence: top[0].confidence, latencyMs: 0, top3: top }
}

/**
 * Load ONLY the bundled procedural fallback model
 * (`models/omnisign_tfjs/` — Conv1D, trained on procedural 225-dim
 * sequences for the 88-word conversation vocabulary). This is the model
 * Simulate Sign must use: procedural motion is the only input it can
 * recognise (the real LexiSight weights were trained on real signers
 * and return low-confidence garbage on procedural sequences — live
 * test 23 Nov 2026: TOMORROW -> Afternoon 22.7%, THANKYOU ->
 * Afternoon 29.3%). It is NOT trained on real signers; the live
 * camera path must never use it.
 */
async function loadFallbackModel(tf: Tfjs): Promise<LoadedModel> {
  const labels = await fetchJson<string[]>(`${baseUrl()}models/omnisign_tfjs/labels.json`)
  if (!labels) {
    throw new Error('Fallback model labels not found (models/omnisign_tfjs/labels.json).')
  }
  const model = await tf.loadLayersModel(`${baseUrl()}models/omnisign_tfjs/model.json`)
  return {
    source: 'synthetic',
    labels,
    numClasses: labels.length,
    predict: async (tensor) => {
      const input = tf.tensor(tensor, [1, FRAMES, FEATURE_DIM], 'float32')
      const t0 = performance.now()
      const out = model.predict(input) as import('@tensorflow/tfjs').Tensor
      const data = (await out.data()) as Float32Array
      const latencyMs = performance.now() - t0
      input.dispose()
      out.dispose()
      // Fallback output is already a softmax — use it directly.
      return { ...rankProbs(data, labels), latencyMs }
    },
  }
}

/** Simulate Sign's model: the procedural fallback only, never LexiSight. */
export function getFallbackModel(): Promise<LoadedModel> {
  if (!fallbackPromise) {
    fallbackPromise = (async () => {
      const tf = await loadTfjs()
      return loadFallbackModel(tf)
    })()
    fallbackPromise.catch(() => {
      fallbackPromise = null
    })
  }
  return fallbackPromise
}

export function getModel(): Promise<LoadedModel> {
  if (!modelPromise) {
    modelPromise = (async (): Promise<LoadedModel> => {
      const tf = await loadTfjs()
      // 1) Real LexiSight weights.
      const [temporalJson, temporalBin, labelMap] = await Promise.all([
        fetchJson<ModelJsonFile>(`${baseUrl()}models/lexisight/tfjs_model/model.json`),
        fetchBytes(`${baseUrl()}models/lexisight/tfjs_model/group1-shard1of1.bin`),
        fetchJson<Record<string, string>>(`${baseUrl()}models/lexisight/label_map.json`),
      ])
      if (temporalJson && temporalBin && labelMap) {
        try {
          const temporal = await loadLexisightFromArtifacts(tf, temporalJson, temporalBin)
          const labels = labelsFromMap(labelMap)
          if (temporal.inputs[0].shape?.join(',') !== ',30,126' && temporal.inputs[0].shape) {
            console.warn('[omnisign] LexiSight temporal input shape is not [null,30,126]:', temporal.inputs[0].shape)
          }
          // Best-effort static fingerspelling model.
          let staticModel: LayersModel | null = null
          let staticLabels: string[] = []
          try {
            const [staticJson, staticBin, staticMap] = await Promise.all([
              fetchJson<ModelJsonFile>(`${baseUrl()}models/lexisight/static_tfjs_model/model.json`),
              fetchBytes(`${baseUrl()}models/lexisight/static_tfjs_model/group1-shard1of1.bin`),
              fetchJson<Record<string, string>>(`${baseUrl()}models/lexisight/static_label_map.json`),
            ])
            if (staticJson && staticBin && staticMap) {
              staticModel = await loadLexisightFromArtifacts(tf, staticJson, staticBin)
              staticLabels = labelsFromMap(staticMap)
            }
          } catch (e) {
            console.warn('[omnisign] LexiSight static model unavailable, fingerspelling disabled:', e)
          }
          const loaded: LoadedModel = {
            source: 'lexisight',
            labels,
            numClasses: labels.length,
            predict: async (tensor) => {
              // LexiSight takes hand landmarks only: slice dims 99..224.
              const hands = sliceHands(tensor, FRAMES)
              const input = tf.tensor(hands, [1, FRAMES, LEXISIGHT_HAND_DIM], 'float32')
              const t0 = performance.now()
              const out = temporal.predict(input) as import('@tensorflow/tfjs').Tensor
              const data = (await out.data()) as Float32Array
              const latencyMs = performance.now() - t0
              input.dispose()
              out.dispose()
              // Output is already a softmax; apply only temperature scaling.
              const probs = applyTemperature(data, LEXISIGHT_TEMPERATURE)
              return { ...rankProbs(probs, labels), latencyMs }
            },
          }
          if (staticModel) {
            const sm = staticModel
            loaded.predictStatic = async (frame225) => {
              const hands = sliceHands(frame225, 1)
              const input = tf.tensor(hands, [1, LEXISIGHT_HAND_DIM], 'float32')
              const t0 = performance.now()
              const out = sm.predict(input) as import('@tensorflow/tfjs').Tensor
              const data = (await out.data()) as Float32Array
              const latencyMs = performance.now() - t0
              input.dispose()
              out.dispose()
              return { ...rankProbs(data, staticLabels), latencyMs }
            }
          }
          return loaded
        } catch (e) {
          console.warn('[omnisign] LexiSight weights present but unloadable, using fallback:', e)
        }
      }
      // 2) Bundled procedural fallback (LexiSight missing or unloadable).
      return loadFallbackModel(tf)
    })()
    modelPromise.catch(() => {
      modelPromise = null
    })
  }
  return modelPromise
}

/** Speak text via the Web Speech Synthesis API. */
export function speak(text: string): void {
  try {
    if (!('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.rate = 1
    window.speechSynthesis.speak(u)
  } catch {
    /* TTS unavailable — non-fatal */
  }
}
