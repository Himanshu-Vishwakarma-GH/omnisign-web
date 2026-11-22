import * as ort from 'onnxruntime-web'
import { FRAMES, LANDMARKS, SIGNS, type SignWord } from './constants'

export interface InferenceResult {
  word: SignWord
  confidence: number
  latencyMs: number
  top3: { word: SignWord; confidence: number }[]
}

let sessionPromise: Promise<ort.InferenceSession> | null = null

function modelUrl(): string {
  // import.meta.env.BASE_URL is '/omnisign-web/' on Pages, '/' locally.
  return `${import.meta.env.BASE_URL}models/omnisign_bilstm.onnx`
}

export async function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      // WASM execution provider; single thread keeps low-end devices smooth.
      ort.env.wasm.numThreads = 1
      try {
        return await ort.InferenceSession.create(modelUrl(), {
          executionProviders: ['wasm'],
        })
      } catch (err) {
        sessionPromise = null
        throw err
      }
    })()
  }
  return sessionPromise
}

/** Run one [1,30,48,3] inference. `tensor` is a flat Float32Array (4320). */
export async function runInference(tensor: Float32Array): Promise<InferenceResult> {
  const session = await getSession()
  const input = new ort.Tensor('float32', tensor, [1, FRAMES, LANDMARKS, 3])
  const t0 = performance.now()
  const out = await session.run({ input })
  const latencyMs = performance.now() - t0
  const logits = out.logits.data as Float32Array
  const max = Math.max(...logits)
  const exps = Array.from(logits, (v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map((e) => e / sum)
  const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0])
  return {
    word: SIGNS[order[0][1]],
    confidence: order[0][0],
    latencyMs,
    top3: order.slice(0, 3).map(([p, i]) => ({ word: SIGNS[i], confidence: p })),
  }
}

/** Speak a word via the Web Speech Synthesis API. */
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
