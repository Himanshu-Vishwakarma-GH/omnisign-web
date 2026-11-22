import { useCallback, useEffect, useRef, useState } from 'react'
import Avatar from './components/Avatar'
import { LandmarkBuffer } from './lib/buffer'
import { CONFIDENCE_THRESHOLD, FRAMES, SIGNS, type SignWord } from './lib/constants'
import { Debouncer } from './lib/debounce'
import { tokenizeGloss } from './lib/gloss'
import { normalizeLandmarks } from './lib/normalization'
import { getSession, runInference, speak, type InferenceResult } from './lib/onnxClient'
import { generateSyntheticSequence } from './lib/synthetic'

type Mode = 'standard' | 'performance'
type Status = 'idle' | 'loading-model' | 'ready' | 'error'

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: Float32Array,
  w: number,
  h: number,
) {
  ctx.clearRect(0, 0, w, h)
  ctx.strokeStyle = '#22d3ee'
  ctx.lineWidth = 2
  for (const handOffset of [0, 21]) {
    for (const [a, b] of HAND_CONNECTIONS) {
      const ax = landmarks[(handOffset + a) * 3] * w
      const ay = landmarks[(handOffset + a) * 3 + 1] * h
      const bx = landmarks[(handOffset + b) * 3] * w
      const by = landmarks[(handOffset + b) * 3 + 1] * h
      ctx.beginPath()
      ctx.moveTo(ax, ay)
      ctx.lineTo(bx, by)
      ctx.stroke()
    }
    for (let i = 0; i < 21; i++) {
      const x = landmarks[(handOffset + i) * 3] * w
      const y = landmarks[(handOffset + i) * 3 + 1] * h
      ctx.fillStyle = i === 0 ? '#f472b6' : '#22d3ee'
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

export default function App() {
  const [mode, setMode] = useState<Mode>('standard')
  const [status, setStatus] = useState<Status>('idle')
  const [statusMsg, setStatusMsg] = useState('Model not loaded yet — run a simulation or start the camera.')
  const [simWord, setSimWord] = useState<SignWord>('Help')
  const [detected, setDetected] = useState<InferenceResult | null>(null)
  const [committed, setCommitted] = useState<string | null>(null)
  const [cameraOn, setCameraOn] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [fps, setFps] = useState(0)

  const [glossTokens, setGlossTokens] = useState<string[]>([])
  const [speechText, setSpeechText] = useState('')
  const [replayKey, setReplayKey] = useState(0)
  const [listening, setListening] = useState(false)
  const [speechSupported, setSpeechSupported] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef(0)
  const debouncerRef = useRef(new Debouncer())
  const bufferRef = useRef(new LandmarkBuffer())
  const lastFrameRef = useRef(0)
  const frameCountRef = useRef(0)
  const fpsTimeRef = useRef(performance.now())
  const runningRef = useRef(false)

  useEffect(() => {
    setSpeechSupported(
      'SpeechRecognition' in window || 'webkitSpeechRecognition' in window,
    )
    return () => {
      cancelAnimationFrame(rafRef.current)
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const commitWord = useCallback((word: string, confidence: number) => {
    setCommitted(word)
    speak(word)
    // Keep the committed word visible; the next detection replaces it.
    void confidence
  }, [])

  /** Run the ONNX model on one normalized 30-frame buffer. */
  const inferBuffer = useCallback(async () => {
    if (!bufferRef.current.isFull) return
    try {
      const result = await runInference(bufferRef.current.toTensor())
      setDetected(result)
      setStatus('ready')
      setStatusMsg(`Model ready — last inference ${result.latencyMs.toFixed(1)} ms`)
      const committedWord = debouncerRef.current.push(result.word, result.confidence)
      if (committedWord) commitWord(committedWord, result.confidence)
    } catch (err) {
      setStatus('error')
      setStatusMsg(`Inference failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }, [commitWord])

  /** Demo mode: feed a synthetic 30-frame sequence for the chosen sign. */
  const simulateSign = useCallback(async () => {
    setStatus('loading-model')
    setStatusMsg(`Loading ONNX model… (first run downloads the WASM runtime)`)
    setCommitted(null)
    debouncerRef.current.reset()
    try {
      await getSession()
      const raw = generateSyntheticSequence(simWord, Math.floor(Math.random() * 1e6))
      // Draw the last frame's skeleton on the canvas so the demo is visible.
      const canvas = canvasRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        if (ctx) {
          const lastFrame = raw.slice((FRAMES - 1) * 144)
          drawSkeleton(ctx, lastFrame, canvas.width, canvas.height)
        }
      }
      const buf = new LandmarkBuffer()
      for (let f = 0; f < FRAMES; f++) {
        buf.push(normalizeLandmarks(raw.slice(f * 144, (f + 1) * 144)))
      }
      bufferRef.current = buf
      await inferBuffer()
    } catch (err) {
      setStatus('error')
      setStatusMsg(
        `Could not load the ONNX model: ${err instanceof Error ? err.message : String(err)} — check that web/public/models/omnisign_bilstm.onnx exists in the build.`,
      )
    }
  }, [simWord, inferBuffer])

  /** Webcam loop: MediaPipe HandLandmarker -> normalize -> buffer -> infer. */
  const startCamera = useCallback(async () => {
    setCameraError(null)
    setStatus('loading-model')
    setStatusMsg('Loading hand-tracking model…')
    try {
      const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision')
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
      )
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      })
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      })
      streamRef.current = stream
      const video = videoRef.current
      if (!video) return
      video.srcObject = stream
      await video.play()
      setCameraOn(true)
      setStatus('ready')
      setStatusMsg('Camera live — sign in front of the webcam.')
      runningRef.current = true
      const interval = mode === 'performance' ? 1000 / 15 : 1000 / 30

      const loop = async (now: number) => {
        if (!runningRef.current) return
        rafRef.current = requestAnimationFrame(loop)
        if (now - lastFrameRef.current < interval) return
        lastFrameRef.current = now
        frameCountRef.current += 1
        if (now - fpsTimeRef.current >= 1000) {
          setFps(frameCountRef.current)
          frameCountRef.current = 0
          fpsTimeRef.current = now
        }
        if (video.readyState < 2) return
        const res = landmarker.detectForVideo(video, now)
        const canvas = canvasRef.current
        const ctx = canvas?.getContext('2d')
        const frame = new Float32Array(144)
        // Default: rest pose at the detected wrist positions is not known
        // without detections, so only run inference while hands are visible.
        if (res.landmarks && res.landmarks.length > 0) {
          // Fill right hand (landmarks[0]) and left hand (landmarks[1] if present).
          const fillHand = (lm: { x: number; y: number; z: number }[], offset: number) => {
            for (let i = 0; i < 21; i++) {
              frame[(offset + i) * 3] = lm[i].x
              frame[(offset + i) * 3 + 1] = lm[i].y
              frame[(offset + i) * 3 + 2] = lm[i].z ?? 0
            }
          }
          fillHand(res.landmarks[0], 0)
          if (res.landmarks[1]) fillHand(res.landmarks[1], 21)
          else {
            // Mirror the right hand as a neutral left hand.
            for (let i = 0; i < 21; i++) {
              frame[(21 + i) * 3] = 1 - frame[i * 3]
              frame[(21 + i) * 3 + 1] = frame[i * 3 + 1]
              frame[(21 + i) * 3 + 2] = frame[i * 3 + 2]
            }
          }
          // Upper body: derive from wrist/shoulder geometry (fixed torso).
          const body: [number, number, number][] = [
            [0.35, 0.25, 0], [0.65, 0.25, 0],
            [0.3, 0.45, 0], [0.7, 0.45, 0],
            [frame[21 * 3], frame[21 * 3 + 1], 0],
            [frame[0], frame[1], 0],
          ]
          body.forEach(([x, y, z], k) => {
            frame[(42 + k) * 3] = x
            frame[(42 + k) * 3 + 1] = y
            frame[(42 + k) * 3 + 2] = z
          })
          if (canvas && ctx) drawSkeleton(ctx, frame, canvas.width, canvas.height)
          bufferRef.current.push(normalizeLandmarks(frame))
          if (bufferRef.current.isFull) void inferBuffer()
        }
      }
      rafRef.current = requestAnimationFrame(loop)
    } catch (err) {
      setCameraError(
        err instanceof Error ? err.message : 'Camera or hand-tracking failed to start.',
      )
      setStatus('error')
      setStatusMsg('Camera unavailable — Demo Mode (Simulate Sign) still works without a webcam.')
    }
  }, [mode, inferBuffer])

  const stopCamera = useCallback(() => {
    runningRef.current = false
    cancelAnimationFrame(rafRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setCameraOn(false)
  }, [])

  const submitSpeechText = useCallback((text: string) => {
    const tokens = tokenizeGloss(text)
    setGlossTokens(tokens)
    setReplayKey((k) => k + 1)
  }, [])

  const toggleListening = useCallback(() => {
    const w = window as unknown as {
      SpeechRecognition?: new () => SpeechRecognitionLike
      webkitSpeechRecognition?: new () => SpeechRecognitionLike
    }
    const Rec = w.SpeechRecognition ?? w.webkitSpeechRecognition
    if (!Rec) {
      setSpeechSupported(false)
      return
    }
    if (listening) {
      setListening(false)
      return
    }
    try {
      const rec = new Rec()
      rec.lang = 'en-US'
      rec.interimResults = false
      rec.onresult = (e: { results: { [i: number]: { [j: number]: { transcript: string } } } }) => {
        const transcript = e.results[0][0].transcript
        setSpeechText(transcript)
        submitSpeechText(transcript)
      }
      rec.onend = () => setListening(false)
      rec.onerror = () => setListening(false)
      rec.start()
      setListening(true)
    } catch {
      setListening(false)
    }
  }, [listening, submitSpeechText])

  return (
    <div className="min-h-screen bg-ink text-slate-100">
      <header className="border-b border-slate-800 px-4 py-3 sm:px-6">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              OmniSign <span className="text-accent">— Two-Way Sign Language AI</span>
            </h1>
            <p className="text-xs text-slate-400">
              100% on-device · no video ever leaves your browser · emergency signs, spoken &amp; signed
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2 text-xs">
            <span
              className={`rounded-full px-2.5 py-1 font-medium ${
                status === 'ready'
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : status === 'error'
                    ? 'bg-rose-500/20 text-rose-300'
                    : status === 'loading-model'
                      ? 'bg-amber-500/20 text-amber-300'
                      : 'bg-slate-700/60 text-slate-300'
              }`}
            >
              {status === 'ready' ? '● Ready' : status === 'error' ? '● Error' : status === 'loading-model' ? '● Loading…' : '○ Idle'}
            </span>
            <span className="rounded-full bg-slate-700/60 px-2.5 py-1 text-slate-300">
              {mode === 'standard' ? 'Standard · 30 FPS' : 'Performance · 15 FPS'}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-4 p-4 sm:p-6 lg:grid-cols-2">
        {/* ============ LEFT: Sign-to-Speech ============ */}
        <section className="rounded-2xl border border-slate-800 bg-panel p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-semibold">Sign → Speech</h2>
            <span className="text-xs text-slate-400">MediaPipe hands · ONNX BiLSTM</span>
          </div>

          <div className="relative aspect-video overflow-hidden rounded-xl bg-black">
            <video ref={videoRef} className="absolute inset-0 h-full w-full object-cover" muted playsInline />
            <canvas ref={canvasRef} width={640} height={480} className="absolute inset-0 h-full w-full" />
            {!cameraOn && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-900/70 p-4 text-center">
                <p className="text-sm text-slate-300">Webcam off — use Demo Mode below, or start the camera.</p>
                {cameraError && <p className="max-w-md text-xs text-rose-300">{cameraError}</p>}
              </div>
            )}
            {cameraOn && (
              <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs text-emerald-300">
                ● LIVE · {fps} FPS
              </span>
            )}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            {cameraOn ? (
              <button onClick={stopCamera} className="rounded-lg bg-rose-600 px-3 py-2 text-sm font-medium hover:bg-rose-500">
                Stop camera
              </button>
            ) : (
              <button onClick={startCamera} className="rounded-lg bg-cyan-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-cyan-400">
                Start camera
              </button>
            )}
            <div className="flex items-center rounded-lg border border-slate-700 text-sm">
              <button
                onClick={() => setMode('standard')}
                className={`rounded-l-lg px-3 py-2 ${mode === 'standard' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-300'}`}
              >
                Standard · 30 FPS
              </button>
              <button
                onClick={() => setMode('performance')}
                className={`rounded-r-lg px-3 py-2 ${mode === 'performance' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-300'}`}
              >
                Performance · 15 FPS
              </button>
            </div>
          </div>

          {/* Demo mode */}
          <div className="mt-3 rounded-xl border border-slate-700/80 bg-slate-900/60 p-3">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Demo mode — no webcam needed
            </p>
            <div className="flex flex-wrap gap-2">
              <select
                value={simWord}
                onChange={(e) => setSimWord(e.target.value as SignWord)}
                className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                aria-label="Sign to simulate"
              >
                {SIGNS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
              <button
                onClick={simulateSign}
                className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium hover:bg-violet-500"
              >
                ▶ Simulate Sign
              </button>
            </div>
          </div>

          {/* Detection readout */}
          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            <div className="rounded-xl bg-slate-900/70 p-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-400">Detected sign</p>
              <p className="truncate text-lg font-bold text-cyan-300">{detected?.word ?? '—'}</p>
            </div>
            <div className="rounded-xl bg-slate-900/70 p-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-400">Confidence</p>
              <p className="text-lg font-bold">
                {detected ? `${(detected.confidence * 100).toFixed(1)}%` : '—'}
              </p>
              {detected && detected.confidence < CONFIDENCE_THRESHOLD && (
                <p className="text-[10px] text-amber-300">below 85% — not spoken</p>
              )}
            </div>
            <div className="rounded-xl bg-slate-900/70 p-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-400">Inference latency</p>
              <p className="text-lg font-bold">{detected ? `${detected.latencyMs.toFixed(1)} ms` : '—'}</p>
            </div>
          </div>

          {detected?.top3 && (
            <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
              {detected.top3.map((t) => (
                <span key={t.word} className="rounded-full bg-slate-800 px-2 py-0.5 text-slate-300">
                  {t.word} {(t.confidence * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          )}

          {committed && (
            <p className="mt-3 rounded-lg bg-emerald-500/15 px-3 py-2 text-sm text-emerald-300">
              🔊 Spoken: <strong>{committed}</strong>
            </p>
          )}
          <p className="mt-2 text-xs text-slate-400">{statusMsg}</p>
          <p className="mt-1 text-[11px] text-slate-500">
            A word is spoken only when confidence &gt; 85% for more than 5 consecutive frames.
            Raw video frames stay in browser memory and are discarded after inference.
          </p>
        </section>

        {/* ============ RIGHT: Speech-to-Sign ============ */}
        <section className="rounded-2xl border border-slate-800 bg-panel p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-semibold">Speech → Sign</h2>
            <span className="text-xs text-slate-400">Web Speech API · Three.js avatar · 60 FPS</span>
          </div>

          <div className="aspect-video overflow-hidden rounded-xl bg-gradient-to-b from-slate-800 to-slate-900">
            <Avatar tokens={glossTokens} replayKey={replayKey} />
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={toggleListening}
              disabled={!speechSupported}
              className={`rounded-lg px-4 py-2 text-sm font-medium ${
                listening
                  ? 'animate-pulse bg-rose-600 hover:bg-rose-500'
                  : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
              } disabled:cursor-not-allowed disabled:opacity-40`}
            >
              {listening ? '● Listening… (tap to stop)' : '🎙️ Mic — speak'}
            </button>
            <button
              onClick={() => glossTokens.length > 0 && setReplayKey((k) => k + 1)}
              className="rounded-lg border border-slate-700 px-4 py-2 text-sm hover:bg-slate-800"
            >
              ↻ Replay
            </button>
          </div>
          {!speechSupported && (
            <p className="mt-2 text-xs text-amber-300">
              Speech recognition isn&apos;t supported in this browser — use the text box below instead.
            </p>
          )}

          <form
            className="mt-3 flex gap-2"
            onSubmit={(e) => {
              e.preventDefault()
              submitSpeechText(speechText)
            }}
          >
            <input
              value={speechText}
              onChange={(e) => setSpeechText(e.target.value)}
              placeholder='Type something — try "What is your name?"'
              className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm outline-none focus:border-cyan-400"
              aria-label="Text to sign"
            />
            <button type="submit" className="rounded-lg bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-950 hover:bg-cyan-400">
              Sign it
            </button>
          </form>

          <div className="mt-3">
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">Gloss tokens</p>
            {glossTokens.length === 0 ? (
              <p className="text-sm text-slate-500">No gloss yet — speak or type above.</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {glossTokens.map((t, i) => (
                  <span key={`${t}-${i}`} className="rounded-full bg-cyan-500/15 px-3 py-1 text-sm font-semibold text-cyan-300 ring-1 ring-cyan-500/40">
                    {t}
                  </span>
                ))}
              </div>
            )}
            <p className="mt-2 text-[11px] text-slate-500">
              Filler words (is, are, the, a, an, am) are removed and WH-words move to the end,
              following sign-language gloss order. Unknown words are finger-spelled A–Z.
            </p>
          </div>
        </section>
      </main>

      <footer className="mx-auto max-w-7xl px-6 pb-6 text-center text-[11px] text-slate-500">
        OmniSign Phase 1 Emergency MVP — 20 emergency signs · BiLSTM (ONNX, &lt; 3 MB) · MediaPipe hands ·
        Three.js avatar · All processing happens on your device. Trained on synthetic motion patterns;
        swap in real WLASL weights for production accuracy (see README).
      </footer>
    </div>
  )
}

interface SpeechRecognitionLike {
  lang: string
  interimResults: boolean
  onresult: ((e: { results: { [i: number]: { [j: number]: { transcript: string } } } }) => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
  start: () => void
}
