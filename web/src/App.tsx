import { useCallback, useEffect, useRef, useState } from 'react'
import HandAvatar from './components/HandAvatar'
import SkeletonView from './components/SkeletonView'
import {
  CONFIDENCE_THRESHOLD,
  EMERGENCY_WORDS,
  VOCAB_WORDS,
  displayWord,
} from './lib/constants'
import { Debouncer } from './lib/debounce'
import { tokenizeGloss } from './lib/gloss'
import {
  HolisticBuffer,
  drawHolisticSkeleton,
  extractFeatures,
} from './lib/holistic'
import { generateWordSequence } from './lib/motionGen.mjs'
import { getFallbackModel, getModel, speak, type InferenceResult, type ModelSource } from './lib/tfjsClient'

type Mode = 'standard' | 'performance'
type Status = 'idle' | 'loading-model' | 'ready' | 'error'
type Turn = { from: 'sign' | 'speech'; text: string; time: string }

export default function App() {
  const [mode, setMode] = useState<Mode>('standard')
  const [status, setStatus] = useState<Status>('idle')
  const [statusMsg, setStatusMsg] = useState(
    'Model not loaded yet — run a simulation or start the camera.',
  )
  const [modelSource, setModelSource] = useState<ModelSource>('none')
  const [emergencyMode, setEmergencyMode] = useState(false)
  const [avatar3d, setAvatar3d] = useState(true)

  const [simWord, setSimWord] = useState<string>('HELLO')
  const [detected, setDetected] = useState<InferenceResult | null>(null)
  const [committed, setCommitted] = useState<string | null>(null)
  const [cameraOn, setCameraOn] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [fps, setFps] = useState(0)

  const [turns, setTurns] = useState<Turn[]>([])
  const [glossTokens, setGlossTokens] = useState<string[]>([])
  const [speechText, setSpeechText] = useState('')
  const [interimText, setInterimText] = useState('')
  const [replayKey, setReplayKey] = useState(0)
  const [listening, setListening] = useState(false)
  const [speechSupported, setSpeechSupported] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef(0)
  const debouncerRef = useRef(new Debouncer())
  const bufferRef = useRef(new HolisticBuffer())
  const lastFrameRef = useRef(0)
  const frameCountRef = useRef(0)
  const fpsTimeRef = useRef(performance.now())
  const runningRef = useRef(false)
  const emergencyRef = useRef(emergencyMode)
  emergencyRef.current = emergencyMode
  const recRef = useRef<SpeechRecognitionLike | null>(null)
  const transcriptEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setSpeechSupported(
      'SpeechRecognition' in window || 'webkitSpeechRecognition' in window,
    )
    return () => {
      cancelAnimationFrame(rafRef.current)
      recRef.current?.stop()
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  const addTurn = useCallback((from: Turn['from'], text: string) => {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    setTurns((prev) => [...prev.slice(-49), { from, text, time }])
  }, [])

  const commitWord = useCallback(
    (word: string, confidence: number) => {
      void confidence
      setCommitted(word)
      const spoken = displayWord(word)
      speak(spoken)
      addTurn('sign', spoken)
    },
    [addTurn],
  )

  /** Run the TF.js model on one full 30-frame buffer. */
  const inferBuffer = useCallback(async () => {
    if (!bufferRef.current.isFull) return
    try {
      const model = await getModel()
      setModelSource(model.source)
      let result = await model.predict(bufferRef.current.toTensor())
      // Emergency mode: prefer emergency words — if the top prediction is
      // not an emergency word but one is in the top 3, surface that instead.
      if (emergencyRef.current && !EMERGENCY_WORDS.includes(result.word)) {
        const emergencyPick = result.top3.find((t) => EMERGENCY_WORDS.includes(t.word))
        if (emergencyPick) {
          result = { ...result, word: emergencyPick.word, confidence: emergencyPick.confidence }
        }
      }
      setDetected(result)
      setStatus('ready')
      setStatusMsg(
        `Model ready (${model.source === 'lexisight' ? 'LexiSight weights' : 'synthetic-trained fallback'}) — last inference ${result.latencyMs.toFixed(1)} ms`,
      )
      const committedWord = debouncerRef.current.push(result.word, result.confidence)
      if (committedWord) commitWord(committedWord, result.confidence)
    } catch (err) {
      setStatus('error')
      setStatusMsg(`Inference failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }, [commitWord])

  /**
   * Demo mode: feed a procedural 30-frame landmark sequence for the
   * sign, recognised by the PROCEDURAL demo model only
   * (getFallbackModel). The real LexiSight weights must NOT be used
   * here: they were trained on real signers and return low-confidence
   * garbage on procedural motion (live test 23 Nov 2026: TOMORROW ->
   * Afternoon 22.7%, THANKYOU -> Afternoon 29.3%, nothing spoken).
   * Unlike the camera path, a demo sign commits on a single
   * high-confidence recognition — there is no 5-frame consensus to
   * build up from a generated sequence.
   */
  const simulateSign = useCallback(async () => {
    setStatus('loading-model')
    setStatusMsg('Loading demo model… (simulated landmarks, not a real signer)')
    setCommitted(null)
    debouncerRef.current.reset()
    try {
      const model = await getFallbackModel()
      setModelSource(model.source)
      // Procedural landmarks in the 225-dim format (seeded demo).
      const raw = generateWordSequence(simWord, Math.floor(Math.random() * 1e6))
      const canvas = canvasRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        if (ctx) drawHolisticSkeleton(ctx, raw.slice(29 * 225), canvas.width, canvas.height)
      }
      const result = await model.predict(raw)
      setDetected(result)
      setStatus('ready')
      setStatusMsg(
        `Demo sign recognised by the procedural demo model — last inference ${result.latencyMs.toFixed(1)} ms. Live camera uses LexiSight weights.`,
      )
      if (result.confidence >= CONFIDENCE_THRESHOLD) {
        commitWord(result.word, result.confidence)
      }
    } catch (err) {
      setStatus('error')
      setStatusMsg(
        `Could not load the demo model: ${err instanceof Error ? err.message : String(err)}`,
      )
    }
  }, [simWord, commitWord])

  /** Webcam loop: MediaPipe Holistic -> 225-dim features -> buffer -> infer. */
  const startCamera = useCallback(async () => {
    setCameraError(null)
    setStatus('loading-model')
    setStatusMsg('Loading holistic tracking model…')
    try {
      const { FilesetResolver, HolisticLandmarker } = await import('@mediapipe/tasks-vision')
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
      )
      const landmarker = await HolisticLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
      })
      // Warm the TF.js model in parallel so the first inference is fast.
      void getModel().then((m) => setModelSource(m.source)).catch(() => undefined)
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
        const features = extractFeatures({
          poseLandmarks: res.poseLandmarks?.[0] ?? null,
          leftHandLandmarks: res.leftHandLandmarks?.[0] ?? null,
          rightHandLandmarks: res.rightHandLandmarks?.[0] ?? null,
        })
        const hasSignal =
          (res.poseLandmarks?.length ?? 0) > 0 ||
          (res.leftHandLandmarks?.length ?? 0) > 0 ||
          (res.rightHandLandmarks?.length ?? 0) > 0
        if (canvas && ctx) drawHolisticSkeleton(ctx, features, canvas.width, canvas.height)
        if (hasSignal) {
          bufferRef.current.push(features)
          if (bufferRef.current.isFull) void inferBuffer()
        }
      }
      rafRef.current = requestAnimationFrame(loop)
    } catch (err) {
      setCameraError(
        err instanceof Error ? err.message : 'Camera or holistic tracking failed to start.',
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

  const submitSpeechText = useCallback(
    (text: string) => {
      const tokens = tokenizeGloss(text)
      setGlossTokens(tokens)
      setReplayKey((k) => k + 1)
      if (text.trim()) addTurn('speech', text.trim())
    },
    [addTurn],
  )

  const signEmergencyWord = useCallback(
    (word: string) => {
      setGlossTokens([word])
      setReplayKey((k) => k + 1)
      addTurn('speech', displayWord(word))
    },
    [addTurn],
  )

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
      // Stop the live recognition (previously this only flipped the
      // button state and left the recogniser running).
      recRef.current?.stop()
      recRef.current = null
      setListening(false)
      setInterimText('')
      return
    }
    try {
      const rec = new Rec()
      rec.lang = 'en-US'
      rec.interimResults = true
      rec.onresult = (e: SpeechRecognitionEventLike) => {
        let interim = ''
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const res = e.results[i]
          if (res.isFinal) {
            const transcript = res[0].transcript
            setSpeechText(transcript)
            setInterimText('')
            submitSpeechText(transcript)
          } else {
            interim += res[0].transcript
          }
        }
        if (interim) setInterimText(interim)
      }
      rec.onend = () => {
        setListening(false)
        setInterimText('')
      }
      rec.onerror = () => {
        setListening(false)
        setInterimText('')
      }
      recRef.current = rec
      rec.start()
      setListening(true)
    } catch {
      setListening(false)
    }
  }, [listening, submitSpeechText])

  const vocabOptions = emergencyMode ? EMERGENCY_WORDS : VOCAB_WORDS

  return (
    <div className="min-h-screen bg-ink text-slate-100">
      <header className="border-b border-slate-800 px-4 py-3 sm:px-6">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              OmniSign <span className="text-accent">— Everyday Conversation, Signed &amp; Spoken</span>
            </h1>
            <p className="text-xs text-slate-400">
              100% on-device · no video ever leaves your browser · {VOCAB_WORDS.length} conversation
              signs + A–Z finger-spelling · emergency mode included
            </p>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2 text-xs">
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
            {modelSource !== 'none' && (
              <span
                className={`rounded-full px-2.5 py-1 ${
                  modelSource === 'lexisight'
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-amber-500/20 text-amber-300'
                }`}
                title={
                  modelSource === 'lexisight'
                    ? 'Real LexiSight weights loaded — live camera recognition'
                    : 'Procedural demo model — trained on simulated motion, not real signers'
                }
              >
                {modelSource === 'lexisight' ? 'LexiSight weights' : 'Synthetic model'}
              </span>
            )}
            <button
              onClick={() => setEmergencyMode((v) => !v)}
              className={`rounded-full px-3 py-1 font-semibold ${
                emergencyMode ? 'bg-rose-600 text-white' : 'bg-slate-700/60 text-slate-300'
              }`}
              aria-pressed={emergencyMode}
            >
              {emergencyMode ? '🚨 Emergency mode ON' : 'Emergency mode'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-4 p-4 sm:p-6 lg:grid-cols-[1fr_340px_1fr]">
        {/* ============ LEFT: Sign-to-Speech ============ */}
        <section className="rounded-2xl border border-slate-800 bg-panel p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-semibold">Sign → Speech</h2>
            <span className="text-xs text-slate-400">MediaPipe Holistic · TF.js · 30×225</span>
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
              Demo mode — no webcam needed · simulated landmarks · procedural demo model
            </p>
            <p className="mb-2 text-[11px] text-slate-500">
              Simulated sign (demo motion) — recognised by the bundled demo model. Live camera
              uses the real LexiSight weights.
            </p>
            <div className="flex flex-wrap gap-2">
              <select
                value={simWord}
                onChange={(e) => setSimWord(e.target.value)}
                className="max-w-[220px] rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                aria-label="Sign to simulate"
              >
                {vocabOptions.map((s) => (
                  <option key={s} value={s}>{displayWord(s)}</option>
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
              <p className="truncate text-lg font-bold text-cyan-300">{detected ? displayWord(detected.word) : '—'}</p>
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
                  {displayWord(t.word)} {(t.confidence * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          )}

          {committed && (
            <p className="mt-3 rounded-lg bg-emerald-500/15 px-3 py-2 text-sm text-emerald-300">
              🔊 Spoken: <strong>{displayWord(committed)}</strong>
            </p>
          )}
          <p className="mt-2 text-xs text-slate-400">{statusMsg}</p>
          <p className="mt-1 text-[11px] text-slate-500">
            Live camera: a word is spoken only when confidence &gt; 85% for more than 5
            consecutive frames. Demo mode commits a single high-confidence recognition.
            Raw video frames stay in browser memory and are discarded after inference.
          </p>
        </section>

        {/* ============ MIDDLE: Conversation transcript ============ */}
        <section className="flex flex-col rounded-2xl border border-slate-800 bg-panel p-4">
          <h2 className="mb-3 text-base font-semibold">Conversation</h2>
          <div className="min-h-[320px] flex-1 space-y-2 overflow-y-auto rounded-xl bg-slate-900/60 p-3">
            {turns.length === 0 ? (
              <p className="text-sm text-slate-500">
                Nothing yet. Sign in front of the camera, simulate a sign, or type/speak a reply —
                the conversation will appear here, both sides.
              </p>
            ) : (
              turns.map((t, i) => (
                <div
                  key={i}
                  className={`max-w-[90%] rounded-xl px-3 py-2 text-sm ${
                    t.from === 'sign'
                      ? 'ml-auto bg-cyan-500/15 text-cyan-200 ring-1 ring-cyan-500/30'
                      : 'bg-slate-800 text-slate-200'
                  }`}
                >
                  <p className="mb-0.5 text-[10px] uppercase tracking-wide opacity-60">
                    {t.from === 'sign' ? '🤟 Signed' : '🗣️ Spoken'} · {t.time}
                  </p>
                  {t.text}
                </div>
              ))
            )}
            <div ref={transcriptEndRef} />
          </div>

          {emergencyMode && (
            <div className="mt-3 rounded-xl border border-rose-500/40 bg-rose-500/10 p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-rose-300">
                Emergency quick signs — tap to sign
              </p>
              <div className="flex flex-wrap gap-1.5">
                {EMERGENCY_WORDS.map((w) => (
                  <button
                    key={w}
                    onClick={() => signEmergencyWord(w)}
                    className="rounded-full bg-rose-600/80 px-2.5 py-1 text-xs font-medium text-white hover:bg-rose-500"
                  >
                    {displayWord(w)}
                  </button>
                ))}
              </div>
            </div>
          )}
          <button
            onClick={() => setTurns([])}
            className="mt-3 self-start rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-400 hover:bg-slate-800"
          >
            Clear conversation
          </button>
        </section>

        {/* ============ RIGHT: Speech-to-Sign ============ */}
        <section className="rounded-2xl border border-slate-800 bg-panel p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-semibold">Speech → Sign</h2>
            <span className="text-xs text-slate-400">Rigged 3D hands · 60 FPS</span>
          </div>

          {/* Mic-first: tap the big mic and speak — the hands sign it live. */}
          <div className="mb-3 flex items-center gap-3 rounded-xl border border-slate-700/80 bg-slate-900/60 p-3">
            <button
              onClick={toggleListening}
              disabled={!speechSupported}
              aria-label={listening ? 'Stop listening' : 'Tap to speak'}
              aria-pressed={listening}
              className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-full text-2xl shadow-lg transition ${
                listening
                  ? 'animate-pulse bg-rose-600 text-white ring-4 ring-rose-500/40'
                  : 'bg-cyan-500 text-slate-950 ring-4 ring-cyan-500/20 hover:bg-cyan-400'
              } disabled:cursor-not-allowed disabled:opacity-40`}
            >
              🎙️
            </button>
            <div className="min-w-0">
              <p className="text-sm font-semibold">
                {listening ? 'Listening… tap the mic to stop' : 'Tap the mic and speak'}
              </p>
              <p className="truncate text-xs text-slate-400">
                {listening
                  ? interimText || 'Say something — e.g. “Hello, how are you today?”'
                  : 'Your words are signed by the 3D hands below as you speak.'}
              </p>
              {listening && interimText && (
                <p className="mt-1 truncate text-sm italic text-cyan-300">“{interimText}…”</p>
              )}
            </div>
          </div>

          <div className="aspect-video overflow-hidden rounded-xl bg-gradient-to-b from-slate-800 to-slate-900">
            {avatar3d ? (
              <HandAvatar tokens={glossTokens} replayKey={replayKey} />
            ) : (
              <SkeletonView tokens={glossTokens} replayKey={replayKey} />
            )}
          </div>
          <div className="mt-2 flex items-center rounded-lg border border-slate-700 text-xs">
            <button
              onClick={() => setAvatar3d(true)}
              className={`flex-1 rounded-l-lg px-3 py-1.5 ${avatar3d ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-300'}`}
            >
              3D hands
            </button>
            <button
              onClick={() => setAvatar3d(false)}
              className={`flex-1 rounded-r-lg px-3 py-1.5 ${!avatar3d ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-300'}`}
            >
              Skeleton view
            </button>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
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
              placeholder='Type something — try "Hello, how are you today?"'
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
              Filler words (is, are, the, a, an, am, do, will…) are removed and WH-words move to
              the end, following sign-language gloss order. Words outside the {VOCAB_WORDS.length}
              -word lexicon are finger-spelled A–Z.
            </p>
          </div>
        </section>
      </main>

      <footer className="mx-auto max-w-7xl px-6 pb-6 text-center text-[11px] text-slate-500">
        OmniSign v2 — everyday conversation first, emergency mode as a feature · MediaPipe Holistic
        (33 pose + 2×21 hand landmarks) · TF.js · LexiSight hands-only 126-dim model (30×225 capture) · Pose lexicon avatar ·
        All processing happens on your device. Live camera recognition uses the real LexiSight
        weights (263 ISL words) — live human-signer accuracy not yet verified. Simulate Sign is a
        demo: procedural motion recognised by the bundled demo model, not by LexiSight.
        <br />
        3D hand model: “Rigged Hand” by J-Toastie, CC-BY, via Poly Pizza (
        <a
          className="underline hover:text-slate-300"
          href="https://poly.pizza/m/BEy8jbxm6A"
          target="_blank"
          rel="noreferrer"
        >
          poly.pizza/m/BEy8jbxm6A
        </a>
        ).
      </footer>
    </div>
  )
}

interface SpeechRecognitionResultLike {
  isFinal: boolean
  [j: number]: { transcript: string }
}

interface SpeechRecognitionEventLike {
  resultIndex: number
  results: { length: number; [i: number]: SpeechRecognitionResultLike }
}

interface SpeechRecognitionLike {
  lang: string
  interimResults: boolean
  onresult: ((e: SpeechRecognitionEventLike) => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
  start: () => void
  stop: () => void
}
