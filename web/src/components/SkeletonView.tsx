import { useEffect, useRef } from 'react'
import { drawHolisticSkeleton } from '../lib/holistic'
import { loadLexicon, restFrame, type Lexicon } from '../lib/lexicon'

interface SkeletonViewProps {
  /** Gloss tokens currently being signed, e.g. ["YOUR","NAME","WHAT"]. */
  tokens: string[]
  /** Increment to replay the current tokens. */
  replayKey: number
}

const FRAME_MS = 1000 / 12 // 12 fps playback of lexicon frames

/**
 * 2D skeleton renderer for Pipeline B — the reliable layer. Draws the
 * same 225-dim lexicon frames as the 3D avatar, as a MediaPipe-style
 * skeleton (pose in cyan, hands in pink).
 */
export default function SkeletonView({ tokens, replayKey }: SkeletonViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tokensRef = useRef(tokens)
  tokensRef.current = tokens
  const replayRef = useRef(replayKey)
  replayRef.current = replayKey

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let lexicon: Lexicon | null = null
    let frames: Float32Array[] = [restFrame()]
    let frameIdx = 0
    let lastTick = performance.now()
    let lastTokens: string[] = []
    let lastReplay = replayKey
    let raf = 0
    let cancelled = false

    void loadLexicon().then((lx) => {
      if (!cancelled) lexicon = lx
    })

    const draw = (frame: Float32Array) => {
      const w = canvas.width
      const h = canvas.height
      drawHolisticSkeleton(ctx, frame, w, h)
    }

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      if (tokensRef.current !== lastTokens || replayRef.current !== lastReplay) {
        lastTokens = tokensRef.current
        lastReplay = replayRef.current
        if (lexicon && tokensRef.current.length > 0) {
          frames = lexicon.framesForTokens(tokensRef.current).frames
        } else {
          frames = [restFrame()]
        }
        frameIdx = 0
        lastTick = now
      }
      if (now - lastTick >= FRAME_MS && frames.length > 1) {
        frameIdx = (frameIdx + 1) % frames.length
        lastTick = now
      }
      draw(frames[Math.min(frameIdx, frames.length - 1)])
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <canvas
      ref={canvasRef}
      width={640}
      height={480}
      className="h-full w-full"
      aria-label="Signing skeleton view"
    />
  )
}
