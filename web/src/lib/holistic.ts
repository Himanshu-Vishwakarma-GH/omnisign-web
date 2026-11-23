/**
 * LexiSight-format Holistic feature extraction.
 *
 * One frame = 225 floats, MediaPipe Holistic order:
 *   pose landmarks 0..32   -> dims 0..98
 *   left hand 0..20        -> dims 99..161
 *   right hand 0..20       -> dims 162..224
 * Coordinates are MediaPipe normalized image coords (x, y in [0,1],
 * z relative depth). A missing hand/pose is zeros.
 *
 * This is the documented LexiSight v2 convention (README: 30 frames x 225
 * dims = pose 33 + left hand 21 + right hand 21, no face mesh). The exact
 * landmark ORDER inside the 225 vector and the training-time normalization
 * were NOT published in the repo, so this file implements the standard
 * Selvaraj-et-al. convention (pose, left hand, right hand, raw normalized
 * coords) and flags it as UNVERIFIED in VERIFICATION.md. If the real
 * LexiSight weights are dropped in and predictions look wrong, this order
 * is the first thing to check.
 */

export const FRAMES = 30
export const FEATURE_DIM = 225
export const POSE_LANDMARKS = 33
export const HAND_LANDMARKS = 21

export interface HolisticResult {
  poseLandmarks?: { x: number; y: number; z: number }[] | null
  leftHandLandmarks?: { x: number; y: number; z: number }[] | null
  rightHandLandmarks?: { x: number; y: number; z: number }[] | null
}

/** Flatten one Holistic result into a 225-dim feature vector. */
export function extractFeatures(res: HolisticResult): Float32Array {
  const out = new Float32Array(FEATURE_DIM)
  const fill = (
    landmarks: { x: number; y: number; z: number }[] | null | undefined,
    offset: number,
    count: number,
  ) => {
    if (!landmarks) return
    for (let i = 0; i < Math.min(count, landmarks.length); i++) {
      const lm = landmarks[i]
      if (!lm) continue
      out[offset + i * 3] = lm.x ?? 0
      out[offset + i * 3 + 1] = lm.y ?? 0
      out[offset + i * 3 + 2] = lm.z ?? 0
    }
  }
  fill(res.poseLandmarks, 0, POSE_LANDMARKS)
  fill(res.leftHandLandmarks, 99, HAND_LANDMARKS)
  fill(res.rightHandLandmarks, 162, HAND_LANDMARKS)
  return out
}

/** Rolling 30-frame ring buffer of 225-dim frames. */
export class HolisticBuffer {
  private frames: Float32Array[] = []

  push(frame: Float32Array): void {
    if (frame.length !== FEATURE_DIM) {
      throw new Error(`HolisticBuffer expects ${FEATURE_DIM} values per frame`)
    }
    this.frames.push(frame)
    if (this.frames.length > FRAMES) this.frames.shift()
  }

  get length(): number {
    return this.frames.length
  }

  get isFull(): boolean {
    return this.frames.length === FRAMES
  }

  clear(): void {
    this.frames = []
  }

  /** Flattened [1, 30, 225] tensor data (frame-major). */
  toTensor(): Float32Array {
    const out = new Float32Array(FRAMES * FEATURE_DIM)
    for (let f = 0; f < this.frames.length; f++) {
      out.set(this.frames[f], f * FEATURE_DIM)
    }
    return out
  }
}

// --- Skeleton drawing (shared by the camera overlay + SkeletonView) --------

export const POSE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 9], [9, 10],
  [7, 8], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [27, 31], [28, 30], [28, 32],
]

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]

/** Draw a 225-dim frame as a 2D skeleton on a canvas context. */
export function drawHolisticSkeleton(
  ctx: CanvasRenderingContext2D,
  frame: Float32Array,
  w: number,
  h: number,
): void {
  ctx.clearRect(0, 0, w, h)
  const px = (i: number) => frame[i * 3] * w
  const py = (i: number) => frame[i * 3 + 1] * h
  const drawSet = (
    offset: number,
    connections: [number, number][],
    color: string,
    dot: string,
  ) => {
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()
    for (const [a, b] of connections) {
      ctx.moveTo(px(offset + a), py(offset + a))
      ctx.lineTo(px(offset + b), py(offset + b))
    }
    ctx.stroke()
    void dot // dots are drawn per landmark below
  }
  // Pose (offset 0) in cyan, hands in pink.
  drawSet(0, POSE_CONNECTIONS, '#22d3ee', '#22d3ee')
  drawSet(33, HAND_CONNECTIONS, '#f472b6', '#f472b6')
  drawSet(54, HAND_CONNECTIONS, '#f472b6', '#f472b6')
  // Landmark dots.
  for (let i = 0; i < 75; i++) {
    const x = px(i)
    const y = py(i)
    if (x === 0 && y === 0) continue // missing landmark
    ctx.fillStyle = i < 33 ? '#22d3ee' : '#f472b6'
    ctx.beginPath()
    ctx.arc(x, y, i === 0 ? 4 : 2.5, 0, Math.PI * 2)
    ctx.fill()
  }
}
