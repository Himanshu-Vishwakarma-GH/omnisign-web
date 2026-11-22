import { COORDS, FEATURES_PER_FRAME, FRAMES, LANDMARKS } from './constants'

/**
 * Rolling 30-frame ring buffer of normalized landmark frames.
 * toTensor() returns a Float32Array of shape [1, 30, 48, 3] flattened in
 * frame-major order (frame 0 first), ready for onnxruntime-web.
 */
export class LandmarkBuffer {
  private frames: Float32Array[] = []

  push(frame: Float32Array): void {
    if (frame.length !== FEATURES_PER_FRAME) {
      throw new Error(`LandmarkBuffer expects ${FEATURES_PER_FRAME} values per frame`)
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

  /** Flattened [1, 30, 48, 3] tensor data (frame-major). */
  toTensor(): Float32Array {
    const out = new Float32Array(FRAMES * LANDMARKS * COORDS)
    for (let f = 0; f < this.frames.length; f++) {
      out.set(this.frames[f], f * FEATURES_PER_FRAME)
    }
    return out
  }
}
