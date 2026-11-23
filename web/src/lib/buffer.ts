import { FEATURE_DIM, FRAMES } from './holistic'

/**
 * Rolling 30-frame ring buffer of 225-dim Holistic feature frames.
 * toTensor() returns a Float32Array of shape [1, 30, 225] flattened in
 * frame-major order (frame 0 first), ready for the TF.js model.
 *
 * Kept as a thin wrapper around HolisticBuffer for the older import name;
 * new code should import HolisticBuffer from './holistic' directly.
 */
export class LandmarkBuffer {
  private frames: Float32Array[] = []

  push(frame: Float32Array): void {
    if (frame.length !== FEATURE_DIM) {
      throw new Error(`LandmarkBuffer expects ${FEATURE_DIM} values per frame`)
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
