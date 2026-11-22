/**
 * Wrist-centric normalization (scale + translation invariant).
 *
 *   p_hat_i = (p_i - p_wrist) / || p_mcp_middle - p_wrist ||_2
 *
 * Landmark layout (48 x 3):
 *   0..20   right hand, 21 joints (0 = wrist, 9 = middle-finger MCP)
 *   21..41  left hand,  21 joints (21 = wrist, 30 = middle-finger MCP)
 *   42..47  upper body: L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist
 *
 * Each hand is normalized by its own wrist / middle-MCP scale; the upper
 * body is normalized by the shoulder midpoint and shoulder width.
 * A degenerate (near-zero) scale falls back to 1 so the output stays finite.
 */

export function normalizeLandmarks(input: Float32Array): Float32Array {
  if (input.length !== 48 * 3) {
    throw new Error(`normalizeLandmarks expects 144 values, got ${input.length}`)
  }
  const out = new Float32Array(144)
  const get = (i: number): [number, number, number] => [
    input[i * 3],
    input[i * 3 + 1],
    input[i * 3 + 2],
  ]
  const set = (i: number, x: number, y: number, z: number) => {
    out[i * 3] = x
    out[i * 3 + 1] = y
    out[i * 3 + 2] = z
  }
  const norm = (a: [number, number, number], b: [number, number, number]) =>
    Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

  const normalizeGroup = (
    indices: number[],
    originIdx: number,
    scaleRefIdx: number,
  ) => {
    const origin = get(originIdx)
    const scale = Math.max(norm(get(scaleRefIdx), origin), 1e-6)
    for (const i of indices) {
      const p = get(i)
      set(
        i,
        (p[0] - origin[0]) / scale,
        (p[1] - origin[1]) / scale,
        (p[2] - origin[2]) / scale,
      )
    }
  }

  // Right hand (0..20): wrist 0, middle MCP 9.
  normalizeGroup(Array.from({ length: 21 }, (_, k) => k), 0, 9)
  // Left hand (21..41): wrist 21, middle MCP 30.
  normalizeGroup(Array.from({ length: 21 }, (_, k) => 21 + k), 21, 30)
  // Upper body (42..47): origin = shoulder midpoint, scale = shoulder width.
  const shoulderL = get(42)
  const shoulderR = get(43)
  const mid: [number, number, number] = [
    (shoulderL[0] + shoulderR[0]) / 2,
    (shoulderL[1] + shoulderR[1]) / 2,
    (shoulderL[2] + shoulderR[2]) / 2,
  ]
  const shoulderWidth = Math.max(norm(shoulderR, shoulderL), 1e-6)
  for (let i = 42; i < 48; i++) {
    const p = get(i)
    set(
      i,
      (p[0] - mid[0]) / shoulderWidth,
      (p[1] - mid[1]) / shoulderWidth,
      (p[2] - mid[2]) / shoulderWidth,
    )
  }
  return out
}
