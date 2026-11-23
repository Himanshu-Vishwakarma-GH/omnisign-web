/**
 * Type declarations for motionGen.mjs (plain-JS procedural generator).
 * The implementation is shared between the browser app and the Node
 * training script, so it stays dependency-free JavaScript.
 */
export const FRAMES: number
export const POSE_LANDMARKS: number
export const HAND_LANDMARKS: number
export const FEATURE_DIM: number
export const EMERGENCY_WORDS: string[]
export const VOCAB_WORDS: string[]
export const SHAPES: Record<string, number[]>
export const LETTER_SHAPES: Record<string, number[]>
export function mulberry32(seed: number): () => number
export function handLandmarks(
  wrist: [number, number, number],
  curls: number[],
  side: number,
  spread?: number,
): [number, number, number][]
export function buildFrame(
  wristL: [number, number, number],
  wristR: [number, number, number],
  curlsL: number[],
  curlsR: number[],
  headDx?: number,
  headDy?: number,
): Float32Array
export function generateWordSequence(word: string, seed?: number): Float32Array
export function generateWordKeyframes(word: string): number[][]
export function generateLetterKeyframes(letter: string): number[][]
