/** Shared constants for OmniSign v2 (conversation-first). */
import { EMERGENCY_WORDS, VOCAB_WORDS } from './motionGen.mjs'

export { EMERGENCY_WORDS, VOCAB_WORDS }
export type SignWord = (typeof VOCAB_WORDS)[number]

export const FRAMES = 30
export const FEATURE_DIM = 225
export const CONFIDENCE_THRESHOLD = 0.85
export const DEBOUNCE_FRAMES = 5 // commit only after MORE than 5 sequential passes

/** Display form of a gloss label, e.g. THANKYOU -> "Thank you". */
export function displayWord(gloss: string): string {
  if (gloss === 'THANKYOU') return 'Thank you'
  return gloss.charAt(0) + gloss.slice(1).toLowerCase()
}
