/**
 * Rule-based English -> Sign Language Gloss tokenizer.
 *
 *  - Upper-cases, strips punctuation.
 *  - Removes filler words: is, are, the, a, an, am.
 *  - Moves WH-words (what, where, when, who, why, how) to the END,
 *    matching ASL/ISL gloss word order.
 *  - Collapses "thank you" into the single gloss THANKYOU.
 *
 * Example: "What is your name?" -> ["YOUR", "NAME", "WHAT"]
 */

const FILLERS = new Set([
  'is', 'are', 'the', 'a', 'an', 'am', 'it', 'to', 'of',
  'do', 'does', 'did', 'will', 'would', 'should', 'just',
])
const WH_WORDS = new Set(['what', 'where', 'when', 'who', 'why', 'how', 'which'])

export function tokenizeGloss(text: string): string[] {
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s']/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .filter((w) => !FILLERS.has(w))

  // Collapse "thank you" -> "thankyou".
  const merged: string[] = []
  for (let i = 0; i < words.length; i++) {
    if (words[i] === 'thank' && words[i + 1] === 'you') {
      merged.push('thankyou')
      i++
    } else {
      merged.push(words[i])
    }
  }

  const wh: string[] = []
  const rest: string[] = []
  for (const w of merged) {
    if (WH_WORDS.has(w)) wh.push(w.toUpperCase())
    else rest.push(w.toUpperCase())
  }
  return [...rest, ...wh]
}
