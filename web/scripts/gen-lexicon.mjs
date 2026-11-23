/**
 * Generate web/public/lexicon/lexicon.json from the procedural motion
 * generator (motionGen.mjs): 5 keyframes x 225 dims per word + per letter.
 *
 * Run:  node scripts/gen-lexicon.mjs
 *
 * The lexicon is PROCEDURAL (authored motion parameters), not motion
 * captured from real signers — see README "Limitations". It is the
 * fallback because no word-level ISL pose archive is downloadable without
 * gated access (CISLR is gated on Hugging Face; ISLTranslate poses are
 * sentence-level with no word alignment; iSign is gated).
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  VOCAB_WORDS,
  generateLetterKeyframes,
  generateWordKeyframes,
} from '../src/lib/motionGen.mjs'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const outDir = join(root, 'public', 'lexicon')
mkdirSync(outDir, { recursive: true })

const round = (arr) => arr.map((v) => Math.round(v * 1000) / 1000)

const words = {}
for (const w of VOCAB_WORDS) {
  words[w] = generateWordKeyframes(w).map(round)
}
const letters = {}
for (let c = 65; c <= 90; c++) {
  const letter = String.fromCharCode(c)
  letters[letter] = generateLetterKeyframes(letter).map(round)
}

const lexicon = {
  version: 2,
  generatedFrom:
    'procedural motionGen.mjs v2 (authored trajectories + hand shapes, 225-dim Holistic frames)',
  words,
  letters,
}

writeFileSync(join(outDir, 'lexicon.json'), JSON.stringify(lexicon))
console.log(
  `lexicon.json: ${Object.keys(words).length} words, ${Object.keys(letters).length} letters`,
)
