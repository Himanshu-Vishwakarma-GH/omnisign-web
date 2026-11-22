// Generates web/public/poses.json from the deterministic pose generator.
// Run: node scripts/gen-poses.mjs  (no build step needed — pure ESM, no deps)
import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

// Inline the same hash-based generator as src/lib/poses.ts so the JSON is
// generated without a TS toolchain.
function hashStr(s) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return Math.abs(h)
}
function poseForToken(token) {
  const h = hashStr(token)
  const a = (n) => ((h >> n) % 100) / 100
  const arm = (off) => ({
    shoulder: [-0.4 + a(off) * 1.2, -0.5 + a(off + 3) * 1.0, -0.6 + a(off + 6) * 1.2],
    elbow: 0.3 + a(off + 9) * 1.1,
    hand: 0.2 + a(off + 12) * 0.8,
  })
  return {
    left: arm(0),
    right: arm(16),
    head: [a(24) * 0.2 - 0.1, a(27) * 0.3 - 0.15, 0],
  }
}

const WORDS = [
  'HELP', 'DOCTOR', 'PAIN', 'WATER', 'YES', 'NO', 'FAMILY', 'EMERGENCY',
  'HOSPITAL', 'MEDICINE', 'STOP', 'WAIT', 'CALL', 'POLICE', 'FIRE', 'HURT',
  'BREATHE', 'ALLERGY', 'THANKYOU', 'PLEASE',
]
const out = {}
for (const w of WORDS) out[w] = poseForToken(w)
for (let c = 65; c <= 90; c++) {
  const letter = String.fromCharCode(c)
  const p = poseForToken(letter)
  p.right.shoulder = [0.5, -0.3, 0.9]
  p.right.elbow = 1.4
  p.left.shoulder = [0.1, 0.2, 0.15]
  p.left.elbow = 0.4
  out[letter] = p
}

const here = dirname(fileURLToPath(import.meta.url))
const dest = join(here, '..', 'public', 'poses.json')
writeFileSync(dest, JSON.stringify(out, null, 2))
console.log(`wrote ${dest} (${Object.keys(out).length} poses)`)
