/**
 * Train the bundled fallback TF.js model on procedural 225-dim sequences.
 *
 * Architecture: Conv1D(48, k=5) -> MaxPool -> Conv1D(64, k=3) ->
 * GlobalMaxPool -> Dense(128) -> Dense(vocab) softmax.
 * Input [30, 225] (LexiSight format), output = conversation vocabulary.
 *
 * Run:  node scripts/train-tfjs.mjs [--samples 120] [--epochs 25]
 * Writes: web/public/models/omnisign_tfjs/{model.json, group1-shard*.bin}
 *         web/public/models/omnisign_tfjs/labels.json
 *
 * Uses @tensorflow/tfjs-node (native, fast) when available, otherwise the
 * pure-JS @tensorflow/tfjs CPU backend (slower).
 *
 * NOTE: this model is trained on PROCEDURAL sequences from motionGen.mjs,
 * not on real signers. It proves the pipeline end-to-end in the exact
 * LexiSight input format; real accuracy needs the LexiSight weights
 * (see README "Adding real LexiSight weights").
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { VOCAB_WORDS, generateWordSequence } from '../src/lib/motionGen.mjs'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const args = process.argv.slice(2)
const opt = (name, def) => {
  const i = args.indexOf(name)
  return i >= 0 && args[i + 1] ? Number(args[i + 1]) : def
}
const TRAIN_PER_CLASS = opt('--samples', 120)
const VAL_PER_CLASS = 30
const EPOCHS = opt('--epochs', 25)
const FRAMES = 30
const DIMS = 225

let tf
// NOTE: tfjs-node needs a one-line node_modules patch to run on Node 24:
// its nodejs_kernel_backend.js calls Node's `util.isNullOrUndefined`,
// removed from Node long ago. Patch (applied 2026-11-23, training-only,
// browser bundle unaffected):
//   node_modules/@tensorflow/tfjs-node/dist/nodejs_kernel_backend.js
//   after `var util_1 = require("util");` add:
//   if (typeof util_1.isNullOrUndefined !== "function") {
//     util_1.isNullOrUndefined = function (x) { return x == null; };
//   }
// Default backend is pure-JS CPU (no patch needed, but ~10x slower);
// pass --node for the native backend (requires the patch above).
if (args.includes('--node')) {
  tf = await import('@tensorflow/tfjs-node')
  console.log('backend: tfjs-node (native)')
} else {
  tf = await import('@tensorflow/tfjs')
  await tf.setBackend('cpu')
  console.log('backend: tfjs CPU (pure JS)')
}

const numClasses = VOCAB_WORDS.length
console.log(`vocab: ${numClasses} words, train ${TRAIN_PER_CLASS}/class, val ${VAL_PER_CLASS}/class`)

function buildDataset(perClass, seedBase) {
  const n = numClasses * perClass
  const xs = new Float32Array(n * FRAMES * DIMS)
  const ys = new Float32Array(n * numClasses)
  let row = 0
  for (let c = 0; c < numClasses; c++) {
    for (let s = 0; s < perClass; s++) {
      const seq = generateWordSequence(VOCAB_WORDS[c], seedBase + s * 7 + c * 131)
      xs.set(seq, row * FRAMES * DIMS)
      ys[row * numClasses + c] = 1
      row++
    }
  }
  return { xs, ys, n }
}

console.log('generating training data…')
const train = buildDataset(TRAIN_PER_CLASS, 1000)
const val = buildDataset(VAL_PER_CLASS, 90000)
const xTrain = tf.tensor(train.xs, [train.n, FRAMES, DIMS])
const yTrain = tf.tensor(train.ys, [train.n, numClasses])
const xVal = tf.tensor(val.xs, [val.n, FRAMES, DIMS])
const yVal = tf.tensor(val.ys, [val.n, numClasses])

const model = tf.sequential()
model.add(
  tf.layers.conv1d({
    inputShape: [FRAMES, DIMS],
    filters: 48,
    kernelSize: 5,
    activation: 'relu',
    padding: 'same',
  }),
)
model.add(tf.layers.maxPooling1d({ poolSize: 2 }))
model.add(
  tf.layers.conv1d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }),
)
model.add(tf.layers.globalMaxPooling1d())
model.add(tf.layers.dense({ units: 128, activation: 'relu' }))
model.add(tf.layers.dropout({ rate: 0.2 }))
model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }))
model.compile({
  optimizer: tf.train.adam(0.001),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
})
model.summary()

await model.fit(xTrain, yTrain, {
  epochs: EPOCHS,
  batchSize: 64,
  validationData: [xVal, yVal],
  callbacks: {
    onEpochEnd: (epoch, logs) => {
      console.log(
        `epoch ${epoch + 1}/${EPOCHS} loss=${logs.loss.toFixed(4)} acc=${logs.acc.toFixed(4)} ` +
          `val_loss=${logs.val_loss.toFixed(4)} val_acc=${logs.val_acc.toFixed(4)}`,
      )
    },
  },
})

const outDir = join(root, 'public', 'models', 'omnisign_tfjs')
mkdirSync(outDir, { recursive: true })
await model.save(`file://${outDir}`)
writeFileSync(join(outDir, 'labels.json'), JSON.stringify(VOCAB_WORDS, null, 2))
console.log(`saved to ${outDir}`)

// Final held-out check (fresh seeds the model never saw).
const test = buildDataset(10, 777000)
const xTest = tf.tensor(test.xs, [test.n, FRAMES, DIMS])
const pred = model.predict(xTest)
const predIdx = pred.argMax(-1).dataSync()
let correct = 0
const perClass = new Map()
for (let i = 0; i < test.n; i++) {
  const truth = Math.floor(i / 10)
  const ok = predIdx[i] === truth
  if (ok) correct++
  const e = perClass.get(truth) ?? { ok: 0, total: 0 }
  e.total++
  if (ok) e.ok++
  perClass.set(truth, e)
}
console.log(`held-out accuracy: ${(correct / test.n).toFixed(4)} (${correct}/${test.n})`)
const worst = [...perClass.entries()]
  .filter(([, e]) => e.ok < e.total)
  .map(([c, e]) => `${VOCAB_WORDS[c]} ${e.ok}/${e.total}`)
console.log('imperfect classes:', worst.length ? worst.join(', ') : 'none')
