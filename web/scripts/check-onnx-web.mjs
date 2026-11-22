// Validate the browser inference path in Node: same onnxruntime-web package
// the app uses, same model file, same normalized tensor the TS app feeds it.
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web'

const SIGNS = ['Help','Doctor','Pain','Water','Yes','No','Family','Emergency','Hospital','Medicine','Stop','Wait','Call','Police','Fire','Hurt','Breathe','Allergy','ThankYou','Please']

const session = await ort.InferenceSession.create(
  new URL('../dist/models/omnisign_bilstm.onnx', import.meta.url).pathname,
  { executionProviders: ['wasm'] },
)
for (const word of ['help', 'water', 'police']) {
  const data = new Float32Array(JSON.parse(readFileSync(`/tmp/tensor_${word}.json`, 'utf8')))
  const input = new ort.Tensor('float32', data, [1, 30, 48, 3])
  const t0 = performance.now()
  const out = await session.run({ input })
  const ms = performance.now() - t0
  const logits = Array.from(out.logits.data)
  const max = Math.max(...logits)
  const exps = logits.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map((e) => e / sum)
  const top = probs.indexOf(Math.max(...probs))
  console.log(`${word}: predicted=${SIGNS[top]} conf=${probs[top].toFixed(3)} latency=${ms.toFixed(1)}ms`)
}
