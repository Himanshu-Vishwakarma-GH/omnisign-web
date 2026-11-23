/**
 * LexiSight model loading helpers (pure functions — no TF.js import).
 *
 * What the real LexiSight weights turned out to be (verified 2026-11-23
 * from the downloaded files in web/public/models/lexisight/):
 *
 * - Temporal word model (`tfjs_model/`): Keras 3 Functional model.
 *   Input [null, 30, 126] — 30 frames of HAND-ONLY landmarks
 *   (21 left-hand + 21 right-hand joints x 3 coords = 126).
 *   NOT 225-dim: the pose landmarks are NOT an input to this model.
 *   Output 263 classes (label_map.json). Transformer encoder with two
 *   custom layers (PositionalEncoding, TransformerEncoderBlock).
 * - Static fingerspelling model (`static_tfjs_model/`): plain Dense stack.
 *   Input [null, 126] (one frame of hand landmarks), output 36 classes
 *   (0-9, A-Z from static_label_map.json).
 *
 * The shipped model.json files are Keras 3 exports, which stock TF.js
 * cannot load directly:
 *   1. InputLayer uses `batch_shape` instead of `batch_input_shape`.
 *   2. `inbound_nodes` use the Keras 3 {args:[__keras_tensor__]} format
 *      instead of the Keras 2 [[name, node, tensor, {}]] format.
 *   3. `dtype` is a DTypePolicy object instead of a plain string.
 *   4. `input_layers`/`output_layers` are flat triples, not [triple].
 * `patchTopology` rewrites all four in memory (downloaded files untouched).
 *
 * The weights manifest uses bare duplicate names ('kernel', 'bias', ...),
 * so stock loadLayersModel cannot map them. `assignWeights` below maps
 * manifest tensors to model variables by SHAPE-group FIFO order instead
 * (see tfjsClient.ts).
 *
 * UNVERIFIED (cannot be confirmed from the model files alone — LexiSight
 * source was NOT fetched, per the user's fetch-only-models approval):
 * - The 126-dim hand order: assumed left hand (63) then right hand (63),
 *   matching the 225-dim capture order (dims 99..161 left, 162..224 right).
 * - Whether training-time inputs were normalised (raw MediaPipe coords
 *   are used here, as captured).
 * - TransformerEncoderBlock norm placement: implemented POST-norm
 *   (attention -> add -> norm -> FFN -> add -> norm), the standard
 *   keras.io TransformerBlock pattern. A PRE-norm variant was also
 *   load-tested in Node and produces plausible outputs; post-norm was
 *   chosen as the most likely original. If live predictions look
 *   systematically wrong, this is the second thing to check (after the
 *   hand order above).
 * - Calibration: calibration_temperature.json says T=1.05 (ECE 0.034).
 *   Applied to the output probabilities as p^(1/T), renormalised.
 */

export const LEXISIGHT_HAND_DIM = 126
export const LEXISIGHT_FRAMES = 30
/** Calibration temperature from LexiSight's calibration_temperature.json. */
export const LEXISIGHT_TEMPERATURE = 1.05

/** Label maps are Records {"0": "ACTOR", ...} — return labels in index order. */
export function labelsFromMap(map: Record<string, string>): string[] {
  return Object.keys(map)
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => map[k])
}

/**
 * Slice the hand landmarks out of 225-dim Holistic frames:
 * dims 99..161 (left hand) + 162..224 (right hand) = 126 per frame.
 * `frames225` is frame-major: frame f occupies [f*225, (f+1)*225).
 */
export function sliceHands(frames225: Float32Array, frames = LEXISIGHT_FRAMES): Float32Array {
  const out = new Float32Array(frames * LEXISIGHT_HAND_DIM)
  for (let f = 0; f < frames; f++) {
    for (let d = 0; d < LEXISIGHT_HAND_DIM; d++) {
      out[f * LEXISIGHT_HAND_DIM + d] = frames225[f * 225 + 99 + d]
    }
  }
  return out
}

/** Temperature scaling on already-softmaxed probabilities: p^(1/T), renormalised. */
export function applyTemperature(probs: ArrayLike<number>, temperature: number): number[] {
  if (temperature === 1) return Array.from(probs)
  const inv = 1 / temperature
  const adj = Array.from(probs, (p) => Math.pow(Math.max(p, 1e-12), inv))
  const sum = adj.reduce((a, b) => a + b, 0)
  return adj.map((v) => v / sum)
}

/** Top-k ranking of a probability vector against labels. */
export function topK(
  probs: ArrayLike<number>,
  labels: string[],
  k = 3,
): { word: string; confidence: number }[] {
  const order = Array.from(probs, (p, i) => i).sort((a, b) => probs[b] - probs[a])
  return order.slice(0, k).map((i) => ({ word: labels[i], confidence: probs[i] }))
}

/**
 * Rewrite a Keras 3 model topology JSON into the format TF.js expects.
 * Returns a new object; the input is not mutated.
 */
export function patchTopology(modelJson: unknown): Record<string, unknown> {
  const topo = JSON.parse(JSON.stringify(modelJson)) as {
    modelTopology: {
      config: {
        layers: Record<string, any>[]
        input_layers: unknown
        output_layers: unknown
      }
    }
  }
  const cfg = topo.modelTopology.config
  for (const layer of cfg.layers) {
    const c = (layer.config ?? {}) as Record<string, unknown>
    if (c.dtype && typeof c.dtype === 'object') {
      const d = c.dtype as { config?: { name?: string } }
      c.dtype = d.config?.name ?? 'float32'
    }
    if (layer.class_name === 'InputLayer' && c.batch_shape && !c.batch_input_shape) {
      c.batch_input_shape = c.batch_shape
      delete c.batch_shape
    }
    delete layer.build_config
    const nodes = (layer.inbound_nodes ?? []) as {
      args?: { config?: { keras_history?: [string, number, number] } }[]
    }[]
    layer.inbound_nodes = nodes.map((call) =>
      (call.args ?? [])
        .map((t) => {
          const h = t?.config?.keras_history
          return h ? [h[0], h[1], h[2], {}] : null
        })
        .filter(Boolean),
    )
  }
  const wrap = (v: unknown): unknown =>
    Array.isArray(v) && v.length === 3 && typeof v[0] === 'string' ? [v] : v
  cfg.input_layers = wrap(cfg.input_layers)
  cfg.output_layers = wrap(cfg.output_layers)
  return topo.modelTopology as unknown as Record<string, unknown>
}
