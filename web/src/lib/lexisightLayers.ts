/**
 * Custom TF.js layers for the LexiSight temporal model.
 *
 * The temporal model.json (Keras 3 export) references two custom layers
 * from LexiSight's own training code (module `ml.models.temporal_v1`),
 * which stock TF.js does not know:
 *   - `PositionalEncoding` (max_len 30, d_model 128): a trainable
 *     `pos_embedding` weight of shape [1, 30, 128], added to the input.
 *   - `TransformerEncoderBlock` (d_model 128, num_heads 4, d_ff 256,
 *     dropout 0.3): standard multi-head self-attention + feed-forward.
 *
 * These are re-implementations written from the model.json topology +
 * weights-manifest shapes alone (Q/K/V kernels [128,4,32], output
 * projection [4,32,128], FFN [128,256] -> [256,128], two LayerNorms per
 * block). No LexiSight source was read. The weight ORDER created here
 * (Q, K, V, output, FFN1, FFN2, norm1, norm2 per block) matches the
 * manifest order, which is what the positional weight assignment in
 * tfjsClient.ts relies on.
 *
 * Norm placement is POST-norm (attention -> residual add -> norm ->
 * FFN -> residual add -> norm), the standard keras.io TransformerBlock
 * pattern. UNVERIFIED — see lexisight.ts. Dropout is inert at inference.
 */
import * as tf from '@tensorflow/tfjs'

type LayerConfig = Record<string, unknown>

function registerSafe(cls: { className: string }): void {
  try {
    tf.serialization.registerClass(cls as never)
  } catch {
    // Already registered (e.g. HMR / double import) — safe to ignore.
  }
}

export class PositionalEncoding extends tf.layers.Layer {
  static className = 'PositionalEncoding'
  private maxLen: number
  private dModel: number
  private posEmb!: tf.LayerVariable

  constructor(config: LayerConfig) {
    super({ name: config.name as string, trainable: config.trainable as boolean | undefined })
    this.maxLen = (config.max_len as number) ?? 30
    this.dModel = (config.d_model as number) ?? 128
  }

  build(): void {
    this.posEmb = this.addWeight('pos_embedding', [1, this.maxLen, this.dModel], 'float32', tf.initializers.zeros())
    this.built = true
  }

  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[] {
    const x = (Array.isArray(inputs) ? inputs[0] : inputs) as tf.Tensor
    const len = x.shape[1] as number
    const pe = this.posEmb.read().slice([0, 0, 0], [1, len, this.dModel])
    return x.add(pe)
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    return inputShape
  }

  getConfig(): tf.serialization.ConfigDict {
    return { ...super.getConfig(), max_len: this.maxLen, d_model: this.dModel }
  }
}

export class TransformerEncoderBlock extends tf.layers.Layer {
  static className = 'TransformerEncoderBlock'
  private dModel: number
  private numHeads: number
  private dff: number
  private dropoutRate: number
  private qW!: tf.LayerVariable
  private qB!: tf.LayerVariable
  private kW!: tf.LayerVariable
  private kB!: tf.LayerVariable
  private vW!: tf.LayerVariable
  private vB!: tf.LayerVariable
  private oW!: tf.LayerVariable
  private oB!: tf.LayerVariable
  private ffn1W!: tf.LayerVariable
  private ffn1B!: tf.LayerVariable
  private ffn2W!: tf.LayerVariable
  private ffn2B!: tf.LayerVariable
  private n1g!: tf.LayerVariable
  private n1b!: tf.LayerVariable
  private n2g!: tf.LayerVariable
  private n2b!: tf.LayerVariable

  constructor(config: LayerConfig) {
    super({ name: config.name as string, trainable: config.trainable as boolean | undefined })
    this.dModel = (config.d_model as number) ?? 128
    this.numHeads = (config.num_heads as number) ?? 4
    this.dff = (config.d_ff as number) ?? 256
    this.dropoutRate = (config.dropout as number) ?? 0.3
  }

  build(): void {
    const k = this.dModel
    const h = this.numHeads
    const kd = Math.floor(k / h)
    // Creation order MUST match the weights-manifest order:
    // Q, K, V, output-projection, FFN1, FFN2, norm1, norm2.
    this.qW = this.addWeight('q_kernel', [k, h, kd], 'float32', tf.initializers.glorotUniform({}))
    this.qB = this.addWeight('q_bias', [h, kd], 'float32', tf.initializers.zeros())
    this.kW = this.addWeight('k_kernel', [k, h, kd], 'float32', tf.initializers.glorotUniform({}))
    this.kB = this.addWeight('k_bias', [h, kd], 'float32', tf.initializers.zeros())
    this.vW = this.addWeight('v_kernel', [k, h, kd], 'float32', tf.initializers.glorotUniform({}))
    this.vB = this.addWeight('v_bias', [h, kd], 'float32', tf.initializers.zeros())
    this.oW = this.addWeight('o_kernel', [h, kd, k], 'float32', tf.initializers.glorotUniform({}))
    this.oB = this.addWeight('o_bias', [k], 'float32', tf.initializers.zeros())
    this.ffn1W = this.addWeight('ffn1_kernel', [k, this.dff], 'float32', tf.initializers.glorotUniform({}))
    this.ffn1B = this.addWeight('ffn1_bias', [this.dff], 'float32', tf.initializers.zeros())
    this.ffn2W = this.addWeight('ffn2_kernel', [this.dff, k], 'float32', tf.initializers.glorotUniform({}))
    this.ffn2B = this.addWeight('ffn2_bias', [k], 'float32', tf.initializers.zeros())
    this.n1g = this.addWeight('norm1_gamma', [k], 'float32', tf.initializers.ones())
    this.n1b = this.addWeight('norm1_beta', [k], 'float32', tf.initializers.zeros())
    this.n2g = this.addWeight('norm2_gamma', [k], 'float32', tf.initializers.ones())
    this.n2b = this.addWeight('norm2_beta', [k], 'float32', tf.initializers.zeros())
    this.built = true
  }

  private layerNorm(x: tf.Tensor, gamma: tf.LayerVariable, beta: tf.LayerVariable): tf.Tensor {
    return tf.tidy(() => {
      const mean = x.mean(-1, true)
      const variance = x.sub(mean).square().mean(-1, true)
      const normed = x.sub(mean).div(variance.add(1e-6).sqrt())
      return normed.mul(gamma.read()).add(beta.read())
    })
  }

  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[] {
    const x = (Array.isArray(inputs) ? inputs[0] : inputs) as tf.Tensor
    const h = this.numHeads
    const kd = Math.floor(this.dModel / h)
    const batch = x.shape[0] as number
    const steps = x.shape[1] as number
    return tf.tidy(() => {
      const proj = (inp: tf.Tensor, w: tf.LayerVariable, b: tf.LayerVariable): tf.Tensor =>
        tf.einsum('btd,dhk->bthk', inp, w.read()).add(b.read())
      const q = proj(x, this.qW, this.qB)
      const k = proj(x, this.kW, this.kB)
      const v = proj(x, this.vW, this.vB)
      const scores = tf.einsum('bthk,bshk->bhts', q, k).div(Math.sqrt(kd))
      const att = tf.einsum('bhts,bshk->bthk', tf.softmax(scores, -1), v)
      const attFlat = att.reshape([batch, steps, this.dModel])
      const out = tf
        .einsum(
          'bd,dk->bk',
          attFlat.reshape([batch * steps, this.dModel]),
          this.oW.read().reshape([this.dModel, this.dModel]),
        )
        .reshape([batch, steps, this.dModel])
        .add(this.oB.read())
      // POST-norm: attention -> residual -> norm -> FFN -> residual -> norm.
      const y1 = this.layerNorm(x.add(out), this.n1g, this.n1b)
      const f1 = tf.relu(y1.matMul(this.ffn1W.read()).add(this.ffn1B.read()))
      const f2 = f1.matMul(this.ffn2W.read()).add(this.ffn2B.read())
      return this.layerNorm(y1.add(f2), this.n2g, this.n2b)
    })
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    return inputShape
  }

  getConfig(): tf.serialization.ConfigDict {
    return {
      ...super.getConfig(),
      d_model: this.dModel,
      num_heads: this.numHeads,
      d_ff: this.dff,
      dropout: this.dropoutRate,
    }
  }
}

registerSafe(PositionalEncoding)
registerSafe(TransformerEncoderBlock)
