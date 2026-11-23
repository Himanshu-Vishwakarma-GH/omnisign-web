# OmniSign — Two-Way Sign Language AI (v2, conversation-first)

Real-time two-way sign-language conversation, **100% in the browser, zero
backend**. A deaf signer and a hearing speaker talk to each other: the signer
signs into the webcam and the app speaks the words aloud; the speaker talks or
types and a signing avatar signs the reply back. No video or audio ever leaves
the device.

**Live demo:** https://himanshu-vishwakarma-gh.github.io/omnisign-web/

## What changed in v2

v1 was an emergency-only MVP (20 emergency words, synthetic ONNX model). The
user tested it and reported most things didn't work, and redirected the goal:
**normal everyday conversation first, emergency as one feature/mode** — and to
build on open-source sign-language work instead of invented models.

v2 therefore:

- **Conversation-first UI** — a chat-style transcript in the middle (signed
  turns on the right, spoken/text turns on the left), webcam + skeleton
  overlay on the left, signing avatar on the right.
- **88-word conversation vocabulary** — greetings, introductions, questions,
  common verbs/nouns, politeness (hello, thank you, please, yes, no, name,
  what, where, when, how, help, water, food, doctor, hospital, family,
  friend, today, tomorrow, go, come, want, need, like, good, bad, sorry…).
  The 20 emergency words are still there, surfaced by an **Emergency mode**
  toggle (quick-sign chips + recognition prioritised toward them).
- **LexiSight-format recognition** — MediaPipe Holistic landmarks
  (33 pose + 21 left-hand + 21 right-hand = **225 dims/frame**), 30-frame
  window, 5-frame consensus smoothing, TF.js model — the input format of the
  open-source [LexiSight](https://github.com/Yuvanshankarraj5/lexisight)
  ISL recogniser (see "LexiSight verification" below).
- **Pose lexicon for speech→sign** — every word and letter (A–Z) has
  5 keyframes × 225 dims in `web/public/lexicon/lexicon.json`, rendered in
  **two layers from the same data**: a 2D skeleton canvas and a Three.js
  avatar retargeted from the same pose frames. Unknown words are
  fingerspelled letter by letter.
- **Simulate Sign demo mode** — works with no webcam: realistic landmark
  sequences are fed through the *real* TF.js pipeline.

## Architecture

```
webcam ──▶ MediaPipe HolisticLandmarker (@mediapipe/tasks-vision)
             33 pose + 21+21 hand landmarks / frame
        ──▶ extractFeatures → 225-dim vector, missing → zeros
        ──▶ HolisticBuffer (30 frames) → [1, 30, 225]
        ──▶ slice hand landmarks (dims 99..224) → [1, 30, 126]
        ──▶ TF.js model → word + confidence + latency
        ──▶ 5-frame consensus debounce → transcript → Web Speech TTS
                ▲
                │  model loader tries in order:
                │  1. models/lexisight/  (real LexiSight weights, if added)
                │  2. models/omnisign_tfjs/ (bundled fallback, trained here)

text / mic ──▶ Web Speech STT or typed text
        ──▶ gloss tokenizer ("What is your name?" → YOUR, NAME, WHAT)
        ──▶ lexicon keyframes (5 × 225 per word; A–Z fingerspelling fallback)
        ──▶ 2D skeleton canvas  +  Three.js avatar (same frames)
```

## LexiSight verification (important — read this)

**Update 2026-11-23: the real LexiSight weights ARE now bundled** (user
approved fetching *only the model data files*). What actually happened:

- A full-repo clone was BLOCKED by a system security notice (execution-risk
  check at ~69 MB). No LexiSight code was cloned, run or copied — ever.
- Instead, only 10 data files were downloaded via curl from
  `raw.githubusercontent.com/Yuvanshankarraj5/lexisight/main/public/model/`:
  the temporal + static `model.json` files, their weight shards, 5 label
  maps and `calibration_temperature.json`. Every file's byte size matched
  the GitHub API listing exactly; SHA-256 recorded in `VERIFICATION.md`.
- **What the files actually contain (this differs from the README claims
  the v2 rebuild relied on):** the temporal model takes **[1, 30, 126]** —
  hand landmarks only (21 left + 21 right × 3), NOT 225-dim — and outputs
  **263 classes** (not 278). The static fingerspelling model takes
  [1, 126] and outputs 36 classes (0–9, A–Z). Both are Keras 3 exports,
  which stock TF.js cannot load directly: the app patches the topology in
  memory (InputLayer `batch_shape`, Keras 3 `inbound_nodes`, dtype
  objects) and re-implements the two custom layers (`PositionalEncoding`,
  `TransformerEncoderBlock`, post-norm) from the topology + weight shapes
  alone. Weights are assigned by shape-group FIFO order because the
  manifest uses bare duplicate names.
- **Verified in Node (tfjs-node):** both models load (41/41 and 16/16
  weights, all bytes consumed), inputs/outputs as above, softmax outputs
  sum to 1, predictions are input-sensitive (random inputs give varied
  labels), warm inference ~12–18 ms. Temperature T=1.05 is applied per
  LexiSight's calibration file.
- **Vocabulary coverage:** the 263-label vocab covers **34 of our 88**
  conversation words (plus phrases THANK_YOU, HOW ARE YOU, GOOD MORNING…).
  It does NOT contain HELP, WATER, YES, NO, WHAT, WHERE, WHEN, GO, COME,
  WANT, NEED, EAT, DRINK, SLEEP, NAME, FOOD, EMERGENCY, PAIN, STOP, WAIT,
  CALL, FIRE, HURT, BREATHE, ALLERGY, SORRY or PLEASE — only 5 of the 20
  emergency words (DOCTOR, FAMILY, HOSPITAL, MEDICINE, POLICE). The app
  only ever shows labels the loaded model can produce.
- **Still UNVERIFIED:** the 126-dim hand order (assumed left-then-right,
  matching capture dims 99..224), training-time normalisation, and the
  transformer's norm placement (post-norm assumed — the standard keras.io
  pattern; a pre-norm variant was also load-tested). If live predictions
  look systematically wrong, check those first, in that order. Live
  human-signer accuracy has NOT been measured — see `VERIFICATION.md`.

## The bundled fallback model (what actually ships today)

`web/public/models/omnisign_tfjs/` — a small Conv1D TF.js model (83k params,
~325 KB weights) trained by `web/scripts/train-tfjs.mjs` on **procedural**
225-dim sequences from `web/src/lib/motionGen.mjs` (authored hand shapes +
arm trajectories, seeded variations), 120 train + 30 val samples per word,
25 epochs. Held-out accuracy on fresh procedural seeds: **92.84% (817/880)**,
validation accuracy 92.35%. Weak classes: HE 0/10, DAY 1/10, I 3/10,
PAIN 4/10 — procedurally similar motions (rest-pose pronouns, waving
HELLO/BYE) confuse it. **This is a pipeline-proving model, not a
human-signer recogniser.** Real accuracy with a live human signer is
unverified and will be much lower until real LexiSight weights are used.

## The lexicon (procedural — be honest about this)

`web/public/lexicon/lexicon.json` (88 words + 26 letters, 5 keyframes × 225
dims each) is generated by `web/scripts/gen-lexicon.mjs` from
`motionGen.mjs`. The poses are **authored procedural trajectories, not
motion-captured from real ISL signers** — no word-level ISL pose archive is
downloadable without gated access (CISLR is gated on Hugging Face;
ISLTranslate poses are sentence-level with no word alignment; iSign is
gated). The avatar signs are therefore *representative motions*, not
verified ISL. Replace them with real pose data when a licensed source is
available.

## The LexiSight weights (bundled — what they are)

`web/public/models/lexisight/` (1.6 MB, committed so GitHub Pages serves them):

- `tfjs_model/model.json` (10,173 B) + `group1-shard1of1.bin` (1,242,652 B)
  — temporal Transformer, [1,30,126] → 263 ISL words.
- `static_tfjs_model/model.json` (9,084 B) + `group1-shard1of1.bin` (310,160 B)
  — Dense fingerspelling stack, [1,126] → 36 classes (0–9, A–Z).
- `label_map.json` (4,819 B, 263 labels), `temporal_label_map.json`,
  `label_map_holistic_v2.json`, `label_map_holistic_v3.json` (261 labels
  each — do NOT match the 263-output model; kept for reference only),
  `static_label_map.json`, `calibration_temperature.json` (T=1.05).
- Only these 10 data files were fetched (user-approved, 2026-11-23). **No
  LexiSight source code is in this repo.** If the weights are ever
  refreshed, re-download the same 10 URLs and re-check sizes + SHA-256
  against `VERIFICATION.md`.

## Setup

```bash
cd web
npm install
npm test        # vitest — 20 tests
npm run build   # tsc + vite build → web/dist
npm run preview # serve dist; smoke-test with curl (see VERIFICATION.md)
```

Regenerate the lexicon / retrain the fallback model:

```bash
node scripts/gen-lexicon.mjs                                  # → public/lexicon/lexicon.json
node scripts/train-tfjs.mjs --samples 120 --epochs 25 --node  # → public/models/omnisign_tfjs/
```

`--node` uses the native tfjs-node backend (much faster). It needs a
one-line node_modules patch on Node 24 — see the comment at the top of
`scripts/train-tfjs.mjs`. Without `--node` it trains on the pure-JS CPU
backend (very slow: epoch 1 took >5 min).

## Extending the vocabulary

1. Add the word to `VOCAB_WORDS` in `web/src/lib/motionGen.mjs` with a
   motion entry in `WORD_MOTIONS` (hand, shape, waypoint path, repeat, head).
2. Run `node scripts/gen-lexicon.mjs` and
   `node scripts/train-tfjs.mjs --samples 120 --epochs 25 --node`.
3. Add tests in `web/src/lib/omnisign.test.ts`.

## Attribution & licences

- **LexiSight** — https://github.com/Yuvanshankarraj5/lexisight —
  the real TF.js model weights + label maps bundled in
  `web/public/models/lexisight/` (10 data files only, user-approved fetch;
  no LexiSight code). Note: GitHub's API reports **no licence file** on
  that repo, despite an MIT badge in its README — treat the weights'
  licence as unconfirmed.
- **"Rigged Hand" 3D hand model** — by J-Toastie, licensed CC-BY, via
  Poly Pizza (https://poly.pizza/m/BEy8jbxm6A) — the rigged GLB hand
  (`web/public/models/hand.glb`, 91,424 bytes) used by the Speech-to-Sign
  3D hands. Credit is also shown in the app footer. No Blender build
  was used: the model is the Poly Pizza asset, unmodified.
- **MediaPipe Tasks Vision** — Apache-2.0 — Holistic landmarker.
- **spoken-to-signed-translation / pose-format** (sign.mt) — MIT — the
  pose-lexicon architecture this app follows; no code copied.
- **ISL data licensing warning:** the best ISL datasets (WLASL, CISLR /
  ISLTranslate / iSign) are non-commercial (CC-BY-NC[-SA] / C-UDA). LexiSight
  code is MIT but its weights were trained on CC-BY-NC data, so downstream
  commercial use of those weights is legally unclear. This project is a
  research/learning demo, not a commercial product.

## Limitations (honest)

- The LexiSight model is real and loads, but **live human-signer accuracy
  has NOT been measured** — no real-webcam test with a live signer has been
  run. Its 263-word vocab covers only 34/88 of our conversation words and
  5/20 emergency words; words outside it can never be recognised by it.
- The 126-dim hand order, training-time normalisation and transformer
  norm placement are **unverified** (see above).
- The signing avatar's motions are **procedural, not verified ISL** — do not
  present them as correct ISL to learners.
- MediaPipe Holistic model asset URL is assumed by analogy with the
  hand-landmarker URL and is not verified; camera failure degrades
  gracefully (status shown, app still boots, Simulate Sign still works).
- The app boots and renders even if MediaPipe/TF.js models fail to load;
  a clear status is shown, never a blank page.
- If the LexiSight weights fail to load at runtime, the app falls back to
  the procedural-trained model — which is a pipeline demo, not a
  human-signer recogniser. See `VERIFICATION.md`.
