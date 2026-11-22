# OmniSign — Two-Way Sign Language AI (100% On-Device)

Real-time, bidirectional sign-language translation that runs **entirely in the
browser** — no server, no cloud API, no video ever leaving the user's device.

- **Sign → Speech**: a webcam tracks both hands + upper body (MediaPipe
  HandLandmarker), a lightweight ONNX BiLSTM recognizes 20 critical emergency
  signs, and the word is spoken aloud via the Web Speech Synthesis API.
- **Speech → Sign**: spoken English (Web Speech API, with a text-input
  fallback) is converted to Sign Language Gloss grammar and signed back by a
  procedural Three.js 3D avatar at 60 FPS, with A–Z finger-spelling fallback.

Built for high-stakes everyday moments — emergency hospital rooms (where
masks block lip-reading), public transit announcements, workplace meetings —
where a human interpreter ($60–$120/hr, 48–72 hr booking) simply isn't there.

## The 20 emergency signs

Help · Doctor · Pain · Water · Yes · No · Family · Emergency · Hospital ·
Medicine · Stop · Wait · Call · Police · Fire · Hurt · Breathe · Allergy ·
ThankYou · Please

## Architecture

```
webcam ──▶ MediaPipe HandLandmarker (WASM, CDN, lazy)
              48 landmarks (21 joints × 2 hands + 6 upper-body)
              ──▶ wrist-centric normalization  p̂ = (p − p_wrist) / ‖p_mcp − p_wrist‖
              ──▶ 30-frame ring buffer  [1, 30, 48, 3]
              ──▶ ONNX BiLSTM (onnxruntime-web, WASM)  < 3 MB
              ──▶ debounce: conf > 0.85 for > 5 consecutive frames
              ──▶ Web Speech Synthesis (TTS)

mic / text ──▶ Web Speech API (or typed text)
              ──▶ rule-based gloss tokenizer
                    ("What is your name?" → ["YOUR", "NAME", "WHAT"])
              ──▶ poses.json keyframes (20 words + A–Z)
              ──▶ Three.js avatar, quaternion SLERP @ 60 FPS
```

Privacy: raw webcam frames live only in local browser memory and are
discarded after inference. The only network requests are the lazy
MediaPipe model download (CDN) and the app's own static assets.

## Performance targets

| Metric | Target | Status (Phase 1) |
|---|---|---|
| Sign inference latency | < 100 ms (goal < 60 ms) | measured in-app, live badge |
| Avatar rendering | 60 FPS | requestAnimationFrame + SLERP |
| Boot time | < 1.5 s | static Vite bundle |
| ONNX model | < 3 MB | verified at export |
| Avatar assets | < 5 MB | procedural geometry, 0 external assets |

**Adaptive FPS**: Standard Mode (30 FPS) / Performance Mode (15 FPS) toggle
for older laptops and budget phones.

## Local setup

```bash
cd web
npm install
npm run dev        # dev server
npm run build      # production build -> web/dist
npm run preview    # serve the build: http://localhost:4173/omnisign-web/
npm test           # vitest unit tests (normalization, buffer, debounce, gloss)
```

### Train / re-export the model

```bash
cd ml
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu onnx onnxruntime numpy
.venv/bin/python export_onnx.py --epochs 30 --samples 400 --hidden 64
# writes web/public/models/omnisign_bilstm.onnx (< 3 MB), verifies with
# onnx.checker + an onnxruntime inference pass on a synthetic "Help" sample.
```

### Demo mode (no webcam needed)

Pick a sign from the **Simulate Sign** dropdown and click **Simulate Sign**.
A deterministic synthetic 30-frame landmark sequence (the same motion model
used to train the BiLSTM) is fed through the ONNX model locally and the
detected word, confidence and latency are shown.

## Project layout

```
omnisign-web/
├── ml/
│   ├── model.py          # BiLSTM + synthetic motion dataset (mirrors TS generator)
│   └── export_onnx.py    # train -> export ONNX opset 14 -> verify
└── web/
    ├── public/
    │   ├── models/omnisign_bilstm.onnx
    │   └── poses.json    # avatar keyframes: 20 words + A–Z (gen: scripts/gen-poses.mjs)
    └── src/
        ├── lib/          # normalization, buffer, debounce, gloss, synthetic, onnxClient, poses
        ├── components/Avatar.tsx   # procedural Three.js avatar
        └── App.tsx       # split-screen UI
```

## Swapping in real WLASL weights later

1. **Data**: download [WLASL](https://dxli94.github.io/WLASL/) (or
   [Voxel51](https://voxel51.com/) fiftyone WLASL) videos/landmarks for the
   20 emergency glosses (map WLASL labels to the 20 classes in
   `ml/model.py::SIGNS`).
2. **Preprocessing**: extract the same 48 landmarks per frame with MediaPipe,
   apply the identical wrist-centric normalization (`ml/model.py::normalize`,
   mirrored in `web/src/lib/normalization.ts`), and build 30-frame windows.
3. **Train**: replace `generate_dataset()` in `ml/model.py` with a real-data
   loader; the BiLSTM input shape `[B, 30, 48, 3]` and the export script stay
   unchanged. Keep the model small (hidden ≤ 64) to stay under the 3 MB budget.
4. **Export & drop in**: run `export_onnx.py`; the new
   `web/public/models/omnisign_bilstm.onnx` is picked up automatically — the
   web app's input pipeline and label order (`SIGNS`) are the contract.
5. **Avatar**: extend `web/public/poses.json` (regenerate via
   `web/scripts/gen-poses.mjs` after editing `src/lib/poses.ts`) with
   motion-captured keyframes per gloss for more natural signing.

## Known limitations (Phase 1)

- The shipped model is trained on **synthetic motion patterns**, not real
  signers — it proves the full edge pipeline end-to-end, but real-world
  accuracy requires WLASL training (see above).
- Webcam sign recognition uses hand landmarks only (no face/upper-body pose
  model yet); the 6 upper-body landmarks are derived geometrically.
- Speech recognition depends on the browser's Web Speech API (Chrome/Edge);
  Firefox/Safari fall back to the text input box.
- MediaPipe's hand-landmarker model downloads from a CDN on first camera use;
  everything else is fully offline after first load.

## License

MIT — free to use, adapt, and deploy.
