"""OmniSign BiLSTM sign classifier.

Input:  [batch, 30, 48, 3] wrist-centric normalized landmarks
         (21 joints x 2 hands + 6 upper-body landmarks, x/y/z)
Model:  flatten per frame -> 144 features -> BiLSTM(hidden, 2 layers)
        -> last hidden state -> Linear -> 20 emergency-sign logits.

Training data is SYNTHETIC: each class is a distinct sinusoidal motion
prototype over 30 frames (mirrors web/src/lib/synthetic.ts exactly).
Swap in real WLASL data later by replacing `generate_dataset` — the
model, normalization and ONNX export stay unchanged.
"""

import math
import os

import numpy as np

SIGNS = [
    "Help", "Doctor", "Pain", "Water", "Yes", "No", "Family", "Emergency",
    "Hospital", "Medicine", "Stop", "Wait", "Call", "Police", "Fire", "Hurt",
    "Breathe", "Allergy", "ThankYou", "Please",
]
FRAMES = 30
LANDMARKS = 48
NUM_CLASSES = len(SIGNS)


def mulberry32(seed: int):
    """Deterministic PRNG matching the TypeScript implementation."""
    a = seed & 0xFFFFFFFF

    def rand():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rand


def base_position(i: int):
    if i < 42:
        hand = 0 if i < 21 else 1
        j = i % 21
        ox = 0.25 if hand == 0 else -0.25
        if j == 0:
            return [ox, 0.45, 0.0]
        finger = (j - 1) // 4
        seg = (j - 1) % 4
        x = ox + (finger - 2) * 0.04 + (-0.03 if finger == 0 else 0.0)
        y = 0.45 + 0.03 + seg * 0.035
        z = 0.01 * seg
        return [x, y, z]
    body = [
        [-0.2, 0.95, 0.0], [0.2, 0.95, 0.0],
        [-0.32, 0.68, 0.0], [0.32, 0.68, 0.0],
        [-0.25, 0.45, 0.0], [0.25, 0.45, 0.0],
    ]
    return body[i - 42]


def class_motion(class_index: int):
    return {
        "freq": 0.5 + 0.09 * class_index,  # unique per class
        "amp": 0.05 + 0.02 * (class_index % 3),
        "axis": class_index % 3,
        "hand": class_index % 3,  # 0 right, 1 left, 2 both
    }


def generate_sequence(class_index: int, seed: int) -> np.ndarray:
    """One synthetic [30, 48, 3] landmark sequence (raw, unnormalized)."""
    m = class_motion(class_index)
    rand = mulberry32(seed * 1000 + class_index * 97 + 13)
    out = np.zeros((FRAMES, LANDMARKS, 3), dtype=np.float32)
    for t in range(FRAMES):
        phase = 2 * math.pi * m["freq"] * t / FRAMES
        drift = 0.02 * math.sin(2 * math.pi * 0.5 * t / FRAMES + class_index)
        for i in range(LANDMARKS):
            base = base_position(i)
            disp = drift
            is_right = i < 21
            is_left = 21 <= i < 42
            active = (
                m["hand"] == 2
                or (m["hand"] == 0 and is_right)
                or (m["hand"] == 1 and is_left)
            )
            if active:
                disp += m["amp"] * math.sin(phase + 0.3 * i)
            elif i >= 42:
                disp += m["amp"] * 0.3 * math.sin(phase + 0.3 * i)
            noise = (rand() + rand() + rand() - 1.5) * 0.01
            p = list(base)
            p[m["axis"]] += disp
            p[0] += noise
            p[1] += noise * 0.5
            p[2] += noise * 0.5
            out[t, i] = p
    return out


def normalize_batch(X: np.ndarray) -> np.ndarray:
    """Vectorized wrist-centric normalization for a [N, 30, 48, 3] batch.

    Identical math to normalize() / web/src/lib/normalization.ts, applied
    to the whole batch at once with NumPy broadcasting.
    """
    out = np.empty_like(X)
    for lo, hi, origin_idx, scale_idx in ((0, 21, 0, 9), (21, 42, 21, 30)):
        origin = X[:, :, origin_idx, :]  # [N, 30, 3]
        scale = np.maximum(
            np.linalg.norm(X[:, :, scale_idx, :] - origin, axis=-1, keepdims=True),
            1e-6,
        )
        out[:, :, lo:hi, :] = (X[:, :, lo:hi, :] - origin[:, :, None, :]) / scale[:, :, None, :]
    shoulder_l, shoulder_r = X[:, :, 42, :], X[:, :, 43, :]
    mid = (shoulder_l + shoulder_r) / 2
    width = np.maximum(
        np.linalg.norm(shoulder_r - shoulder_l, axis=-1, keepdims=True), 1e-6
    )
    out[:, :, 42:, :] = (X[:, :, 42:, :] - mid[:, :, None, :]) / width[:, :, None, :]
    return out


def normalize(seq: np.ndarray) -> np.ndarray:
    """Wrist-centric normalization for one [30, 48, 3] sequence."""
    return normalize_batch(seq[None, :, :, :])[0]


def generate_dataset(samples_per_class: int = 400, seed: int = 0):
    """Vectorized dataset generation.

    Same motion model as generate_sequence() / web/src/lib/synthetic.ts:
    per-class sinusoidal prototype + slow drift + small noise. Noise is drawn
    from a normal distribution with the same scale as the TS generator's
    (u1+u2+u3-1.5)*0.01 term (std ~0.005 on x, half that on y/z); the exact
    noise realization is random per sample in both implementations, so only
    the prototype math needs to match — and it does, element for element.
    """
    rng = np.random.default_rng(seed)
    base = np.array([base_position(i) for i in range(LANDMARKS)], dtype=np.float32)
    t = np.arange(FRAMES, dtype=np.float32)
    idx = np.arange(LANDMARKS, dtype=np.float32)
    X, y = [], []
    for c in range(NUM_CLASSES):
        m = class_motion(c)
        phase = 2 * math.pi * m["freq"] * t / FRAMES  # [30]
        sine = m["amp"] * np.sin(phase[:, None] + 0.3 * idx[None, :])  # [30, 48]
        drift = 0.02 * np.sin(2 * math.pi * 0.5 * t / FRAMES + c)  # [30]
        # Every landmark drifts; active hand(s) get the full sine, the
        # upper body gets 0.3x — exactly as in synthetic.ts.
        disp = np.tile(drift[:, None], (1, LANDMARKS))
        hand_active = np.zeros(LANDMARKS, dtype=bool)
        if m["hand"] in (0, 2):
            hand_active[0:21] = True
        if m["hand"] in (1, 2):
            hand_active[21:42] = True
        disp[:, hand_active] += sine[:, hand_active]
        # Body: full sine when both hands move (hand==2), else 0.3x —
        # mirrors the if/elif in synthetic.ts exactly.
        disp[:, 42:] += (sine[:, 42:] if m["hand"] == 2 else 0.3 * sine[:, 42:])
        motion = np.zeros((FRAMES, LANDMARKS, 3), dtype=np.float32)
        motion[:, :, m["axis"]] = disp
        noise = rng.normal(0, 0.005, size=(samples_per_class, FRAMES, LANDMARKS, 3))
        noise[:, :, :, 1] *= 0.5
        noise[:, :, :, 2] *= 0.5
        seqs = base[None, None, :, :] + motion[None, :, :, :] + noise
        X.append(normalize_batch(seqs.astype(np.float32)))
        y.extend([c] * samples_per_class)
    X = np.concatenate(X).astype(np.float32)  # [N, 30, 48, 3]
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def build_model(hidden: int = 64, layers: int = 2):
    import torch
    import torch.nn as nn

    class SignBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=LANDMARKS * 3,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
            )
            self.head = nn.Linear(hidden * 2, NUM_CLASSES)

        def forward(self, x):
            # x: [B, 30, 48, 3] -> [B, 30, 144]
            x = x.reshape(x.shape[0], FRAMES, LANDMARKS * 3)
            out, (h, _) = self.lstm(x)
            last = torch.cat([h[-2], h[-1]], dim=1)  # both directions, last layer
            return self.head(last)

    return SignBiLSTM()


def train(epochs: int = 30, samples_per_class: int = 400, hidden: int = 64):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    X, y = generate_dataset(samples_per_class)
    n_train = int(0.9 * len(y))
    Xtr, ytr, Xva, yva = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    model = build_model(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    bs = 128
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss, correct = 0.0, 0
        for i in range(0, n_train, bs):
            idx = perm[i : i + bs]
            xb = torch.from_numpy(Xtr[idx.numpy()])
            yb = torch.from_numpy(ytr[idx.numpy()])
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        model.eval()
        with torch.no_grad():
            va_logits = model(torch.from_numpy(Xva))
            va_acc = (va_logits.argmax(1).numpy() == yva).mean()
        print(
            f"epoch {epoch + 1:>2}/{epochs}  loss {total_loss / n_train:.4f}  "
            f"train_acc {correct / n_train:.4f}  val_acc {va_acc:.4f}",
            flush=True,
        )
    return model


if __name__ == "__main__":
    m = train()
    os.makedirs("checkpoints", exist_ok=True)
    import torch

    torch.save(m.state_dict(), "checkpoints/bilstm.pt")
    print("saved checkpoints/bilstm.pt")
