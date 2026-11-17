"""
DeepTrace detection engine. No UI code in here.

Two independent detectors, both run on faces found by OpenCV YuNet:

  1. Face-manipulation detector (face swaps, re-enactment)
     EfficientNet-B7 checkpoints from the winning entry of the Deepfake
     Detection Challenge (Seferbekov, 2020, MIT). Trained on DFDC: ~100k clips
     of consented actors filmed on phones and webcams, which is why it stays
     calm on ordinary home video where benchmark-trained models panic.

  2. AI-generation detector (fully synthetic images: diffusion, GAN)
     Community Forensics ViT-S/16-384 (Park & Owens, CVPR 2025, MIT).

Thresholds and the quality gate come from the benchmark in EVALUATION.md.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("DT_WEIGHTS", os.path.join(HERE, "weights"))

# ── decision bands (aggregated probability) ──────────────────────────────────
T_REAL = 0.30          # at or below: no evidence of manipulation
T_FAKE = 0.70          # at or above: likely manipulated
T_AIGEN = 0.70         # AI-generation score at or above: likely AI-generated
# ── quality gate: below these the models cannot tell blur from forgery ───────
MIN_BRIGHT = 40.0      # mean grey level of the face crop (0-255)
MIN_SHARP = 8.0        # variance of the Laplacian of the face at 256x256
MIN_FACE_PX = 48
MIN_FRAMES = 5
CAM_SCALE = 60.0       # activation that maps to full heat; fakes peak ~75, clean faces ~2
CASCADE_SKIP = 0.10    # first checkpoint this sure it's clean: skip the second

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ───────────────────────────── models ─────────────────────────────
class SwapNet(nn.Module):
    """DFDC-winner architecture: EfficientNet-B7 (noisy student) + linear head."""

    def __init__(self):
        super().__init__()
        import timm
        self.encoder = timm.create_model("tf_efficientnet_b7", pretrained=False, num_classes=0, global_pool="")
        self.fc = nn.Linear(2560, 1)

    def forward(self, x):
        return self.fc(self.encoder(x).mean((2, 3)))


def _load_st(path: str) -> dict:
    from safetensors.torch import load_file
    return {k: v.float() for k, v in load_file(path).items()}


def load_swap_models() -> list[nn.Module]:
    paths = sorted(glob.glob(os.path.join(WEIGHTS, "swap_*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No swap_*.safetensors in {WEIGHTS}")
    models = []
    for p in paths:
        m = SwapNet()
        m.load_state_dict(_load_st(p), strict=True)
        models.append(m.eval())
    return models


def load_aigen_model() -> nn.Module:
    import timm
    m = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
    m.load_state_dict(_load_st(os.path.join(WEIGHTS, "aigen_vits384.safetensors")), strict=True)
    return m.eval()


# ───────────────────────── preprocessing ─────────────────────────
def prep_swap(face_bgr: np.ndarray, size: int = 380) -> torch.Tensor:
    """Isotropic resize to 380 and centre on a black square (as in training)."""
    h, w = face_bgr.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(face_bgr, (max(1, int(w * s)), max(1, int(h * s))),
                    interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((size, size, 3), np.uint8)
    h, w = im.shape[:2]
    y0, x0 = (size - h) // 2, (size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = im
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1).copy())


def prep_aigen(bgr: np.ndarray) -> torch.Tensor:
    """Resize short side to 440, centre-crop 384 (the model's eval transform)."""
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    w, h = pil.size
    s = 440 / min(w, h)
    pil = pil.resize((max(384, round(w * s)), max(384, round(h * s))), Image.BILINEAR)
    w, h = pil.size
    l, t = (w - 384) // 2, (h - 384) // 2
    a = np.asarray(pil.crop((l, t, l + 384, t + 384)), np.float32) / 255.0
    return torch.from_numpy(((a - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1).copy())


def crop_box(bgr: np.ndarray, box, margin: float) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    mw, mh = int(w * margin), int(h * margin)
    return bgr[max(y - mh, 0):y + h + mh, max(x - mw, 0):x + w + mw]


def heat_overlay(face_bgr: np.ndarray, cam: np.ndarray, scale: float = CAM_SCALE) -> np.ndarray:
    """Overlay the positive ("fake") evidence on the face crop, RGB out.
    Undo the letterboxing prep_swap applied, so the map lines up with the crop.
    The scale is fixed (not per image), so a clean face genuinely shows no heat."""
    h, w = face_bgr.shape[:2]
    s = 380 / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    y0, x0 = (380 - nh) // 2, (380 - nw) // 2
    full = cv2.resize(np.maximum(cam, 0), (380, 380), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(full[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m / max(scale, 1e-6), 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, w / 60))
    alpha = (m * 0.85)[..., None]
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    heat = np.zeros_like(rgb)
    heat[..., 0], heat[..., 1], heat[..., 2] = 255, 90 + 120 * (1 - m), 60
    d = 0.55 * float(m.max())                                    # desaturate only where there is heat to show
    grey = rgb.mean(-1, keepdims=True) * d + rgb * (1 - d)
    return np.clip(grey * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


def annotate(frame_bgr: np.ndarray, boxes: list[tuple], max_w: int = 1100) -> np.ndarray:
    """Draw each face box with its score; colours follow the verdict bands. RGB out."""
    s = min(1.0, max_w / frame_bgr.shape[1])
    im = cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame_bgr.copy()
    t = max(2, int(im.shape[1] / 400))
    for box, sc, label in boxes:
        x, y, w, h = [int(v * s) for v in box]
        col = (78, 90, 255) if sc >= T_FAKE else ((106, 243, 201) if sc <= T_REAL else (75, 184, 242))
        cv2.rectangle(im, (x, y), (x + w, y + h), col, t, cv2.LINE_AA)
        txt = f"{label}  {sc * 100:.0f}%"
        fs = max(0.5, im.shape[1] / 1600)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(th + 8, y - 6)
        cv2.rectangle(im, (x - t // 2, ty - th - 8), (x + tw + 10, ty + 4), col, -1)
        cv2.putText(im, txt, (x + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (14, 15, 13), 1, cv2.LINE_AA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def quality(face_bgr: np.ndarray) -> tuple[float, float]:
    g = cv2.cvtColor(cv2.resize(face_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
    return float(g.mean()), float(cv2.Laplacian(g, cv2.CV_64F).var())


# ───────────────────────── face detection ────────────────────────
class FaceDetector:
    """OpenCV YuNet. Boxes are returned in original-image pixels, largest first."""

    def __init__(self, max_side: int = 960, score: float = 0.6):
        path = os.path.join(WEIGHTS, "yunet.onnx")
        self.det = cv2.FaceDetectorYN.create(path, "", (320, 320), score, 0.3, 50)
        self.max_side = max_side

    def detect(self, bgr: np.ndarray) -> list[dict]:
        h, w = bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(bgr, (int(round(w * s)), int(round(h * s)))) if s < 1 else bgr
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, out = self.det.detect(img)
        faces = []
        for r in (out if out is not None else []):
            box = r[:4] / s
            if min(box[2], box[3]) >= MIN_FACE_PX:
                faces.append({"box": box, "score": float(r[14])})
        faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
        return faces


# ───────────────────────────── results ───────────────────────────
class NoFaceError(Exception):
    pass


@dataclass
class Result:
    kind: str                         # "video" | "image"
    verdict: str                      # "real" | "fake" | "aigen" | "inconclusive"
    fake_prob: float                  # headline score, 0..1 (see `headline`)
    headline: str                     # "swap" | "aigen": which detector drives fake_prob
    swap_prob: float                  # face-manipulation score
    aigen_prob: float | None          # AI-generation score (None if not run)
    scores: list[float]               # per analysed frame (video) / per face (image)
    frame_idx: list[int]
    n_sampled: int = 0
    n_dropped: int = 0
    n_faces: int = 1
    low_quality: bool = False
    brightness: float = 0.0
    sharpness: float = 0.0
    faces: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    thumbs: list[tuple] = field(default_factory=list)   # (label, score, rgb array)
    heat: list[tuple] = field(default_factory=list)     # (label, score, rgb overlay), same order as thumbs
    evidence: np.ndarray | None = None                  # annotated frame / photo, rgb
    evidence_label: str = ""


def decide(swap: float, aigen: float | None, low_quality: bool, enough: bool = True) -> tuple[str, str, float]:
    """-> (verdict, headline detector, headline score)."""
    ai_lead = aigen is not None and aigen >= T_AIGEN and aigen >= swap
    if low_quality or not enough:      # the gate applies to both detectors
        return ("inconclusive", "aigen", aigen) if ai_lead else ("inconclusive", "swap", swap)
    if ai_lead:
        return "aigen", "aigen", aigen
    if swap >= T_FAKE:
        return "fake", "swap", swap
    if swap <= T_REAL and (aigen is None or aigen < 0.5):
        return "real", "swap", swap
    if aigen is not None and aigen >= 0.5 and aigen > swap:
        return "inconclusive", "aigen", aigen
    return "inconclusive", "swap", swap


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class Engine:
    def __init__(self, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.det = FaceDetector()
        self.swap = load_swap_models()
        self.aigen = load_aigen_model()

    @torch.inference_mode()
    def score_swap(self, crops: list[np.ndarray], batch: int = 4, progress=None) -> list[float]:
        """Mean of the checkpoints. To save CPU, later checkpoints are skipped when the
        first one finds the whole set confidently clean (mean <= CASCADE_SKIP)."""
        progress = progress or (lambda p: None)
        xs = [torch.stack([prep_swap(c) for c in crops[i:i + batch]]) for i in range(0, len(crops), batch)]
        per_model, cams = [], []
        for k, m in enumerate(self.swap):
            sc = []
            for j, x in enumerate(xs):
                if k == 0:
                    # Exact class-activation map: the head is a linear layer on
                    # global-average-pooled features, so w . A(h, w) is each spatial
                    # cell's contribution to the "fake" logit. No extra compute.
                    feats = m.encoder(x)
                    logit = m.fc(feats.mean((2, 3)))
                    cams += torch.einsum("c,bchw->bhw", m.fc.weight[0], feats).numpy().tolist()
                else:
                    logit = m(x)
                sc += torch.sigmoid(logit)[:, 0].tolist()
                progress((k + (j + 1) / len(xs)) / len(self.swap))
            per_model.append(sc)
            if k == 0 and float(np.mean(sc)) <= CASCADE_SKIP:
                break
        progress(1.0)
        self.last_cams = [np.asarray(c, np.float32) for c in cams]
        return np.mean(per_model, axis=0).tolist()

    @torch.inference_mode()
    def score_aigen(self, images: list[np.ndarray]) -> list[float]:
        x = torch.stack([prep_aigen(im) for im in images])
        return torch.sigmoid(self.aigen(x))[:, 0].tolist()

    # ───────────── video ─────────────
    def analyse_video(self, path: str, max_frames: int = 16, progress=None) -> Result:
        progress = progress or (lambda p: None)
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            raise ValueError("Could not read this video. Is it a valid video file?")
        # Evenly spaced over the whole clip; a second set of in-between frames is
        # held in reserve for clips where the face is often missing.
        idxs = sorted(set(np.linspace(0, total - 1, min(total, max_frames * 2)).astype(int).tolist()))
        primary = idxs[::2] if len(idxs) > max_frames else idxs
        reserve = [i for i in idxs if i not in set(primary)]

        tracks: list[dict] = []
        stats = {"sampled": 0, "dropped": 0}
        frames_kept: list[np.ndarray] = []

        def handle(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                return
            stats["sampled"] += 1
            faces = self.det.detect(fr)[:4]
            if not faces:
                stats["dropped"] += 1
                return
            if len(frames_kept) < 64:
                frames_kept.append(fr)
            for f in faces:
                x, y, w, h = f["box"]
                c = np.array([x + w / 2, y + h / 2])
                tr = next((t for t in tracks if np.linalg.norm(t["c"] - c) < 0.75 * max(t["s"], w)), None)
                if tr is None:
                    tr = {"c": c, "s": w, "items": []}
                    tracks.append(tr)
                tr["c"], tr["s"] = c, w
                tr["items"].append({"idx": i, "crop": crop_box(fr, f["box"], 1 / 3),
                                    "px": float(min(w, h)), "frame": fr, "box": f["box"]})

        for n, i in enumerate(primary):
            handle(i)
            progress(0.25 * (n + 1) / len(primary))
        if max((len(t["items"]) for t in tracks), default=0) < min(max_frames, 8):
            for i in reserve:
                handle(i)
        cap.release()

        if not tracks:
            raise NoFaceError(
                "DeepTrace couldn't find a clear face in this video. It analyses faces "
                "specifically, so clips without a visible face aren't supported.")
        top = max(len(t["items"]) for t in tracks)
        # People who are on screen for a meaningful share of the clip, up to three.
        tracks = sorted((t for t in tracks if len(t["items"]) >= max(2, 0.4 * top) or len(t["items"]) == top),
                        key=lambda t: -len(t["items"]))[:3]
        budget = max(6, max_frames // len(tracks)) if len(tracks) > 1 else max_frames
        for t in tracks:
            if len(t["items"]) > budget:
                keep = np.linspace(0, len(t["items"]) - 1, budget).astype(int)
                t["items"] = [t["items"][k] for k in keep]

        all_crops = [it["crop"] for t in tracks for it in t["items"]]
        all_scores = self.score_swap(all_crops, progress=lambda p: progress(0.25 + 0.65 * p))
        all_cams = self.last_cams
        people, k = [], 0
        for t in tracks:
            n = len(t["items"])
            sc = all_scores[k:k + n]
            for it, cam in zip(t["items"], all_cams[k:k + n]):
                it["cam"] = cam
            k += n
            q = np.array([quality(it["crop"]) for it in t["items"]])
            people.append({"scores": sc, "items": t["items"], "prob": float(np.mean(sc)),
                           "bright": float(np.median(q[:, 0])), "sharp": float(np.median(q[:, 1]))})

        # A clip is as suspicious as its worst face, but only faces clear enough to
        # judge get a vote. If none is clear enough, the clip gets no verdict.
        for r in people:
            r["lowq"] = r["bright"] < MIN_BRIGHT or r["sharp"] < MIN_SHARP
        judged = [r for r in people if not r["lowq"]] or people
        skipped = len(people) - len(judged)
        lead = max(judged, key=lambda r: r["prob"])
        self.last_items = lead["items"]

        # AI-generation check on a handful of whole frames + the lead face.
        pick = np.linspace(0, len(lead["items"]) - 1, min(4, len(lead["items"]))).astype(int)
        ag_in = [lead["items"][j]["frame"] for j in pick] + \
                [crop_box(lead["items"][j]["frame"], lead["items"][j]["box"], 0.4) for j in pick]
        ag = self.score_aigen(ag_in)
        n4 = len(pick)
        aigen = float(max(np.mean(ag[:n4]), np.mean(ag[n4:])))
        progress(1.0)

        low_q = lead["lowq"]
        enough = len(lead["scores"]) >= MIN_FRAMES
        verdict, headline, p = decide(lead["prob"], aigen, low_q, enough)

        notes = []
        if low_q:
            why = "too dark" if lead["bright"] < MIN_BRIGHT else "too blurry or too compressed"
            notes.append(f"The face is {why} for a reliable call. Heavy blur and noise look the same as "
                         "forgery artefacts to a detector, so DeepTrace won't give a verdict on this footage.")
        if not enough:
            notes.append(f"Only {len(lead['scores'])} usable face frame(s) were found, which is too few for a verdict.")
        if np.median([it["px"] for it in lead["items"]]) < 100:
            notes.append("The face is small in frame. Detection is less reliable on small faces.")
        if len(judged) > 1:
            notes.append(f"{len(judged)} people were analysed separately. The verdict follows the most suspicious face.")
        if skipped:
            notes.append(f"{skipped} other face(s) in the clip were too dark or blurry to judge and were left out of the verdict.")
        spread = float(np.std(lead["scores"]))
        if verdict == "inconclusive" and not low_q and enough and spread > 0.3:
            notes.append("Scores swing a lot between frames. That can mean a partial edit, or just difficult footage.")

        order = np.argsort(lead["scores"])[::-1][:6]
        heat = [(f"frame {lead['items'][j]['idx']}", lead["scores"][j],
                 heat_overlay(lead["items"][j]["crop"], lead["items"][j]["cam"]))
                for j in order]
        top = lead["items"][order[0]]
        boxes = []
        for pi, r in enumerate(people):
            for it, sc_ in zip(r["items"], r["scores"]):
                if it["idx"] == top["idx"]:
                    boxes.append((it["box"], sc_, f"person {pi + 1}" if len(people) > 1 else "face"))
        evidence = annotate(top["frame"], boxes)
        return Result(
            heat=heat, evidence=evidence,
            evidence_label=f"Frame {top['idx']}, the most suspicious moment for the lead face",
            kind="video", verdict=verdict, fake_prob=p, headline=headline,
            swap_prob=lead["prob"], aigen_prob=aigen,
            scores=lead["scores"], frame_idx=[it["idx"] for it in lead["items"]],
            n_sampled=stats["sampled"], n_dropped=stats["dropped"], n_faces=len(people),
            low_quality=low_q, brightness=lead["bright"], sharpness=lead["sharp"],
            faces=[{"prob": r["prob"], "n": len(r["scores"]),
                    "thumb": _rgb(r["items"][len(r["items"]) // 2]["crop"])} for r in people],
            notes=notes,
            thumbs=[(f"frame {lead['items'][j]['idx']}", lead["scores"][j], _rgb(lead["items"][j]["crop"])) for j in order],
        )

    # ───────────── photo ─────────────
    def analyse_image(self, bgr: np.ndarray) -> Result:
        faces = self.det.detect(bgr)[:4]
        if not faces:
            raise NoFaceError(
                "DeepTrace couldn't find a clear face in this photo. It analyses faces "
                "specifically, so images without a visible face aren't supported.")
        crops = [crop_box(bgr, f["box"], 1 / 3) for f in faces]
        # Flip-averaging steadies single-image scores.
        sc = self.score_swap([c for crop in crops for c in (crop, cv2.flip(crop, 1))])
        cams = self.last_cams[0::2]
        per_face = [float(np.mean(sc[i:i + 2])) for i in range(0, len(sc), 2)]
        lead = int(np.argmax(per_face))

        ag = self.score_aigen([bgr] + [crop_box(bgr, f["box"], 0.4) for f in faces])
        aigen = float(max(ag))

        bright, sharp = quality(crops[lead])
        low_q = bright < MIN_BRIGHT or sharp < MIN_SHARP
        verdict, headline, p = decide(per_face[lead], aigen, low_q)

        notes = ["A single photo carries far less evidence than a video. Treat this as a lead, not proof."]
        if low_q:
            notes.append("The face is too dark or too blurry for a reliable call on face manipulation.")
        if min(faces[lead]["box"][2:]) < 100:
            notes.append("The face is small in frame, which reduces reliability.")
        if len(faces) > 1:
            notes.append(f"{len(faces)} faces were analysed. The verdict follows the most suspicious one.")
        order = np.argsort(per_face)[::-1]
        heat = [(f"face {k + 1}", per_face[j], heat_overlay(crops[j], cams[j]))
                for k, j in enumerate(order)]
        evidence = annotate(bgr, [(faces[j]["box"], per_face[j], f"face {k + 1}" if len(faces) > 1 else "face")
                                  for k, j in enumerate(order)])
        return Result(
            heat=heat, evidence=evidence, evidence_label="Every face found in the photo",
            kind="image", verdict=verdict, fake_prob=p, headline=headline,
            swap_prob=per_face[lead], aigen_prob=aigen,
            scores=per_face, frame_idx=list(range(len(per_face))), n_sampled=1, n_faces=len(per_face),
            low_quality=low_q, brightness=bright, sharpness=sharp,
            faces=[{"prob": per_face[j], "aigen": ag[1 + j], "n": 1, "thumb": _rgb(crops[j])} for j in order],
            notes=notes,
            thumbs=[(f"face {k + 1}", per_face[j], _rgb(crops[j])) for k, j in enumerate(order)],
        )
