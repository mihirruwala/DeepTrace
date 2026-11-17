"""
File provenance: what the file says about itself, independent of any model.

Platforms strip metadata, so *absence* of a marker proves nothing. *Presence* of an
AI-generation marker is strong evidence, because people rarely forge it in that
direction. Content Credentials (C2PA) are detected, not cryptographically verified.
"""
from __future__ import annotations

import os

# IPTC digital source types (the standard way generators label their output)
AI_SOURCE = b"trainedAlgorithmicMedia"
AI_COMPOSITE = b"compositeWithTrainedAlgorithmicMedia"
# Settings blocks that image generators write into PNG text chunks
GEN_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "invokeai_metadata", "dream", "negative_prompt")
GEN_NAMES = ("midjourney", "dall-e", "dall·e", "firefly", "stable diffusion", "comfyui", "automatic1111",
             "novelai", "leonardo", "ideogram", "imagen", "flux", "sora", "runway", "pika", "kling", "veo",
             "chatgpt", "openai", "grok", "gemini")
EDITORS = ("photoshop", "lightroom", "gimp", "snapseed", "facetune", "picsart", "faceapp", "capcut", "remini")


def _scan(path: str, window: int = 4_000_000) -> bytes:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(window)
        if size > 2 * window:
            f.seek(size - window)
            return head + f.read()
        return head + f.read()


def inspect(path: str) -> dict:
    """-> {"ai_declared": bool, "findings": [(level, title, detail)], "camera": str|None}
    level: "ai" (declares AI), "info" (neutral facts), "edit" (edited with a known tool)."""
    blob = _scan(path)
    low = blob.lower()
    findings, camera, ai = [], None, False

    if AI_COMPOSITE in blob:
        ai = True
        findings.append(("ai", "Declared AI-edited",
                         "The file's metadata uses the IPTC label for media edited with generative AI."))
    elif AI_SOURCE in blob:
        ai = True
        findings.append(("ai", "Declared AI-generated",
                         "The file's metadata uses the IPTC label for media created by a generative model."))
    if b"c2pa" in low and (b"jumb" in low or b"jumd" in low):
        findings.append(("info", "Content Credentials present",
                         "This file carries a C2PA manifest. DeepTrace detects it but does not verify the "
                         "signature; check it at contentcredentials.org/verify."))

    try:
        from PIL import Image, ExifTags
        with Image.open(path) as im:
            info = {str(k).lower(): v for k, v in (im.info or {}).items()}
            keys = [k for k in GEN_KEYS if k in info]
            if keys:
                ai = True
                findings.append(("ai", "Generator settings embedded",
                                 f"The image contains a '{keys[0]}' block of the kind image generators "
                                 "(Stable Diffusion, ComfyUI and similar) write into their output."))
            ex = im.getexif()
            tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()} if ex else {}
            make, model = str(tags.get("Make", "")).strip(), str(tags.get("Model", "")).strip()
            if make or model:
                camera = (model if make.lower() in model.lower() else f"{make} {model}").strip()
            soft = " ".join(str(tags.get(k, "")) for k in ("Software", "ProcessingSoftware", "Artist"))
            soft += " " + str(info.get("software", ""))
            sl = soft.lower()
            if any(n in sl for n in GEN_NAMES):
                ai = True
                findings.append(("ai", "Made with an AI tool", f"Software tag: {soft.strip()[:80]}"))
            elif any(n in sl for n in EDITORS):
                findings.append(("edit", "Edited", f"Software tag: {soft.strip()[:80]}. Editing is not "
                                 "forgery, but the image is not straight from a camera."))
    except Exception:
        pass   # not an image, or unreadable metadata

    if not ai:
        hit = next((n for n in GEN_NAMES if n.encode() in low and n not in ("flux", "veo", "gemini", "grok")), None)
        if hit and (b"xmp" in low or b"<x:xmpmeta" in low):
            findings.append(("edit", "Mentions an AI tool",
                             f"The metadata mentions '{hit}', which may mean it was made or edited with it."))
    if camera:
        findings.append(("info", "Camera", camera))
    if not findings:
        findings.append(("info", "No provenance markers",
                         "Nothing in the metadata says how this file was made. That's normal: most apps and "
                         "platforms strip it, so this neither helps nor hurts the file's case."))
    return {"ai_declared": ai, "findings": findings, "camera": camera}
