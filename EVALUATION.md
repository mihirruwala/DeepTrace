# DeepTrace v5 evaluation

Run on 16 Nov 2026 with the exact engine and settings this Space ships
(16 frames per clip, two checkpoints, quality gate on, bands 0.30 / 0.70).
Every number below is a full-pipeline result: file in, verdict out.

## Why v4 was replaced

v4 used MesoNet-4 (about 28,000 parameters) trained on 332 face crops. It scored a
plain, well-lit, unedited webcam recording at 92% "manipulated", and did the same to
most ordinary home video. A model that small, trained on that little, memorises its
training set and treats anything unfamiliar as fake. Retraining it for more epochs
would not have helped; the fix was a different model and a different way of deciding.

## What was tried

| Candidate | Result on 14 real home-webcam clips | Kept? |
|---|---|---|
| MesoNet-4 (v4) | flagged the clear clip at 92% | no |
| GenD CLIP ViT-L/14 (WACV 2026, state of the art on academic benchmarks) | median score 0.90, every clip looked "fake" | no |
| DFDC-winner EfficientNet-B7 | median score 0.05 | **yes** |

GenD is very good at catching fakes, but it was trained on FaceForensics++ only, and
on consumer webcam footage it cannot tell a real face from a forged one. The
DFDC-winner models were trained on about 100,000 clips of paid actors filmed on phones
and webcams, which is the kind of video people actually upload. Two of its seven
checkpoints are used (`final_111_..._0_36` and `final_777_..._0_31`); the full seven
were no better on the development set and would be 3.5x slower.

For images made from scratch by a generator, the Community Forensics ViT-S/16-384
detector was added as a second, independent check.

## Results

"Correct" means a real file got *no manipulation found*, or a fake file got *likely
manipulated / AI-generated*. "Wrong" is the opposite call. "Inconclusive" includes
files where no face was found.

| Test set | Files | Correct | Inconclusive | Wrong |
|---|---:|---:|---:|---:|
| Home webcam clips, real (Photo Booth) | 14 | 9 | 5 | 0 |
| Google DFD, real | 16 | 14 | 2 | 0 |
| Celeb-DF v2, real (held-out) | 40 | 28 | 10 | 2 |
| WildDeepfake, real | 13 | 7 | 6 | 0 |
| Phone selfie clips, real (held-out) | 5 | 5 | 0 | 0 |
| Google DFD, face swaps | 16 | 11 | 3 | 2 |
| Celeb-DF v2, face swaps (held-out) | 40 | 30 | 8 | 2 |
| WildDeepfake, internet fakes | 14 | 3 | 11 | 0 |
| Commercial face swaps (held-out) | 5 | 1 | 3 | 1 |
| Real photos: FFHQ, CelebA-HQ, selfies (held-out) | 85 | 79 | 3 | 3 |
| AI-generated faces: Flux, SDXL (held-out) | 120 | 87 | 14 | 19 |

Totals:

- **Real videos (88):** 63 clean (72%), 23 inconclusive (26%), **2 wrongly flagged (2%)**.
- **Fake videos (75):** 45 flagged (60%), 25 inconclusive (33%), **5 passed as clean (7%)**.
- **Held-out videos only (45 real, 45 fake):** 4% of real wrongly flagged, 69% of fakes flagged, 7% of fakes passed as clean.
- **Real photos (85):** 3 wrongly flagged (4%). **AI-generated photos (120):** 87 flagged (73%), 19 passed as clean (16%).

Held-out sets played no part in choosing models, thresholds or the quality gate. The
other sets were used for those choices, so their numbers are optimistic.

## How the verdict is made

1. Mean manipulation score across frames. At or below 0.30: clean. At or above 0.70: flagged.
2. In between: inconclusive. Most wrong answers a single threshold would give live in this band.
3. Quality gate: if the face crop's median brightness is under 40 (of 255) or its
   Laplacian variance is under 8, no verdict. Every real clip that scored high in
   development was dark, smeared by noise reduction, or very blurry. To a detector
   those look like forgery artefacts.
4. With several people in frame, each is scored separately and the verdict follows
   the most suspicious face that is clear enough to judge.
5. The AI-generation score flags a file when it is 0.70 or higher and above the
   manipulation score.

## v5.1 additions (16 Nov 2026)

- **Heatmaps** are exact class activation maps from the first face-manipulation
  checkpoint (its head is a linear layer on pooled features). The colour scale is
  fixed at an activation of 60: on the test clips fake faces peaked around 75 and
  clean faces around 2, so a clean face shows no heat.
- **Provenance** reads IPTC `DigitalSourceType` AI labels, generator settings in PNG
  text chunks, AI tool names in software tags, and the presence of a C2PA manifest
  (detected, not verified). An AI label flags the file even when the models miss it.
  None of the 327 benchmark files carried such a label, so the numbers above are
  unchanged by this step.

## What these numbers do not show

- The samples are small. A rate measured on 40 files can easily be off by 10 points.
- WildDeepfake is low-resolution, so most of it is stopped by the quality gate. That
  is the gate working as intended, but it means DeepTrace says little about
  low-quality internet reposts.
- Only five clips from a current commercial face-swap tool were tested, and one of
  five was flagged. Expect modern, high-quality swaps to be missed more often than
  the older benchmark fakes.
- No fully AI-generated *videos* (Sora, Veo and similar) were tested. The
  AI-generation detector was validated on still images only.
- Nobody tried to fool it on purpose. A motivated forger can beat any public detector.

## Sources

- DFDC winner: https://github.com/selimsef/dfdc_deepfake_challenge
- Community Forensics: https://arxiv.org/abs/2411.04125
- GenD (evaluated, not used): https://arxiv.org/abs/2508.06248
- Test data: Google DFD, WildDeepfake, Celeb-DF v2, UniData deepfake videos, FFHQ,
  CelebA-HQ and bitmind SyntheticFacesHighQuality-T2I, all via Hugging Face.
