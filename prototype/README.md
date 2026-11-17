# The first DeepTrace prototype

An early Streamlit prototype for checking whether a video has been manipulated. Upload a video (up to 200 MB), and the app pulls out up to 10 faces and scores each one with FuNet-A, a PyTorch model (`funet_a_full.pth`).

**The current version is at the [root of this repository](../README.md)** and runs live on [Hugging Face Spaces](https://huggingface.co/spaces/ruwalamihir/deepfake_detector). This folder is kept as a record of how it started.

## Files

| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `model_definitions.py`, `my_models.py` | Model code |
| `funet_a_full.pth` | Trained weights used by the prototype |
| `test*_app.py` | Earlier experiments |

## Run it

```bash
cd prototype
pip install -r requirements.txt torch facenet-pytorch
streamlit run app.py
```

`requirements.txt` doesn't list PyTorch yet, hence the extra packages above.

## Limits

A detector like this can be wrong, especially on compressed video or on generators it never saw in training. Treat its output as a signal to look closer, not as proof.
