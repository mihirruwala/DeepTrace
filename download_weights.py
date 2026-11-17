"""Fetch the model weights from the DeepTrace Space on Hugging Face into ./weights.

GitHub can't hold files over 100 MB, so the weights live with the Space instead:
https://huggingface.co/spaces/ruwalamihir/deepfake_detector/tree/main/weights

    python3 download_weights.py
"""

import os
import sys
import urllib.request

BASE = "https://huggingface.co/spaces/ruwalamihir/deepfake_detector/resolve/main/weights/"
FILES = ("swap_a.safetensors", "swap_b.safetensors", "aigen_vits384.safetensors")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def main():
    os.makedirs(HERE, exist_ok=True)
    for name in FILES:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            print("have", name)
            continue
        print("downloading", name, "...", flush=True)
        urllib.request.urlretrieve(BASE + name, path + ".part")
        os.replace(path + ".part", path)
    print("weights ready in", HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
