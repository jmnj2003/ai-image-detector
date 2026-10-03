import csv
import itertools
import os
import random
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from diffusers import StableDiffusionXLPipeline

OUT_DIR = os.environ.get("OUT_DIR", "downloaded_data/attribution_train/opensource")
META = os.path.join(OUT_DIR, "metadata.csv")
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS, GUIDANCE, SIZE = 30, 5.0, 1024
N_IMAGES = int(os.environ.get("N_IMAGES", "400"))
SEED_BASE = int(os.environ.get("SEED_BASE", "1000"))  # offset from the original 50's seeds (1-50)

# --- prompt generation: combinatorial, so no manual writing of 400 sentences,
# and no overlap with the original 50 shared prompts used for the cross-
# generator test set. ---
SUBJECTS = [
    "a delivery courier", "a street musician", "a construction worker", "a yoga instructor",
    "a chess player", "a bookstore owner", "a mechanic", "a nurse", "a beekeeper",
    "a night-shift baker", "a tour guide", "a farmer", "a photographer", "a tailor",
    "a lighthouse keeper", "a barista", "a firefighter", "a violinist", "a potter",
    "a park ranger",
]
SETTINGS = [
    "in a crowded night market", "on a quiet mountain trail", "inside a glass greenhouse",
    "at a busy harbor at dawn", "in an old brick workshop", "on a rooftop terrace at dusk",
    "beside a canal in an old town", "in a sunlit university library", "at a rural train platform",
    "inside a vintage record shop", "on a foggy vineyard hillside", "in a bustling spice market",
    "at an outdoor ice rink", "inside a small pottery studio", "on a desert campsite at night",
]
STYLES = [
    "photographed in natural light", "shot on a rainy afternoon", "captured mid-motion",
    "with soft golden-hour lighting", "in a candid documentary style", "with dramatic shadows",
    "on an overcast morning", "with warm indoor lighting",
]


def build_prompts(n, seed=7):
    combos = list(itertools.product(SUBJECTS, SETTINGS, STYLES))
    random.Random(seed).shuffle(combos)
    prompts = []
    for subj, setting, style in combos[:n]:
        subj_clean = subj[2:] if subj.startswith("a ") else subj
        prompts.append(f"A {subj_clean} {setting}, {style}.")
    return prompts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prompts = build_prompts(N_IMAGES)
    print(f"{len(prompts)} prompts ready (0 overlap with the original 50 test prompts)")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, add_watermarker=False,
    )
    if os.environ.get("LOWVRAM") == "1":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    else:
        pipe.to("cuda")

    new_file = not os.path.exists(META)
    with open(META, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["filename", "prompt", "seed", "steps", "guidance", "date"])

        for idx, prompt in enumerate(prompts, 1):
            fname = f"sdxl_bulk_{idx:04d}.jpg"
            path = os.path.join(OUT_DIR, fname)
            if os.path.exists(path):
                continue

            seed = SEED_BASE + idx
            gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
            t0 = time.time()
            image = pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                         height=SIZE, width=SIZE, generator=gen).images[0]
            image.convert("RGB").save(path, "JPEG", quality=92)

            writer.writerow([fname, prompt, seed, STEPS, GUIDANCE, time.strftime("%Y-%m-%d")])
            f.flush()
            if idx % 20 == 0:
                print(f"[{idx}/{len(prompts)}] ({time.time() - t0:.1f}s last image)")

    print(f"Done. Images + metadata.csv are in {OUT_DIR}")


if __name__ == "__main__":
    main()