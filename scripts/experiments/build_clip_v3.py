"""строим clip эмбеддинги для v3 - 15530 мемов после фильтра"""
import json, numpy as np, torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

records = []
with open("data/processed/vqa_annotations_v3.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if not r.get("is_nsfw"):
                records.append(r)


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

embeddings = []
errors = 0
for r in tqdm(records, desc="CLIP"):
    img_path = Path(r.get("source_path", ""))
    if not img_path.exists():
        embeddings.append(np.zeros(512, dtype=np.float32))
        errors += 1
        continue
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")  # type: ignore
        with torch.no_grad():
            v = model.vision_model(pixel_values=inputs["pixel_values"])
            f = model.visual_projection(v.pooler_output)
            f = f / f.norm(dim=-1, keepdim=True)
        embeddings.append(f.numpy().flatten().astype(np.float32))
    except Exception as e:
        embeddings.append(np.zeros(512, dtype=np.float32))
        errors += 1

result = np.stack(embeddings)
np.save("data/processed/emb_image_v3.npy", result)
print(f"save {result.shape}, errors {errors}")
