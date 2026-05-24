"""эксперимент: добавление CLIP image search к текстовым индексам.
Сравнивает Hit@5 с CLIP и без, перебирая разные RRF-веса.
Индексы: mega-text (all fields), search_queries (EN), CLIP ViT-B/32.
"""
import json, numpy as np, faiss
from pathlib import Path
from tqdm import tqdm

VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
QUERIES_FILE = Path("eval/validation_set/queries_v3.json")
EXP_DIR = Path("data/experiments")

# загрузка данных
records = []
with open(VQA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if not r.get("is_nsfw"):
                records.append(r)

with open(QUERIES_FILE) as f:
    queries_data = json.load(f)

filename_to_idx = {r["filename"]: i for i, r in enumerate(records)}


idx_all = faiss.read_index(str(EXP_DIR / "faiss_C_all.index"))
idx_sq = faiss.read_index(str(EXP_DIR / "faiss_D_search_queries.index"))
clip_emb = np.load("data/processed/emb_image_v3.npy").astype(np.float32)
idx_clip = faiss.IndexFlatIP(clip_emb.shape[1])
idx_clip.add(clip_emb)  # type: ignore
del clip_emb
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("BAAI/bge-m3", device="cpu")
clip_text = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", device="cpu")


def encode_clip_text(query):
    return clip_text.encode(query, normalize_embeddings=True).reshape(1, -1).astype(np.float32)


def rrf_fusion(index_results, weights, k=60):
    scores = {}
    for (idxs, _), w in zip(index_results, weights):
        for rank, idx in enumerate(idxs):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = [idx for idx, score in sorted_pairs]
    return result


# тестируем разные веса
configs = {
    "только F": ([idx_all, idx_sq], [0.6, 0.4], False),
    "F+clip [0.5,0.3,0.2]": ([idx_all, idx_sq, idx_clip], [0.5, 0.3, 0.2], True),
    "F+clip [0.4,0.3,0.3]": ([idx_all, idx_sq, idx_clip], [0.4, 0.3, 0.3], True),
    "F+clip [0.5,0.5,0.5]": ([idx_all, idx_sq, idx_clip], [0.5, 0.5, 0.5], True),
}

for cfg_name, (indices, weights, use_clip) in configs.items():
    print(f"{cfg_name}")
    for lang in ["en", "ru"]:
        hits1, hits5, hits10, total = 0, 0, 0, 0
        total_rr = 0.0
        query_key = "queries_en" if lang == "en" else "queries_ru"

        for item in tqdm(queries_data, desc=f"{cfg_name} {lang}"):
            filename = item["filename"]
            if filename not in filename_to_idx:
                continue
            target_idx = filename_to_idx[filename]

            for query in item.get(query_key, []):
                q_text = encoder.encode(query, normalize_embeddings=True).reshape(1, -1).astype(np.float32)
                results = []
            
                for idx in indices[:2]:  # текстовые индексы
                    s, i = idx.search(q_text, 50)  # type: ignore
                    results.append((i[0], s[0]))
             
                if use_clip:    # clip
                    q_clip = encode_clip_text(query)
                    s, i = idx_clip.search(q_clip, 50)  # type: ignore
                    results.append((i[0], s[0]))

                fused = rrf_fusion(results, weights)

                if target_idx in fused[:1]:
                    hits1 += 1
                if target_idx in fused[:5]:
                    hits5 += 1
                if target_idx in fused[:10]:
                    hits10 += 1
                for rank, idx_val in enumerate(fused[:10]):
                    if idx_val == target_idx:
                        total_rr += 1.0 / (rank + 1)
                        break
                total += 1

        print(f"{lang} Hit@1={hits1/total:.1%} Hit@5={hits5/total:.1%} Hit@10={hits10/total:.1%} MRR={total_rr/total:.3f}")
