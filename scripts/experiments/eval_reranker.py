"""Reranker: загружаем модели по очереди чтобы не было OOM"""
import json, numpy as np, faiss, gc, torch
from pathlib import Path
from tqdm import tqdm

VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
QUERIES_FILE = Path("eval/validation_set/queries_v3.json")
EXP_DIR = Path("data/experiments")


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


def rrf_fusion(index_results, weights, k=60):
    scores = {}
    for (idxs, _), w in zip(index_results, weights):
        for rank, idx in enumerate(idxs):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: -x[1])]


#  кодируем все запросы и получить top10 через rrf
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("BAAI/bge-m3", device="cpu")

weights = [0.6, 0.4]
# для каждого item, query, lang сохраняем top10 и target_idx
tasks = []  

for lang in ["en", "ru"]:
    query_key = "queries_en" if lang == "en" else "queries_ru"
    for item in tqdm(queries_data, desc=f"Encode {lang}"):
        filename = item["filename"]
        if filename not in filename_to_idx:
            continue
        target_idx = filename_to_idx[filename]
        for query in item.get(query_key, []):
            q_emb = encoder.encode(query, normalize_embeddings=True).reshape(1, -1).astype(np.float32)
            s1, i1 = idx_all.search(q_emb, 50)
            s2, i2 = idx_sq.search(q_emb, 50)
            top10 = rrf_fusion([(i1[0], s1[0]), (i2[0], s2[0])], weights)[:10]
            tasks.append((lang, query, target_idx, top10))



del encoder
gc.collect()
torch.mps.empty_cache() if hasattr(torch, 'mps') else None


# rerank top-10 

from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")

results = {"en": {"h1": 0, "h5": 0, "h10": 0, "rr": 0.0, "total": 0},
           "ru": {"h1": 0, "h5": 0, "h10": 0, "rr": 0.0, "total": 0}}

for lang, query, target_idx, top10 in tqdm(tasks, desc="Reranking"):
    pairs = []
    for i in top10:
        caption = records[i].get('caption', '')
        idea = records[i].get('main_idea', '')
        pairs.append((query, f"{caption}. {idea}"))
    scores = reranker.predict(pairs, batch_size=10)
    reranked = [top10[j] for j in sorted(range(len(scores)), key=lambda j: -scores[j])]

    r = results[lang]
    if target_idx in reranked[:1]:
        r["h1"] += 1
    if target_idx in reranked[:5]:
        r["h5"] += 1
    if target_idx in reranked[:10]:
        r["h10"] += 1
    for rank, idx in enumerate(reranked[:10]):
        if idx == target_idx:
            r["rr"] += 1.0 / (rank + 1)
            break
    r["total"] += 1

for lang in ["en", "ru"]:
    r = results[lang]
    t = r["total"]
    print(f"{lang.upper()} + reranker ({t} запросов ")
    print(f"Hit@1 = {r['h1']/t:.1%}")
    print(f"Hit@5= {r['h5']/t:.1%}")
    print(f"Hit@10= {r['h10']/t:.1%}")
    print(f"MRR@10 = {r['rr']/t:.3f}")
