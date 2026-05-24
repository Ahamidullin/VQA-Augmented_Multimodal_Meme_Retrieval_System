"""финальные метрики hit@1 hit@5 hit@10 mrr@10 - rrf двух индексов c_all + d_search_queries"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

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

filename_to_idx = {}
for i, r in enumerate(records):
    filename_to_idx[r["filename"]] = i

idx_all = faiss.read_index(str(EXP_DIR / "faiss_C_all.index"))
idx_sq = faiss.read_index(str(EXP_DIR / "faiss_D_search_queries.index"))

model = SentenceTransformer("BAAI/bge-m3", device="cpu")


def rrf_fusion(index_results, weights, k=60):
    scores = {}
    for (idxs, _), w in zip(index_results, weights):
        for rank, idx in enumerate(idxs):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for idx, score in sorted_pairs:
        result.append(idx)
    return result


def evaluate(lang="en"):
    query_key = "queries_en" if lang == "en" else "queries_ru"
    weights = [0.6, 0.4]
    hits1, hits5, hits10, total = 0, 0, 0, 0
    total_rr = 0.0

    for item in queries_data:
        filename = item["filename"]
        if filename not in filename_to_idx:
            continue
        target_idx = filename_to_idx[filename]

        for query in item.get(query_key, []):
            q_emb = model.encode(query, normalize_embeddings=True).reshape(1, -1).astype(np.float32)  # type: ignore

            s1, i1 = idx_all.search(q_emb, 50)  # type: ignore
            s2, i2 = idx_sq.search(q_emb, 50)  # type: ignore
            results = [(i1[0], s1[0]), (i2[0], s2[0])]
            fused = rrf_fusion(results, weights)

            if target_idx in fused[:1]:
                hits1 += 1
            if target_idx in fused[:5]:
                hits5 += 1
            if target_idx in fused[:10]:
                hits10 += 1

            for rank, idx in enumerate(fused[:10]):
                if idx == target_idx:
                    total_rr += 1.0 / (rank + 1)
                    break

            total += 1

    return {
        "hit1": hits1 / total,
        "hit5": hits5 / total,
        "hit10": hits10 / total,
        "mrr10": total_rr / total,
        "total": total,
    }


for lang in ["en", "ru"]:
    m = evaluate(lang)
    print(f"{lang} ({m['total']}) hit@1={m['hit1']:.1%} hit@5={m['hit5']:.1%} hit@10={m['hit10']:.1%} mrr@10={m['mrr10']:.3f}")
