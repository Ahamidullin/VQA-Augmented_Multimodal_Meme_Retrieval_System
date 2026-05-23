"""Строим индекс RU search_queries и оцениваем F + ru_sq через RRF"""
import json, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
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

# загрузка существующих индексов
idx_all = faiss.read_index(str(EXP_DIR / "faiss_C_all.index"))
idx_sq_en = faiss.read_index(str(EXP_DIR / "faiss_D_search_queries.index"))

# загрузка модели
model = SentenceTransformer("BAAI/bge-m3", device="cpu")

# строим RU search_queries индекс
ru_sq_path = EXP_DIR / "faiss_ru_search_queries.index"
if ru_sq_path.exists():
    idx_sq_ru = faiss.read_index(str(ru_sq_path))
else:
    ru_texts = []
    for r in records:
        sq_ru = r.get("search_queries_ru", [])
        ru_texts.append(", ".join(sq_ru) if sq_ru else "")

    non_empty = sum(1 for t in ru_texts if t)
    

    safe = [t if t.strip() else "empty" for t in ru_texts]
    emb = model.encode(safe, show_progress_bar=True, batch_size=64, normalize_embeddings=True).astype(np.float32)

    # зануляем пустые
    for i, t in enumerate(ru_texts):
        if not t.strip():
            emb[i] = 0.0

    idx_sq_ru = faiss.IndexFlatIP(emb.shape[1])
    idx_sq_ru.add(emb)
    faiss.write_index(idx_sq_ru, str(ru_sq_path))
    print(f"  сохранено: {ru_sq_path}, {emb.shape}")


def rrf_fusion(index_results, weights, k=60):
    scores = {}
    for (idxs, _), w in zip(index_results, weights):
        for rank, idx in enumerate(idxs):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: -x[1])]


def evaluate(indices_list, weights, label=""):
    for lang in ["en", "ru"]:
        hits1, hits5, hits10, total = 0, 0, 0, 0
        total_rr = 0.0
        query_key = "queries_en" if lang == "en" else "queries_ru"

        for item in queries_data:
            filename = item["filename"]
            if filename not in filename_to_idx:
                continue
            target_idx = filename_to_idx[filename]
            for query in item.get(query_key, []):
                q_emb = model.encode(query, normalize_embeddings=True).reshape(1, -1).astype(np.float32)
                results = []
                for idx in indices_list:
                    s, i = idx.search(q_emb, 50)
                    results.append((i[0], s[0]))
                fused = rrf_fusion(results, weights)

                if target_idx in fused[:1]: hits1 += 1
                if target_idx in fused[:5]: hits5 += 1
                if target_idx in fused[:10]: hits10 += 1
                for rank, idx_val in enumerate(fused[:10]):
                    if idx_val == target_idx:
                        total_rr += 1.0 / (rank + 1)
                        break
                total += 1

        print(f"{label} {lang}: Hit@1={hits1/total:.1%} Hit@5={hits5/total:.1%} Hit@10={hits10/total:.1%} MRR={total_rr/total:.3f}")


# сравнение конфигов
print("\n F: C + EN_sq [0.6, 0.4]")
evaluate([idx_all, idx_sq_en], [0.6, 0.4], "F")

print("\n G: C + EN_sq + RU_sq [0.5, 0.25, 0.25]")
evaluate([idx_all, idx_sq_en, idx_sq_ru], [0.5, 0.25, 0.25], "G")

print("\nG2: C + EN_sq + RU_sq [0.4, 0.3, 0.3]")
evaluate([idx_all, idx_sq_en, idx_sq_ru], [0.4, 0.3, 0.3], "G2")

print("\n G3: C + RU_sq [0.6, 0.4]")
evaluate([idx_all, idx_sq_ru], [0.6, 0.4], "G3")
