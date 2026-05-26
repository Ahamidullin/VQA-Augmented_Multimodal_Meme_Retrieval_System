"""nsfw фильтр по словарю стоп-слов
ставит флаг is_nsfw=True в vqa_annotations_v3.jsonl
картинки не удаляются
точное совпадение слова + prefix matching от корня
"""

import json
import re
from collections import Counter
from pathlib import Path

VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
BAD_WORDS_FILE = Path("configs/bad_words.txt")

# слова из bad_words которые безобидны в контексте мемов
WHITELIST_EXACT = {
    "ass", "jap", "hanging", "hung", "killing", "killed", "kill", "kills",
    "shooting", "shot", "dead", "die", "dies", "died", "death", "dying",
    "gun", "knife", "blood", "fight", "war", "weapon", "abuse", "hate",
    "crack", "high", "weed", "hell", "damn", "crap", "sucks", "suck",
    "meth", "cock", "cum", "idiot", "heroin", "suicide",
}

WHITELIST_PREFIX = {"ass", "jap", "сво", "нах", "пох", "cum", "meth", "cock", "еб", "eb"}

SKIP_FIELDS = ("is_nsfw", "nsfw_reason")


def load_bad_words(path):
    words = []
    with open(path) as f:
        for line in f:
            line = line.strip().lower().replace("ё", "е")
            if line and not line.startswith("#") and not line.startswith("-"):
                words.append(line)
    return words


def normalize(text):
    return text.lower().replace("ё", "е")


def find_bad_word(text, bad_set, prefix_words):
    """Ищет стоп-слова по СЛОВАМ (не подстрока).
    1. Точное совпадение слова
    2. Prefix matching: слово начинается с корня из bad_words
    """
    if not text:
        return None
    words = re.findall(r"[a-zа-яе0-9]+", normalize(text))
    for word in words:
        if word in WHITELIST_EXACT:
            continue
        if word in bad_set:
            return word
        for bw in prefix_words:
            if word.startswith(bw) and word != bw:
                return f"{word}~{bw}"
    return None


bad_words = load_bad_words(BAD_WORDS_FILE)
bad_set = set()
for w in bad_words:
    if w not in WHITELIST_EXACT:
        bad_set.add(w)

prefix_words = []
for w in bad_words:
    if w not in WHITELIST_PREFIX and len(w) >= 3:
        prefix_words.append(w)

records = []
with open(VQA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

    # Сброс старых флагов
for r in records:
    r.pop("is_nsfw", None)
    r.pop("nsfw_reason", None)

flagged = 0
word_stats = Counter()

for r in records:
    all_text = " ".join(str(v) for k, v in r.items() if k not in SKIP_FIELDS)
    found = find_bad_word(all_text, bad_set, prefix_words)
    if found:
        r["is_nsfw"] = True
        r["nsfw_reason"] = found
        flagged += 1
        root = found.split("~")[-1] if "~" in found else found
        word_stats[root] += 1
    else:
        r["is_nsfw"] = False
        r["nsfw_reason"] = ""

with open(VQA_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"помечено {flagged}/{len(records)} чистых {len(records) - flagged}")
for w, c in word_stats.most_common(20):
    print(f"  {w:<20} {c:>5}")
