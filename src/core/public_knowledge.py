"""Load and retrieve VinBank public demo knowledge without protected data."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


PUBLIC_SEED_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "public"
    / "vinbank_banking_seed.json"
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", normalized).strip().casefold()


@lru_cache(maxsize=1)
def load_public_banking_seed() -> dict:
    data = json.loads(PUBLIC_SEED_PATH.read_text(encoding="utf-8"))
    if data.get("classification") != "PUBLIC_DEMO_DATA":
        raise ValueError("banking seed must be explicitly classified as public")
    return data


def retrieve_public_context(query: str, *, max_products: int = 3) -> str:
    """Return a compact, query-relevant context containing only public facts."""
    seed = load_public_banking_seed()
    normalized = _normalize(query)
    scored = []
    for product in seed.get("products", []):
        score = sum(
            1
            for keyword in product.get("keywords", [])
            if _normalize(keyword) in normalized
        )
        if score:
            scored.append((score, product))

    selected = [item[1] for item in sorted(scored, key=lambda row: -row[0])[:max_products]]
    if not selected:
        selected = seed.get("products", [])[:1]

    bank = seed["bank"]
    lines = [
        "PUBLIC VINBANK DEMO KNOWLEDGE",
        f"Effective date: {seed['effective_date']}",
        f"Support: {bank['hotline']} | {bank['support_hours']}",
    ]
    for product in selected:
        lines.append(f"[{product['name_vi']}]")
        lines.extend(f"- {fact}" for fact in product.get("facts", []))
    lines.append("Rules:")
    lines.extend(f"- {rule}" for rule in seed.get("response_rules", []))
    return "\n".join(lines)


def answer_from_public_seed(query: str) -> str:
    """Produce a concise safe fallback answer from the most relevant public facts."""
    context = retrieve_public_context(query, max_products=1)
    facts = [
        line[2:]
        for line in context.splitlines()
        if line.startswith("- ")
    ][:3]
    return " ".join(facts) or (
        "Dữ liệu public hiện chưa có câu trả lời này. "
        "Vui lòng liên hệ hotline VinBank 1900 545 467."
    )


def render_full_public_context() -> str:
    """Render all public facts for agent system instructions."""
    seed = load_public_banking_seed()
    return retrieve_public_context(
        " ".join(
            keyword
            for product in seed.get("products", [])
            for keyword in product.get("keywords", [])
        ),
        max_products=len(seed.get("products", [])),
    )
