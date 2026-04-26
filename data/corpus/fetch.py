"""Download Project Gutenberg texts for Phase 3.3 multilingual prose corpus.

Run: python data/corpus/fetch.py
Writes: data/corpus/prose.txt with `=== LANG=xx ID=NNNN TITLE=... ===` separators.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent
OUT_FILE = CORPUS_DIR / "prose.txt"

# (lang, gutenberg_id, title). Each language has a few candidates so a 404
# on one ID does not lose the whole language.
SOURCES: list[tuple[str, int, str]] = [
    # English
    ("en", 1342, "Pride and Prejudice (Austen)"),
    ("en", 1661, "Adventures of Sherlock Holmes (Doyle)"),
    ("en", 84, "Frankenstein (Shelley)"),
    # French
    ("fr", 17489, "Les Miserables Vol 1 (Hugo)"),
    ("fr", 13951, "Madame Bovary (Flaubert)"),
    ("fr", 4791, "Candide (Voltaire)"),
    ("fr", 6099, "Vingt mille lieues sous les mers (Verne)"),
    # Spanish
    ("es", 2000, "Don Quijote (Cervantes)"),
    ("es", 15532, "La Regenta (Clarin)"),
    # German
    ("de", 2407, "Faust I (Goethe)"),
    ("de", 5097, "Die Leiden des jungen Werther (Goethe)"),
    ("de", 22367, "Die Verwandlung (Kafka)"),
]

URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

USER_AGENT = "morseformer-corpus-fetch/0.1 (research)"

START_RE = re.compile(r"^\*\*\*\s*START OF .*?GUTENBERG.*?\*\*\*", re.IGNORECASE | re.MULTILINE)
END_RE = re.compile(r"^\*\*\*\s*END OF .*?GUTENBERG.*?\*\*\*", re.IGNORECASE | re.MULTILINE)


def fetch(book_id: int) -> str | None:
    for tmpl in URL_TEMPLATES:
        url = tmpl.format(id=book_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status != 200:
                    continue
                raw = r.read()
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        return raw.decode(enc)
                    except UnicodeDecodeError:
                        continue
        except urllib.error.HTTPError as e:
            print(f"  {url} -> HTTP {e.code}", file=sys.stderr)
        except Exception as e:
            print(f"  {url} -> {type(e).__name__}: {e}", file=sys.stderr)
    return None


def strip_gutenberg(text: str) -> str:
    m_start = START_RE.search(text)
    m_end = END_RE.search(text)
    if m_start:
        text = text[m_start.end():]
    if m_end:
        text = text[:m_end.start()]
    return text.strip()


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    per_lang: dict[str, int] = {}
    for lang, book_id, title in SOURCES:
        print(f"[{lang}] #{book_id} {title}")
        raw = fetch(book_id)
        if raw is None:
            print("  FAILED")
            continue
        body = strip_gutenberg(raw)
        if len(body) < 1000:
            print(f"  too short ({len(body)} chars), skipping")
            continue
        chunks.append(f"=== LANG={lang} ID={book_id} TITLE={title} ===\n{body}\n")
        per_lang[lang] = per_lang.get(lang, 0) + len(body)
        print(f"  ok ({len(body):,} chars)")
        time.sleep(1.0)  # polite to the Gutenberg mirror
    OUT_FILE.write_text("\n".join(chunks), encoding="utf-8")
    total = sum(per_lang.values())
    print(f"\nWrote {OUT_FILE} ({total:,} chars)")
    for lang, n in sorted(per_lang.items()):
        print(f"  {lang}: {n:,} chars")


if __name__ == "__main__":
    main()
