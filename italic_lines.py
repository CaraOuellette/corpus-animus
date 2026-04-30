#!/usr/bin/env python3
"""
Italic line extraction for local conversation exports.

An "italic line" is a line consisting entirely of *...* - single asterisks
wrapping content, with line breaks (or message boundaries) on either side.
Distinct from inline emphasis like *very* important, which has surrounding
text on the same line.

Format-neutral by design: classification (action descriptions, mood tags,
sound effects, etc.) is left to the study layer.

Reads assistant messages via `corpus.iter_messages`, which in this workspace
supports export files such as `conversations.json` and `chat.html`.

Usage:
  python italic_lines.py                # top 50 by frequency
  python italic_lines.py --top 200
  python italic_lines.py --min-count 3
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from corpus import DEFAULT_DB_PATH, iter_messages

# Strip code blocks first - fenced blocks and inline code can contain `*`.
CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```|`[^`\n]+`')

# A line consisting entirely of *...* (with optional surrounding whitespace).
# [^*\n]+? excludes nested `*`, so **bold** lines and ***bolditalic*** lines
# don't match. MULTILINE anchors ^/$ to line boundaries inside the message.
ITALIC_LINE_PATTERN = re.compile(
    r'^[ \t]*\*([^*\n]+?)\*[ \t]*$',
    re.MULTILINE,
)


def extract_italic_lines(text: str) -> list[str]:
    """Return all italic-line contents found in a message body."""
    if not text:
        return []
    cleaned = CODE_BLOCK_PATTERN.sub('', text)
    return ITALIC_LINE_PATTERN.findall(cleaned)


def normalize(phrase: str) -> str:
    return phrase.lower().strip().rstrip('.,;:!?')


def iter_corpus_italic_lines(db_path=None):
    """Yield raw italic-line strings from all assistant messages, in corpus order."""
    for _month, sender, content, _convo_uuid in iter_messages(db_path):
        if sender != 'assistant':
            continue
        for line in extract_italic_lines(content):
            yield line


def index_italic_lines(db_path=None) -> Counter:
    """Count normalized italic lines across all assistant messages."""
    counter: Counter[str] = Counter()
    for line in iter_corpus_italic_lines(db_path):
        counter[normalize(line)] += 1
    return counter


def export_jsonl(db_path, output_path: Path) -> int:
    """Write all italic lines (raw) to a JSONL file, one JSON string per line."""
    n = 0
    with output_path.open('w', encoding='utf-8') as f:
        for line in iter_corpus_italic_lines(db_path):
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            n += 1
    return n


WORD_PATTERN = re.compile(r'\w+')


def ngram_shape(db_path=None, n_max: int = 3) -> dict:
    """
    For n in 1..n_max, count the number of distinct (normalized) italic lines
    that contain each n-gram. Each line contributes at most once per n-gram,
    so high-frequency lines like `*pauses*` don't dominate the ranking.
    """
    seen_lines: set[str] = set()
    counters: dict[int, Counter] = {n: Counter() for n in range(1, n_max + 1)}
    for line in iter_corpus_italic_lines(db_path):
        norm = normalize(line)
        if norm in seen_lines:
            continue
        seen_lines.add(norm)
        tokens = WORD_PATTERN.findall(norm)
        for n in range(1, n_max + 1):
            ngs_in_line = {
                ' '.join(tokens[i:i + n])
                for i in range(len(tokens) - n + 1)
            }
            for ng in ngs_in_line:
                counters[n][ng] += 1
    return counters


def main():
    parser = argparse.ArgumentParser(
        description='Index italic-line frequency across assistant turns.'
    )
    parser.add_argument('--db', type=Path, default=None,
                        help=('Path to input export file '
                              f'(default: {DEFAULT_DB_PATH})'))
    parser.add_argument('--top', type=int, default=50,
                        help='Number of top italic lines to show (default: 50)')
    parser.add_argument('--min-count', type=int, default=1,
                        help='Hide italic lines occurring fewer than N times')
    parser.add_argument('--export-jsonl', type=Path, default=None,
                        help='Write all italic lines (raw) to JSONL and exit')
    parser.add_argument('--ngrams', action='store_true',
                        help='Print top uni/bi/trigrams across italic lines '
                             '(ranked by # distinct lines containing each)')
    args = parser.parse_args()

    if args.export_jsonl:
        n = export_jsonl(args.db, args.export_jsonl)
        print(f"Wrote {n:,} italic lines to {args.export_jsonl}")
        return

    if args.ngrams:
        counters = ngram_shape(args.db, n_max=3)
        labels = {1: 'unigrams', 2: 'bigrams', 3: 'trigrams'}
        for n in (1, 2, 3):
            ranked = counters[n].most_common(args.top)
            ranked = [(g, c) for g, c in ranked if c >= args.min_count]
            if not ranked:
                continue
            width = max(len(g) for g, _ in ranked)
            print(f"\nTop {len(ranked)} {labels[n]} (by # distinct lines):\n")
            for i, (gram, count) in enumerate(ranked, 1):
                print(f"  {i:3d}. {count:>5}  {gram:<{width}}")
        return

    counter = index_italic_lines(args.db)
    total = sum(counter.values())
    unique = len(counter)

    print(f"Total italic lines: {total:,}  ({unique:,} unique)")
    if total == 0:
        return

    ranked = [(p, c) for p, c in counter.most_common(args.top) if c >= args.min_count]
    if not ranked:
        print(f"(no italic lines with count >= {args.min_count})")
        return

    width = max(len(p) for p, _ in ranked)
    print(f"\nTop {len(ranked)}:\n")
    for i, (phrase, count) in enumerate(ranked, 1):
        print(f"  {i:3d}. {count:>5}  {phrase:<{width}}")


if __name__ == '__main__':
    main()
