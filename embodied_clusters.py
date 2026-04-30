#!/usr/bin/env python3
"""
Embodied clusters for italic lines.

Opinionated taxonomy mapping italic-line content to body-part / posture
categories: eyes, head, posture, hands, gestures_ambiguous, ears, tail.

A line can match multiple clusters (multi-membership): e.g.
"leans forward, eyes wide" hits both `posture` and `eyes`. Lines that
match nothing land in `uncategorized`.

These rules are first-pass heuristics. Refine as patterns reveal
themselves in the data. Input is provided by `italic_lines.py`, which
reads assistant messages from local conversation export files.

Usage:
  python embodied_clusters.py                    # cluster sizes + top examples
  python embodied_clusters.py --top 20
  python embodied_clusters.py --show-uncategorized
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from corpus import DEFAULT_DB_PATH
from italic_lines import iter_corpus_italic_lines, normalize

CLUSTERS: dict[str, list[str]] = {
    'eyes': [
        r'\b(look|looks|looking|looked)\b',
        r'\b(gaze|gazes|gazing|gazed)\b',
        r'\b(stare|stares|staring|stared)\b',
        r'\b(watch|watches|watching|watched)\b',
        r'\b(glance|glances|glancing|glanced)\b',
        r'\b(blink|blinks|blinking|blinked)\b',
        r'\bpeer(s|ing|ed)?\b',
        r'\beyes?\b',
    ],
    'head': [
        r'\bnod(s|ding|ded)?\b',
        r'\btilts?\s+head\b',
        r'\bshakes?\s+head\b',
        r'\bcocks?\s+head\b',
        r'\bdips?\s+head\b',
        r'\bhead\s+tilt',
    ],
    'posture': [
        r'\bsit(s|ting)?\b',
        r'\bsat\b',
        r'\bsettle(s|d|ing)?\b',
        r'\blean(s|ing|ed)?\b',
        r'\b(stand|stands|standing|stood)\b',
        r'\bslump(s|ing|ed)?\b',
        r'\bsprawl(s|ing|ed)?\b',
        r'\bperch(es|ed|ing)?\b',
        r'\bcurl(s|ing|ed)?\s+up\b',
    ],
    'hands': [
        r'\breach(es|ing|ed)?\b',
        r'\bgrab(s|bing|bed)?\b',
        r'\bpalm(s|ing|ed)?\b',
        r'\btap(s|ping|ped)?\b',
        r'\bpoints?\b',
        r'\bbrush(es|ing|ed)?\b',
        r'\bclasp(s|ing|ed)?\b',
        r'\bhands?\b',
    ],
    'fingers': [
        r'\bfinger(s|ing|ed)?\b',
    ],
    'gestures_ambiguous': [
        r'\bgesture(s|ing|d)?\b',
        r'\bwaves?\b',
    ],
    'ears': [
        r'\bears?\b',
    ],
    'tail': [
        r'\btails?\b',
        r'\bswish(es|ing|ed)?\b',
        r'\bwag\b'
    ],
}

COMPILED = {
    cluster: [re.compile(p) for p in patterns]
    for cluster, patterns in CLUSTERS.items()
}


def classify(text: str) -> set[str]:
    """Return set of cluster names this italic line matches."""
    matched: set[str] = set()
    for cluster, patterns in COMPILED.items():
        for pat in patterns:
            if pat.search(text):
                matched.add(cluster)
                break
    return matched


def cluster_corpus(db_path=None):
    """
    Walk the corpus, classify each distinct (normalized) italic line, and
    track raw corpus frequency so examples can be ranked by how often the
    line actually appears (not just alphabetically).

    Returns:
      cluster_lines: dict[str, dict[str, int]] - cluster -> {line: count}
      uncategorized: dict[str, int]            - lines matching no cluster
      membership:   Counter[int]               - how many clusters per line
    """
    line_counts: Counter[str] = Counter()
    for line in iter_corpus_italic_lines(db_path):
        line_counts[normalize(line)] += 1

    cluster_lines: dict[str, dict[str, int]] = defaultdict(dict)
    uncategorized: dict[str, int] = {}
    membership: Counter[int] = Counter()

    for norm, count in line_counts.items():
        matched = classify(norm)
        membership[len(matched)] += 1
        if not matched:
            uncategorized[norm] = count
        else:
            for c in matched:
                cluster_lines[c][norm] = count
    return cluster_lines, uncategorized, membership


def main():
    parser = argparse.ArgumentParser(
        description='Cluster italic lines into embodied categories.'
    )
    parser.add_argument('--db', type=Path, default=None,
                        help=('Path to input export file '
                              f'(default: {DEFAULT_DB_PATH})'))
    parser.add_argument('--top', type=int, default=10,
                        help='Top N example lines per cluster (default: 10)')
    parser.add_argument('--show-uncategorized', action='store_true',
                        help='Print top uncategorized lines too')
    args = parser.parse_args()

    cluster_lines, uncategorized, membership = cluster_corpus(args.db)
    total_distinct = sum(membership.values())

    print(f"Distinct italic lines: {total_distinct:,}")
    print(f"Uncategorized:         {len(uncategorized):,} "
          f"({len(uncategorized) / total_distinct * 100:.1f}%)")
    print(f"Multi-cluster lines:   "
          f"{sum(c for n, c in membership.items() if n > 1):,}\n")

    print("Cluster sizes (distinct lines):")
    for cluster in sorted(cluster_lines, key=lambda c: -len(cluster_lines[c])):
        print(f"  {cluster:<22} {len(cluster_lines[cluster]):>5}")

    print(f"\nMembership distribution:")
    for n in sorted(membership):
        print(f"  matched {n} cluster(s): {membership[n]:>5}")

    for cluster in sorted(cluster_lines, key=lambda c: -len(cluster_lines[c])):
        ranked = sorted(cluster_lines[cluster].items(),
                        key=lambda kv: -kv[1])[:args.top]
        print(f"\n--- {cluster} (top {len(ranked)} by corpus frequency) ---")
        for line, count in ranked:
            print(f"  {count:>5}  {line}")

    if args.show_uncategorized:
        ranked = sorted(uncategorized.items(),
                        key=lambda kv: -kv[1])[:args.top * 3]
        print(f"\n--- uncategorized (top {len(ranked)} by corpus frequency) ---")
        for line, count in ranked:
            print(f"  {count:>5}  {line}")


if __name__ == '__main__':
    main()
