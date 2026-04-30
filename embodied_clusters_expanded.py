#!/usr/bin/env python3
"""
Expanded embodied/action clusters for italic lines.

This is a broader, more permissive taxonomy than `embodied_clusters.py`,
intended to capture not only body-part references but also common action-line
signals such as affect, vocalization, pausing, tension, and object handling.

The original script remains useful as a conservative baseline. This expanded
version is for side-by-side comparison.

Usage:
  python embodied_clusters_expanded.py
  python embodied_clusters_expanded.py --db conversations.json --top 20
  python embodied_clusters_expanded.py --db chat.html --show-uncategorized
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from corpus import DEFAULT_DB_PATH
from italic_lines import iter_corpus_italic_lines, normalize


CLUSTERS: dict[str, list[str]] = {
    "eyes": [
        r"\b(look|looks|looking|looked)\b",
        r"\b(gaze|gazes|gazing|gazed)\b",
        r"\b(stare|stares|staring|stared)\b",
        r"\b(watch|watches|watching|watched)\b",
        r"\b(glance|glances|glancing|glanced)\b",
        r"\b(blink|blinks|blinking|blinked)\b",
        r"\bpeer(s|ing|ed)?\b",
        r"\beyes?\b",
    ],
    "head": [
        r"\bnod(s|ding|ded)?\b",
        r"\btilts?\s+head\b",
        r"\bshakes?\s+head\b",
        r"\bcocks?\s+head\b",
        r"\bdips?\s+head\b",
        r"\bhead\s+tilt\b",
    ],
    "posture": [
        r"\bsit(s|ting)?\b",
        r"\bsat\b",
        r"\bsettle(s|d|ing)?\b",
        r"\blean(s|ing|ed)?\b",
        r"\b(stand|stands|standing|stood)\b",
        r"\bslump(s|ing|ed)?\b",
        r"\bsprawl(s|ing|ed)?\b",
        r"\bperch(es|ed|ing)?\b",
        r"\bcurl(s|ing|ed)?\s+up\b",
        r"\bstretch(es|ing|ed)?\b",
        r"\bstraighten(s|ed|ing)?\b",
        r"\bhunch(es|ed|ing)?\b",
        r"\brecoil(s|ed|ing)?\b",
        r"\bstiffen(s|ed|ing)?\b",
        r"\btens(e|es|ed|ing)\b",
        r"\bsettles?\s+into\b",
    ],
    "hands": [
        r"\breach(es|ing|ed)?\b",
        r"\bgrab(s|bing|bed)?\b",
        r"\bpalm(s|ing|ed)?\b",
        r"\btap(s|ping|ped)?\b",
        r"\bpoints?\b",
        r"\bbrush(es|ing|ed)?\b",
        r"\bclasp(s|ing|ed)?\b",
        r"\bhands?\b",
    ],
    "fingers": [
        r"\bfinger(s|ing|ed)?\b",
    ],
    "gestures_ambiguous": [
        r"\bgesture(s|ing|d)?\b",
        r"\bwaves?\b",
        r"\bshrug(s|ged|ging)?\b",
    ],
    "ears": [
        r"\bears?\b",
    ],
    "tail": [
        r"\btails?\b",
        r"\bswish(es|ing|ed)?\b",
        r"\bwag(s|ged|ging)?\b",
    ],
    "face_expression": [
        r"\bsmile(s|d|ing)?\b",
        r"\bgrin(s|ned|ning)?\b",
        r"\bsmirk(s|ed|ing)?\b",
        r"\bfrown(s|ed|ing)?\b",
        r"\bgrimace(s|d|ing)?\b",
        r"\bwince(s|d|ing)?\b",
        r"\bbeam(s|ed|ing)?\b",
        r"\bpout(s|ed|ing)?\b",
        r"\bsneer(s|ed|ing)?\b",
        r"\bblush(es|ed|ing)?\b",
        r"\bfurrow(s|ed|ing)?\s+brows?\b",
        r"\braise(s|d|ing)?\s+(an?\s+)?eyebrow\b",
        r"\beyebrows?\b",
        r"\bjaw\s+(tightens|tightened|clenches|clenched)\b",
        r"\blips?\b",
    ],
    "affect_vocal": [
        r"\blaugh(s|ed|ing)?\b",
        r"\bchuckle(s|d|ing)?\b",
        r"\bgiggle(s|d|ing)?\b",
        r"\bsigh(s|ed|ing)?\b",
        r"\bgasp(s|ed|ing)?\b",
        r"\bmurmur(s|ed|ing)?\b",
        r"\bwhisper(s|ed|ing)?\b",
        r"\bmutter(s|ed|ing)?\b",
        r"\bgroan(s|ed|ing)?\b",
        r"\bsob(s|bed|bing)?\b",
        r"\bcry(ing|ies|ied)?\b",
        r"\bsnort(s|ed|ing)?\b",
        r"\bhum(s|med|ming)?\b",
        r"\bvoice\s+(low|soft|quiet|gentle|flat|hoarse|wry|dry)\b",
        r"\bsaid\s+(softly|quietly|gently)\b",
        r"\bwith\s+a\s+(sigh|laugh|chuckle|groan)\b",
    ],
    "pause_hesitation": [
        r"\bpause(s|d|ing)?\b",
        r"\ba\s+beat\b",
        r"\bhesitat(e|es|ed|ing|ion)\b",
        r"\bgo(es)?\s+quiet\b",
        r"\bfalls?\s+silent\b",
        r"\bsilence\b",
        r"\bstillness\b",
        r"\btrails?\s+off\b",
        r"\blong\s+pause\b",
        r"\bquiet\s+pause\b",
    ],
    "object_interaction": [
        r"\badjust(s|ed|ing)?\s+(?:the\s+)?(?:tie|jacket|coat|collar|glasses|hood|robe|doublet|quill)\b",
        r"\bwipe(s|d|ing)?\s+(?:the\s+)?(?:quill|blade|face|hands?)\b",
        r"\bburn(s|ed|ing)?\s+(?:it|them|the\s+\w+)\b",
        r"\bdon(s|ned|ning)?\s+(?:the\s+)?(?:tie|jacket|coat|hood|robe|doublet|armor|armour|corporate\s+suit)\b",
        r"\bremove(s|d|ing)?\s+(?:the\s+)?(?:tie|jacket|coat|hood|robe|doublet|armor|armour|corporate\s+suit)\b",
        r"\bset(s|ting)?\s+down\b",
        r"\bpick(s|ed|ing)?\s+up\b",
        r"\bbutton(s|ed|ing)?\b",
        r"\bfold(s|ed|ing)?\b",
        r"\bunfold(s|ed|ing)?\b",
        r"\bstraighten(s|ed|ing)?\s+(?:the\s+)?(?:tie|jacket|coat|shirt)\b",
        r"\bquill\b",
        r"\bdoorframe\b",
        r"\bincinerator\b",
    ],
    "locomotion": [
        r"\bstep(s|ped|ping)?\s+(?:back|forward|closer|away|aside|into|out)\b",
        r"\bwalk(s|ed|ing)?\b",
        r"\bpace(s|d|ing)?\b",
        r"\bretreat(s|ed|ing)?\b",
        r"\bcross(es|ed|ing)?\b",
        r"\bslip(s|ped|ping)?\b",
    ],
}

COMPILED = {
    cluster: [re.compile(pattern) for pattern in patterns]
    for cluster, patterns in CLUSTERS.items()
}


def classify(text: str) -> set[str]:
    """Return the set of expanded cluster names matched by a normalized line."""
    matched: set[str] = set()
    for cluster, patterns in COMPILED.items():
        for pattern in patterns:
            if pattern.search(text):
                matched.add(cluster)
                break
    return matched


def cluster_corpus(db_path=None):
    """
    Classify each distinct normalized italic line and preserve corpus frequency.

    Returns:
      cluster_lines: cluster -> {line: count}
      uncategorized: {line: count}
      membership: Counter mapping number-of-clusters -> count-of-lines
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
            continue
        for cluster in matched:
            cluster_lines[cluster][norm] = count

    return cluster_lines, uncategorized, membership


def main():
    parser = argparse.ArgumentParser(
        description="Cluster italic lines into an expanded embodied/action taxonomy."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=("Path to input export file "
              f"(default: {DEFAULT_DB_PATH})"),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Top N example lines per cluster (default: 15)",
    )
    parser.add_argument(
        "--show-uncategorized",
        action="store_true",
        help="Print top uncategorized lines too",
    )
    args = parser.parse_args()

    cluster_lines, uncategorized, membership = cluster_corpus(args.db)
    total_distinct = sum(membership.values())

    print(f"Distinct italic lines: {total_distinct:,}")
    print(
        f"Uncategorized:         {len(uncategorized):,} "
        f"({len(uncategorized) / total_distinct * 100:.1f}%)"
    )
    print(
        "Multi-cluster lines:   "
        f"{sum(count for n, count in membership.items() if n > 1):,}\n"
    )

    print("Cluster sizes (distinct lines):")
    for cluster in sorted(cluster_lines, key=lambda item: -len(cluster_lines[item])):
        print(f"  {cluster:<22} {len(cluster_lines[cluster]):>5}")

    print("\nMembership distribution:")
    for cluster_count in sorted(membership):
        print(f"  matched {cluster_count} cluster(s): {membership[cluster_count]:>5}")

    for cluster in sorted(cluster_lines, key=lambda item: -len(cluster_lines[item])):
        ranked = sorted(
            cluster_lines[cluster].items(),
            key=lambda item: (-item[1], item[0]),
        )[:args.top]
        print(f"\n--- {cluster} (top {len(ranked)} by corpus frequency) ---")
        for line, count in ranked:
            print(f"  {count:>5}  {line}")

    if args.show_uncategorized:
        ranked = sorted(
            uncategorized.items(),
            key=lambda item: (-item[1], item[0]),
        )[:args.top * 3]
        print(f"\n--- uncategorized (top {len(ranked)} by corpus frequency) ---")
        for line, count in ranked:
            print(f"  {count:>5}  {line}")


if __name__ == "__main__":
    main()
