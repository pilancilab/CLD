#!/usr/bin/env python3
import os
import csv
import argparse
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Common Voice validated.tsv files across language directories "
            "to produce accent value counts."
        )
    )
    parser.add_argument(
        "cv_dataset_path",
        type=str,
        help="Path to Common Voice root directory containing language subdirectories (e.g., en, id, hi).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="accent_counts.csv",
        help="Output CSV file path (default: accent_counts.csv)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include rows where accent value is empty/blank as an empty string entry.",
    )
    return parser.parse_args()


def find_accent_column(fieldnames):
    """Return the matching accent column name from fieldnames, if any."""
    if not fieldnames:
        return None
    lowered = {name.lower(): name for name in fieldnames}
    # Prefer singular 'accent' if both exist
    if "accent" in lowered:
        return lowered["accent"]
    if "accents" in lowered:
        return lowered["accents"]
    return None


def analyze_language_dir(lang_dir_path, include_empty=False):
    validated_path = os.path.join(lang_dir_path, "validated.tsv")
    if not os.path.isfile(validated_path):
        return None  # No validated.tsv, skip

    with open(validated_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        accent_col = find_accent_column(reader.fieldnames)
        if accent_col is None:
            return Counter()  # No accent column present

        counts = Counter()
        for row in reader:
            val = row.get(accent_col, "")
            if val is None:
                val = ""
            val = val.strip()
            if val == "" and not include_empty:
                continue
            counts[val] += 1

        return counts


def main():
    args = parse_args()
    root = os.path.abspath(args.cv_dataset_path)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Not a directory: {root}")

    # Collect all first-level subdirectories (language codes)
    lang_dirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    lang_dirs.sort()

    rows = []  # (lang, accent, count)
    for lang_dir in lang_dirs:
        lang_code = os.path.basename(lang_dir)

        counts = analyze_language_dir(lang_dir, include_empty=args.include_empty)
        if counts is None:
            # validated.tsv missing; skip
            continue

        if len(counts) == 0:
            # Either no accent column or all filtered out due to empties
            continue

        for accent_value, cnt in counts.items():
            rows.append((lang_code, accent_value, cnt))

    # Write output CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lang", "accent", "count"])  # header
        for lang_code, accent_value, cnt in sorted(rows):
            writer.writerow([lang_code, accent_value, cnt])

    print(f"Wrote accent counts to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()


