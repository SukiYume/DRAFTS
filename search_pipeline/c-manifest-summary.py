"""Aggregate per-beam candidate_manifest.jsonl files into review tables.

Example:
    python c-manifest-summary.py \
        --root /path/to/observations/CRAFTS \
        --csv c-candidates-zd202x.csv \
        --json c-candidates-zd202x.json
"""

import argparse
import csv
import json
from pathlib import Path


PREFERRED_COLUMNS = [
    'manifest_path',
    'task_identifier',
    'root',
    'source',
    'date',
    'beam',
    'input_start_mjd',
    'block_start_mjd',
    'block_start_fits_path',
    'block_start_fits_index',
    'block_fits_count',
    'toa_sec',
    'signal_mjd',
    'toa_sample_from_block_start',
    'toa_sample_from_input_start',
    'dm',
    'det_score',
    'class_score',
    'det_prob',
    'class_threshold',
    'dm_threshold',
    'slice_index',
    'freq_slice',
    'dm_chunk',
    'box_idx',
    'cx',
    'cy',
    'w_box',
    'h_box',
    'jpg_path',
    'npy_path',
]


def flatten_record(record, manifest_path):
    flat = {'manifest_path': str(manifest_path)}
    for key, value in record.items():
        if key == 'task_info' and isinstance(value, dict):
            for info_key, info_value in value.items():
                flat[f'task_{info_key}'] = info_value
        else:
            flat[key] = value
    return flat


def load_records(root, manifest_name):
    records = []
    for manifest_path in sorted(Path(root).rglob(manifest_name)):
        with manifest_path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{manifest_path}:{line_no}: invalid JSONL line: {exc}') from exc
                records.append(flatten_record(record, manifest_path))
    return records


def write_csv(records, csv_path):
    all_keys = set()
    for record in records:
        all_keys.update(record)
    fieldnames = [key for key in PREFERRED_COLUMNS if key in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))

    with Path(csv_path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(records, json_path):
    with Path(json_path).open('w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Search output root containing per-beam manifest files')
    parser.add_argument('--manifest-name', default='candidate_manifest.jsonl')
    parser.add_argument('--csv', default='c-candidates.csv', help='Output CSV path; use empty string to disable')
    parser.add_argument('--json', default='', help='Optional output JSON array path')
    args = parser.parse_args()

    records = load_records(args.root, args.manifest_name)
    if args.csv:
        write_csv(records, args.csv)
        print(f'[CSV] {args.csv} rows={len(records)}')
    if args.json:
        write_json(records, args.json)
        print(f'[JSON] {args.json} rows={len(records)}')
    if not args.csv and not args.json:
        print(f'[INFO] rows={len(records)}; no output requested')


if __name__ == '__main__':
    main()
