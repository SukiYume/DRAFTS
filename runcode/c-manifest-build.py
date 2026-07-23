"""Build a CRAFTS search task manifest for d-center-binary-gate.py.

This script intentionally stays lightweight and does not import
d-center-binary-gate.py, because that would also import torch/cupy/model code
that is unnecessary for path discovery.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_DATA_PATHS = [
    '/data31/ZD2020_1_1_2bit/',
    '/data31/ZD2021_1_1_2bit/',
    '/data31/ZD2022_1_1_2bit/',
    '/data31/ZD2023_1_1_2bit/',
    '/data31/ZD2024_1_1_2bit/',
    '/data32/ZD2025_1_1_2bit/',
]
DEFAULT_OUTPUT = '/path/to/observations/CRAFTS/task_manifest_zd202x_1_1_2bit.json'


def organize_file_lists(data_path: str, beam_filter: Optional[str] = None) -> List[Tuple[str, List[str], dict]]:
    result = []
    path = Path(data_path)
    all_fits = sorted([
        f.name for f in path.iterdir()
        if f.name.endswith('.fits') and not f.name.startswith('.')
        and '_N_' not in f.name and '_W_' not in f.name and '_F_' not in f.name
    ])

    if all_fits:
        beam_match = {f: re.search(r'-(M\d{2})_', f) for f in all_fits}
        skipped = [f for f, m in beam_match.items() if m is None]
        if skipped:
            print(f'[organize] {path}: skipped {len(skipped)} fits without -Mxx_ beam tag')
        all_fits = [f for f in all_fits if beam_match[f] is not None]
        beams = sorted({beam_match[f].group(1) for f in all_fits})
        if beam_filter and beam_filter != 'all':
            beams = [beam_filter] if beam_filter in beams else []

        date_name = path.name
        source_name = path.parent.name
        for beam in beams:
            beam_files = sorted([str(path / f) for f in all_fits if beam in f])
            if beam_files:
                info = {'source': source_name, 'date': date_name, 'beam': beam}
                identifier = f"{source_name}_{date_name}_{beam}"
                result.append((identifier, beam_files, info))
    else:
        subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
        for subdir in subdirs:
            result.extend(organize_file_lists(str(subdir), beam_filter))

    return result


def organize_file_lists_from_roots(data_paths: List[str], beam_filter: Optional[str] = None) -> List[Tuple[str, List[str], dict]]:
    result = []
    for data_path in data_paths:
        root_label = Path(data_path).name
        for identifier, files, info in organize_file_lists(data_path, beam_filter):
            info = {**info, 'root': root_label}
            result.append((f"{root_label}_{identifier}", files, info))
    return result


def write_manifest(tasks, output_path: str, data_paths: List[str], beam_filter: Optional[str]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_paths': data_paths,
        'beam_filter': beam_filter,
        'task_count': len(tasks),
        'tasks': [
            {'identifier': identifier, 'files': files, 'info': info}
            for identifier, files, info in tasks
        ],
    }
    tmp_path = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write('\n')
    os.replace(tmp_path, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT,
                        help=f'manifest output path, default: {DEFAULT_OUTPUT}')
    parser.add_argument('--beam-filter', default='all',
                        help="beam filter: M01 / M02 / all / none; default: all")
    parser.add_argument('--data-path', action='append', dest='data_paths',
                        help='data root path; repeat to override the default CRAFTS roots')
    args = parser.parse_args()

    beam_filter = None if args.beam_filter.lower() == 'none' else args.beam_filter
    data_paths = args.data_paths or DEFAULT_DATA_PATHS

    print(f'Building manifest from {len(data_paths)} data roots')
    for data_path in data_paths:
        print(f'  {data_path}')
    print(f'Beam filter: {beam_filter}')

    tasks = organize_file_lists_from_roots(data_paths, beam_filter)
    write_manifest(tasks, args.output, data_paths, beam_filter)
    print(f'Wrote {len(tasks)} tasks to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
