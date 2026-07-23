#!/bin/bash
# 批量并发解压 xz 文件。source 既可以是「含 xz 路径的列表文件」也可以是「含 xz 文件的目录」，
# 脚本自动判断；目录模式下可指定文件名过滤模式（默认 'M01'）。
#
# Usage:
#   ./extract_xz.sh <source>  <destination>  [name_filter]
#
# Examples:
#   ./extract_xz.sh  ./files.txt          /path/to/output          # 文件列表 → 输出
#   ./extract_xz.sh  /path/to/raw_xz/     /path/to/output          # 目录（默认按 'M01' 过滤）
#   ./extract_xz.sh  /path/to/raw_xz/     /path/to/output  'all'   # 目录，不过滤
#   ./extract_xz.sh  /path/to/raw_xz/     /path/to/output  'M02'   # 目录，按 'M02' 过滤
#
# 行为说明:
#   - 已存在且无 .temp 标记的目标文件会被跳过
#   - 存在 .temp 标记说明上次解压中途失败，会重新尝试
#   - concurrent_tasks 控制并发；默认 50，环境变量 CONCURRENT_TASKS 可覆盖

set -euo pipefail

usage() {
  echo "Usage: $0 <source> <destination> [name_filter]" >&2
  echo "  source      : list file (one xz path per line) OR directory containing .xz files" >&2
  echo "  destination : output directory" >&2
  echo "  name_filter : optional, only used when source is a directory;" >&2
  echo "                pass 'all' to disable filter, defaults to 'M01'" >&2
}

extract_one() {
  local xz_file="$1"
  local base_name destination_path temp_file

  base_name=$(basename -s .xz "$xz_file")
  echo "$base_name"

  destination_path="$destination_folder/$base_name"
  temp_file="$destination_path.temp"

  if [ -e "$destination_path" ] && [ ! -e "$temp_file" ]; then
    echo "  [skip] already extracted: $xz_file"
    return 0
  fi

  if [ -e "$temp_file" ]; then
    echo "  [retry] previously failed: $xz_file"
    rm -f "$destination_path" "$temp_file"
  fi

  touch "$temp_file"
  xz -d "$xz_file" -c > "$destination_path" && rm -f "$temp_file"
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

source_path="$1"
destination_folder="$2"
name_filter="${3:-M01}"
concurrent_tasks="${CONCURRENT_TASKS:-50}"

mkdir -p "$destination_folder"

# ─── 自动识别 source 类型 ───
if [ -d "$source_path" ]; then
  echo "[INFO] source is a directory; scanning for .xz files (filter: ${name_filter})"
  if [ "$name_filter" = "all" ] || [ -z "$name_filter" ]; then
    mapfile -t xz_files < <(find "$source_path" -type f -name '*.xz' | sort)
  else
    mapfile -t xz_files < <(find "$source_path" -type f -name "*${name_filter}*.xz" | sort)
  fi
elif [ -f "$source_path" ]; then
  echo "[INFO] source is a list file; reading paths from ${source_path}"
  mapfile -t xz_files < "$source_path"
else
  echo "[ERROR] source path does not exist: $source_path" >&2
  exit 2
fi

n_files="${#xz_files[@]}"
echo "[INFO] found $n_files xz file(s); dest=$destination_folder concurrency=$concurrent_tasks"

# ─── 并发解压 ───
for ((i = 0; i < n_files; i += concurrent_tasks)); do
  end=$((i + concurrent_tasks))

  for ((j = i; j < end && j < n_files; j++)); do
    extract_one "${xz_files[j]}" &
  done

  wait
done

echo "[DONE] extracted $n_files xz file(s) to $destination_folder"
