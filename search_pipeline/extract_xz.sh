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
#   - 已存在、无 .temp/.partial 标记且大小正确的目标文件会被跳过
#   - 存在 .temp/.partial 标记说明上次解压中途失败，会重新尝试
#   - 解压先写入 .partial，大小校验通过后再原子替换正式文件
#   - concurrent_tasks 控制并发；默认 min(nproc, 16)，环境变量 CONCURRENT_TASKS 可覆盖

set -euo pipefail

usage() {
  echo "Usage: $0 <source> <destination> [name_filter]" >&2
  echo "  source      : list file (one xz path per line) OR directory containing .xz files" >&2
  echo "  destination : output directory" >&2
  echo "  name_filter : optional, only used when source is a directory;" >&2
  echo "                pass 'all' to disable filter, defaults to 'M01'" >&2
}

get_uncompressed_size() {
  local xz_file="$1"
  local report expected_size

  if ! report=$(xz --robot --list -- "$xz_file"); then
    return 1
  fi
  expected_size=$(awk -F '\t' '$1 == "file" {print $5; exit}' <<<"$report")
  [[ "$expected_size" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$expected_size"
}

extract_one() {
  local xz_file="$1"
  local base_name destination_path temp_file partial_file
  local expected_size actual_size

  base_name=$(basename -- "$xz_file")
  base_name=${base_name%.xz}
  echo "$base_name"

  destination_path="$destination_folder/$base_name"
  temp_file="$destination_path.temp"
  partial_file="$destination_path.partial"

  if ! expected_size=$(get_uncompressed_size "$xz_file"); then
    echo "[FAIL] cannot read xz metadata: $xz_file" >&2
    return 1
  fi

  if [ -f "$destination_path" ] && [ ! -e "$temp_file" ] && [ ! -e "$partial_file" ]; then
    actual_size=$(stat -c %s -- "$destination_path")
    if [ "$actual_size" -eq "$expected_size" ]; then
      echo "  [skip] already extracted and size verified: $xz_file"
      return 0
    fi
    echo "  [retry] existing output has wrong size: $destination_path" >&2
  fi

  if [ -e "$temp_file" ] || [ -e "$partial_file" ]; then
    echo "  [retry] previously failed: $xz_file"
    if [ -e "$temp_file" ]; then
      # 兼容旧版本：旧脚本会直接把未完成内容写到正式目标文件。
      rm -f -- "$destination_path"
    fi
    rm -f -- "$partial_file" "$temp_file"
  fi

  touch -- "$temp_file"
  if ! xz -d -c -- "$xz_file" > "$partial_file"; then
    echo "[FAIL] decompression failed: $xz_file" >&2
    return 1
  fi

  actual_size=$(stat -c %s -- "$partial_file")
  if [ "$actual_size" -ne "$expected_size" ]; then
    echo "[FAIL] size mismatch: $xz_file expected=$expected_size actual=$actual_size" >&2
    return 1
  fi

  if ! mv -f -- "$partial_file" "$destination_path"; then
    echo "[FAIL] cannot install extracted file: $destination_path" >&2
    return 1
  fi
  rm -f -- "$temp_file"
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

source_path="$1"
destination_folder="$2"
name_filter="${3:-M01}"
default_concurrency=16
if command -v nproc >/dev/null 2>&1; then
  cpu_count=$(nproc)
  if [[ "$cpu_count" =~ ^[1-9][0-9]*$ ]] && ((cpu_count < default_concurrency)); then
    default_concurrency="$cpu_count"
  fi
fi
concurrent_tasks="${CONCURRENT_TASKS:-$default_concurrency}"

if ! [[ "$concurrent_tasks" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] CONCURRENT_TASKS must be a positive integer: $concurrent_tasks" >&2
  exit 2
fi

for required_command in awk basename find sort stat xargs xz; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "[ERROR] required command not found: $required_command" >&2
    exit 2
  fi
done

mkdir -p "$destination_folder"

# ─── 自动识别 source 类型 ───
if [ -d "$source_path" ]; then
  echo "[INFO] source is a directory; scanning for .xz files (filter: ${name_filter})"
  xz_files=()
  if [ "$name_filter" = "all" ] || [ -z "$name_filter" ]; then
    while IFS= read -r -d '' xz_file; do
      xz_files+=("$xz_file")
    done < <(find "$source_path" -type f -name '*.xz' -print0 | sort -z)
  else
    while IFS= read -r -d '' xz_file; do
      xz_files+=("$xz_file")
    done < <(find "$source_path" -type f -name "*${name_filter}*.xz" -print0 | sort -z)
  fi
elif [ -f "$source_path" ]; then
  echo "[INFO] source is a list file; reading paths from ${source_path}"
  xz_files=()
  while IFS= read -r xz_file || [ -n "$xz_file" ]; do
    xz_file=${xz_file%$'\r'}
    [[ "$xz_file" =~ ^[[:space:]]*$ ]] && continue
    xz_files+=("$xz_file")
  done < "$source_path"
else
  echo "[ERROR] source path does not exist: $source_path" >&2
  exit 2
fi

n_files="${#xz_files[@]}"
if ((n_files == 0)); then
  echo "[ERROR] no xz files matched source=$source_path filter=$name_filter" >&2
  exit 3
fi

declare -A output_sources=()
for xz_file in "${xz_files[@]}"; do
  if [ ! -f "$xz_file" ] || [ ! -r "$xz_file" ]; then
    echo "[ERROR] xz input is not a readable regular file: $xz_file" >&2
    exit 3
  fi
  if [[ "$xz_file" != *.xz ]]; then
    echo "[ERROR] input does not end in .xz: $xz_file" >&2
    exit 3
  fi
  base_name=$(basename -- "$xz_file")
  base_name=${base_name%.xz}
  if [[ -n "${output_sources[$base_name]+present}" ]]; then
    echo "[ERROR] output basename collision: $base_name" >&2
    echo "        first: ${output_sources[$base_name]}" >&2
    echo "        second: $xz_file" >&2
    exit 3
  fi
  output_sources["$base_name"]="$xz_file"
done

echo "[INFO] found $n_files xz file(s); dest=$destination_folder concurrency=$concurrent_tasks"

# ─── 并发解压：单个任务结束后立刻补位，不等待整批完成 ───
export -f get_uncompressed_size extract_one
export destination_folder
if ! printf '%s\0' "${xz_files[@]}" \
  | xargs -0 -r -n 1 -P "$concurrent_tasks" bash -euo pipefail -c 'extract_one "$1"' _; then
  exit 1
fi
echo "[DONE] extracted $n_files xz file(s) to $destination_folder"
