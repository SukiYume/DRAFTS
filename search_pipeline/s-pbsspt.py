# fmt: off
"""gate 与 d-dm-time 的 PBS 提交器。

提交器读取入口脚本中的 ``section_num``，把全部 section 均匀分配给 GPU 作业。
每个 PBS 作业申请一块 GPU，并按 ``workers_per_gpu`` 指定的上限并发执行 section。
默认值 1 会逐个执行 section。入口脚本使用 ``python <script> <section>`` 契约。

常用命令：

    python s-pbsspt.py -s d-center-binary-gate.py -n 1:8 -j gate -d
    python s-pbsspt.py -s d-center-binary-gate.py -n 1:8 -j gate -w 4
    python s-pbsspt.py -s d-center-binary-gate.py -n 1:8 -j gate -S
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CONDA_ENV       = "pytorch"
DEFAULT_THREADS         = 8
DEFAULT_WORKERS_PER_GPU = 1
DEFAULT_SUBMIT_DELAY_S  = 2.0


def run_section(script_path, section):
    """合并并标记子进程输出，把搜索入口捕获的处理异常转换为失败状态。"""
    failed = False
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", script_path, section],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        print(f"[section {section}] Launch failed: {exc}", flush=True)
        return 1
    assert process.stdout is not None
    with process.stdout:
        for line in process.stdout:
            print(f"[section {section}] {line.rstrip()}", flush=True)
            if re.match(r"\s*(Error processing\b|Traceback \(most recent call last\):)", line):
                failed = True
    exit_code = process.wait()
    if exit_code == 0 and failed:
        print(f"[section {section}] Processing error detected; marking section failed", flush=True)
        return 1
    return exit_code if exit_code >= 0 else 128 - exit_code


def parse_node_config(value):
    """解析 ``1:8,2:8`` 为 ``{1: 8, 2: 8}``。"""
    config = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            node_text, count_text = item.split(":", 1)
            node                 = int(node_text)
            count                = int(count_text)
        except ValueError as exc:
            raise ValueError(f"节点配置 {item!r} 无效；格式应为 1:8,2:8") from exc
        if node <= 0 or count <= 0:
            raise ValueError(f"节点号和 GPU 数必须为正整数：{item!r}")
        if node in config:
            raise ValueError(f"节点 gpu{node:02d} 重复出现")
        config[node] = count
    if not config:
        raise ValueError("至少需要配置一个 GPU 节点")
    return config


def resolve_entry_script(root_path, script_name):
    """解析入口脚本，并限制其位于运行目录内。"""
    root        = Path(root_path).expanduser().resolve()
    script_path = (root / script_name).resolve()
    try:
        script_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"入口脚本必须位于运行目录内：{script_path}") from exc
    if not script_path.is_file():
        raise FileNotFoundError(f"搜索入口不存在：{script_path}")
    return root, script_path


def read_section_num(script_path):
    """读取入口脚本主配置中的唯一 ``section_num``。"""
    text    = Path(script_path).read_text(encoding="utf-8")
    matches = re.findall(r"^\s*section_num\s*=\s*(\d+)\s*,", text, re.M)
    if len(matches) != 1:
        raise ValueError(
            f"{script_path} 需要且只能包含一个字面量 section_num，实际找到 {len(matches)} 个"
        )
    section_num = int(matches[0])
    if section_num <= 0:
        raise ValueError(f"section_num 必须为正整数，实际为 {section_num}")
    return section_num


def assign_sections(section_num, node_config):
    """把全部 section 连续、均匀地分配给 GPU 作业。"""
    if section_num <= 0:
        raise ValueError("section_num 必须为正整数")

    gpu_slots = [node for node, count in node_config.items() for _ in range(count)]
    if not gpu_slots:
        raise ValueError("至少需要一个 GPU 作业")

    active_slots  = gpu_slots[:section_num]
    base, extra   = divmod(section_num, len(active_slots))
    assignments   = []
    first_section = 0
    for slot_index, node in enumerate(active_slots):
        count          = base + int(slot_index < extra)
        sections       = tuple(range(first_section, first_section + count))
        first_section += count
        assignments.append((node, sections))
    return assignments


def section_tag(sections, width):
    """生成 PBS 文件名和作业名使用的 section 标签。"""
    if len(sections) == 1:
        return f"{sections[0]:0{width}d}"
    return f"{sections[0]:0{width}d}-{sections[-1]:0{width}d}"


def render_pbs_script(
    *,
    root_path,
    script_path,
    pbs_dir,
    node,
    sections,
    section_width,
    job_name,
    conda_env,
    threads,
    workers_per_gpu,
):
    """生成一个单 GPU、按并发上限执行多个 section 的 PBS 脚本。"""
    tag           = section_tag(sections, section_width)
    sections_bash = " ".join(map(str, sections))
    root_shell    = shlex.quote(str(root_path))
    script_shell  = shlex.quote(str(script_path))
    runner_shell  = shlex.quote(str(Path(__file__).resolve()))
    env_shell     = shlex.quote(conda_env)

    return f"""#!/bin/bash
#PBS -N {job_name}-{tag}
#PBS -o {pbs_dir}/{job_name}-job{tag}.log
#PBS -j oe
#PBS -q gpu
#PBS -l nodes=gpu{node:02d}
#PBS -W x=GRES:gpu@1

SHELL_INIT="${{SHELL_INIT:-${{HOME}}/.bashrc}}"
source "$SHELL_INIT" || exit 1
conda activate {env_shell} || exit 1
if command -v nomps >/dev/null 2>&1; then
    nomps
fi

export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}
export OPENBLAS_NUM_THREADS={threads}
export NUMEXPR_NUM_THREADS={threads}
export VECLIB_MAXIMUM_THREADS={threads}
export BLIS_NUM_THREADS={threads}

ROOT_PATH={root_shell}
SCRIPT_PATH={script_shell}
RUNNER_PATH={runner_shell}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
sections=({sections_bash})
workers_per_gpu={workers_per_gpu}

cd "$ROOT_PATH" || exit 1
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Sections: ${{sections[*]}}"
echo "Workers per GPU: $workers_per_gpu"
nvidia-smi

status=0
pids=()
batch_sections=()
successful_sections=()
failed_sections=()

run_section() {{
    local section="$1" exit_code
    echo "Starting section $section"

    "$PYTHON_BIN" -u "$RUNNER_PATH" --run-section "$SCRIPT_PATH" "$section"
    exit_code=$?
    if (( exit_code == 0 )); then
        echo "Section $section finished successfully"
    else
        echo "Section $section failed with exit code $exit_code"
    fi
    return "$exit_code"
}}

wait_batch() {{
    local index
    for index in "${{!pids[@]}}"; do
        if wait "${{pids[$index]}}"; then
            successful_sections+=("${{batch_sections[$index]}}")
        else
            status=1
            failed_sections+=("${{batch_sections[$index]}}")
        fi
    done
    pids=()
    batch_sections=()
}}

for section in "${{sections[@]}}"; do
    run_section "$section" &
    pids+=("$!")
    batch_sections+=("$section")
    if (( ${{#pids[@]}} >= workers_per_gpu )); then
        wait_batch
    fi
done

wait_batch
echo "Successful sections (${{#successful_sections[@]}}): ${{successful_sections[*]}}"
echo "Failed sections (${{#failed_sections[@]}}): ${{failed_sections[*]}}"
echo "Job exit status: $status"
echo "Job finished at: $(date)"
exit "$status"
"""


def create_job_files(
    *,
    root_path,
    script_path,
    node_config,
    section_num,
    job_name,
    conda_env,
    threads,
    workers_per_gpu,
):
    """生成全部 PBS 文件，返回 ``(node, sections, path)`` 列表。"""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_name):
        raise ValueError("job_name 只能包含字母、数字、点、下划线和短横线")
    if threads <= 0:
        raise ValueError("threads 必须为正整数")
    if workers_per_gpu <= 0:
        raise ValueError("workers_per_gpu 必须为正整数")

    pbs_dir       = root_path / "pbsspt"
    pbs_dir.mkdir(parents=True, exist_ok=True)
    section_width = max(2, len(str(section_num - 1)))
    assignments   = assign_sections(section_num, node_config)
    jobs          = []

    for node, sections in assignments:
        tag      = section_tag(sections, section_width)
        pbs_path = pbs_dir / f"{job_name}-node-{node:02d}-section-{tag}.pbs"
        content  = render_pbs_script(
            root_path       = root_path,
            script_path     = script_path,
            pbs_dir         = pbs_dir,
            node            = node,
            sections        = sections,
            section_width   = section_width,
            job_name        = job_name,
            conda_env       = conda_env,
            threads         = threads,
            workers_per_gpu = workers_per_gpu,
        )
        pbs_path.write_text(content, encoding="utf-8", newline="\n")
        jobs.append((node, sections, pbs_path))
    return jobs


def submit_jobs(jobs, submit_delay_s):
    """提交 PBS 文件，并返回失败作业数。"""
    failure_count = 0
    for index, (node, sections, pbs_path) in enumerate(jobs):
        try:
            result = subprocess.run(
                ["qsub", str(pbs_path)],
                capture_output = True,
                text           = True,
            )
        except FileNotFoundError:
            failure_count += 1
            print("[ERROR] 找不到 qsub", file=sys.stderr)
            break

        if result.returncode == 0:
            print(
                f"[SUBMIT] gpu{node:02d} sections={list(sections)} "
                f"job_id={result.stdout.strip()}"
            )
        else:
            error = result.stderr.strip() or result.stdout.strip()
            failure_count += 1
            print(f"[ERROR] sections={list(sections)} {error}", file=sys.stderr)

        if index + 1 < len(jobs) and submit_delay_s > 0:
            time.sleep(submit_delay_s)
    return failure_count


def check_status():
    """显示当前用户的 PBS 作业。"""
    command = ["qstat"]
    user    = os.environ.get("USER") or os.environ.get("USERNAME")
    if user:
        command.extend(("-u", user))
    try:
        return subprocess.run(command).returncode
    except FileNotFoundError:
        print("[ERROR] 找不到 qstat", file=sys.stderr)
        return 127


def build_parser():
    parser = argparse.ArgumentParser(description="为 gate 或 d-dm-time 生成并提交单 GPU PBS 作业。")
    parser.add_argument("-s", "--script", required=True, help="入口脚本文件名")
    parser.add_argument("-n", "--nodes", required=True, help="节点及 GPU 作业数，例如 1:8,2:8")
    parser.add_argument("-j", "--job-name", required=True, help="PBS 作业名前缀")
    parser.add_argument(
        "-r", "--root", type=Path, default=Path(__file__).resolve().parent,
        help="搜索代码运行目录",
    )
    parser.add_argument("-e", "--conda-env", default=DEFAULT_CONDA_ENV, help="PBS 作业使用的 conda 环境")
    parser.add_argument(
        "-t", "--threads", type=int, default=DEFAULT_THREADS,
        help="每个搜索进程允许使用的 CPU 线程数",
    )
    parser.add_argument(
        "-w", "--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU,
        help="每块 GPU 同时运行的搜索进程上限",
    )
    parser.add_argument(
        "-p", "--submit-delay", type=float, default=DEFAULT_SUBMIT_DELAY_S,
        help="两次 qsub 之间的秒数",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--dry-run", action="store_true", help="生成 PBS 文件并跳过 qsub")
    mode.add_argument("-S", "--status", action="store_true", help="显示当前用户的 PBS 作业")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "--run-section":
        if len(argv) != 3:
            print("--run-section requires SCRIPT_PATH SECTION", file=sys.stderr)
            return 2
        return run_section(argv[1], argv[2])
    args = build_parser().parse_args(argv)
    if args.status:
        return check_status()
    if args.submit_delay < 0:
        print("[ERROR] submit-delay 必须大于等于 0", file=sys.stderr)
        return 2

    try:
        node_config      = parse_node_config(args.nodes)
        root_path, entry = resolve_entry_script(args.root, args.script)
        section_num      = read_section_num(entry)
        jobs             = create_job_files(
            root_path       = root_path,
            script_path     = entry,
            node_config     = node_config,
            section_num     = section_num,
            job_name        = args.job_name,
            conda_env       = args.conda_env,
            threads         = args.threads,
            workers_per_gpu = args.workers_per_gpu,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(
        f"[PLAN] script={entry.name} section_num={section_num} jobs={len(jobs)} "
        f"workers_per_gpu={args.workers_per_gpu}"
    )
    for node, sections, pbs_path in jobs:
        print(
            f"[CREATE] gpu{node:02d} sections={list(sections)} "
            f"pbs={pbs_path.name}"
        )

    if args.dry_run:
        print("[DRY-RUN] PBS 文件生成完成")
        return 0

    failure_count = submit_jobs(jobs, args.submit_delay)
    if failure_count:
        print(f"[SUMMARY] {failure_count} 个作业提交失败", file=sys.stderr)
        return 1
    print(f"[SUMMARY] {len(jobs)} 个作业提交成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
