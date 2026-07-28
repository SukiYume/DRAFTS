"""PBS 作业批量提交与监控。

把 ``node_config`` 里指定的 ``{节点号: GPU数}`` 展开成连续编号的 section，每块 GPU 生成一个
PBS 脚本（写到 ``<root_path>pbsspt/``）并 ``qsub`` 一次。可通过 ``workers_per_gpu`` 让同一块
GPU（同一次 qsub）上并发跑多个 section——PBS 只申请了 1 块 GPU，脚本内部起多个后台进程天然共享
这块卡，不需要调度器额外支持。所有作业最终用 section 编号区分，传给搜索脚本作为 ``sys.argv[1]``。

脚本也可以从 ``d-center-binary-gate.py`` 模板渲染目标专用入口脚本，并在提交前检查
``node_config`` 展开后的总 section 数（= GPU 数 x ``workers_per_gpu``）是否等于入口脚本中的
``process_config.section_num``。

Usage:
    python s-pbsspt.py                # 提交全部作业
    python s-pbsspt.py --dry-run      # 只生成 PBS 文件，不实际 qsub
    python s-pbsspt.py --status       # 查 qstat 状态
    python s-pbsspt.py --help         # 显示帮助

Generic example:
    node_config = {1: 8}
    workers_per_gpu = 4
    # total sections = 8 * 4 = 32, matching d-center-binary-gate.py section_num=32
"""

import os, sys, time, subprocess, re
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. PBS 脚本模板
# ---------------------------------------------------------------------------

def create_pbs_script(sections, node, root_path, script_name, job_name_prefix="center"):
    """生成一个 PBS 脚本：在这块 GPU 上并发跑 ``sections`` 里的所有 section。

    PBS 只申请 1 块 GPU（``GRES:gpu@1``），``sections`` 里的每个 section 各起一个
    后台进程，天然共享这块卡；每个 section 的输出单独重定向到自己的日志文件，
    避免和其它并发 section 的输出混在一起。
    """
    section_tag = f"{sections[0]:02d}" if len(sections) == 1 else f"{sections[0]:02d}-{sections[-1]:02d}"
    run_lines = "\n".join(
        f'python {script_name} {s} > {root_path}pbsspt/{job_name_prefix}-section{s:02d}.log 2>&1 &\n'
        f'pids+=($!)'
        for s in sections
    )
    return f'''#!/bin/bash
#PBS -N {job_name_prefix}-{section_tag}
#PBS -o {root_path}pbsspt/{job_name_prefix}-cm{section_tag}-output.log
#PBS -e {root_path}pbsspt/{job_name_prefix}-cm{section_tag}-error.err
#PBS -q gpu
#PBS -l nodes=gpu{node:02d}
#PBS -W x=GRES:gpu@1

# 环境
SHELL_INIT="${{SHELL_INIT:-${{HOME}}/.bashrc}}"
source "$SHELL_INIT"
conda activate "${{CONDA_ENV:-pytorch}}"
nomps

# 限制每个 Python worker 的 CPU 线程池，避免 OpenBLAS/PyTorch/OpenCV 默认按整机核数开线程
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export BLIS_NUM_THREADS=8

# 工作目录
cd {root_path}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# 运行搜索（同一块 GPU 上并发跑 {len(sections)} 个 section，各自日志见 section*.log）
echo "Starting sections {sections} processing..."
pids=()
{run_lines}

# 逐个 wait 取子进程真实退出码；任一 section 失败（OOM/异常退出）都让整个 PBS 作业标记为失败，
# 避免裸 wait 只返回最后一个子进程状态、把失败悄悄吃掉。
status=0
for pid in "${{pids[@]}}"; do
    wait "$pid" || status=1
done

echo "Job finished at: $(date)"
exit "$status"
'''


# ---------------------------------------------------------------------------
# 2. gate 脚本渲染 / 自检
# ---------------------------------------------------------------------------

def _python_literal(value):
    """把配置值写回 Python 源码。"""
    return repr(value)


def _replace_assignment(text, name, value):
    pattern = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)[^#\n]+(.*)$", re.M)
    text, count = pattern.subn(
        lambda match: f"{match.group(1)}{_python_literal(value)}{match.group(2)}",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"未找到赋值项: {name}")
    return text


def _replace_config_value(text, name, value):
    pattern = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)[^,\n]+(,.*)$", re.M)
    text, count = pattern.subn(
        lambda match: f"{match.group(1)}{_python_literal(value)}{match.group(2)}",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"未找到 ProcessConfig 参数: {name}")
    return text


def prepare_gate_script(root_path, template_name, output_name, *,
                        process_config=None, data_path=None, save_base=None,
                        beam_filter=None, log_file=None, detector_type=None,
                        detector_ckpt=None, classifier_model_name=None,
                        classifier_ckpt=None):
    """从 d-center-binary-gate.py 生成某个 FRB 专用入口脚本。

    可替换 ``ProcessConfig``、数据路径、输出路径、beam、日志文件、模型类型和模型路径。
    返回值是生成后的入口脚本文件名，可直接传给 ``submit_jobs``。
    """
    root = Path(root_path)
    template_path = root / template_name
    output_path = root / output_name
    text = template_path.read_text(encoding='utf-8')

    for key, value in (process_config or {}).items():
        text = _replace_config_value(text, key, value)
    for key, value in {
        'log_file': log_file,
        'data_path': data_path,
        'save_base': save_base,
        'beam_filter': beam_filter,
        'DETECTOR_TYPE': detector_type,
        'DETECTOR_CKPT': detector_ckpt,
        'CLASSIFIER_MODEL_NAME': classifier_model_name,
        'CLASSIFIER_CKPT': classifier_ckpt,
    }.items():
        if value is not None:
            text = _replace_assignment(text, key, value)

    output_path.write_text(text, encoding='utf-8')
    print(f"[PREPARE] {template_name} -> {output_name}")
    return output_name


def read_script_section_num(root_path, script_name):
    script_path = Path(root_path) / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"搜索脚本不存在: {script_path}")
    text = script_path.read_text(encoding='utf-8')
    match = re.search(r"^\s*section_num\s*=\s*(\d+)\s*,", text, re.M)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# 3. 批量提交
# ---------------------------------------------------------------------------

def submit_jobs(root_path, script_name, node_config, job_name="center", dry_run=False, workers_per_gpu=1):
    """按 ``node_config`` 展开 section 并 qsub。

    Args
    ----
    root_path :
        搜索代码所在目录（结尾要带 ``/``）；PBS 脚本会写在 ``<root_path>pbsspt/``。
    script_name :
        搜索入口文件名（``d-center-binary-gate.py`` 或 ``d-dm-time-predown.py``）。
    node_config :
        ``{节点号: GPU数}``，如 ``{1: 4, 2: 4}`` 表示两个计算节点各使用 4 块 GPU
        （每块 GPU 对应一次 qsub）。节点号应按目标 PBS 集群配置。
    job_name :
        PBS Job 名前缀，便于 ``qstat`` 过滤。
    dry_run :
        True → 只生成 PBS 文件，不 qsub。
    workers_per_gpu :
        每块 GPU（每次 qsub）上并发跑多少个 section，取值 1~4。默认 1，等价于原来的一卡一
        section。总 section 数 = ``sum(node_config.values()) * workers_per_gpu``，须等于
        ``script_name`` 里的 ``process_config.section_num``。
    """
    script_path = f'{root_path}pbsspt/'
    os.makedirs(script_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = f'{script_path}submission_log_{timestamp}.txt'

    submitted_jobs = []
    failed_jobs    = []
    total_gpus         = sum(node_config.values())
    total_sections     = total_gpus * workers_per_gpu
    script_section_num = read_script_section_num(root_path, script_name)
    if script_section_num is not None and script_section_num != total_sections:
        raise ValueError(
            f"node_config 总 GPU 数({total_gpus}) x workers_per_gpu({workers_per_gpu}) = "
            f"{total_sections}，必须等于 {script_name} 里的 process_config.section_num"
            f"({script_section_num})"
        )

    print(f"[INFO] 准备提交 {total_gpus} 个作业，每卡并发 {workers_per_gpu} 个 section，"
          f"共 {total_sections} 个 section")
    print(f"[INFO] 节点分配:")
    for node, count in node_config.items():
        print(f"       Node {node:02d}: {count} 块 GPU")
    print(f"[INFO] 脚本目录: {script_path}")
    print(f"[INFO] 日志文件: {log_file}")

    with open(log_file, 'w') as log_f:
        log_f.write(f"PBS Job Submission Log - {timestamp}\n"
                    f"Script: {script_name}\n"
                    f"Total Sections: {total_sections}\n"
                    f"Workers per GPU: {workers_per_gpu}\n"
                    f"Node Config: {node_config}\n"
                    f"{'-' * 50}\n")

        section = 0
        for node, job_count in node_config.items():
            log_f.write(f"\nNode {node:02d} - 提交 {job_count} 块 GPU:\n")
            print(f"\n[NODE {node:02d}] 开始提交 {job_count} 块 GPU...")

            for _ in range(job_count):
                sections      = list(range(section, section + workers_per_gpu))
                section_tag   = f"{sections[0]:02d}" if workers_per_gpu == 1 else f"{sections[0]:02d}-{sections[-1]:02d}"
                pbs_file_name = f'{job_name}-node-{node:02d}-section-{section_tag}.pbs'
                pbs_file_path = os.path.join(script_path, pbs_file_name)

                try:
                    pbs_content = create_pbs_script(sections, node, root_path, script_name, job_name)
                    with open(pbs_file_path, 'w') as pbs_f:
                        pbs_f.write(pbs_content)

                    print(f"[CREATE] Sections {sections} -> Node {node:02d}: {pbs_file_name}")
                    log_f.write(f"  Sections {sections} -> Node {node:02d}: {pbs_file_name}\n")

                    if not dry_run:
                        result = subprocess.run(
                            f'qsub {pbs_file_path}', shell=True,
                            capture_output=True, text=True,
                        )
                        if result.returncode == 0:
                            job_id = result.stdout.strip()
                            submitted_jobs.append((sections, node, job_id))
                            print(f"[SUBMIT] Sections {sections} 提交成功: {job_id}")
                            log_f.write(f"    -> 提交成功: {job_id}\n")
                        else:
                            failed_jobs.append((sections, node, result.stderr))
                            print(f"[ERROR] Sections {sections} 提交失败: {result.stderr}")
                            log_f.write(f"    -> 提交失败: {result.stderr}\n")
                        time.sleep(5)                                     # 避免过快提交
                    else:
                        print(f"[DRY-RUN] 将提交: {pbs_file_name}")
                        log_f.write(f"    -> DRY-RUN模式\n")

                except Exception as e:
                    failed_jobs.append((sections, node, str(e)))
                    print(f"[ERROR] Sections {sections} (Node {node:02d}) 处理失败: {e}")
                    log_f.write(f"  Sections {sections} 处理失败: {e}\n")

                section += workers_per_gpu

        log_f.write(f"{'-' * 50}\n总结:\n"
                    f"  成功提交: {len(submitted_jobs)}\n"
                    f"  提交失败: {len(failed_jobs)}\n")

    print(f"\n[SUMMARY]"
          f"\n  成功提交: {len(submitted_jobs)} 个作业"
          f"\n  提交失败: {len(failed_jobs)} 个作业")

    if submitted_jobs:
        print("  成功的作业ID:")
        for sections, node, job_id in submitted_jobs:
            print(f"    Sections {sections} (Node {node:02d}): {job_id}")
    if failed_jobs:
        print("  失败的作业:")
        for sections, node, error in failed_jobs:
            print(f"    Sections {sections} (Node {node:02d}): {error}")

    return submitted_jobs, failed_jobs


# ---------------------------------------------------------------------------
# 4. 作业状态查询
# ---------------------------------------------------------------------------

def check_job_status(job_ids=None):
    """``qstat`` 查作业状态。``job_ids=None`` 时查当前用户所有作业。"""
    if job_ids:
        for job_id in job_ids:
            result = subprocess.run(f'qstat {job_id}', shell=True, capture_output=True, text=True)
            print(f"Job {job_id}: {result.stdout.strip()}")
    else:
        result = subprocess.run('qstat -u $USER', shell=True, capture_output=True, text=True)
        print(result.stdout)


def print_help():
    print(__doc__)
    print("配置说明:")
    print("  在脚本里修改 root_path / script_name / node_config / job_name / workers_per_gpu 五项即可")
    print("  例如: node_config = {1: 4, 2: 4} 表示两个计算节点各使用 4 块 GPU")
    print("  workers_per_gpu=4 表示每块 GPU 上并发跑 4 个 section（1~4 皆可）")
    print("  通用示例：node_config={1: 8}, workers_per_gpu=4, section_num=32")
    print("  总 section 数 = sum(node_config.values()) * workers_per_gpu，应等于搜索脚本里的")
    print("  process_config.section_num")
    print("常用 script_name:")
    print("  d-center-binary-gate.py  -> PBS 内执行 python d-center-binary-gate.py <section>")
    print("  d-dm-time-predown.py     -> PBS 内执行 python d-dm-time-predown.py <section>")
    print("  c-data-check.py          -> PBS 内执行 python c-data-check.py <section>")
    print("目标专用 gate 脚本:")
    print("  prepare_gate_script(...) -> 生成 d-center-binary-<target>.py 并作为 script_name 提交")


# ---------------------------------------------------------------------------
# 5. CLI 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # ---- 配置 ----
    root_path       = '/path/to/drafts_runs/search_pipeline/'                # 末尾要带 '/'
    workers_per_gpu = 4                                                   # 每块 GPU 并发跑几个 section（1~4）
    script_name     = 'd-center-binary-gate.py'                           # 或 'd-dm-time-predown.py'
    node_config     = {1: 8}                                              # {节点号: GPU数}
    job_name        = 'drafts'
    # sum(node_config.values()) * workers_per_gpu 必须等于 process_config.section_num
    # 当前配置：8 块 GPU x 4 workers = 32，对应 d-center-binary-gate.py 里 section_num=32

    # ---- 从 gate 模板生成目标专用入口脚本 ----
    # 多个目标可分别调用 prepare_gate_script，并把返回值作为 script_name 提交。
    # script_name = prepare_gate_script(
    #     root_path, 'd-center-binary-gate.py', 'd-center-binary-target.py',
    #     process_config={
    #         'dm_range': 4096,
    #         'dm_scale': 1,
    #         'dm_offset': 0,
    #         'dm_threshold': 10,
    #         'block_size': 4096,
    #         'dm_span': 1024,
    #         'det_prob': 0.45,
    #         'section_num': sum(node_config.values()) * workers_per_gpu,
    #         'time_factor': 8,
    #     },
    #     data_path='/path/to/observations/source/date/',
    #     save_base='/path/to/observations/',
    #     beam_filter='M01',
    #     log_file='./processing_log_target.txt',
    # )

    # ---- CLI 解析 ----
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--dry-run':
            print("[INFO] DRY-RUN 模式，不会实际提交作业")
            submit_jobs(root_path, script_name, node_config, job_name, dry_run=True,
                        workers_per_gpu=workers_per_gpu)
        elif arg == '--status':
            check_job_status()
        elif arg in ('--help', '-h'):
            print_help()
        else:
            print(f"[ERROR] 未知参数: {arg}")
            print_help()
    else:
        submitted, _ = submit_jobs(root_path, script_name, node_config, job_name,
                                    workers_per_gpu=workers_per_gpu)
        if submitted:
            print("\n[INFO] 检查作业状态:")
            print("  qstat -u $USER")
            print(f"  或: python {os.path.basename(sys.argv[0])} --status")
