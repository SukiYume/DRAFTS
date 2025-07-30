import sys, time, subprocess, os
from datetime import datetime

def create_pbs_script(section, node, root_path, script_name, job_name_prefix="center"):
    """创建PBS脚本内容"""
    pbs_content = f'''#!/bin/bash
#PBS -N {job_name_prefix}-{section:02d}
#PBS -o {root_path}pbsspt/cm{section:02d}-output.log
#PBS -e {root_path}pbsspt/cm{section:02d}-error.err
#PBS -q gpu
#PBS -l nodes=gpu{node:02d}
#PBS -W x=GRES:gpu@1

# 设置环境
source /home/ykzhang/.bashrc
conda activate pytorch

# 进入工作目录
cd {root_path}

# 记录任务开始时间
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# 运行脚本
echo "Starting section {section} processing..."
python {script_name} {section}

# 记录任务结束时间
echo "Job finished at: $(date)"
'''
    return pbs_content


def submit_jobs(root_path, script_name, sections=5, node_list=None, job_name="center", dry_run=False):
    """批量提交PBS作业"""

    script_path = f'{root_path}pbsspt/'

    # 创建PBS脚本目录
    os.makedirs(script_path, exist_ok=True)

    # 记录提交信息
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{script_path}submission_log_{timestamp}.txt'

    submitted_jobs = []
    failed_jobs = []

    print(f"[INFO] 准备提交 {sections} 个作业")
    print(f"[INFO] 使用节点: {node_list}")
    print(f"[INFO] 脚本路径: {script_path}")
    print(f"[INFO] 日志文件: {log_file}")

    with open(log_file, 'w') as log_f:
        log_f.write(f"PBS Job Submission Log - {timestamp}\n")
        log_f.write(f"Script: {script_name}\n")
        log_f.write(f"Sections: {sections}\n")
        log_f.write(f"Nodes: {node_list}\n")
        log_f.write("-" * 50 + "\n")

        for i in range(sections):
            try:
                # 分配节点（循环使用）
                node = node_list[i % len(node_list)]
                section = i

                # 创建PBS脚本
                pbs_content = create_pbs_script(section, node, root_path, script_name, job_name)
                pbs_file_name = f'{job_name}-node-{node:02d}-section-{section:02d}.pbs'
                pbs_file_path = os.path.join(script_path, pbs_file_name)

                # 写入PBS文件
                with open(pbs_file_path, 'w') as pbs_f:
                    pbs_f.write(pbs_content)

                print(f"[CREATE] Section {section:02d} -> Node {node:02d}: {pbs_file_name}")
                log_f.write(f"Section {section:02d} -> Node {node:02d}: {pbs_file_name}\n")

                if not dry_run:
                    # 提交作业
                    cmd = f'qsub {pbs_file_path}'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                    if result.returncode == 0:
                        job_id = result.stdout.strip()
                        submitted_jobs.append((section, node, job_id))
                        print(f"[SUBMIT] Section {section:02d} 提交成功: {job_id}")
                        log_f.write(f"  -> 提交成功: {job_id}\n")
                    else:
                        failed_jobs.append((section, node, result.stderr))
                        print(f"[ERROR] Section {section:02d} 提交失败: {result.stderr}")
                        log_f.write(f"  -> 提交失败: {result.stderr}\n")

                    # 避免过快提交
                    time.sleep(5)
                else:
                    print(f"[DRY-RUN] 将提交: {pbs_file_name}")
                    log_f.write(f"  -> DRY-RUN模式\n")

            except Exception as e:
                failed_jobs.append((i, node_list[i % len(node_list)], str(e)))
                print(f"[ERROR] Section {i:02d} 处理失败: {e}")
                log_f.write(f"Section {i:02d} 处理失败: {e}\n")

        # 写入总结
        log_f.write("-" * 50 + "\n")
        log_f.write(f"总结:\n")
        log_f.write(f"  成功提交: {len(submitted_jobs)}\n")
        log_f.write(f"  提交失败: {len(failed_jobs)}\n")

    # 打印总结
    print(f"\n[SUMMARY]")
    print(f"  成功提交: {len(submitted_jobs)} 个作业")
    print(f"  提交失败: {len(failed_jobs)} 个作业")

    if submitted_jobs:
        print(f"  成功的作业ID:")
        for section, node, job_id in submitted_jobs:
            print(f"    Section {section:02d} (Node {node:02d}): {job_id}")

    if failed_jobs:
        print(f"  失败的作业:")
        for section, node, error in failed_jobs:
            print(f"    Section {section:02d} (Node {node:02d}): {error}")

    return submitted_jobs, failed_jobs


def check_job_status(job_ids=None):
    """检查作业状态"""
    if job_ids:
        for job_id in job_ids:
            cmd = f'qstat {job_id}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(f"Job {job_id}: {result.stdout.strip()}")
    else:
        # 检查所有作业
        cmd = 'qstat -u $USER'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)


if __name__ == '__main__':

    # 配置参数
    root_path   = '/home/ykzhang/CRAFTS/'
    script_name = 'd-center-binary.py'  # 修改为对应的脚本名
    sections    = 5
    node_list   = [13]
    job_name    = "center"

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--dry-run':
            print("[INFO] 运行在DRY-RUN模式，不会实际提交作业")
            submit_jobs(root_path, script_name, sections, node_list, job_name, dry_run=True)
        elif sys.argv[1] == '--status':
            print("[INFO] 检查作业状态")
            check_job_status()
        elif sys.argv[1] == '--help':
            print("用法:")
            print("  python searchpbs_center.py           # 提交所有作业")
            print("  python searchpbs_center.py --dry-run # 预览模式，不实际提交")
            print("  python searchpbs_center.py --status  # 检查作业状态")
            print("  python searchpbs_center.py --help    # 显示帮助")
        else:
            sections = int(sys.argv[1])
            submit_jobs(root_path, script_name, sections, node_list, job_name)
    else:
        # 默认提交作业
        submitted, failed = submit_jobs(root_path, script_name, sections, node_list, job_name)

        if submitted:
            print(f"\n[INFO] 可以使用以下命令检查作业状态:")
            print(f"  qstat -u $USER")
            print(f"  或运行: python {sys.argv[0]} --status")