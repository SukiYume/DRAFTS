import os, sys

from multiprocessing import Pool, set_start_method

import torch
from centernet_utils import get_res
from centernet_model import centernet
from binary_model import BinaryNet

import importlib
d_center_binary = importlib.import_module('d-center-binary')
process_path    = d_center_binary.process_path
ProcessConfig   = d_center_binary.ProcessConfig

device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR         = 'pbsspt'
COMPLETED_LOG   = os.path.join(LOG_DIR, 'completed_paths.log')
ERROR_LOG       = os.path.join(LOG_DIR, 'error_paths.log')


# Helper functions for logging
def log_completion(section, data_path, beam, save_path):
    """Log successful completion of a path processing."""
    path_key = f"{section}|{data_path}|{beam}|{save_path}"
    os.makedirs(os.path.dirname(COMPLETED_LOG), exist_ok=True)
    with open(COMPLETED_LOG, 'a') as f:
        f.write(f"{path_key}\n")
    return path_key


def log_error(section, data_path, beam, save_path, error_msg):
    """Log error during path processing."""
    path_key = f"{section}|{data_path}|{beam}|{save_path}"
    os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)
    with open(ERROR_LOG, 'a') as f:
        f.write(f"{path_key}|{error_msg}\n")
    return path_key


def get_completed_paths(section=None):
    """Get set of completed paths, optionally filtered by section."""
    completed = set()
    if os.path.exists(COMPLETED_LOG):
        with open(COMPLETED_LOG, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('|', 1)
                    if section is None or (len(parts) > 1 and parts[0] == str(section)):
                        completed.add(parts[1] if len(parts) > 1 else parts[0])
    return completed


# Simplified safe_process_path function
def safe_process_path(*args):
    section = args[-1]  # Section ID is passed as the last argument
    data_path, beam, save_path = args[:3]
    args = args[:-1]  # Remove section from args
    try:
        print(f"[START] Section {section} processing: {data_path}|{beam}|{save_path}")
        process_path(*args)
        path_key = log_completion(section, data_path, beam, save_path)
        print(f"[SUCCESS] Task completed and logged: {path_key}")
        return True
    except Exception as e:
        import traceback
        print(f"[ERROR] Section {section} task {data_path}|{beam}|{save_path} failed: {e}")
        traceback.print_exc()
        path_key = log_error(section, data_path, beam, save_path, str(e)[:100])
        print(f"[ERROR] Failure logged to {ERROR_LOG}")
        return False


if __name__ == '__main__':

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    process_config = ProcessConfig(
        DM_range        = 4096,
        DM_SCALE        = 1, # 1 / 4
        DM_OFFSET       = 0, # - 128
        DM_THRESHOLD    = 50,
        block_size      = 8192,
        det_prob        = 0.3,
        gpu_num         = 5
    )

    ## 载入模型
    base_model    = 'resnet18'
    model         = centernet(model_name=base_model).to(device)
    model.load_state_dict(torch.load('./best_model_resnet18.pth', map_location=device, weights_only=True))
    model.eval()

    class_model   = BinaryNet(base_model, num_classes=2).to(device)
    class_model.load_state_dict(torch.load('./best_model_resnet18_fix_n4.pth', map_location=device, weights_only=True))
    class_model.eval()

    ## 处理路径
    section             = int(sys.argv[1])

    ## 获取所有路径
    import glob
    root_path           = '/data31/ZD2023_1_1_2bit/'
    data_list           = glob.glob(root_path + '*/*/')

    # 首先生成所有参数列表
    all_params_list     = []
    for data_path in data_list:
        project, source, date = data_path.split('/')[-4: -1]
        for beam in range(1, 20):
            save_dir    = f'/home/ykzhang/low.iops.files/CRAFTS/2023/{source}/{date}/{beam:0>2d}/'
            all_params_list.append((data_path, beam, save_dir, model, class_model, process_config, section))
    print(f"[INFO] 总共生成 {len(all_params_list)} 个任务")

    # 检查已经完成的路径
    completed_paths     = get_completed_paths()
    print(f"[INFO] 已加载 {len(completed_paths)} 个已完成路径，将跳过这些路径")

    # 过滤所有已经完成的路径
    filtered_all_params = []
    for params in all_params_list:
        data_path, beam, save_path = params[:3]
        path_key        = f"{data_path}|{beam}|{save_path}"
        if path_key in completed_paths:
            print(f"[SKIP] 跳过已完成的路径: {path_key}")
        else:
            filtered_all_params.append(params)
    print(f"[INFO] 过滤后剩余 {len(filtered_all_params)} 个待处理任务")

    # 然后按section分割
    params_list_section = filtered_all_params[section::process_config.gpu_num]
    print(f"[INFO] 本节区域处理 {len(params_list_section)} 个路径")

    # No wrapper function needed now
    with Pool(4) as pool:
        pool.starmap(safe_process_path, params_list_section) # pool.starmap(process_path, params_list_section)