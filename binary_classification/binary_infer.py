"""在全量数据集上跑 binary 分类器，输出每条样本的 ``label/pred/prob``，并打印准确率。

``--weights`` 既可以传训练得到的 ``best_model.pth``（raw），也可以传 ``best_model_ema.pth``
（EMA）；后者通常更稳，建议两个都跑一下、用验证集判定哪个更好。
"""

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from binary_data import BurstDataset, build_catalog
from binary_model import build_binary_model


# ---------------------------------------------------------------------------
# 1. 设备 + 权重加载
# ---------------------------------------------------------------------------

def setup_device(args):
    if args.device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.device:
        return torch.device(f"cuda:{args.device}")
    return torch.device("cuda")


def load_weights(model, weights, device):
    """兼容 ``state_dict`` 和完整 checkpoint dict。"""
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# 2. 推理：对整个 catalog 跑一遍，每条样本输出 (label, pred, prob_false, prob_true)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    """返回与 catalog 行一一对应的 dict 列表（顺序与 DataLoader 一致）。"""
    model.eval()
    rows = []
    for inputs, labels in tqdm(loader, dynamic_ncols=True, ascii=True, desc="Infer"):
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        for label, p, pr in zip(labels.cpu().numpy(), pred.cpu().numpy(), prob.cpu().numpy()):
            rows.append({
                "label": int(label),
                "pred": int(p),
                "prob_false": float(pr[0]),
                "prob_true": float(pr[1]),
            })
    return rows


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def main(args):
    device = setup_device(args)
    model = build_binary_model(
        model_type=args.model_type, model_name=args.model_name,
        num_classes=2, pretrained=False, dropout=args.dropout,
    ).to(device)
    load_weights(model, args.weights, device)

    catalog = build_catalog(args.data_path)
    # 与训练时的 val 数据流一致：单图加载、不做随机增强
    dataset = BurstDataset(catalog, val=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    pred_rows = predict(model, loader, device)
    meta_cols = ["h5_file", "local_idx", "name", "source", "sample_key"]
    out = pd.concat(
        [catalog[meta_cols].reset_index(drop=True), pd.DataFrame(pred_rows)],
        axis=1,
    )

    correct = int((out["label"] == out["pred"]).sum())
    print(f"Accuracy: {correct / len(out):.4f} ({correct}/{len(out)})")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"Saved predictions: {args.output}")


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="./Data/")
    parser.add_argument("--output", type=str, default="binary_predictions.csv",
                        help="预测结果 CSV 输出路径；置空字符串可只打印 accuracy 不落盘")

    # model（须与训练时一致）
    parser.add_argument("--model-type", type=str, default="ConvNeXtNet", choices=["ConvNeXtNet", "SPPConvNeXt"])
    parser.add_argument("--model-name", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"])
    parser.add_argument("--dropout", type=float, default=0.5, help="仅 SPPConvNeXt 用到")

    # hardware
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)

    main(parser.parse_args())
