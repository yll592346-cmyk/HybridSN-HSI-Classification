"""Lightweight HybridSN reproduction for Pavia University.

Expected input files: PaviaU.mat and PaviaU_gt.mat. Run with the bundled
PyTorch environment, for example:
  python run_hybridsn.py --data-dir work/data --output-dir outputs/results
"""
import argparse
import csv
import json
import random
import shutil
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
               "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"]


class PatchSet(Dataset):
    def __init__(self, cube, coords, labels, patch):
        self.cube = cube
        self.coords = coords
        self.labels = labels
        self.patch = patch
        self.pad = patch // 2
        self.padded = np.pad(cube, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        r, c = self.coords[i]
        p = self.padded[r:r + self.patch, c:c + self.patch]
        # NCHW-style input: spectral channels first, then spatial dimensions.
        x = torch.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1))).float().unsqueeze(0)
        return x, int(self.labels[i])


class HybridSN(nn.Module):
    def __init__(self, bands, classes):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3)), nn.BatchNorm3d(8), nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, (5, 3, 3)), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, (3, 3, 3)), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        reduced_bands = bands - 12
        self.conv2d = nn.Sequential(
            nn.Conv2d(32 * reduced_bands, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(32, classes)

    def forward(self, x):
        x = self.conv3d(x)
        n, c, b, h, w = x.shape
        x = x.reshape(n, c * b, h, w)
        return self.head(self.conv2d(x).flatten(1))


def load_data(data_dir):
    data_dir = Path(data_dir)
    img_file, gt_file = data_dir / "PaviaU.mat", data_dir / "PaviaU_gt.mat"
    if not img_file.exists() or not gt_file.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urls = {
            img_file: "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            gt_file: "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        }
        for path, url in urls.items():
            if not path.exists():
                print("Downloading {}".format(path.name), flush=True)
                try:
                    with urllib.request.urlopen(url, timeout=20) as response, open(path, "wb") as target:
                        shutil.copyfileobj(response, target)
                except Exception as exc:
                    raise RuntimeError("自动下载失败（{}）。请从 Pavia University 官方数据页下载该文件，放入 {} 后重试。详情：{}".format(path.name, data_dir.resolve(), exc))
    if not img_file.exists() or not gt_file.exists():
        raise FileNotFoundError("需要 PaviaU.mat 与 PaviaU_gt.mat；请将两文件放入 --data-dir。")
    image = sio.loadmat(img_file)["paviaU"].astype(np.float32)
    gt = sio.loadmat(gt_file)["paviaU_gt"].astype(np.int64)
    return image, gt


def prepare(image, gt, bands, patch, seed):
    h, w, raw_bands = image.shape
    flat = image.reshape(-1, raw_bands)
    scaler = StandardScaler()
    flat = scaler.fit_transform(flat).astype(np.float32)
    pca = PCA(n_components=bands, whiten=True, random_state=seed)
    cube = pca.fit_transform(flat).reshape(h, w, bands).astype(np.float32)
    coords = np.argwhere(gt > 0)
    labels = gt[gt > 0] - 1
    return cube, coords, labels


def split_indices(labels, ratio, seed):
    idx = np.arange(len(labels))
    return train_test_split(idx, train_size=ratio, random_state=seed, stratify=labels)


def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    per = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {"OA": float(accuracy_score(y_true, y_pred)), "AA": float(per.mean()),
            "Kappa": float(cohen_kappa_score(y_true, y_pred)),
            "per_class_accuracy": [float(v) for v in per], "confusion_matrix": cm.tolist()}


def evaluate(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).argmax(1).cpu().numpy()
            ys.extend(y.numpy().tolist()); ps.extend(pred.tolist())
    return np.asarray(ys), np.asarray(ps)


def train_run(cube, coords, labels, gt, args, ratio, epochs, seed, run_name, device, full_eval=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    train_idx, test_idx = split_indices(labels, ratio, seed)
    train_ds = PatchSet(cube, coords[train_idx], labels[train_idx], args.patch)
    test_ds = PatchSet(cube, coords[test_idx], labels[test_idx], args.patch)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False, num_workers=0,
                             pin_memory=(device.type == "cuda"))
    model = HybridSN(cube.shape[-1], len(CLASS_NAMES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y); loss.backward(); opt.step()
            total_loss += loss.item() * len(y)
        history.append({"epoch": epoch + 1, "loss": total_loss / len(train_ds)})
        print("{} epoch {}/{} loss={:.4f}".format(run_name, epoch + 1, epochs, history[-1]["loss"]), flush=True)
    y_true, y_pred = evaluate(model, test_loader, device)
    result = metrics(y_true, y_pred)
    result.update({"run": run_name, "train_ratio": ratio, "epochs": epochs,
                   "bands": int(cube.shape[-1]), "patch": int(args.patch),
                   "train_samples": len(train_ds), "test_samples": len(test_ds),
                   "history": history})
    if full_eval:
        all_coords = np.argwhere(np.ones(gt.shape, dtype=bool))
        all_ds = PatchSet(cube, all_coords, np.zeros(len(all_coords), dtype=np.int64), args.patch)
        all_loader = DataLoader(all_ds, batch_size=args.batch * 2, shuffle=False, num_workers=0,
                                pin_memory=(device.type == "cuda"))
        preds = []
        model.eval()
        with torch.no_grad():
            for x, _ in all_loader:
                preds.extend(model(x.to(device, non_blocking=True)).argmax(1).cpu().numpy().tolist())
        result["map"] = np.asarray(preds, dtype=np.uint8).reshape(gt.shape) + 1
        result["model_state"] = model.state_dict()
    return result


def save_map(image, gt, pred, path):
    rgb = image[:, :, [55, 41, 12]]
    lo, hi = np.percentile(rgb, [2, 98]); rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    cmap = matplotlib.colors.ListedColormap(["black"] + list(plt.get_cmap("tab10").colors[:9]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax[0].imshow(rgb); ax[0].set_title("PaviaU false-color (55/41/12)")
    ax[1].imshow(gt, cmap=cmap, vmin=0, vmax=9); ax[1].set_title("Ground Truth")
    ax[2].imshow(pred, cmap=cmap, vmin=0, vmax=9); ax[2].set_title("HybridSN prediction")
    for a in ax: a.axis("off")
    fig.savefig(path, dpi=160); plt.close(fig)


def save_cm(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(np.asarray(cm), cmap="Blues")
    ax.set_xticks(range(9), CLASS_NAMES, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(9), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title("Confusion Matrix")
    for i in range(9):
        for j in range(9): ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_training_curve(histories, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, history in histories:
        ax.plot([x["epoch"] for x in history], [x["loss"] for x in history], marker="o", label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training cross-entropy loss")
    ax.set_title("HybridSN training loss"); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_comparison_plot(main, rows, path):
    selected = [r for r in rows if r["run"] != "comparison_baseline_4ep"]
    names = ["4-epoch\ncontrol"] + [r["run"].replace("_", "\n") for r in selected]
    data = [next(r for r in rows if r["run"] == "comparison_baseline_4ep")] + selected
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(data)); width = 0.24
    for offset, metric, label in [(-width, "OA", "OA"), (0, "AA", "AA"), (width, "Kappa", "Kappa")]:
        ax.bar(x + offset, [100*r[metric] for r in data], width, label=label)
    ax.set_xticks(x, names); ax.set_ylim(0, 100); ax.set_ylabel("Score (%)")
    ax.set_title("Controlled parameter comparisons (4 epochs each)")
    ax.grid(axis="y", alpha=0.2); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def spatial_analysis(gt, pred, result):
    cm = np.asarray(result["confusion_matrix"])
    gt_counts = np.bincount(gt[gt > 0] - 1, minlength=9)
    pred_counts = np.bincount(pred.reshape(-1) - 1, minlength=9)
    low = np.argsort(result["per_class_accuracy"])[:3]
    pairs = [(int(cm[i, j]), i, j) for i in range(9) for j in range(9) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    confusions = "；".join("{} 被分成 {}（{} 个）".format(CLASS_NAMES[i], CLASS_NAMES[j], n)
                          for n, i, j in pairs[:3]) or "测试集没有出现明显的类间错分"
    lows = "、".join("{}（{:.1%}）".format(CLASS_NAMES[i], result["per_class_accuracy"][i]) for i in low)
    most, pred_most = int(np.argmax(gt_counts)), int(np.argmax(pred_counts))
    count_note = "真实标注中像元最多的是{}（{}），预测图中最多的是{}（{}）。".format(
        CLASS_NAMES[most], gt_counts[most], CLASS_NAMES[pred_most], pred_counts[pred_most])
    return ("预测图覆盖整幅影像，包括 Ground Truth 未标注区域；Ground Truth 图中的黑色仅表示该像元没有人工标签，不能据此判定为某一种地物，也不能把预测图中相同位置的模型输出纳入测试精度。" + count_note +
            "空间上成片且边界清楚的区域通常更容易利用 Patch 邻域信息保持类别一致；道路、草地、屋顶等地物交界处的混合像元会同时包含多种光谱响应，容易沿边界形成细碎错分。较小或狭长的地物只占 Patch 的一部分，模型可能被周围占比更高的类别带偏；阴影、裸土、沥青及深色屋顶之间的光谱相似也会增加混淆。" +
            "本次测试集中逐类召回率最低的三类为：" + lows + "。混淆矩阵中数量较多的错分方向为：" + confusions + "。若预测图中某类斑块明显扩大而其召回率并不高，需检查该类精确率及其对应列；若斑块破碎并且召回率偏低，则应优先检查该类训练样本数量及其与周边类别的混淆。" +
            "OA 汇总所有测试像元，可能被大类主导；AA 对九类召回率等权平均，因此更能显示小类表现；Kappa 补偿随机一致性。阅读结果时应将三个指标与混淆矩阵、空间图共同判断，不能仅凭总体精度宣称所有类别都分类良好。当前为快速实验，训练样本比例、轮数和模型容量均受运行时间约束，边界噪声和少数类误分应作为误差分析重点；增大训练样本或增加轮数后应重新生成全图、重新计算指标，再比较改动前后的实测变化。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="work/data")
    p.add_argument("--output-dir", default="outputs/results")
    p.add_argument("--bands", type=int, default=30)
    p.add_argument("--patch", type=int, default=11)
    p.add_argument("--ratio", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--comparisons", action="store_true", help="run three concise ablation experiments")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device={}".format(torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"), flush=True)
    image, gt = load_data(args.data_dir)
    labels = gt[gt > 0] - 1
    print("image={} bands={} labeled_pixels={} classes={}".format(image.shape, image.shape[-1], len(labels), len(np.unique(labels))), flush=True)
    with open(out / "class_counts.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["label", "class_name", "pixel_count"])
        for i, name in enumerate(CLASS_NAMES): writer.writerow([i + 1, name, int(np.sum(gt == i + 1))])
    rows = []
    start = time.time()
    cube, coords, labels = prepare(image, gt, args.bands, args.patch, args.seed)
    main_run = train_run(cube, coords, labels, gt, args, args.ratio, args.epochs, args.seed,
                         "HybridSN-main", device, full_eval=True)
    main_map = main_run.pop("map"); state = main_run.pop("model_state")
    torch.save(state, out / "hybridsn_paviau.pt")
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(main_run, f, ensure_ascii=False, indent=2)
    np.save(out / "prediction_map.npy", main_map)
    save_map(image, gt, main_map, out / "classification_map.png")
    save_cm(main_run["confusion_matrix"], out / "confusion_matrix.png")
    with open(out / "experiment_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["run", "train_ratio", "bands", "patch", "epochs", "train_samples", "test_samples", "OA", "AA", "Kappa"])
        writer.writerow([main_run[k] for k in ["run", "train_ratio", "bands", "patch", "epochs", "train_samples", "test_samples", "OA", "AA", "Kappa"]])
    if args.comparisons:
        settings = [("comparison_baseline_4ep", args.ratio, args.bands, args.patch),
                    ("train_ratio_5pct", 0.05, args.bands, args.patch),
                    ("pca_bands_15", args.ratio, 15, args.patch),
                    ("patch_size_7", args.ratio, args.bands, 7)]
        for name, ratio, bands, patch in settings:
            local_args = argparse.Namespace(**vars(args)); local_args.patch = patch
            cube_c, coords_c, labels_c = prepare(image, gt, bands, patch, args.seed)
            res = train_run(cube_c, coords_c, labels_c, gt, local_args, ratio,
                            max(3, args.epochs // 2), args.seed, name, device, full_eval=False)
            rows.append(res)
        with open(out / "experiment_log.csv", "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for r in rows: writer.writerow([r[k] for k in ["run", "train_ratio", "bands", "patch", "epochs", "train_samples", "test_samples", "OA", "AA", "Kappa"]])
    histories = [(main_run["run"], main_run["history"])] + [(r["run"], r["history"]) for r in rows]
    save_training_curve(histories, out / "training_loss.png")
    if rows:
        save_comparison_plot(main_run, rows, out / "comparison_metrics.png")
    with open(out / "training_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["run", "epoch", "training_loss"])
        for name, history in histories:
            for item in history: writer.writerow([name, item["epoch"], item["loss"]])
    elapsed = time.time() - start
    with open(out / "report_zh.md", "w", encoding="utf-8") as f:
        f.write("# HybridSN 高光谱图像分类实验报告\n\n")
        f.write("## 实验设置\n\nPavia University 原始数据尺寸为 {}×{}×{}，含 {} 个有标签像元、9 个地物类别。采用 StandardScaler 与 PCA（{} 维）预处理，{}×{} 空间 Patch，按类别分层以 {:.0%} 像元训练、其余测试；HybridSN 使用 3D 卷积提取光谱-空间特征并接 2D 卷积分类，训练 {} 轮，批大小 {}，设备 {}。\n\n".format(*image.shape, int((gt>0).sum()), args.bands, args.patch, args.patch, args.ratio, args.epochs, args.batch, torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"))
        f.write("## 主实验结果\n\n| OA | AA | Kappa |\n|---:|---:|---:|\n| {:.4f} | {:.4f} | {:.4f} |\n\n".format(main_run["OA"], main_run["AA"], main_run["Kappa"]))
        f.write("混淆矩阵和逐类精度见配套图像与 metrics.json。OA 为整体精度，AA 为 9 类召回率的宏平均，Kappa 衡量扣除随机一致性后的分类一致程度。\n\n")
        f.write("精度解释注意：本实验按像元随机分层划分训练与测试集，邻近像元可能分别进入两组，而且 11×11 Patch 会产生空间重叠，因此 OA、AA、Kappa 可能偏乐观，更适合验证流程和同场景插值效果；若要评价跨区域泛化，应按空间区域留出测试集。标准化与 PCA 使用整幅影像的无标签光谱分布，属于转导式预处理，没有使用测试标签。\n\n")
        f.write("## 空间分布分析\n\n" + spatial_analysis(gt, main_map, main_run) + "\n\n")
        if rows:
            f.write("## 对比与消融实验\n\n| 设置 | 训练比例 | PCA 波段 | Patch | 轮数 | OA | AA | Kappa |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in rows: f.write("| {} | {:.0%} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(r["run"],r["train_ratio"],r["bands"],r["patch"],r["epochs"],r["OA"],r["AA"],r["Kappa"]))
            control = next(r for r in rows if r["run"] == "comparison_baseline_4ep")
            names = {"train_ratio_5pct": "训练比例降至 5%", "pca_bands_15": "PCA 降至 15 维", "patch_size_7": "Patch 缩至 7×7"}
            f.write("\n为控制训练轮数影响，以下差值均相对同样训练 4 轮的基准（OA {:.4f}、AA {:.4f}、Kappa {:.4f}）计算。".format(control["OA"], control["AA"], control["Kappa"]))
            for r in rows:
                if r["run"] == "comparison_baseline_4ep": continue
                f.write("{}：OA {:+.4f}、AA {:+.4f}、Kappa {:+.4f}。".format(names[r["run"]], r["OA"]-control["OA"], r["AA"]-control["AA"], r["Kappa"]-control["Kappa"]))
            f.write("这些差值来自同一随机种子和分层划分设置，但参数间仍可能有训练随机性；主实验 8 轮指标另列，不与 4 轮对照直接混作单变量差异。\n\n")
        f.write("总运行时间：{:.1f} 秒。复现实验命令：`python run_hybridsn.py --data-dir work/data --output-dir outputs/results --comparisons`。\n".format(elapsed))
    print("Saved results to {} in {:.1f}s".format(out.resolve(), elapsed), flush=True)


if __name__ == "__main__": main()
