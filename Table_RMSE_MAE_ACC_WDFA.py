import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta

# 假设这些模块在你本地存在
from openstl.models.ambhfn import AMBHFN_Model
from utils.data_sliding import data_process
from config import configs

# ================= 逻辑区：评估函数 (返回逐小时数组) =================
def weighted_rmse(y_pred, y_true):
    lat = np.linspace(38, 54, num=64, dtype=float, endpoint=True)
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    w = torch.tensor(weights_lat, device=y_pred.device, dtype=y_pred.dtype)

    RMSE_hourly = np.empty([y_pred.size(1)])
    for i in range(y_pred.size(1)):
        diff_sq = (y_pred[:, i, :, :] - y_true[:, i, :, :]).permute(0, 2, 1) ** 2
        RMSE_hourly[i] = np.sqrt((diff_sq * w).mean([-2, -1]).cpu().numpy()).mean()
    return RMSE_hourly


def weighted_mae(y_pred, y_true):
    lat = np.linspace(38, 54, num=64, dtype=float, endpoint=True)
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    w = torch.tensor(weights_lat, device=y_pred.device, dtype=y_pred.dtype)

    MAE_hourly = np.empty([y_pred.size(1)])
    for i in range(y_pred.size(1)):
        diff_abs = abs(y_pred[:, i, :, :] - y_true[:, i, :, :]).permute(0, 2, 1)
        MAE_hourly[i] = (diff_abs * w).mean([0, -2, -1]).cpu().numpy()
    return MAE_hourly


def weighted_acc(y_pred, y_true):
    lat = np.linspace(38, 54, num=64, dtype=float, endpoint=True)
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    w = torch.tensor(weights_lat, device=y_pred.device, dtype=y_pred.dtype)

    ACC_hourly = np.empty([y_pred.size(1)])
    for i in range(y_pred.size(1)):
        clim = y_true[:, i, :, :].mean(0)
        a = y_true[:, i, :, :] - clim
        a_prime = (a - a.mean()).permute(0, 2, 1)
        fa = y_pred[:, i, :, :] - clim
        fa_prime = (fa - fa.mean()).permute(0, 2, 1)
        num = torch.sum(w * fa_prime * a_prime)
        den = torch.sqrt(torch.sum(w * fa_prime ** 2) * torch.sum(w * a_prime ** 2))
        ACC_hourly[i] = (num / den).cpu().item()
    return ACC_hourly


def calculate_wdfa_all_thresholds(u_pred, v_pred, u_true, v_true):
    angle_pred = np.degrees(np.arctan2(v_pred, u_pred))
    angle_true = np.degrees(np.arctan2(v_true, u_true))
    diff = np.abs(angle_pred - angle_true)
    diff = np.where(diff > 180, 360 - diff, diff)

    wdfa_90 = np.mean(diff <= 90.0, axis=(0, 2, 3)) * 100
    wdfa_45 = np.mean(diff <= 45.0, axis=(0, 2, 3)) * 100
    wdfa_22 = np.mean(diff <= 22.5, axis=(0, 2, 3)) * 100
    return wdfa_90, wdfa_45, wdfa_22


# ================= 配置区 =================
MODEL_PATH = './chkfile/checkpoint_ambhfn.chk'
DATA_DIR = './data/Northeast'
GAP = 12


# ==========================================

class FastTestDataset(Dataset):
    def __init__(self, base_dataset): self.base = base_dataset

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx].astype(np.float32)
        return sample[:24], sample[24:]


def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    print("Loading test data...")
    uv_test = np.load(os.path.join(DATA_DIR, 'uv100_test.npy'))
    zt_test = np.load(os.path.join(DATA_DIR, '1000zt_test.npy'))
    ele = np.load(os.path.join(DATA_DIR, 'DEM_northeast.npy'))
    ele[ele < 0] = 0
    ele = torch.from_numpy((ele - ele.mean()) / ele.std()).float().to(device)

    # 2. 构建 Dataloader
    base_ds = data_process((uv_test, zt_test), samples_gap=GAP)
    test_loader = DataLoader(FastTestDataset(base_ds), batch_size=4, shuffle=False)

    # 3. 加载模型
    model = AMBHFN_Model(
        disable_attention=getattr(configs, 'disable_attention', False),
        disable_conv=getattr(configs, 'disable_conv', False),
        simplify_depth=getattr(configs, 'simplify_depth', False)
    ).to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        state_dict = ckpt['net'] if 'net' in ckpt else (ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"✅ Loaded model successfully")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
        return

    # 4. 推理
    preds, trues = [], []
    print("Inference in progress...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            output = model(inputs.to(device), ele)
            preds.append(output.cpu().numpy())
            trues.append(targets.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 5. 反归一化
    u_true = trues[:, :, 0] * 3.520639587404695 + 1.3793918561383813
    v_true = trues[:, :, 1] * 4.020371225633169 - 0.1864951025062636
    u_pred = preds[:, :, 0] * 3.520639587404695 + 1.3793918561383813
    v_pred = preds[:, :, 1] * 4.020371225633169 - 0.1864951025062636

    w_true = np.sqrt(u_true ** 2 + v_true ** 2)
    w_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    # 6. 计算季节索引
    print("Calculating seasonal indices...")
    start_date = datetime(2023, 1, 1, 0, 0, 0)
    seasons = []

    for i in range(len(preds)):
        target_start_hour = i * GAP + 24
        current_date = start_date + timedelta(hours=int(target_start_hour))
        month = current_date.month

        if month in [3, 4, 5]:
            seasons.append('Spring')
        elif month in [6, 7, 8]:
            seasons.append('Summer')
        elif month in [9, 10, 11]:
            seasons.append('Autumn')
        else:
            seasons.append('Winter')

    seasons = np.array(seasons)
    season_names = ['Spring', 'Summer', 'Autumn', 'Winter']

    # 定义要切分的时间段字典 (名称: [切片起始, 切片结束])
    time_intervals = {
        '1h': (0, 1),
        '2h': (1, 2),
        '3h': (2, 3),
        '4h': (3, 4),
        '5-7h': (4, 7),
        '7-13h': (7, 13),
        '14-24h': (13, 24)
    }

    # 7. 打印分段报表
    print("\n" + "=" * 115)
    header = f"{'Season':<8} | {'Interval':<8} | {'RMSE':<8} | {'MAE':<8} | {'ACC':<8} | {'WDFA 90°':<10} | {'WDFA 45°':<10} | {'WDFA 22.5°':<10}"
    print(header)
    print("-" * 115)

    for season in season_names:
        mask = (seasons == season)
        if not np.any(mask):
            continue

        u_t_sub, v_t_sub = u_true[mask], v_true[mask]
        u_p_sub, v_p_sub = u_pred[mask], v_pred[mask]
        w_t_sub, w_p_sub = w_true[mask], w_pred[mask]

        # 计算出当前季节长达24小时的指标数组
        rmse_h = weighted_rmse(torch.tensor(w_p_sub), torch.tensor(w_t_sub))
        mae_h = weighted_mae(torch.tensor(w_p_sub), torch.tensor(w_t_sub))
        acc_h = weighted_acc(torch.tensor(w_p_sub), torch.tensor(w_t_sub))
        wdfa90, wdfa45, wdfa22 = calculate_wdfa_all_thresholds(u_p_sub, v_p_sub, u_t_sub, v_t_sub)

        # 遍历时间段进行数组切片求平均
        for interval_name, (start, end) in time_intervals.items():
            row = (f"{season:<8} | {interval_name:<8} | {rmse_h[start:end].mean():<8.4f} | {mae_h[start:end].mean():<8.4f} | {acc_h[start:end].mean():<8.4f} | "
                   f"{wdfa90[start:end].mean():<10.2f}% | {wdfa45[start:end].mean():<10.2f}% | {wdfa22[start:end].mean():<10.2f}%")
            print(row)
        print("-" * 115)

    # 打印全年的分段平均值
    rmse_h = weighted_rmse(torch.tensor(w_pred), torch.tensor(w_true))
    mae_h = weighted_mae(torch.tensor(w_pred), torch.tensor(w_true))
    acc_h = weighted_acc(torch.tensor(w_pred), torch.tensor(w_true))
    wdfa90, wdfa45, wdfa22 = calculate_wdfa_all_thresholds(u_pred, v_pred, u_true, v_true)

    for interval_name, (start, end) in time_intervals.items():
        row = (f"{'ANNUAL':<8} | {interval_name:<8} | {rmse_h[start:end].mean():<8.4f} | {mae_h[start:end].mean():<8.4f} | {acc_h[start:end].mean():<8.4f} | "
               f"{wdfa90[start:end].mean():<10.2f}% | {wdfa45[start:end].mean():<10.2f}% | {wdfa22[start:end].mean():<10.2f}%")
        print(row)
    print("=" * 115)


if __name__ == '__main__':
    run_test()