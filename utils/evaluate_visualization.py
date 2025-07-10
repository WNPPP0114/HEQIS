# 文件名: utils/evaluate_visualization.py (修正后)

import torch
import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import pandas as pd


def validate(model, val_x, val_y):
    """验证模型在验证集上的MSE损失。"""
    model.eval()
    with torch.no_grad():
        # val_x 已经是在正确的设备上 (由调用者保证)
        device = val_x.device  # <--- 获取 val_x 所在的设备

        # 将 val_y 移动到与 val_x 相同的设备
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y, dtype=torch.float32, device=device)
        else:
            val_y = val_y.clone().detach().float().to(device)

        predictions, _ = model(val_x)

        # 现在 predictions 和 val_y 都在同一个设备上
        mse_loss = F.mse_loss(predictions.squeeze(), val_y.squeeze())

        return mse_loss


def validate_with_label(model, val_x, val_y, val_labels):
    """验证模型在验证集上的MSE和分类准确率。"""
    model.eval()
    with torch.no_grad():
        # val_x 已经是在正确的设备上 (由调用者保证)
        device = val_x.device  # <--- 获取 val_x 所在的设备

        # --- 核心修正：将 val_y 移动到与 val_x 相同的设备 ---
        if isinstance(val_y, np.ndarray):
            val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)
        else:
            val_y_t = val_y.clone().detach().float().to(device)

        # --- 核心修正：将 val_labels 移动到与 val_x 相同的设备 ---
        if isinstance(val_labels, np.ndarray):
            val_lbl_t = torch.tensor(val_labels, dtype=torch.long, device=device)
        else:
            val_lbl_t = val_labels.clone().detach().long().to(device)

        # 使用模型进行预测，predictions 和 logits 都在 val_x 的设备上
        predictions, logits = model(val_x)

        # --- 核心修正：现在所有张量都在同一个设备上进行计算 ---
        mse_loss = F.mse_loss(predictions.squeeze(), val_y_t.squeeze())

        true_cls = val_lbl_t[:, -1].squeeze()
        pred_cls = logits.argmax(dim=1)
        acc = (pred_cls == true_cls).float().mean()

        return mse_loss, acc


# ... evaluate_best_models, plot_fitting_curve 等其他函数保持不变 ...
# 我将把完整的 evaluate_visualization.py 文件内容附在下面，以防万一

def plot_fitting_curve(true_values, predicted_values, dates, output_dir, model_name):
    """绘制拟合曲线，横坐标为日期。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    try:
        plt.rcParams.update({'font.size': 12, 'font.family': 'SimHei'})
    except:
        print("警告: 无法设置'SimHei'字体，中文可能无法正常显示。")
        plt.rcParams.update({'font.size': 12})

    plt.figure(figsize=(15, 7))
    plt.plot(dates, true_values, label='真实值', linewidth=2, color='royalblue', marker='o', markersize=2,
             linestyle='-')
    plt.plot(dates, predicted_values, label='预测值', linewidth=1.5, color='darkorange', marker='x', markersize=3,
             linestyle='--')
    plt.title(f'{model_name} 拟合曲线', fontsize=18)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('值', fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_fitting_curve.png', dpi=300)
    plt.close()


def compute_metrics(true_values, predicted_values):
    """计算MSE, MAE, RMSE, MAPE等指标。"""
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / (true_values + 1e-8))) * 100
    per_target_mse = np.mean((true_values - predicted_values) ** 2, axis=0)
    return mse, mae, rmse, mape, per_target_mse


def evaluate_best_models(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir,
                         date_series=None):
    """评估最佳模型，并使用日期进行绘图。"""
    N = len(generators)
    for i in range(N):
        if best_model_state[i] is None:
            print(f"警告: G{i + 1} 没有找到最佳模型状态，将跳过评估。")
            continue
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()

    # 这里的 train_y 和 test_y 已经是CPU上的tensor，所以 .cpu() 是安全的
    train_y_inv = y_scaler.inverse_transform(train_y.cpu().numpy().reshape(-1, 1)).flatten()
    test_y_inv = y_scaler.inverse_transform(test_y.cpu().numpy().reshape(-1, 1)).flatten()

    train_dates, test_dates = None, None
    if date_series is not None and isinstance(date_series, pd.Series):
        train_size = len(train_y)
        test_size = len(test_y)
        total_known_size = len(date_series)

        # 假设训练集和测试集来自原始数据（减去头部窗口）的末尾部分
        # train_df 的总长度是 train_size + test_size
        full_data_len = train_size + test_size
        train_start_idx = total_known_size - full_data_len
        test_start_idx = total_known_size - test_size

        if train_start_idx >= 0 and test_start_idx > train_start_idx:
            train_dates = date_series.iloc[train_start_idx:test_start_idx]
            test_dates = date_series.iloc[test_start_idx:]
        else:
            print("警告: 日期序列与数据长度不匹配，无法为绘图分配日期。")

    train_preds_inv, test_preds_inv = [], []
    train_metrics_list, test_metrics_list = [], []

    with torch.no_grad():
        for i in range(N):
            if best_model_state[i] is None: continue
            train_pred, _ = generators[i](train_xes[i])
            train_pred_inv_i = y_scaler.inverse_transform(train_pred.cpu().numpy()).flatten()
            train_preds_inv.append(train_pred_inv_i)
            # train_y_inv 的长度可能比 train_pred_inv_i 长（因为序列创建）
            true_vals_for_metric = train_y_inv[-len(train_pred_inv_i):]
            train_metrics = compute_metrics(true_vals_for_metric, train_pred_inv_i)
            train_metrics_list.append(train_metrics)

            if train_dates is not None:
                offset = len(train_dates) - len(train_pred_inv_i)
                plot_fitting_curve(true_vals_for_metric, train_pred_inv_i, train_dates.iloc[offset:], output_dir,
                                   f'G{i + 1}_Train')
            else:
                plot_fitting_curve(true_vals_for_metric, train_pred_inv_i, range(len(true_vals_for_metric)), output_dir,
                                   f'G{i + 1}_Train')
            logging.info(
                f"Train Metrics for G{i + 1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")

        for i in range(N):
            if best_model_state[i] is None: continue
            test_pred, _ = generators[i](test_xes[i])
            test_pred_inv_i = y_scaler.inverse_transform(test_pred.cpu().numpy()).flatten()
            test_preds_inv.append(test_pred_inv_i)
            true_vals_for_metric = test_y_inv[-len(test_pred_inv_i):]
            test_metrics = compute_metrics(true_vals_for_metric, test_pred_inv_i)
            test_metrics_list.append(test_metrics)

            if test_dates is not None:
                offset = len(test_dates) - len(test_pred_inv_i)
                plot_fitting_curve(true_vals_for_metric, test_pred_inv_i, test_dates.iloc[offset:], output_dir,
                                   f'G{i + 1}_Test')
            else:
                plot_fitting_curve(true_vals_for_metric, test_pred_inv_i, range(len(true_vals_for_metric)), output_dir,
                                   f'G{i + 1}_Test')
            logging.info(
                f"Test Metrics for G{i + 1}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

    while len(train_metrics_list) < N:
        train_metrics_list.append((np.nan, np.nan, np.nan, np.nan, np.nan))
    while len(test_metrics_list) < N:
        test_metrics_list.append((np.nan, np.nan, np.nan, np.nan, np.nan))

    result = {
        "train_mse": [m[0] for m in train_metrics_list], "train_mae": [m[1] for m in train_metrics_list],
        "train_rmse": [m[2] for m in train_metrics_list], "train_mape": [m[3] for m in train_metrics_list],
        "train_mse_per_target": [m[4] for m in train_metrics_list],
        "test_mse": [m[0] for m in test_metrics_list], "test_mae": [m[1] for m in test_metrics_list],
        "test_rmse": [m[2] for m in test_metrics_list], "test_mape": [m[3] for m in test_metrics_list],
        "test_mse_per_target": [m[4] for m in test_metrics_list],
    }
    return result


def plot_generator_losses(data_G, output_dir): pass


def plot_discriminator_losses(data_D, output_dir): pass


def visualize_overall_loss(histG, histD, output_dir): pass


def plot_mse_loss(hist_MSE_G, hist_val_loss, num_epochs, output_dir): pass