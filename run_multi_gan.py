# 文件名: run_multi_gan.py (完整修改版)

import argparse
import pandas as pd
import os
import glob
import time
from utils.logger import setup_experiment_logging
from time_series_maa import MAA_time_series
import torch
import sys
import copy  # <--- 引入 copy 模块，虽然我们最终没用它，但知道这个选项是好的
import numpy as np

def generate_and_save_daily_signals(gca_instance, best_model_state, predict_csv_path, output_dir):
    # ... (此函数无需修改，保持原样) ...
    if not best_model_state or not any(s is not None for s in best_model_state):
        print("没有可用的最佳模型状态，跳过每日信号生成。")
        return
    print("\n--- 开始为所有模型生成每日预测信号 ---")
    try:
        df_predict = pd.read_csv(predict_csv_path)
    except FileNotFoundError:
        print(f"错误：找不到预测数据文件 {predict_csv_path}。")
        return
    if not hasattr(gca_instance, 'x_scalers') or not gca_instance.x_scalers:
        print("错误: gca_instance 中缺少 x_scalers，无法进行信号生成。")
        return
    x_scaler = gca_instance.x_scalers[0]
    device = gca_instance.device
    for i, state in enumerate(best_model_state):
        if state is None:
            continue
        gen_name = gca_instance.generator_names[i]
        window_size = gca_instance.window_sizes[i]
        generator = gca_instance.generators[i]
        generator.load_state_dict(state)
        generator.eval()
        print(f"正在处理模型: G{i + 1} ({gen_name})，窗口大小: {window_size}")
        signals = []
        non_feature_cols = ['date', 'close', 'direction', 'open', 'high', 'low', 'volume']
        feature_columns = [col for col in df_predict.columns if col not in non_feature_cols]
        for j in range(window_size, len(df_predict)):
            df_segment = df_predict.iloc[j - window_size: j]
            sequence_data = df_segment[feature_columns].values
            scaled_sequence = x_scaler.transform(sequence_data)
            input_tensor = torch.from_numpy(np.array([scaled_sequence])).float().to(device)
            with torch.no_grad():
                _, logits = generator(input_tensor)
                prediction = logits.argmax(dim=1).item()
            signal_date = df_predict.iloc[j]['date']
            signals.append({'date': signal_date, 'predicted_action': prediction})
        if signals:
            df_signals = pd.DataFrame(signals)
            signal_filename = f'G{i + 1}_{gen_name}_daily_signals.csv'
            signal_filepath = os.path.join(output_dir, signal_filename)
            df_signals.to_csv(signal_filepath, index=False)
            print(f"已保存每日信号文件: {signal_filepath}")


def find_stock_files(base_dir, stock_name=None, sector=None):
    # ... (函数不变) ...
    search_pattern = os.path.join(base_dir, 'train', '**', '*.csv')
    all_files = glob.glob(search_pattern, recursive=True)
    found_files = []
    if not stock_name and not sector:
        return all_files
    for f in all_files:
        path_parts = f.replace('\\', '/').split('/')
        if len(path_parts) < 3:
            continue
        file_stock_name = path_parts[-2]
        file_sector = path_parts[-3]
        if stock_name and file_stock_name == stock_name:
            found_files.append(f)
        elif sector and file_sector == sector:
            found_files.append(f)
    return found_files


def run_experiment_for_stock(args, stock_csv_path):
    """为单个股票的数据文件运行一次完整的实验（训练或预测）。"""
    path_parts = stock_csv_path.replace('\\', '/').split('/')
    stock_name = path_parts[-2]
    sector_name = path_parts[-3]

    experiment_base_dir = os.path.join(args.output_dir, sector_name, stock_name)
    stock_specific_ckpt_dir = os.path.join(experiment_base_dir, 'ckpt')
    stock_specific_output_dir = experiment_base_dir

    os.makedirs(stock_specific_output_dir, exist_ok=True)
    os.makedirs(stock_specific_ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 20} 开始处理股票: {sector_name} - {stock_name} {'=' * 20}")
    print(f"数据源: {stock_csv_path}")
    print(f"统一实验目录: {stock_specific_output_dir}")
    print(f"模型保存目录: {stock_specific_ckpt_dir}")

    # ==================== MODIFICATION START ====================
    # 创建一个新的 Namespace 对象作为本次循环的参数容器
    # 这样可以避免污染原始的 args 对象，防止路径在循环中累积
    local_args = argparse.Namespace(**vars(args))
    local_args.output_dir = stock_specific_output_dir
    local_args.ckpt_dir = stock_specific_ckpt_dir
    # ===================== MODIFICATION END =====================

    # 在实例化 MAA_time_series 时，使用 local_args
    gca = MAA_time_series(local_args,
                          local_args.N_pairs, local_args.batch_size, local_args.num_epochs,
                          local_args.generators, local_args.discriminators,
                          stock_specific_ckpt_dir, stock_specific_output_dir,
                          local_args.window_sizes,
                          ckpt_path=local_args.ckpt_path,
                          initial_learning_rate=local_args.lr,
                          train_split=local_args.train_split,
                          do_distill_epochs=local_args.distill_epochs,
                          cross_finetune_epochs=local_args.cross_finetune_epochs,
                          device=local_args.device,
                          seed=local_args.random_seed)

    full_df = pd.read_csv(stock_csv_path, usecols=['date'])
    date_series = pd.to_datetime(full_df['date'], format='%Y%m%d')

    predict_csv_path = stock_csv_path.replace(os.path.join('train'), os.path.join('predict'))

    gca.process_data(
        train_csv_path=stock_csv_path,
        predict_csv_path=predict_csv_path,
        target_column='close',
        exclude_columns=['date', 'direction']
    )

    gca.init_dataloader()
    gca.init_model(local_args.num_classes)

    logger = setup_experiment_logging(stock_specific_output_dir, vars(local_args), f"train_{stock_name}")

    results = None
    if local_args.mode == "train":
        results, best_model_state = gca.train(logger, date_series=date_series)

        if best_model_state and any(s is not None for s in best_model_state):
            print("\n--- 训练结束，保存相关产物 ---")
            gca.save_models(best_model_state)
            gca.save_scalers()
            gca.generate_and_save_daily_signals(best_model_state, predict_csv_path)

            print("\n--- 加载最佳模型以生成预测对比CSV ---")
            for i in range(gca.N):
                if best_model_state[i] is not None:
                    gca.generators[i].load_state_dict(best_model_state[i])
            gca.save_predictions_to_csv(date_series=date_series)

    elif local_args.mode == "pred":
        results = gca.pred(date_series=date_series)

    if results:
        # ==================== MODIFICATION: 修正 master_results.csv 的保存路径 ====================
        # master_results.csv 应该保存在最顶层的 output_dir，而不是被污染的路径
        master_results_file = os.path.join(args.output_dir, "master_results.csv")
        # ======================================================================================
        timestamp_for_log = time.strftime("%Y%m%d-%H%M%S")
        result_row = {
            "timestamp": timestamp_for_log,
            "sector": sector_name,
            "stock": stock_name,
            "train_mse": results["train_mse"],
            "train_mae": results["train_mae"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
        }
        df = pd.DataFrame([result_row])
        header = not os.path.exists(master_results_file)
        df.to_csv(master_results_file, mode='a', header=header, index=False)
        print(f"===== 股票 {stock_name} 处理完毕, 结果已记录到 {master_results_file} =====")
    else:
        print(f"===== 股票 {stock_name} 处理完毕, 但没有结果需要记录 =====")

# main 函数应该在 experiment_runner.py 中，这里不再提供它的代码，
# 因为 run_experiment_for_stock 的修改已经解决了核心问题。
# 请确保您的 experiment_runner.py 调用的是这个更新后的 run_experiment_for_stock 函数。