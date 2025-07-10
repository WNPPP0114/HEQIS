# 文件名: run_multi_gan.py

import argparse
import pandas as pd
import os
import glob
import time
from utils.logger import setup_experiment_logging
from time_series_maa import MAA_time_series
import torch


def find_stock_files(base_dir, stock_name=None, sector=None):
    """根据股票名称或板块名称在指定的基础目录中查找预处理好的CSV文件。"""
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path_parts = stock_csv_path.replace('\\', '/').split('/')
    stock_name = path_parts[-2]
    sector_name = path_parts[-3]

    stock_specific_output_dir = os.path.join(args.output_dir, sector_name, stock_name, timestamp)
    stock_specific_ckpt_dir = os.path.join(args.ckpt_dir, sector_name, stock_name, timestamp)
    os.makedirs(stock_specific_output_dir, exist_ok=True)
    os.makedirs(stock_specific_ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 20} 开始处理股票: {sector_name} - {stock_name} (时间戳: {timestamp}) {'=' * 20}")
    print(f"数据源: {stock_csv_path}")
    print(f"输出目录: {stock_specific_output_dir}")
    print(f"模型保存目录: {stock_specific_ckpt_dir}")

    framework_args = args
    if not isinstance(framework_args, argparse.Namespace):
        framework_args = argparse.Namespace(**vars(args))

    framework_args.output_dir = stock_specific_output_dir
    framework_args.ckpt_dir = stock_specific_ckpt_dir

    gca = MAA_time_series(framework_args,
                          framework_args.N_pairs, framework_args.batch_size, framework_args.num_epochs,
                          framework_args.generators, framework_args.discriminators,
                          stock_specific_ckpt_dir, stock_specific_output_dir,
                          framework_args.window_sizes,
                          ckpt_path=framework_args.ckpt_path,
                          initial_learning_rate=framework_args.lr,
                          train_split=framework_args.train_split,
                          do_distill_epochs=framework_args.distill_epochs,
                          cross_finetune_epochs=framework_args.cross_finetune_epochs,
                          device=framework_args.device,
                          seed=framework_args.random_seed)

    full_df = pd.read_csv(stock_csv_path, usecols=['date'])
    date_series = pd.to_datetime(full_df['date'], format='%Y%m%d')

    target_column = 'close'
    exclude_columns = ['date', 'direction']
    predict_csv_path = stock_csv_path.replace(os.path.join('train'), os.path.join('predict'))

    gca.process_data(
        train_csv_path=stock_csv_path,
        predict_csv_path=predict_csv_path,
        target_column=target_column,
        exclude_columns=exclude_columns
    )

    gca.init_dataloader()
    gca.init_model(framework_args.num_classes)

    logger = setup_experiment_logging(stock_specific_output_dir, vars(framework_args), f"train_{stock_name}")

    results = None
    if framework_args.mode == "train":
        results, best_model_state = gca.train(logger)
        if best_model_state and any(s is not None for s in best_model_state):
            print("\n--- 训练结束，保存最佳模型状态 ---")
            gca.save_models(best_model_state)

            print("\n--- 加载最佳模型以生成预测对比CSV ---")
            for i in range(gca.N):
                if best_model_state[i] is not None:
                    gca.generators[i].load_state_dict(best_model_state[i])
            gca.save_predictions_to_csv(date_series=date_series)

    elif framework_args.mode == "pred":
        results = gca.pred(date_series=date_series)

    if results:
        master_results_file = os.path.join(args.output_dir, "master_results.csv")
        result_row = {
            "timestamp": timestamp,
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