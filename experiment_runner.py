# 文件名: experiment_runner.py

import argparse
import torch
import os
import sys
import time
import pandas as pd
import logging # <--- 导入 logging 模块
# 导入功能模块
import run_multi_gan # <--- 确保导入 run_multi_gan 模块
# 假设 setup_arg_parser 和 validate_args 在这个文件的其他地方（比如顶部）
# from .run_multi_gan import find_stock_files, run_experiment_for_stock # 或者这样导入
# from experiment_runner import setup_arg_parser, validate_args # 假设 setup_arg_parser 和 validate_args 是全局函数或在这个文件顶部

# ==============================================================================
# 默认参数配置中心 (COMMON_ARGS)
# ==============================================================================
COMMON_ARGS = {
    "stock_name": None,
    "sector": None,
    "data_base_dir": "csv_data",
    "output_dir": "output/multi_gan",
    "ckpt_dir": "ckpt/multi_gan", # 这个参数在 run_multi_gan 中会被覆盖，但保留以便 argparse 使用
    "notes": "Default run from COMMON_ARGS",
    "window_sizes": [5, 10, 15],
    "N_pairs": 3,
    "num_classes": 3,
    "generators": ["gru", "lstm", "transformer"],
    "discriminators": None,
    "distill_epochs": 1,
    "cross_finetune_epochs": 5,
    "num_epochs": 200,
    "lr": 2e-5,
    "batch_size": 64,
    "train_split": 0.8,
    "random_seed": 3407,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp_dtype": "none",
    "mode": "train",
    "ckpt_path": "auto", # 这个参数在 run_multi_gan 中会被覆盖
}


# ==============================================================================
# 假设 setup_arg_parser 和 validate_args 定义在此处
# ==============================================================================

def setup_arg_parser():
    """设置命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="为预处理好的股票数据运行Multi-GAN模型。这是实验的主控制脚本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for key, value in COMMON_ARGS.items():
        if isinstance(value, list):
            parser.add_argument(f'--{key}', type=type(value[0]) if value else str, nargs='+', default=value,
                                help=f"默认为: {value}")
        elif isinstance(value, bool):
            # action='store_true' 表示如果命令行中出现了 --key，则 args.key 为 True，否则为默认值（这里默认值是 False）
            # 为了支持默认值为 True 的情况，需要根据默认值调整 action
            if value is True:
                 parser.add_argument(f'--{key}', action='store_false', default=value, help=f"设置此项以禁用 {key}")
            else:
                 parser.add_argument(f'--{key}', action='store_true', default=value, help=f"设置此项以激活 {key}")
        else:
            # 尝试根据默认值类型判断，如果默认值是 None，则使用 str 类型
            arg_type = type(value) if value is not None else str
            parser.add_argument(f'--{key}', type=arg_type, default=value,
                                help=f"默认为: {value}")
    return parser


def validate_args(args):
    """验证和修正参数的逻辑。"""
    # 确保 N_pairs 与 generators 和 window_sizes 长度一致
    if len(args.generators) != args.N_pairs or len(args.window_sizes) != args.N_pairs:
        print("错误: --generators, --window_sizes, 和 --N_pairs 的数量必须一致!")
        sys.exit(1)

    # 如果 discriminators 为 None，根据 N_pairs 设置一个默认列表
    if args.discriminators is None:
        args.discriminators = ["default"] * args.N_pairs
    elif len(args.discriminators) != args.N_pairs:
         print("错误: --discriminators 的数量必须与 --N_pairs 一致，或者为 None!")
         sys.exit(1)


    # 检查 CUDA 设备可用性
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"警告: 指定了设备 {args.device} 但CUDA不可用，将使用CPU。")
        args.device = "cpu"

    # 检查 amp_dtype 是否有效
    valid_amp_dtypes = ['none', 'float16', 'bfloat16', 'mixed']
    if args.amp_dtype.lower() not in valid_amp_dtypes:
        print(f"警告: 无效的 --amp_dtype 值 '{args.amp_dtype}'。可选值: {valid_amp_dtypes}。将使用 'none'。")
        args.amp_dtype = 'none'
    # 强制 amp_dtype 小写
    args.amp_dtype = args.amp_dtype.lower()

    return args


# ==============================================================================


def main():
    """实验的主函数，负责解析参数和调度实验。"""
    # ==================== MODIFICATION START ====================
    # Configure basic console logging ONCE at the beginning
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler() # Add stream handler for console output
        ]
    )
    # Get the root logger to ensure its level is set if basicConfig didn't do it (it should)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Ensure root logger level is appropriate
    # ===================== MODIFICATION END =====================


    parser = setup_arg_parser()
    args = parser.parse_args()
    args = validate_args(args)

    print("===== 当前最终运行参数 (已合并默认值和命令行参数) =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================================")

    stock_files_to_process = run_multi_gan.find_stock_files(args.data_base_dir, args.stock_name, args.sector)
    if not stock_files_to_process:
        print(f"错误: 在 '{args.data_base_dir}' 目录中找不到任何匹配的股票数据。请确认 get_stock_data.py 已成功运行。")
        sys.exit(1)

    print(f"\n成功找到 {len(stock_files_to_process)} 个股票数据文件待处理。")

    for stock_file in stock_files_to_process:
        # ==================== MODIFICATION START ====================
        # Check if experiment output directory already exists for this stock
        path_parts = stock_file.replace('\\', '/').split('/')
        stock_name_from_path = path_parts[-2]
        sector_name_from_path = path_parts[-3]

        # Construct the expected output directory path WITHOUT timestamp
        expected_output_dir = os.path.join(args.output_dir, sector_name_from_path, stock_name_from_path)

        # Check if the directory exists and contains model checkpoints
        # This is the logic to skip training if results likely exist
        if args.mode == "train" and os.path.isdir(expected_output_dir):
             # Check for existence of checkpoint files as a proxy for successful training
             ckpt_dir_check = os.path.join(expected_output_dir, 'ckpt', 'generators')
             if os.path.isdir(ckpt_dir_check) and any(f.endswith('.pt') for f in os.listdir(ckpt_dir_check)):
                 print(f"\n--- 已找到股票 {stock_name_from_path} 的训练结果在 {expected_output_dir}，跳过训练。 ---")
                 continue # Skip to the next stock_file in the loop
             else:
                 # Directory exists but no checkpoints, maybe a failed previous run?
                 # Decide whether to re-run or skip. Current logic re-runs.
                 print(f"\n--- 找到股票 {stock_name_from_path} 的目录 {expected_output_dir}，但未找到检查点，将尝试重新运行。 ---")


        # ===================== MODIFICATION END =====================

        try:
            # Pass the original args object to run_experiment_for_stock.
            # run_experiment_for_stock will create a local copy (local_args) inside
            run_multi_gan.run_experiment_for_stock(args, stock_file)

        except Exception as e:
            # Get stock name again in case of error
            stock_name_from_path_in_err = stock_file.replace('\\', '/').split('/')[-2]
            print(f"\n!!!!!! 在处理股票 {stock_name_from_path_in_err} 时发生严重错误: {e} !!!!!!")
            import traceback # ensure traceback is imported
            traceback.print_exc()
            print(f"!!!!!! 跳过股票 {stock_name_from_path_in_err}，继续处理下一个... !!!!!!")
            continue

    print("\n所有指定的实验已全部完成。")


if __name__ == "__main__":
    main()