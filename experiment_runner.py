# 文件名: experiment_runner.py (已修改为更灵活的配置方式)

import argparse
import torch
import os
import sys
import time
import pandas as pd
import logging
import run_multi_gan
import random
import numpy as np


# ==============================================================================
# 独立的、可重用的设置随机种子的函数
# ==============================================================================
def set_seed(seed):
    """
    为所有相关的随机数生成器设置种子，以确保实验的可复现性。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"--- 随机种子已设置为: {seed} ---")


# ==============================================================================
# --- 实验配置中心 (COMMON_ARGS) ---
# 这是您进行实验时唯一需要修改的地方。
# ==============================================================================
COMMON_ARGS = {
    # --- 基本路径配置 ---
    "stock_name": None,  # 指定单个股票名称进行训练，如 "黄河旋风"
    "sector": None,  # 指定整个板块进行训练，如 "培育钻石"
    "data_base_dir": "csv_data",
    "output_dir": "output/multi_gan",
    "ckpt_dir": "ckpt/multi_gan",
    "notes": "Default run from COMMON_ARGS",

    # ==========================================================================
    # --- 核心模型数量与类型配置 (★★★ 增删模型在此处修改 ★★★) ---
    #
    # 规则:
    # 1. "N_pairs" 的值必须等于下面三个列表的长度。
    # 2. "generators", "window_sizes", "use_rope" 三个列表的长度必须严格相等。
    #
    # 示例：当前配置为 4 个模型
    # ==========================================================================
    "N_pairs": 3,  # <-- 模型对的数量

    "generators": ["gru", "bilstm", "transformer"],  # <-- 模型名称列表 (长度必须为 N_pairs)
    "window_sizes": [60, 60, 60],  # <-- 每个模型对应的窗口大小 (长度必须为 N_pairs)
    "use_rope": [True, True, True],  # <-- 每个模型是否使用旋转位置编码 (长度必须为 N_pairs)

    "discriminators": None,  # 保持为 None 即可，代码会自动生成 N_pairs 个 'default' 判别器

    # --- 训练超参数 ---
    "num_classes": 3,
    "distill_epochs": 1,
    "cross_finetune_epochs": 5,
    "num_epochs": 100,
    "lr": 2e-5,
    "batch_size": 64,
    "train_val_test_split": [0.7, 0.1, 0.2],
    "random_seed": 3407,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp_dtype": "none",
    "mode": "train",  # 'train' 或 'pred'
    "ckpt_path": "auto",

    # --- 损失函数与监控指标配置 ---
    "adversarial_loss_mode": "bce",  # 可选 'bce', 'mse'
    "regression_loss_mode": "mse",  # 可选 'mse', 'mae'
    "monitor_metric": "val_mse",  # 可选 'val_mse', 'val_acc', 'val_bce', 'val_cls_loss'

    # --- 预训练相关参数 ---
    "pretrainer_type": "t3vae",  # 可选: 'cae', 't3vae'
    "pretrain": True,
    "pretrain_epochs": 50,
    "lr_cae_finetune_multiplier": 100,
}


# ==============================================================================
# setup_arg_parser 和 validate_args 定义
# (这些函数已经很灵活，无需修改)
# ==============================================================================
def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="为预处理好的股票数据运行Multi-GAN模型。这是实验的主控制脚本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for key, value in COMMON_ARGS.items():
        if isinstance(value, list):
            if value and isinstance(value[0], bool):
                parser.add_argument(f'--{key}', type=lambda x: (str(x).lower() == 'true'), nargs='+', default=value,
                                    help=f"布尔值列表 (e.g., True False True)。默认为: {value}")
            else:
                parser.add_argument(f'--{key}', type=type(value[0]) if value else str, nargs='+', default=value,
                                    help=f"默认为: {value}")
        elif isinstance(value, bool):
            group = parser.add_mutually_exclusive_group(required=False)
            group.add_argument(f'--{key}', dest=key, action='store_true', help=f"激活 {key} (默认为: {value})")
            group.add_argument(f'--no-{key}', dest=key, action='store_false', help=f"禁用 {key}")
            parser.set_defaults(**{key: value})
        else:
            arg_type = type(value) if value is not None else str
            parser.add_argument(f'--{key}', type=arg_type, default=value,
                                help=f"默认为: {value}")
    return parser


def validate_args(args):
    if len(args.generators) != args.N_pairs or len(args.window_sizes) != args.N_pairs:
        print("错误: --generators, --window_sizes 的数量必须与 --N_pairs 一致!")
        sys.exit(1)
    if len(args.use_rope) != args.N_pairs:
        print(f"错误: --use_rope 列表的长度 ({len(args.use_rope)}) 必须与 --N_pairs ({args.N_pairs}) 一致!")
        sys.exit(1)
    if len(args.train_val_test_split) != 3 or not abs(sum(args.train_val_test_split) - 1.0) < 1e-6:
        print(f"错误: --train_val_test_split 必须包含3个数字且总和为1，当前为: {args.train_val_test_split}")
        sys.exit(1)
    if args.discriminators is None:
        args.discriminators = ["default"] * args.N_pairs
    elif len(args.discriminators) != args.N_pairs:
        print("错误: --discriminators 的数量必须与 --N_pairs 一致，或者为 None!")
        sys.exit(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"警告: 指定了设备 {args.device} 但CUDA不可用，将使用CPU。");
        args.device = "cpu"
    valid_amp_dtypes = ['none', 'float16', 'bfloat16', 'mixed']
    if args.amp_dtype.lower() not in valid_amp_dtypes:
        print(f"警告: 无效的 --amp_dtype 值 '{args.amp_dtype}'。将使用 'none'。");
        args.amp_dtype = 'none'
    args.amp_dtype = args.amp_dtype.lower()
    valid_loss_modes = ['bce', 'mse', 'mae']
    if args.adversarial_loss_mode not in valid_loss_modes or args.regression_loss_mode not in valid_loss_modes:
        print(f"警告: 无效的损失函数模式。可选值: {valid_loss_modes}")
    if args.monitor_metric not in ['val_mse', 'val_acc', 'val_bce', 'val_cls_loss']:
        print(f"警告: 无效的监控指标。可选值: 'val_mse', 'val_acc', 'val_bce', 'val_cls_loss'")
    if args.pretrainer_type not in ['cae', 't3vae']:
        print(f"警告: 无效的预训练器类型 '{args.pretrainer_type}'。将使用默认的 'cae'。");
        args.pretrainer_type = 'cae'
    return args


# ==============================================================================
# main 函数 (无需修改)
# ==============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.StreamHandler()])
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    parser = setup_arg_parser()
    args = parser.parse_args()
    args = validate_args(args)

    set_seed(args.random_seed)

    print("===== 当前最终运行参数 (已合并默认值和命令行参数) =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================================")

    stock_files_to_process = run_multi_gan.find_stock_files(args.data_base_dir, args.stock_name, args.sector)
    if not stock_files_to_process:
        print(f"错误: 在 '{args.data_base_dir}' 目录中找不到任何匹配的股票数据。")
        sys.exit(1)

    print(f"\n成功找到 {len(stock_files_to_process)} 个股票数据文件待处理。")

    for stock_file in stock_files_to_process:
        set_seed(args.random_seed)

        path_parts = stock_file.replace('\\', '/').split('/')
        stock_name_from_path = path_parts[-2]
        sector_name_from_path = path_parts[-3]
        expected_output_dir = os.path.join(args.output_dir, sector_name_from_path, stock_name_from_path)

        if args.mode == "train" and os.path.isdir(expected_output_dir):
            ckpt_dir_check = os.path.join(expected_output_dir, 'ckpt', 'generators')
            if os.path.isdir(ckpt_dir_check) and any(f.endswith('.pt') for f in os.listdir(ckpt_dir_check)):
                print(f"\n--- 已找到股票 {stock_name_from_path} 的训练结果在 {expected_output_dir}，跳过训练。 ---")
                continue
            else:
                print(
                    f"\n--- 找到股票 {stock_name_from_path} 的目录 {expected_output_dir}，但未找到检查点，将尝试重新运行。 ---")
        try:
            run_multi_gan.run_experiment_for_stock(args, stock_file)
        except Exception as e:
            stock_name_from_path_in_err = stock_file.replace('\\', '/').split('/')[-2]
            print(f"\n!!!!!! 在处理股票 {stock_name_from_path_in_err} 时发生严重错误: {e} !!!!!!")
            import traceback
            traceback.print_exc()
            print(f"!!!!!! 跳过股票 {stock_name_from_path_in_err}，继续处理下一个... !!!!!!")
            continue
    print("\n所有指定的实验已全部完成。")


if __name__ == "__main__":
    main()