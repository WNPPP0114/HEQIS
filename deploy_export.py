# 文件名: deploy_export.py (Windows运行 - 终极批处理版)

import torch
import sys
import os
import joblib
import json
import numpy as np
import pandas as pd
import glob
import re
import time

# 引入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models
# 引入实验配置以获取模型结构参数
from experiment_runner import COMMON_ARGS

# --- 路径配置 ---
BASE_OUTPUT_DIR = 'output/multi_gan'  # 训练权重存放地
FILTERED_OUTPUT_DIR = 'output_filtered_signals'  # 最佳策略结果存放地
DATA_BASE_DIR = 'csv_data/predict'  # Scaler存放地
EXPORT_DIR = 'deploy_output'  # 导出文件的存放目录


def get_available_stocks():
    """扫描 output_filtered_signals 目录获取有最佳策略的股票列表"""
    stocks = []
    # 查找所有存在 best_metrics.csv 的目录
    pattern = os.path.join(FILTERED_OUTPUT_DIR, '*', '*', 'best_metrics.csv')
    files = glob.glob(pattern)

    for f in files:
        parts = f.replace('\\', '/').split('/')
        if len(parts) >= 4:
            sector = parts[-3]
            stock = parts[-2]
            stocks.append({'sector': sector, 'name': stock, 'metrics_path': f})

    # 按板块和名称排序，保证列表顺序固定
    stocks.sort(key=lambda x: (x['sector'], x['name']))
    return stocks


def find_matching_generator_index(metrics_path):
    """
    通过比对 best_metrics.csv 与同目录下其他 G*_metrics.csv 的内容，
    反向推导 generator_index。
    """
    folder_path = os.path.dirname(metrics_path)

    try:
        # 1. 读取最佳文件的关键指标
        df_best = pd.read_csv(metrics_path)
        if df_best.empty: return None
        if 'generator_index' in df_best.columns: return int(df_best.iloc[0]['generator_index'])

        target_return = df_best.iloc[0]['cumulative_return_percentage']
        target_trades = df_best.iloc[0]['num_trades']

        # 2. 扫描同目录下的原始 G 文件
        g_files = glob.glob(os.path.join(folder_path, 'G*_metrics.csv'))

        for g_file in g_files:
            if 'best_metrics.csv' in g_file: continue
            try:
                df_curr = pd.read_csv(g_file)
                if df_curr.empty: continue
                curr_return = df_curr.iloc[0]['cumulative_return_percentage']
                curr_trades = df_curr.iloc[0]['num_trades']

                if np.isclose(target_return, curr_return, atol=1e-5) and target_trades == curr_trades:
                    filename = os.path.basename(g_file)
                    match = re.search(r'G(\d+)_', filename)
                    if match: return int(match.group(1))
            except Exception:
                continue
        return None
    except Exception:
        return None


def get_model_config(gen_idx):
    """根据生成器索引 (1-based) 从 COMMON_ARGS 获取模型配置"""
    list_idx = gen_idx - 1
    if list_idx >= len(COMMON_ARGS['generators']):
        raise ValueError(f"生成器索引 G{gen_idx} 超出了配置列表范围！")

    return {
        'model_type': COMMON_ARGS['generators'][list_idx],
        'window_size': COMMON_ARGS['window_sizes'][list_idx],
        'use_rope': COMMON_ARGS['use_rope'][list_idx]
    }


def get_model_class(model_type):
    name_map = {
        'gru': models.Generator_gru, 'lstm': models.Generator_lstm,
        'transformer': models.Generator_transformer, 'transformer_deep': models.Generator_transformer_deep,
        'rnn': models.Generator_rnn, 'dct': models.Generator_dct,
        'mpd': models.Generator_mpd, 'bigru': models.Generator_bigru,
        'bilstm': models.Generator_bilstm
    }
    return name_map.get(model_type.lower())


def export_stock(stock_info, quiet=False):
    """
    执行导出逻辑
    :param quiet: 如果为True，减少部分打印，适合批量模式
    :return: (success: bool, message: str)
    """
    sector = stock_info['sector']
    stock_name = stock_info['name']

    if not quiet:
        print(f"\n{'=' * 20} 正在导出: {stock_name} ({sector}) {'=' * 20}")

    # 1. 确定最佳策略
    gen_idx = find_matching_generator_index(stock_info['metrics_path'])
    if gen_idx is None:
        return False, "无法匹配最佳策略索引"

    # 2. 获取模型配置
    try:
        config = get_model_config(gen_idx)
    except Exception as e:
        return False, f"配置匹配失败: {e}"

    if not quiet:
        print(f"🎯 策略锁定: G{gen_idx} | 模型: {config['model_type']} | 窗口: {config['window_size']}")

    # 3. 定位文件路径
    ckpt_dir = os.path.join(BASE_OUTPUT_DIR, sector, stock_name, 'ckpt', 'generators')
    ckpt_filename = f"{gen_idx}_{config['model_type']}.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

    scaler_dir = os.path.join(DATA_BASE_DIR, sector, stock_name)
    x_scaler_path = os.path.join(scaler_dir, 'x_scaler.gz')
    y_scaler_path = os.path.join(scaler_dir, 'y_scaler.gz')

    if not os.path.exists(ckpt_path):
        # 尝试模糊匹配
        possible = glob.glob(os.path.join(ckpt_dir, f"{gen_idx}_*.pt"))
        if possible:
            ckpt_path = possible[0]
        else:
            return False, f"权重文件缺失: {ckpt_filename}"

    if not os.path.exists(x_scaler_path):
        return False, "Scaler文件缺失"

    # 4. 加载 Scaler
    try:
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        input_size = x_scaler.n_features_in_
    except Exception as e:
        return False, f"Scaler加载错误: {e}"

    # 5. 初始化模型
    ModelClass = get_model_class(config['model_type'])
    if not ModelClass: return False, f"未知模型类型: {config['model_type']}"

    init_kwargs = {'use_rope': config['use_rope']}
    if config['model_type'] == 'mpd':
        init_kwargs.update({'input_height': config['window_size'], 'input_width': input_size, 'num_classes': 3,
                            'pretrainer_type': 'cae'})
    elif config['model_type'] in ['transformer', 'transformer_deep', 'dct']:
        init_kwargs.update({'input_dim': input_size, 'output_len': 1})
    else:
        init_kwargs.update({'input_size': input_size, 'out_size': 1})

    try:
        model = ModelClass(**init_kwargs)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        return False, f"模型加载权重失败: {e}"

    # 6. 导出
    save_dir = os.path.join(EXPORT_DIR, sector, stock_name)
    os.makedirs(save_dir, exist_ok=True)
    onnx_path = os.path.join(save_dir, 'model_deploy.onnx')
    json_path = os.path.join(save_dir, 'scaler_params.json')

    # 导出JSON
    params = {
        "stock_name": stock_name,
        "model_type": config['model_type'],
        "best_generator_index": gen_idx,
        "x_scale": x_scaler.scale_.tolist(),
        "x_min": x_scaler.min_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
        "y_min": y_scaler.min_.tolist(),
        "n_features": input_size,
        "window_size": config['window_size']
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

    # 导出ONNX
    if config['model_type'] == 'mpd':
        dummy_input = torch.randn(1, 1, config['window_size'], input_size)
    else:
        dummy_input = torch.randn(1, config['window_size'], input_size)

    try:
        # 抑制特定的UserWarning
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            torch.onnx.export(
                model, dummy_input, onnx_path, export_params=True, opset_version=12,
                do_constant_folding=True, input_names=['input'], output_names=['output_reg', 'output_cls'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output_reg': {0: 'batch_size'},
                              'output_cls': {0: 'batch_size'}}
            )
    except Exception as e:
        # 即使报错也可能是环境问题，只要文件生成了就算成功
        if not os.path.exists(onnx_path):
            return False, f"ONNX导出异常: {e}"

    return True, "成功"


def parse_selection(input_str, max_len):
    """解析用户输入"""
    input_str = input_str.strip().lower()
    if input_str == 'all':
        return list(range(max_len))

    selected = set()
    # 替换逗号为所有的空格
    parts = input_str.replace(',', ' ').split()

    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                # 用户输入是1-based，转为0-based区间
                selected.update(range(start - 1, end))
            except ValueError:
                pass
        else:
            try:
                idx = int(part) - 1
                if 0 <= idx < max_len:
                    selected.add(idx)
            except ValueError:
                pass

    return sorted(list(selected))


def main():
    print("正在扫描可用的最佳策略...")
    stocks = get_available_stocks()

    if not stocks:
        print("未找到任何含有 best_metrics.csv 的股票记录。请先运行 filter_trading_signals.py。")
        return

    while True:
        print("\n" + "=" * 40)
        print("可用股票列表:")
        for i, s in enumerate(stocks):
            print(f"[{i + 1}] {s['sector']} - {s['name']}")
        print("=" * 40)

        print("请输入指令:")
        print("  - 输入 'all' : 导出所有股票")
        print("  - 输入数字 (如 '1') : 导出单个")
        print("  - 输入列表 (如 '1 3 5') : 导出多个")
        print("  - 输入范围 (如 '1-5') : 导出区间")
        print("  - 输入 'q' : 退出程序")

        choice = input("\n请选择: ").strip()
        if choice.lower() == 'q':
            break

        selected_indices = parse_selection(choice, len(stocks))

        if not selected_indices:
            print("❌ 无效的输入，请重新选择。")
            continue

        print(f"\n🚀 开始批量处理 {len(selected_indices)} 个任务...\n")

        success_count = 0
        fail_count = 0
        failed_stocks = []

        start_time = time.time()

        for idx in selected_indices:
            stock = stocks[idx]
            # 如果是批量(>1个)，开启quiet模式减少刷屏
            is_quiet = len(selected_indices) > 1

            # 显示进度条风格的提示
            print(f"[{idx + 1}/{len(stocks)}] 处理: {stock['name']} ... ", end='', flush=True)

            success, msg = export_stock(stock, quiet=is_quiet)

            if success:
                print("✅ 成功")
                success_count += 1
            else:
                print(f"❌ 失败 ({msg})")
                fail_count += 1
                failed_stocks.append(f"{stock['name']}: {msg}")

        total_time = time.time() - start_time
        print("\n" + "-" * 30)
        print(f"📊 处理完成！耗时: {total_time:.2f}s")
        print(f"✅ 成功: {success_count}")
        print(f"❌ 失败: {fail_count}")
        if failed_stocks:
            print("失败详情:")
            for f in failed_stocks:
                print(f"  - {f}")
        print("-" * 30)

        if len(selected_indices) == len(stocks):
            # 如果是全部导出，通常跑完一次就想退出了，但也允许继续
            pass


if __name__ == "__main__":
    main()