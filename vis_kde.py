# 文件名: vis_kde.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import glob


def plot_kde_for_experiment(csv_files: list, output_path: str, stock_name: str, sector_name: str, alpha: float,
                            no_grid: bool):
    """
    为单个实验（即一只股票）的所有生成器的结果绘制一张汇总的KDE图。

    参数:
        csv_files (list): 包含该实验所有生成器预测结果的CSV文件路径列表。
        output_path (str): 生成的图片文件的保存路径。
        stock_name (str): 当前处理的股票名称。
        sector_name (str): 当前处理的股票所属板块。
        alpha (float): 填充区域的透明度。
        no_grid (bool): 是否移除网格线。
    """
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用一个漂亮的主题
    plt.rcParams.update({'font.size': 14, 'font.family': 'SimHei'})  # 设置字体以支持中文
    plt.figure(figsize=(12, 7))

    all_true_series = []

    # 第一次循环：收集所有生成器的真实值和预测值数据
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if 'true' not in df.columns or 'pred' not in df.columns:
                print(f"警告: 文件 {csv_path} 缺少 'true' 或 'pred' 列，已跳过。")
                continue

            # 从文件名中提取生成器名称，例如 'predictions_gen_1_gru.csv' -> 'gru'
            filename = os.path.basename(csv_path)
            parts = filename.replace('.csv', '').split('_')
            gen_name = parts[-1] if len(parts) > 1 else f"Gen-{parts[1]}"

            # 绘制该生成器的预测值分布
            sns.kdeplot(df['pred'].dropna(), label=f'预测值 ({gen_name.upper()})',
                        linewidth=1.5, alpha=alpha, fill=True)

            # 只需从第一个文件中收集真实值数据，因为它们都一样
            if not all_true_series:
                all_true_series.append(df['true'].dropna())

        except Exception as e:
            print(f"处理文件 {csv_path} 时出错: {e}")
            continue

    # 如果收集到了真实值数据，则绘制其统一的分布图
    if all_true_series:
        combined_true = pd.concat(all_true_series).dropna()
        if not combined_true.empty:
            sns.kdeplot(combined_true, label='真实值 (统一)', color='orangered',
                        linewidth=2.5, fill=True, alpha=alpha - 0.1, zorder=0)  # 让真实值在最底层

    # 设置图表标题和标签
    plt.title(f'真实值 vs. 预测值 KDE 分布\n({sector_name} - {stock_name})', fontsize=20, pad=20)
    plt.xlabel('收盘价', fontsize=16)
    plt.ylabel('密度', fontsize=16)

    # 添加图例
    plt.legend(title='数据来源', fontsize=12)

    # 根据参数决定是否显示网格
    if no_grid:
        plt.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已成功保存图表: {output_path}")


def find_and_process_experiments(root_dir: str, output_dir: str, alpha: float, no_grid: bool):
    """
    递归扫描根目录，为每个实验（每只股票）生成一张KDE图。
    """
    # 查找所有名为 'true2pred_csv' 的子目录
    search_pattern = os.path.join(root_dir, '**', 'true2pred_csv')
    experiment_dirs = glob.glob(search_pattern, recursive=True)

    if not experiment_dirs:
        print(f"错误: 在根目录 '{root_dir}' 下未找到任何 'true2pred_csv' 文件夹。")
        print("请确认训练/预测流程已成功运行，并生成了对应的CSV文件。")
        return

    print(f"找到 {len(experiment_dirs)} 个实验结果目录，开始生成KDE图...")

    for exp_dir in experiment_dirs:
        csv_files = glob.glob(os.path.join(exp_dir, '*.csv'))
        if not csv_files:
            continue

        # 从路径中解析出股票和板块信息
        path_parts = exp_dir.replace('\\', '/').split('/')
        # 路径结构: .../{root_dir}/{sector}/{stock_name}/true2pred_csv
        stock_name = path_parts[-2]
        sector_name = path_parts[-3]

        # 定义输出图片的路径和文件名
        output_filename = f"KDE_{sector_name}_{stock_name}.png"
        output_path = os.path.join(output_dir, output_filename)

        plot_kde_for_experiment(csv_files, output_path, stock_name, sector_name, alpha, no_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为每个实验（股票）的真实值和预测值绘制汇总的核密度估计图。')
    parser.add_argument('--input_dir', type=str, default='output/multi_gan',
                        help='包含所有实验结果的根目录 (例如 "output/multi_gan")。')
    parser.add_argument('--output_dir', type=str, default='output_vis/kde_plots',
                        help='保存生成的所有KDE图表的目录。')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='KDE填充区域的透明度。')
    parser.add_argument('--no_grid', action='store_true',
                        help='添加此参数以移除图表中的网格线。')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查是否安装了中文字体，如果没有，则进行提示
    try:
        import matplotlib.font_manager

        if not any(['SimHei' in font.name for font in matplotlib.font_manager.fontManager.ttflist]):
            print("\n警告: 未找到 'SimHei' 字体。图表中的中文可能无法正常显示。")
            print(
                "请安装 'SimHei' 字体 (或者在脚本中修改为其他已安装的中文字体，如 'Microsoft YaHei') 以获得最佳显示效果。\n")
    except ImportError:
        pass

    find_and_process_experiments(args.input_dir, args.output_dir, args.alpha, args.no_grid)

    print("\n所有KDE图表已生成完毕。")