# 文件名: models/pretrainer.py (已修改为在特征相关性热力图中包含'close'列)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.special import loggamma


# --- 预训练器模型定义 ---

class CAE_Encoder(nn.Module):
    """CAE的编码器部分"""

    def __init__(self, input_height, input_width, cae_out_channels=[16, 32, 64], cae_kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential()
        in_channels = 1
        for i, out_ch in enumerate(cae_out_channels):
            self.encoder.add_module(f"conv_{i + 1}",
                                    nn.Conv2d(in_channels, out_ch, kernel_size=cae_kernel_size, padding='same'))
            self.encoder.add_module(f"relu_{i + 1}", nn.ReLU(True))
            target_pool_size = (max(1, input_height // (2 ** (i + 1))), max(1, input_width // (2 ** (i + 1))))
            self.encoder.add_module(f"pool_{i + 1}", nn.AdaptiveMaxPool2d(target_pool_size))
            in_channels = out_ch

    def forward(self, x):
        return self.encoder(x)


class CAE_for_pretrain(nn.Module):
    """一个用于预训练的、完整的卷积自编码器（CAE）。"""

    def __init__(self, input_height, input_width):
        super().__init__()
        self.encoder = CAE_Encoder(input_height, input_width)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.AdaptiveAvgPool2d((input_height, input_width)), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, None, None  # 返回None以匹配t3vae的输出元组格式


class t3VAE_Encoder(nn.Module):
    """t3VAE的编码器部分，输出均值和对数方差"""

    def __init__(self, input_height, input_width, latent_dim=64):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        # 动态计算卷积输出后的维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_height, input_width)
            conv_output_shape = self.encoder_conv(dummy_input).shape
            self.conv_output_dim = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.conv_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_dim, latent_dim)

    def forward(self, x):
        h = self.encoder_conv(x)
        h_flat = self.flatten(h)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar


class t3VAE_for_pretrain(nn.Module):
    """一个用于预训练的、严格遵循t-分布的变分自编码器。"""

    def __init__(self, input_height, input_width, latent_dim=64, degrees_of_freedom=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.v = degrees_of_freedom  # 自由度 ν
        self.encoder = t3VAE_Encoder(input_height, input_width, latent_dim)

        # 动态且稳健地计算解码器维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_height, input_width)
            # 获取编码器卷积部分的输出形状，这将是解码器反卷积部分的输入形状
            self.decoder_unflatten_shape = self.encoder.encoder_conv(dummy_input).shape[1:]  # (C, H, W)
            # 解码器FC层的输出维度必须等于反卷积层输入的总元素数量
            self.decoder_fc_dim = np.prod(self.decoder_unflatten_shape)

        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, self.decoder_fc_dim), nn.ReLU())
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.AdaptiveAvgPool2d((input_height, input_width)),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """严格的t分布重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        device = mu.device
        v_tensor = torch.tensor(self.v, device=device)

        chi2_dist = torch.distributions.Chi2(v_tensor)
        chi2_sample = chi2_dist.rsample(sample_shape=std.shape)

        chi2_sample = torch.clamp(chi2_sample, min=1e-8)

        t_sample = eps * torch.sqrt(v_tensor / chi2_sample)

        return mu + std * t_sample

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon_flat = self.decoder_fc(z)
        x_recon = self.decoder_conv(x_recon_flat.view(x_recon_flat.size(0), *self.decoder_unflatten_shape))
        return x_recon, mu, logvar


def t3vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    修改后的t3VAE损失函数，结合了重构损失和KL散度，更像传统的beta-VAE。
    这牺牲了对gamma-loss的严格遵循，但换来了更好的训练稳定性和可解释性。
    beta参数用于权衡重构和正则化。
    """
    # 1. 重构损失 (Reconstruction Loss)
    recon_loss = F.binary_cross_entropy(recon_x.view(-1), x.view(-1), reduction='sum')

    # 2. 正则化项 (KL Divergence)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))

    # 3. 总损失
    total_loss = recon_loss + beta * kld_loss

    return total_loss


# --- 可视化函数 ---

def plot_pretrain_loss(loss_history, output_dir, pretrainer_type):
    output_path = os.path.join(output_dir, "pretrain_vis", f"{pretrainer_type}_pretrain_loss_curve.png")
    plt.style.use('seaborn-v0_8-whitegrid')
    try:
        plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif'];
        plt.rcParams[
            'axes.unicode_minus'] = False
    except:
        print("警告: 未找到中文字体。")
    plt.figure(figsize=(10, 6));
    plt.plot(loss_history, label=f'{pretrainer_type.upper()} Pre-training Loss', color='deepskyblue')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.title(f'{pretrainer_type.upper()} 预训练损失曲线');
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5);
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150);
    plt.close();
    print(f"已保存{pretrainer_type.upper()}预训练Loss曲线图: {output_path}")


def visualize_reconstruction(ts_maa_instance, pretrainer_ckpt_path, pretrainer_type, num_samples=3,
                             target_window_size=None):
    if target_window_size is None:
        print("错误: visualize_reconstruction 需要 target_window_size 参数。")
        return

    print(f"\n--- 开始为 ws={target_window_size} 生成{pretrainer_type.upper()}的重构可视化图像 ---")

    try:
        mpd_idx = ts_maa_instance.window_sizes.index(target_window_size)
        if ts_maa_instance.generator_names[mpd_idx] != 'mpd':
            print(f"警告: ws={target_window_size} 对应的模型不是 'mpd'。")
            return
    except ValueError:
        print(f"错误: 在配置中找不到 window_size={target_window_size}。")
        return

    if not hasattr(ts_maa_instance, 'test_x_img_all') or len(ts_maa_instance.test_x_img_all) <= mpd_idx or \
            ts_maa_instance.test_x_img_all[mpd_idx].nelement() == 0:
        print(f"信息: ws={target_window_size} 对应的测试数据为空，跳过重构可视化。")
        return

    window_size = ts_maa_instance.window_sizes[mpd_idx];
    num_features = ts_maa_instance.test_x_img_all[mpd_idx].shape[-1]

    if pretrainer_type == 'cae':
        vis_model = CAE_for_pretrain(input_height=window_size, input_width=num_features).to(ts_maa_instance.device)
    elif pretrainer_type == 't3vae':
        vis_model = t3VAE_for_pretrain(input_height=window_size, input_width=num_features).to(ts_maa_instance.device)
    else:
        return

    try:
        vis_model.encoder.load_state_dict(
            torch.load(pretrainer_ckpt_path, map_location=ts_maa_instance.device));
        vis_model.eval()
    except Exception as e:
        print(f"错误: 加载预训练权重 '{pretrainer_ckpt_path}' 失败: {e}");
        return

    test_images = ts_maa_instance.test_x_img_all[mpd_idx];
    num_test_samples = test_images.shape[0]
    actual_num_samples = min(num_samples, num_test_samples)
    if actual_num_samples == 0: return
    sample_indices = np.random.choice(num_test_samples, actual_num_samples, replace=False)

    vis_dir = os.path.join(ts_maa_instance.output_dir, "pretrain_vis",
                           f"{pretrainer_type}_reconstruction_ws{target_window_size}")
    os.makedirs(vis_dir, exist_ok=True)
    for sample_idx in sample_indices:
        with torch.no_grad():
            original_image = test_images[sample_idx:sample_idx + 1].to(ts_maa_instance.device)
            reconstructed_image, _, _ = vis_model(original_image)
        original_np = original_image.cpu().numpy().squeeze();
        reconstructed_np = reconstructed_image.cpu().numpy().squeeze()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5));
        fig.suptitle(f'{pretrainer_type.upper()} Reconstruction (ws={target_window_size}) - Sample {sample_idx}',
                     fontsize=16)
        im1 = ax1.imshow(original_np, cmap='viridis', aspect='auto');
        ax1.set_title('Original Input');
        ax1.set_xlabel('Features');
        ax1.set_ylabel('Time Steps (Window)');
        fig.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(reconstructed_np, cmap='viridis', aspect='auto');
        ax2.set_title('Reconstructed Output');
        ax2.set_xlabel('Features');
        ax2.set_ylabel('Time Steps (Window)');
        fig.colorbar(im2, ax=ax2)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
        save_path = os.path.join(vis_dir, f'reconstruction_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=150);
        plt.close()
        print(f"  已保存{pretrainer_type.upper()}重构图像: {save_path}")


def visualize_feature_correlation(ts_maa_instance):
    # --- 核心修改开始 ---
    print("\n--- 开始生成特征相关性热力图 (已包含 'close' 列) ---")
    if not hasattr(ts_maa_instance, 'feature_columns') or not ts_maa_instance.feature_columns:
        print("警告: ts_maa_instance 中未找到 'feature_columns' 属性，跳过热力图生成。")
        return

    train_csv_path = ts_maa_instance.args.train_csv_path
    if not os.path.exists(train_csv_path):
        print(f"警告: 找不到训练数据文件 '{train_csv_path}'，跳过热力图生成。")
        return

    df_train = pd.read_csv(train_csv_path)

    # 获取原始的、不包含 'close' 的特征列表
    original_feature_cols = ts_maa_instance.feature_columns

    # 显式地将 'close' 添加到要可视化的列列表中
    # 使用 dict.fromkeys 来去重并保持大致顺序，确保 'close' 只被添加一次
    cols_for_heatmap = list(dict.fromkeys(original_feature_cols + ['close']))

    # 从原始数据帧中选择这些列用于绘图
    df_features = df_train[cols_for_heatmap].copy()
    # --- 核心修改结束 ---

    df_features = df_features.loc[:, df_features.var() > 1e-8]
    if df_features.shape[1] < 2:
        print("警告: 用于热力图的有效特征数量少于2，跳过。")
        return

    corr_matrix = df_features.corr(method='pearson')
    plt.style.use('seaborn-v0_8-whitegrid')
    try:
        plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif'];
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    plt.figure(figsize=(22, 18))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)

    path_parts = ts_maa_instance.output_dir.replace('\\', '/').split('/')
    stock_name = path_parts[-1]
    sector_name = path_parts[-2]

    # 更新图表标题以反映 'close' 已被包含
    plt.title(f'特征与收盘价(close)相关性热力图\n股票: {sector_name} - {stock_name}', fontsize=24, pad=20)

    plt.xticks(rotation=90, fontsize=10);
    plt.yticks(rotation=0, fontsize=10)

    vis_dir = os.path.join(ts_maa_instance.output_dir, "pretrain_vis", "feature_analysis")
    os.makedirs(vis_dir, exist_ok=True)

    save_path = os.path.join(vis_dir, 'feature_and_close_correlation_heatmap.png')  # 修改保存文件名

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已成功保存包含 'close' 的特征相关性热力图至: {save_path}")


def visualize_encoded_feature_correlation(ts_maa_instance, pretrainer_ckpt_path, pretrainer_type,
                                          target_window_size=None):
    from tqdm import tqdm
    if target_window_size is None:
        print("错误: visualize_encoded_feature_correlation 需要 target_window_size 参数。")
        return

    print(f"\n--- 开始为 ws={target_window_size} 生成{pretrainer_type.upper()}编码后特征的相关性热力图 ---")

    try:
        mpd_idx = ts_maa_instance.window_sizes.index(target_window_size)
        if ts_maa_instance.generator_names[mpd_idx] != 'mpd':
            print(f"警告: ws={target_window_size} 对应的模型不是 'mpd'。")
            return
    except ValueError:
        print(f"错误: 在配置中找不到 window_size={target_window_size}。")
        return

    if not hasattr(ts_maa_instance, 'train_x_img_all') or len(ts_maa_instance.train_x_img_all) <= mpd_idx or \
            ts_maa_instance.train_x_img_all[mpd_idx].nelement() == 0:
        print(f"信息: ws={target_window_size} 对应的训练数据为空，跳过编码后特征相关性可视化。")
        return

    window_size = ts_maa_instance.window_sizes[mpd_idx];
    num_features = ts_maa_instance.train_x_img_all[mpd_idx].shape[-1]

    if pretrainer_type == 'cae':
        encoder = CAE_Encoder(input_height=window_size, input_width=num_features).to(ts_maa_instance.device)
    elif pretrainer_type == 't3vae':
        encoder = t3VAE_Encoder(input_height=window_size, input_width=num_features).to(ts_maa_instance.device)
    else:
        return

    try:
        encoder.load_state_dict(torch.load(pretrainer_ckpt_path, map_location=ts_maa_instance.device));
        encoder.eval()
    except Exception as e:
        print(f"错误: 加载预训练权重 '{pretrainer_ckpt_path}' 失败: {e}");
        return

    with torch.no_grad():
        train_images = ts_maa_instance.train_x_img_all[mpd_idx];
        batch_size = ts_maa_instance.batch_size;
        encoded_features_list = []
        for i in tqdm(range(0, len(train_images), batch_size),
                      desc=f"{pretrainer_type.upper()}编码特征 (ws={target_window_size})"):
            batch_images = train_images[i:i + batch_size]
            if pretrainer_type == 't3vae':
                mu, _ = encoder(batch_images);
                encoded_batch = mu
            else:
                encoded_batch = encoder(batch_images).view(batch_images.size(0), -1)
            encoded_features_list.append(encoded_batch.cpu().numpy())

    encoded_features = np.vstack(encoded_features_list)
    df_encoded = pd.DataFrame(encoded_features,
                              columns=[f'{pretrainer_type}_feat_{i}' for i in range(encoded_features.shape[1])])
    corr_matrix_encoded = df_encoded.corr(method='pearson')
    plt.figure(figsize=(18, 15));
    sns.heatmap(corr_matrix_encoded, cmap='coolwarm', annot=False, vmin=-1, vmax=1)
    path_parts = ts_maa_instance.output_dir.replace('\\', '/').split('/');
    stock_name = path_parts[-1];
    sector_name = path_parts[-2]
    plt.title(
        f'{pretrainer_type.upper()}编码后特征相关性热力图 (ws={target_window_size})\n股票: {sector_name} - {stock_name}',
        fontsize=24,
        pad=20)

    vis_dir = os.path.join(ts_maa_instance.output_dir, "pretrain_vis", "feature_analysis")
    os.makedirs(vis_dir, exist_ok=True);
    save_path = os.path.join(vis_dir,
                             f'{pretrainer_type}_encoded_feature_correlation_heatmap_ws{target_window_size}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight');
    plt.close()
    print(f"已成功保存{pretrainer_type.upper()}编码后特征相关性热力图至: {save_path}")