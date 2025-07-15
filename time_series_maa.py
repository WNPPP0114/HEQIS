# 文件名: time_series_maa.py

from MAA_base import MAABase
import torch
import torch.nn as nn
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from utils.multiGAN_trainer_disccls import train_multi_gan
from typing import List
import models
import os
import time
import glob
from utils.evaluate_visualization import evaluate_best_models
import joblib
import sys
import logging
import traceback
from tqdm import tqdm
from models.pretrainer import (
    CAE_for_pretrain, t3VAE_for_pretrain, t3vae_loss_function,
    plot_pretrain_loss
)


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time();
        result = func(*args, **kwargs);
        end_time = time.time()
        print(f"MAA_time_series - 方法 '{func.__name__}' 执行耗时: {end_time - start_time:.4f} 秒")
        return result

    return wrapper


def generate_labels(y):
    y = np.array(y).flatten();
    labels = [1]
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            labels.append(2)
        elif y[i] < y[i - 1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


class MAA_time_series(MAABase):
    def __init__(self, args, N_pairs, batch_size, num_epochs, generators_names, discriminators_names, ckpt_dir,
                 output_dir, window_sizes, initial_learning_rate=2e-5, do_distill_epochs=1, cross_finetune_epochs=5,
                 precise=torch.float32, device=None, seed=None, ckpt_path=None, gan_weights=None):
        super().__init__(N_pairs, batch_size, num_epochs, generators_names, discriminators_names, ckpt_dir, output_dir,
                         initial_learning_rate, precise, do_distill_epochs, cross_finetune_epochs, device, seed,
                         ckpt_path)
        self.args = args;
        self.window_sizes = window_sizes;
        self.generator_dict = {};
        self.discriminator_dict = {"default": models.Discriminator3}
        for name in dir(models):
            obj = getattr(models, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and hasattr(models,
                                                                                      '__all__') and name in models.__all__:
                lname = name.lower()
                if "generator" in lname:
                    self.generator_dict[lname.replace("generator_", "")] = obj
                elif "discriminator" in lname:
                    self.discriminator_dict[lname.replace("discriminator", "")] = obj
        self.gan_weights = gan_weights;
        self.init_hyperparameters()

    def run_pretraining_if_needed(self, all_stock_files: List[str], pretrainer_ckpt_path: str, pretrain_epochs: int,
                                  specific_window_size: int):
        pretrainer_type = self.args.pretrainer_type

        all_datasets = []
        window_size = specific_window_size

        temp_df = pd.read_csv(all_stock_files[0])
        temp_feature_cols = [col for col in temp_df.columns if col not in ['date', 'direction']]
        num_features = len(temp_feature_cols)

        for stock_file in tqdm(all_stock_files, desc=f"数据加载 (ws={window_size})"):
            df = pd.read_csv(stock_file)
            features = df[temp_feature_cols].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler.fit_transform(features)
            images = []
            for i in range(window_size, len(scaled_features)):
                image_data = scaled_features[i - window_size: i, :].reshape(1, window_size, num_features)
                images.append(image_data)
            if images: all_datasets.append(TensorDataset(torch.from_numpy(np.array(images)).float()))
        if not all_datasets:
            print(f"错误: 未能为 window_size={window_size} 创建任何预训练数据。")
            return

        full_dataset = ConcatDataset(all_datasets)
        dataloader = DataLoader(full_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        if pretrainer_type == 'cae':
            pretrain_model = CAE_for_pretrain(input_height=window_size, input_width=num_features).to(self.device)
            criterion = nn.MSELoss()
        elif pretrainer_type == 't3vae':
            pretrain_model = t3VAE_for_pretrain(input_height=window_size, input_width=num_features).to(self.device)
            criterion = t3vae_loss_function
        else:
            raise ValueError(f"未知的预训练器类型: {pretrainer_type}")

        optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)
        pretrain_model.train()
        loss_history = []
        with tqdm(total=pretrain_epochs * len(dataloader),
                  desc=f"{pretrainer_type.upper()} 预训练 (ws={window_size})") as pbar:
            for epoch in range(pretrain_epochs):
                epoch_loss = 0.0
                num_batches = 0
                for batch_idx, batch in enumerate(dataloader):
                    images = batch[0].to(self.device)
                    optimizer.zero_grad()
                    recon_images, mu, logvar = pretrain_model(images)
                    if pretrainer_type == 'cae':
                        loss = criterion(recon_images, images)
                    else:
                        loss = criterion(recon_images, images, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix(Epoch=f'{epoch + 1}/{pretrain_epochs}', Loss=f'{epoch_loss / num_batches:.6f}')
                    pbar.update(1)
                avg_epoch_loss = epoch_loss / num_batches
                loss_history.append(avg_epoch_loss)

        os.makedirs(os.path.dirname(pretrainer_ckpt_path), exist_ok=True)
        torch.save(pretrain_model.encoder.state_dict(), pretrainer_ckpt_path)
        plot_pretrain_loss(loss_history, self.output_dir, f"{pretrainer_type}_ws{window_size}")

    @log_execution_time
    def process_data(self, train_csv_path, predict_csv_path, target_column, exclude_columns):
        train_df = pd.read_csv(train_csv_path);
        self.predict_df = pd.read_csv(predict_csv_path)
        feature_columns = [col for col in train_df.columns if col != target_column and col not in exclude_columns]
        self.feature_columns = feature_columns
        x_from_train_df, y_from_train_df = train_df[feature_columns].values, train_df[[target_column]].values
        train_ratio, val_ratio, _ = self.args.train_val_test_split;
        total_len = len(train_df)
        train_end_idx, val_end_idx = int(total_len * train_ratio), int(total_len * (train_ratio + val_ratio))
        train_x_raw, val_x_raw, test_x_raw = x_from_train_df[:train_end_idx], x_from_train_df[
                                                                              train_end_idx:val_end_idx], x_from_train_df[
                                                                                                          val_end_idx:]
        train_y_raw, val_y_raw, test_y_raw = y_from_train_df[:train_end_idx], y_from_train_df[
                                                                              train_end_idx:val_end_idx], y_from_train_df[
                                                                                                          val_end_idx:]
        self.train_size = len(train_x_raw);
        self.x_scalers, self.y_scaler = [MinMaxScaler(feature_range=(0, 1)) for _ in range(self.N)], MinMaxScaler(
            feature_range=(0, 1))
        self.train_x_list = [s.fit_transform(train_x_raw) for s in self.x_scalers];
        self.train_y = self.y_scaler.fit_transform(train_y_raw)
        self.val_x_list = [s.transform(val_x_raw) for s in self.x_scalers];
        self.val_y = self.y_scaler.transform(val_y_raw)
        self.test_x_list = [s.transform(test_x_raw) for s in self.x_scalers];
        self.test_y = self.y_scaler.transform(test_y_raw)
        self.train_labels, self.val_labels, self.test_labels = generate_labels(self.train_y), generate_labels(
            self.val_y), generate_labels(self.test_y)

    def create_sequences_combine(self, x, y, label, window_size, start):
        x_seq, y_reg, y_gan, label_gan, x_img = [], [], [], [], []
        num_features = x.shape[1]
        for i in range(start, x.shape[0]):
            x_seq.append(x[i - window_size: i, :])
            y_reg.append(y[i])
            y_gan.append(y[i - window_size: i + 1])
            label_gan.append(label[i - window_size: i + 1])
            x_img.append(x[i - window_size: i, :].reshape(1, window_size, num_features))
        return (torch.from_numpy(np.array(x_seq)).float(), torch.from_numpy(np.array(x_img)).float(),
                torch.from_numpy(np.array(y_reg)).float(), torch.from_numpy(np.array(y_gan)).float(),
                torch.from_numpy(np.array(label_gan)).long())

    @log_execution_time
    def init_dataloader(self):
        ws = self.window_sizes
        train_data = [self.create_sequences_combine(self.train_x_list[i], self.train_y, self.train_labels, w, ws[-1])
                      for i, w in enumerate(ws)]
        val_data = [self.create_sequences_combine(self.val_x_list[i], self.val_y, self.val_labels, w, ws[-1]) for i, w
                    in enumerate(ws)]
        test_data = [self.create_sequences_combine(self.test_x_list[i], self.test_y, self.test_labels, w, ws[-1]) for
                     i, w in enumerate(ws)]
        self.train_x_all = [d[0].to(self.device) for d in train_data];
        self.train_x_img_all = [d[1].to(self.device) for d in train_data]
        self.train_y_all = train_data[0][2];
        self.train_y_gan_all = [d[3].to(self.device) for d in train_data];
        self.train_label_gan_all = [d[4].to(self.device) for d in train_data]
        self.val_x_all = [d[0].to(self.device) for d in val_data];
        self.val_x_img_all = [d[1].to(self.device) for d in val_data]
        self.val_y_all = val_data[0][2];
        self.val_y_gan_all = [d[3].to(self.device) for d in val_data];
        self.val_label_gan_all = [d[4].to(self.device) for d in val_data]
        self.test_x_all = [d[0].to(self.device) for d in test_data];
        self.test_x_img_all = [d[1].to(self.device) for d in test_data]
        self.test_y_all = test_data[0][2];
        self.test_y_gan_all = [d[3].to(self.device) for d in test_data];
        self.test_label_gan_all = [d[4].to(self.device) for d in test_data]
        self.dataloaders = []
        for i in range(self.N):
            is_mpd = "mpd" in self.generator_names[i]
            x_data = self.train_x_img_all[i] if is_mpd else self.train_x_all[i]
            self.dataloaders.append(
                DataLoader(TensorDataset(x_data, self.train_y_gan_all[i], self.train_label_gan_all[i]),
                           batch_size=self.batch_size, shuffle=True, generator=torch.manual_seed(self.seed),
                           drop_last=True))

    @log_execution_time
    def init_model(self, num_cls):
        self.generators, self.discriminators = [], []

        for i, name in enumerate(self.generator_names):
            input_dim = self.train_x_all[i].shape[-1]
            y_dim = self.train_y_all.shape[-1]
            GenClass = self.generator_dict.get(name)
            if GenClass is None: sys.exit(1)

            init_kwargs = {'use_rope': self.args.use_rope}

            if name == "mpd":
                current_ws = self.window_sizes[i]
                init_kwargs.update({
                    'input_height': current_ws,
                    'input_width': len(self.feature_columns),
                    'num_classes': num_cls,
                    'pretrainer_type': self.args.pretrainer_type
                })
                generator_instance = GenClass(**init_kwargs)

                # 为当前MPD模型构造其特定的预训练权重路径
                pretrainer_ckpt_path = os.path.join(
                    self.args.output_dir,
                    f"{self.args.pretrainer_type}_encoder_ws{current_ws}.pt"
                )

                if os.path.exists(pretrainer_ckpt_path):
                    try:
                        generator_instance.pretrainer_encoder.load_state_dict(
                            torch.load(pretrainer_ckpt_path, map_location=self.device))
                        print(
                            f"成功为 Generator_mpd (G{i + 1}, ws={current_ws}) 加载了预训练权重。")
                    except Exception as e:
                        print(f"警告: 为 G{i + 1} (ws={current_ws}) 加载预训练权重 '{pretrainer_ckpt_path}' 失败: {e}")
                else:
                    print(
                        f"信息: 未找到 G{i + 1} (ws={current_ws}) 对应的预训练权重 '{pretrainer_ckpt_path}'，将随机初始化。")

            elif name in ["transformer", "transformer_deep"]:
                init_kwargs.update({'input_dim': input_dim, 'output_len': y_dim})
                generator_instance = GenClass(**init_kwargs)
            elif name == "dct":
                init_kwargs.update({'input_dim': input_dim, 'out_size': y_dim, 'num_classes': num_cls})
                generator_instance = GenClass(**init_kwargs)
            elif name == "rnn":
                init_kwargs.update({'input_size': input_dim})
                generator_instance = GenClass(**init_kwargs)
            else:
                init_kwargs.update({'input_size': input_dim, 'out_size': y_dim})
                generator_instance = GenClass(**init_kwargs)

            self.generators.append(generator_instance.to(self.device))
            DisClass = self.discriminator_dict.get("default")
            if DisClass is None: sys.exit(1)
            self.discriminators.append(
                DisClass(input_dim=self.window_sizes[i] + 1, out_size=y_dim, num_cls=num_cls).to(self.device))

    def init_hyperparameters(self):
        self.init_GDweight = [[1.0 if i == j else 0.0 for j in range(self.N)] + [1.0] for i in range(self.N)]
        self.final_GDweight = self.gan_weights if self.gan_weights else [[round(1.0 / self.N, 3)] * self.N + [1.0] for _
                                                                         in range(self.N)]

    def train(self, logger, date_series=None):
        is_mpd_run = any("mpd" in name for name in self.generator_names)
        train_xes = self.train_x_img_all if is_mpd_run else self.train_x_all
        val_xes = self.val_x_img_all if is_mpd_run else self.val_x_all
        return train_multi_gan(args=self.args, generators=self.generators, discriminators=self.discriminators,
                               dataloaders=self.dataloaders, window_sizes=self.window_sizes, y_scaler=self.y_scaler,
                               train_xes=train_xes, train_y=self.train_y_all, val_xes=val_xes, val_y=self.val_y_all,
                               val_y_gan=self.val_y_gan_all, val_label_gan=self.val_label_gan_all,
                               output_dir=self.output_dir, device=self.device, init_GDweight=self.init_GDweight,
                               final_GDweight=self.final_GDweight, logger=logger, date_series=date_series)

    def save_models(self, best_model_state):
        gen_dir = os.path.join(self.ckpt_dir, "generators");
        disc_dir = os.path.join(self.ckpt_dir, "discriminators")
        os.makedirs(gen_dir, exist_ok=True);
        os.makedirs(disc_dir, exist_ok=True)
        for i, gen_name in enumerate(self.generator_names):
            if i < len(best_model_state) and best_model_state[i]: torch.save(best_model_state[i], os.path.join(gen_dir,
                                                                                                               f"{i + 1}_{gen_name}.pt"))
        for i, disc in enumerate(self.discriminators):
            if i < len(self.discriminators): torch.save(disc.state_dict(),
                                                        os.path.join(disc_dir, f"{i + 1}_Discriminator3.pt"))
        print(f"所有模型已成功保存至: {self.ckpt_dir}")

    def save_predictions_to_csv(self, date_series=None):
        print("\n--- 开始生成并保存真实值 vs. 预测值对比CSV文件 (基于测试集) ---")
        is_mpd_run = any("mpd" in name for name in self.generator_names)
        test_xes = self.test_x_img_all if is_mpd_run else self.test_x_all
        with torch.no_grad():
            for i, gen in enumerate(self.generators):
                gen.eval()
                try:
                    y_pred_gen, _ = gen(test_xes[i]);
                    y_pred_norm = y_pred_gen.cpu().numpy().reshape(-1, 1);
                    true_y_segment = self.test_y_all[-len(y_pred_gen):];
                    y_true_norm = true_y_segment.cpu().numpy().reshape(-1, 1)
                    y_pred = self.y_scaler.inverse_transform(y_pred_norm).flatten();
                    y_true = self.y_scaler.inverse_transform(y_true_norm).flatten()
                    df_out = pd.DataFrame({'true': y_true, 'pred': y_pred})
                    if date_series is not None:
                        max_ws = max(self.window_sizes);
                        date_aligned = date_series.iloc[max_ws:].reset_index(drop=True)
                        ts_len, vs_len = len(self.train_y_all), len(self.val_y_all)
                        if len(date_aligned) >= (ts_len + vs_len + len(y_pred)):
                            test_dates = date_aligned.iloc[ts_len + vs_len:];
                            dates_csv = test_dates.iloc[-len(y_pred):].reset_index(drop=True)
                            df_out['date'] = dates_csv;
                            df_out = df_out[['date', 'true', 'pred']]
                    csv_dir = os.path.join(self.output_dir, "true2pred_csv");
                    os.makedirs(csv_dir, exist_ok=True)
                    out_path = os.path.join(csv_dir, f'predictions_gen_{i + 1}_{self.generator_names[i]}.csv');
                    df_out.to_csv(out_path, index=False)
                    print(f"已保存 G{i + 1} 的对比: {out_path}")
                except Exception as e:
                    print(f"错误: 为 G{i + 1} 生成预测CSV时出错: {e}");
                    traceback.print_exc()

    def pred(self, date_series=None):
        gen_dir = os.path.join(self.ckpt_dir, "generators")
        if not os.path.isdir(gen_dir): raise FileNotFoundError(f"找不到 'generators' 文件夹: {gen_dir}")
        best_model_state = [None] * self.N;
        loaded_count = 0
        for i, gen_name in enumerate(self.generator_names):
            save_path = os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt")
            if os.path.exists(save_path):
                try:
                    self.generators[i].load_state_dict(torch.load(save_path, map_location=self.device));
                    best_model_state[i] = self.generators[i].state_dict();
                    loaded_count += 1
                except Exception as e:
                    print(f"错误: 加载检查点 {save_path} 失败: {e}");
        if loaded_count == 0: print("错误: 未能加载任何模型。"); return None
        is_mpd_run = any("mpd" in name for name in self.generator_names)
        train_xes, test_xes = (self.train_x_img_all, self.test_x_img_all) if is_mpd_run else (
            self.train_x_all, self.test_x_all)
        results = evaluate_best_models(self.generators, best_model_state, train_xes, self.train_y_all, test_xes,
                                       self.test_y_all, self.y_scaler, self.output_dir, self.window_sizes,
                                       date_series=date_series)
        self.save_predictions_to_csv(date_series=date_series);
        return results

    def save_scalers(self):
        try:
            joblib.dump(self.x_scalers[0], os.path.join(self.output_dir, 'x_scaler.gz'));
            joblib.dump(self.y_scaler, os.path.join(self.output_dir, 'y_scaler.gz'))
            print(f"Scaler 已成功保存至: {self.output_dir}")
        except Exception as e:
            print(f"错误: 保存 scaler 失败: {e}");

    def generate_and_save_daily_signals(self, best_model_state, predict_csv_path):
        if not all(hasattr(self, attr) for attr in
                   ['x_scalers', 'y_scaler', 'feature_columns', 'generators', 'train_size', 'val_y',
                    'window_sizes']): return
        if not best_model_state or not any(s is not None for s in best_model_state): return
        print("\n--- 开始为所有模型生成每日预测信号 ---");
        df_predict = pd.read_csv(predict_csv_path)
        x_scaler, y_scaler = self.x_scalers[0], self.y_scaler;
        val_size = len(self.val_y);
        loop_start_index = self.train_size + val_size + max(self.window_sizes)
        if loop_start_index >= len(df_predict): print(f"警告: 预测数据不足。"); return
        for i, state in enumerate(best_model_state):
            if state is None: continue
            gen_name, window_size, generator = self.generator_names[i], self.window_sizes[i], self.generators[i]
            is_mpd_model = "mpd" in gen_name
            try:
                generator.load_state_dict(state);
                generator.eval()
            except Exception as e:
                print(f"错误: 加载 G{i + 1} 状态失败: {e}");
                continue
            print(f"正在处理模型: G{i + 1} ({gen_name})");
            signals = []
            for j in range(loop_start_index, len(df_predict)):
                df_segment = df_predict.iloc[j - window_size: j];
                if len(df_segment) < window_size: continue
                sequence_data = df_segment[self.feature_columns].values
                if np.isnan(sequence_data).any(): continue
                try:
                    scaled_sequence = x_scaler.transform(sequence_data)
                    input_data = scaled_sequence.reshape(1, 1, window_size,
                                                         len(self.feature_columns)) if is_mpd_model else np.array(
                        [scaled_sequence])
                    input_tensor = torch.from_numpy(input_data).float().to(self.device)
                    with torch.no_grad():
                        gen_output, logits = generator(input_tensor);
                        predicted_action = logits.argmax(dim=1).item()
                        predicted_close_real = y_scaler.inverse_transform(gen_output.cpu().numpy()).flatten()[0]
                    signals.append({'date': df_predict.iloc[j]['date'], 'predicted_action': predicted_action,
                                    'predicted_close': predicted_close_real})
                except Exception as e:
                    print(f"错误: 在索引 {j} 生成信号时出错: {e}");
            if signals:
                df_signals = pd.DataFrame(signals);
                signal_filepath = os.path.join(self.output_dir, f'G{i + 1}_{gen_name}_daily_signals.csv')
                df_signals.to_csv(signal_filepath, index=False, float_format='%.4f');
                print(f"已保存每日信号文件: {signal_filepath}")
            else:
                print(f"警告: G{i + 1} ({gen_name}) 没有生成任何有效信号。")

    def distill(self):
        pass

    def visualize_and_evaluate(self):
        pass

    def init_history(self):
        pass