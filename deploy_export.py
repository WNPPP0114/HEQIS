# 文件名: deploy_export.py (Windows运行)

import torch
import sys
import os
import joblib
import json
import numpy as np

# 1. 路径设置 (根据你的实际路径修改)
# 必须能找到 models 文件夹
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models

# 假设使用 GRU 模型，特征维度 12，窗口 60
MODEL_TYPE = 'gru'
INPUT_SIZE = 12
WINDOW_SIZE = 60

# 你的训练权重路径
CKPT_PATH = "ckpt/multi_gan/培育钻石/黄河旋风/ckpt/generators/1_gru.pt"
# 你的 Scaler 路径
X_SCALER_PATH = "csv_data/predict/培育钻石/黄河旋风/x_scaler.gz"
Y_SCALER_PATH = "csv_data/predict/培育钻石/黄河旋风/y_scaler.gz"

ONNX_NAME = "model_deploy.onnx"
JSON_NAME = "scaler_params.json"


def export_onnx():
    print(f"--- 正在导出 ONNX: {MODEL_TYPE} ---")
    # 初始化模型结构 (必须与训练时一致)
    if MODEL_TYPE == 'gru':
        model = models.Generator_gru(input_size=INPUT_SIZE, out_size=1, use_rope=True)

    # 加载权重
    state_dict = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # 创建 Dummy Input [Batch, Window, Features]
    dummy_input = torch.randn(1, WINDOW_SIZE, INPUT_SIZE)

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_NAME,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ ONNX 导出成功: {ONNX_NAME}")


def export_json_params():
    print(f"--- 正在提取 Scaler 参数到 JSON ---")
    # 加载 joblib 对象
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    # 提取 MinMaxScaler 的核心参数
    # 公式: X_scaled = X * scale_ + min_
    params = {
        "x_scale": x_scaler.scale_.tolist(),
        "x_min": x_scaler.min_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
        "y_min": y_scaler.min_.tolist(),
        "input_features": x_scaler.n_features_in_  # 记录一下特征数量防呆
    }

    # 写入 JSON
    with open(JSON_NAME, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"✅ 参数提取成功: {JSON_NAME}")
    print("   (此文件是纯文本，兼容 Python 3.8/3.9/3.11 等所有版本)")


if __name__ == "__main__":
    export_json_params()
    export_onnx()