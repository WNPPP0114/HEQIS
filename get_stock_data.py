# get_stock_data.py (最终修复版：修复FutureWarning警告)

import tushare as ts
import pandas as pd
import talib
import os
import time
from datetime import datetime, timedelta
import traceback
from typing import Optional, Tuple, List, Dict
import sys
import json
import multiprocessing
import argparse
import numpy as np
import re
import joblib
from sklearn.preprocessing import MinMaxScaler

# --- 配置区 ---
TUSHARE_TOKEN = '5c9bcf56aeee0f2738748e413c0bd3112e22b0897618eaf7f9b4ca41'
BASE_DATA_DIR = 'csv_data'
TRAIN_SUBDIR = 'train'
PREDICT_SUBDIR = 'predict'
USE_ADJUSTED_DATA = True
INDICATOR_WARMUP_PERIOD = 240
TRAIN_DATA_FETCH_START_DATE = '20120101'
USER_TRAIN_END_DATE = '20250630'
PREDICT_DATA_USER_END_DATE = '20251201'

stock_info_dict = {
    "培育钻石": ["黄河旋风"],
    "云计算": ["美利云", "云赛智联"],
    #.................
}

# --- Tushare API 初始化 ---
try:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
except Exception as e:
    print(f"错误: 无法在 get_stock_data.py 中全局初始化 Tushare Pro API 'pro': {e}")
    pro = None


def get_tushare_pro_instance():
    return ts.pro_api(token=TUSHARE_TOKEN)


GLOBAL_TRADE_CAL_DF: Optional[pd.DataFrame] = None


# --- 辅助函数 ---
def find_next_trade_day(start_date_str: str, trade_cal_df: pd.DataFrame) -> str:
    current_date = pd.to_datetime(start_date_str, format='%Y%m%d', errors='coerce')
    if pd.isna(current_date): raise ValueError(f"Invalid start_date_str: {start_date_str}")
    future_cal = trade_cal_df[trade_cal_df['trade_date_dt'] >= current_date].sort_values('trade_date_dt')
    for _, row in future_cal.iterrows():
        if row['is_open'] == 1: return row['trade_date']
    raise ValueError(f"Could not find a trade day from {start_date_str}")


def find_latest_trading_day_on_or_before(date_dt: datetime, trade_cal_df: pd.DataFrame) -> Optional[datetime]:
    filtered_cal = trade_cal_df[
        (trade_cal_df['trade_date_dt'] <= date_dt) & (trade_cal_df['is_open'] == 1)].sort_values('trade_date_dt',
                                                                                                 ascending=False)
    return filtered_cal['trade_date_dt'].iloc[0] if not filtered_cal.empty else None


def create_data_dirs_if_not_exists(sector_name: str, stock_name: str) -> Tuple[str, str]:
    train_stock_dir = os.path.join(BASE_DATA_DIR, TRAIN_SUBDIR, sector_name, stock_name)
    predict_stock_dir = os.path.join(BASE_DATA_DIR, PREDICT_SUBDIR, sector_name, stock_name)
    os.makedirs(train_stock_dir, exist_ok=True)
    os.makedirs(predict_stock_dir, exist_ok=True)
    return train_stock_dir, predict_stock_dir


def get_security_info_from_name(security_name_with_suffix: str, all_securities_basic_df: pd.DataFrame,
                                target_category: Optional[str] = None) -> Tuple[
    Optional[str], Optional[str], Optional[str], Optional[str]]:
    base_name, exchange_preference = security_name_with_suffix, None
    match = re.search(r'\((沪基|深基)\)', security_name_with_suffix)
    if match:
        base_name = re.sub(r'\s*\(.*\)\s*$', '', security_name_with_suffix).strip()
        exchange_preference = '.SH' if match.group(1) == '沪基' else '.SZ'
    df_to_search = all_securities_basic_df.copy()
    if target_category:
        target_type = {"指数": "index", "基金(场内)": "fund"}.get(target_category)
        if target_type:
            df_to_search = df_to_search[df_to_search['type'] == target_type]
        else:
            df_to_search = df_to_search[df_to_search['type'] == 'stock']
    result = df_to_search[df_to_search['name'].str.contains(base_name, case=False, na=False)]
    if not result.empty:
        if exchange_preference and len(result) > 1:
            preferred_result = result[result['ts_code'].str.endswith(exchange_preference)]
            if not preferred_result.empty: result = preferred_result
        exact_match = result[result['name'] == base_name]
        first_result = exact_match.iloc[0] if not exact_match.empty else result.iloc[0]
        return first_result['ts_code'], first_result['type'], first_result['industry'], security_name_with_suffix
    return None, None, None, security_name_with_suffix


# --- 技术指标计算、数据清洗和预处理函数 ---
def calculate_technical_indicators(df_input: pd.DataFrame) -> pd.DataFrame:
    # 这个函数现在只负责计算，不负责格式化
    if df_input.empty or len(df_input) < 30:
        return df_input

    df = df_input.copy().sort_values('date').reset_index(drop=True)

    required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_ohlcv_cols:
        if col not in df.columns:
            return df_input
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=required_ohlcv_cols, inplace=True)
    if df.empty:
        return df_input

    if 'pctChg' not in df.columns or df['pctChg'].isnull().any():
        df['pctChg'] = (df['close'].pct_change()) * 100
        # 修复：fillna({'col': val}) 是安全的，但为了代码一致性，此处保持原样或改为赋值
        df.fillna({'pctChg': 0}, inplace=True)

    op, hi, lo, cl, vo = (df[c].astype(float) for c in ['open', 'high', 'low', 'close', 'volume'])

    try:
        df['ma_5'] = talib.MA(cl, 5)
        df['ma_10'] = talib.MA(cl, 10)
        df['ma_20'] = talib.MA(cl, 20)
        df['ma_30'] = talib.MA(cl, 30)
        df['ma_60'] = talib.MA(cl, 60)

        df['sma_5'] = talib.SMA(cl, 5)
        df['sma_10'] = talib.SMA(cl, 10)
        df['sma_20'] = talib.SMA(cl, 20)
        df['sma_30'] = talib.SMA(cl, 30)
        df['sma_60'] = talib.SMA(cl, 60)

        df['ema_5'] = talib.EMA(cl, 5)
        df['ema_10'] = talib.EMA(cl, 10)
        df['ema_20'] = talib.EMA(cl, 20)
        df['ema_30'] = talib.EMA(cl, 30)
        df['ema_60'] = talib.EMA(cl, 60)

        return df

    except Exception as e:
        print(f"  -> 技术指标计算时发生错误: {e}")
        traceback.print_exc()
        return df_input


def robust_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df_cleaned = df.copy()
    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
        df_cleaned.dropna(subset=['date'], inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'pctChg', 'turn']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    return df_cleaned.sort_values(by='date').reset_index(drop=True)


def preprocess_raw_df(df_raw_daily: pd.DataFrame, df_raw_basic: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df_raw_daily is None or df_raw_daily.empty: return pd.DataFrame()
    df_daily = df_raw_daily.rename(columns={'vol': 'volume', 'trade_date': 'date', 'pct_chg': 'pctChg'})
    daily_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if 'pctChg' in df_daily.columns: daily_cols.append('pctChg')
    df_daily = df_daily[[c for c in daily_cols if c in df_daily.columns]].copy()
    df_merged = df_daily
    if df_raw_basic is not None and not df_raw_basic.empty:
        df_basic = df_raw_basic.rename(columns={'trade_date': 'date', 'turnover_rate': 'turn'})
        basic_cols = ['date', 'turn']
        if any(c in df_basic.columns for c in basic_cols):
            df_merged = pd.merge(df_daily, df_basic[[c for c in basic_cols if c in df_basic.columns]], on='date',
                                 how='left')
    if 'turn' not in df_merged.columns:
        print(f"  -> 警告: 缺少 'turn' (换手率) 列。将添加值为0的占位符列以统一特征维度。")
        df_merged['turn'] = 0.0
    if 'date' in df_merged.columns:
        df_merged['date'] = df_merged['date'].astype(str)
    return df_merged


# --- 主数据处理函数 ---
def fetch_and_process_stock_data(ts_code: str, security_type: str, category_name: str, security_name: str,
                                 fetch_start_date: str, predict_end_date_str: str, user_train_end_date_str: str,
                                 global_trade_cal_df: pd.DataFrame, pro_instance) -> bool:
    try:
        train_dir, pred_dir = create_data_dirs_if_not_exists(category_name, security_name)
        train_fp = os.path.join(train_dir,
                                f"{ts_code.replace('.', '_')}_{category_name}_{security_name}_train_data.csv")
        pred_fp = os.path.join(pred_dir,
                               f"{ts_code.replace('.', '_')}_{category_name}_{security_name}_predict_data.csv")

        target_end_dt = pd.to_datetime(predict_end_date_str, format='%Y%m%d')
        df_existing = pd.DataFrame()

        if os.path.exists(pred_fp):
            try:
                df_existing = pd.read_csv(pred_fp, dtype={'date': str})
                latest_trade_day = find_latest_trading_day_on_or_before(target_end_dt, global_trade_cal_df)
                if latest_trade_day and not df_existing.empty and pd.to_datetime(
                        df_existing['date'].max()) >= latest_trade_day:
                    print(f"证券 {security_name} ({ts_code}): 数据已是最新，跳过获取。")
                    return True
            except (pd.errors.EmptyDataError, KeyError):
                df_existing = pd.DataFrame()

        if not df_existing.empty:
            overlap_days = 60
            last_date_in_file = pd.to_datetime(df_existing['date'].max())
            start_fetch_date = (last_date_in_file - timedelta(days=overlap_days)).strftime('%Y%m%d')
        else:
            start_fetch_date = TRAIN_DATA_FETCH_START_DATE

        df_new_segment_raw = pd.DataFrame()
        if pd.to_datetime(start_fetch_date) <= target_end_dt:
            actual_start = find_next_trade_day(start_fetch_date, global_trade_cal_df)
            if pd.to_datetime(actual_start) <= target_end_dt:
                print(
                    f"证券 {security_name} ({ts_code}): 正在获取从 {actual_start} 到 {predict_end_date_str} 的 {security_type} 数据...")
                api_map = {'stock': (pro_instance.daily, pro_instance.daily_basic),
                           'index': (pro_instance.index_daily, None), 'fund': (pro_instance.fund_daily, None)}
                daily_api, basic_api = api_map[security_type]
                df_raw_daily = daily_api(ts_code=ts_code, start_date=actual_start, end_date=predict_end_date_str)
                df_raw_basic = basic_api(ts_code=ts_code, start_date=actual_start,
                                         end_date=predict_end_date_str) if basic_api else None
                time.sleep(0.25)
                df_new_segment_raw = preprocess_raw_df(df_raw_daily, df_raw_basic)

        if not df_new_segment_raw.empty:
            df_combined_raw = pd.concat([df_existing, df_new_segment_raw]).drop_duplicates(subset=['date'],
                                                                                           keep='last').sort_values(
                'date').reset_index(drop=True)
        else:
            df_combined_raw = df_existing

        if df_combined_raw.empty:
            print(f"  -> 警告: 证券 {security_name} 无有效数据，跳过。")
            return False

        df_to_process = df_combined_raw.copy()
        if security_type == 'stock' and USE_ADJUSTED_DATA:
            print(f"  -> 正在为 {ts_code} 单独获取复权因子并进行前复权...")
            stock_adj_factor_df = pro_instance.adj_factor(ts_code=ts_code, start_date=TRAIN_DATA_FETCH_START_DATE,
                                                          end_date=predict_end_date_str)
            if stock_adj_factor_df is not None and not stock_adj_factor_df.empty:
                base_adj_series = stock_adj_factor_df[stock_adj_factor_df['trade_date'] <= user_train_end_date_str]
                if not base_adj_series.empty:
                    base_adj = base_adj_series.sort_values('trade_date', ascending=False).iloc[0]['adj_factor']
                    df_to_process['trade_date_dt'] = pd.to_datetime(df_to_process['date'], format='%Y%m%d');
                    stock_adj_factor_df['trade_date_dt'] = pd.to_datetime(stock_adj_factor_df['trade_date'],
                                                                          format='%Y%m%d')
                    df_with_adj = pd.merge(df_to_process, stock_adj_factor_df[['trade_date_dt', 'adj_factor']],
                                           on='trade_date_dt', how='left').sort_values('trade_date_dt')

                    # --- 修复部分 开始 ---
                    # 替换原先的 fillna(method=..., inplace=True)
                    df_with_adj['adj_factor'] = df_with_adj['adj_factor'].bfill()
                    df_with_adj['adj_factor'] = df_with_adj['adj_factor'].ffill()
                    # --- 修复部分 结束 ---

                    for col in ['open', 'high', 'low', 'close']:
                        adjusted_price = pd.to_numeric(df_with_adj[col], errors='coerce') * pd.to_numeric(
                            df_with_adj['adj_factor'], errors='coerce') / base_adj
                        df_with_adj[col] = adjusted_price

                    df_with_adj['pctChg'] = (df_with_adj['close'] / df_with_adj['close'].shift(1) - 1) * 100
                    df_to_process = df_with_adj.drop(columns=['trade_date_dt', 'adj_factor'])

        df_cleaned = robust_clean_dataframe(df_to_process)
        if df_cleaned.empty: return False

        print(f"  -> 正在为 {security_name} 计算技术指标...")
        df_final = calculate_technical_indicators(df_cleaned)

        if len(df_final) > INDICATOR_WARMUP_PERIOD:
            df_final = df_final.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)

        df_final.dropna(axis=1, how='all', inplace=True)

        # --- 修复部分 ---
        # 替换 df_final.ffill(inplace=True)
        df_final = df_final.ffill()

        if df_final.empty:
            print(f"  -> 警告: 证券 {security_name} 处理后数据为空，跳过。")
            return False

        # --- 对不同列应用不同的小数位数 ---
        print("  -> 正在格式化数据列的小数位数...")
        cols_to_round_2 = ['open', 'high', 'low', 'close'] + [col for col in df_final.columns if 'ma' in col]
        cols_to_round_4 = ['pctChg', 'turn']

        for col in cols_to_round_2:
            if col in df_final.columns:
                df_final[col] = df_final[col].round(2)

        for col in cols_to_round_4:
            if col in df_final.columns:
                df_final[col] = df_final[col].round(4)

        x_scaler_path = os.path.join(pred_dir, 'x_scaler.gz')
        y_scaler_path = os.path.join(pred_dir, 'y_scaler.gz')

        if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
            print(f"  -> 首次运行，正在为 {security_name} 创建并保存数据归一化 scaler...");
            df_train_for_scaler = df_final[
                pd.to_datetime(df_final['date']) <= pd.to_datetime(user_train_end_date_str)].copy()
            if df_train_for_scaler.empty:
                print(f"  -> 警告: {security_name} 没有足够的训练数据来拟合 scaler，跳过。")
                return False

            target_column = 'close'
            feature_columns = [col for col in df_train_for_scaler.columns if col not in ['date', 'direction']]
            print(f"  -> 用于创建 x_scaler 的特征列: {feature_columns}")

            x_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaler = MinMaxScaler(feature_range=(0, 1))

            x_scaler.fit(df_train_for_scaler[feature_columns].values)
            y_scaler.fit(df_train_for_scaler[[target_column]].values)

            joblib.dump(x_scaler, x_scaler_path)
            joblib.dump(y_scaler, y_scaler_path)
            print(f"  -> Scaler 已成功保存至: {pred_dir}")

        if 'direction' in df_final.columns:
            df_final = df_final.drop(columns=['direction'])

        df_final.to_csv(pred_fp, index=False, na_rep='')
        print(f"  -> 完整预测数据已更新并保存: {pred_fp} ({len(df_final)} 条)")

        if not df_final.empty:
            with open(pred_fp.replace('.csv', '.meta'), 'w') as f:
                json.dump({"last_date": df_final['date'].max()}, f)

        df_train = df_final[pd.to_datetime(df_final['date']) <= pd.to_datetime(user_train_end_date_str)].copy()
        df_train.to_csv(train_fp, index=False, na_rep='')
        print(f"  -> 训练数据已更新并保存: {train_fp} ({len(df_train)} 条)")

        return True
    except Exception as e:
        print(f"证券 {security_name} ({ts_code}): 处理失败 - {e}");
        traceback.print_exc();
        return False


# --- 工作进程函数（无修改）---
def _worker_fetch_data(security_item: dict, global_trade_cal_df: pd.DataFrame, all_securities_basic_df: pd.DataFrame,
                       predict_end_date_str: str) -> dict:
    pro_instance = get_tushare_pro_instance()
    name_with_suffix, category = security_item['name'], security_item['sector']
    ts_code, sec_type, industry, full_name = get_security_info_from_name(name_with_suffix, all_securities_basic_df,
                                                                         target_category=category)
    if not ts_code:
        return {'name': full_name, 'ts_code': 'N/A', 'category': category, 'status': 'skipped_no_ts_code'}
    dir_category = industry if industry and pd.notna(industry) else category
    success = fetch_and_process_stock_data(ts_code, sec_type, dir_category, full_name, TRAIN_DATA_FETCH_START_DATE,
                                           predict_end_date_str, USER_TRAIN_END_DATE, global_trade_cal_df, pro_instance)
    return {'name': full_name, 'ts_code': ts_code, 'category': category, 'status': 'success' if success else 'failed'}


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="并行获取和处理股票、指数、基金数据。")
    parser.add_argument('--num_cores', type=int, default=min(32, multiprocessing.cpu_count()), help=f"指定并行核心数。")
    args = parser.parse_args()
    end_date_str = datetime.now().strftime('%Y%m%d') if str(PREDICT_DATA_USER_END_DATE).lower() == 'latest' else str(
        PREDICT_DATA_USER_END_DATE)
    print("正在获取所有证券基础信息...")
    try:
        stock_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry');
        stock_df['type'] = 'stock'
        markets = ['CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'MSCI', 'OTH'];
        index_dfs = [pro.index_basic(market=m, fields='ts_code,name') for m in markets]
        index_df = pd.concat(index_dfs, ignore_index=True).drop_duplicates(subset=['ts_code']);
        index_df['type'] = 'index';
        index_df['industry'] = '指数'
        fund_df = pro.fund_basic(market='E', fields='ts_code,name');
        fund_df['type'] = 'fund';
        fund_df['industry'] = '基金'
        all_securities_basic = pd.concat(
            [stock_df[['ts_code', 'name', 'industry', 'type']], index_df[['ts_code', 'name', 'industry', 'type']],
             fund_df[['ts_code', 'name', 'industry', 'type']]], ignore_index=True)
        if all_securities_basic.empty: print("错误: 未能获取任何证券基础信息。"); sys.exit(1)
    except Exception as e:
        print(f"错误: 获取证券基础信息失败: {e}");
        sys.exit(1)
    print("正在加载交易日历...")
    try:
        cal_end = (datetime.strptime(end_date_str, '%Y%m%d') + timedelta(days=365)).strftime('%Y%m%d')
        GLOBAL_TRADE_CAL_DF = pro.trade_cal(exchange='SSE', start_date=TRAIN_DATA_FETCH_START_DATE, end_date=cal_end)
        GLOBAL_TRADE_CAL_DF.rename(columns={'cal_date': 'trade_date'}, inplace=True)
        GLOBAL_TRADE_CAL_DF['trade_date_dt'] = pd.to_datetime(GLOBAL_TRADE_CAL_DF['trade_date'])
        GLOBAL_TRADE_CAL_DF['is_open'] = pd.to_numeric(GLOBAL_TRADE_CAL_DF['is_open'])
    except Exception as e:
        print(f"错误: 获取交易日历失败: {e}");
        sys.exit(1)
    securities_to_process = [{'name': name, 'sector': cat} for cat, names in stock_info_dict.items() for name in names]

    num_cores = min(args.num_cores, len(securities_to_process))
    print(f"\n--- 开始并行处理 {len(securities_to_process)} 支证券, 使用 {num_cores} 个进程 ---")
    tasks = [(item, GLOBAL_TRADE_CAL_DF, all_securities_basic, end_date_str) for item in securities_to_process]

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(pool.starmap(_worker_fetch_data, tasks))
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_list = [f"{r['name']} ({r['ts_code']})" for r in results if r['status'] == 'failed']
    skipped_list = [r['name'] for r in results if r['status'] == 'skipped_no_ts_code']
    print("\n--- 证券数据获取与处理完成总结 ---")
    print(f"总计: {len(results)}, 成功: {success_count}, 失败: {len(failed_list)}, 跳过: {len(skipped_list)}")
    if failed_list: print(f"  失败列表: {', '.join(failed_list)}")
    if skipped_list: print(f"  跳过列表 (未找到代码): {', '.join(skipped_list)}")
    print("脚本执行完毕。")