import argparse
import torch
import random
import numpy as np
import pandas as pd
import os
import time
import uuid
from pathlib import Path
import yfinance as yf

from exp.exp_DeepAries import Exp_DeepAries
import torch.multiprocessing as mp
from utils.tools import initialize_logger, fix_seed
from data_provider import data_prep  # data_prep 모듈 내에 YfinancePreprocessor와 관련 함수들이 정의되어 있음

##############################################
# MAIN FUNCTION: DATA PREPARATION & TRAINING
##############################################
def main():
    parser = argparse.ArgumentParser(
        description='Transformer Family and DeepAries for Time Series Forecasting'
    )

    # [General Settings]
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='Model type. Options: [Transformer, iTransformer]')
    parser.add_argument('--is_training', type=int, default=1,
                        help='Training flag (1: train, 0: inference)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature parameter for softmax (used in model components)')

    # [Data Settings]
    parser.add_argument('--market', type=str, default='dj30',
                        help='Market dataset to use. Options: [dj30, kospi, csi300, ftse, etc.]')
    parser.add_argument('--data', type=str, default='general',
                        help='Data type. Options: [general, alpha158]')
    parser.add_argument('--root_path', type=str,
                        help='Root path for the dataset. Defaults to "./data/<market>/" if not provided')
    parser.add_argument('--data_path', type=str,
                        help='CSV filename of the dataset. Defaults to "<market>_data.csv" if not provided')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='Directory to save model checkpoints')

    # [Forecasting Task Settings]
    parser.add_argument('--valid_year', type=str, default='2020-12-31',
                        help='Validation period end date')
    parser.add_argument('--test_year', type=str, default='2021-12-31',
                        help='Test period start date')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=5,
                        help='Label length for decoder start token')
    parser.add_argument('--pred_len', type=int, default=20,
                        help='Prediction sequence length')
    parser.add_argument('--freq', type=str, default='d',
                        help='Frequency for time feature encoding (e.g., s, t, h, d, b, w, m)')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 10],
                        help='List of forecasting horizons for multi-horizon trading (e.g., 1 5 10)')

    # [Model Architecture Settings]
    parser.add_argument('--enc_in', type=int, help='Encoder input size (auto-detected from data)', required=False)
    parser.add_argument('--dec_in', type=int, help='Decoder input size (auto-detected from data)', required=False)
    parser.add_argument('--c_out', type=int, default=1, help='Output size')
    parser.add_argument('--d_model', type=int, default=512, help='Model (hidden) dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of the feed-forward network')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--output_attention', action='store_true',
                        help='Output attention weights from the encoder')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor (specific to certain models)')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time features encoding type. Options: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--use_adappo', type=bool, default=True,
                        help='Use Adappo (reinforcement learning) if True, else use supervised learning')

    # [Optimization Settings]
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for data loading')
    parser.add_argument('--itr', type=int, default=1,
                        help='Number of experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable automatic mixed precision training')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='Learning rate adjustment type')

    # [GPU Settings]
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Use GPU if available')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1',
                        help='Comma separated list of GPU device ids')

    # [Portfolio Management Settings]
    parser.add_argument('--fee_rate', type=float, default=0.0001,
                        help='Transaction fee rate')
    parser.add_argument('--complex_fee', action='store_true', default=False,
                        help='Enable complex fee calculation')
    parser.add_argument('--num_stocks', type=int, default=20,
                        help='Number of stocks to include in the portfolio')
    parser.add_argument('--total_stocks', type=int, required=False,
                        help='Total number of stocks in the dataset')

    args = parser.parse_args()

    # --- Automatically set data paths ---
    if not args.root_path:
        args.root_path = f'./data/{args.market}/'
    if not args.data_path:
        args.data_path = f'{args.market}_data.csv'

    # Create data directory if it doesn't exist
    data_dir = Path(args.root_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Use the last horizon as prediction length ---
    args.pred_len = args.horizons[-1]

    # --- Check if raw data file exists; if not, fetch and save ticker data ---
    raw_data_csv = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(raw_data_csv):
        tickers_csv = f'complete_{args.market.lower()}_tickers.csv'
        if os.path.exists(tickers_csv):
            tickers = [str(ticker) for ticker in pd.read_csv(tickers_csv)['ticker'].tolist()]
            print("Tickers to fetch:", data_prep.format_tickers(tickers))
            data_prep.fetch_and_save_ticker_data(
                ticker_list=tickers,
                output_csv=raw_data_csv,
                start_date='2000-01-01',
                end_date='2025-02-09',
                threshold=0.1
            )
        else:
            print(f"Ticker list file {tickers_csv} not found. Please provide ticker list.")
            return
    else:
        print(f"Raw data file {raw_data_csv} already exists.")

    # --- Check if processed data file exists; if not, generate features ---
    processed_data_csv = os.path.join(args.root_path, f"{args.market}_{args.data}_data.csv")
    if not os.path.exists(processed_data_csv):
        preprocessor = data_prep.YfinancePreprocessor(input_path=raw_data_csv, output_path=processed_data_csv)
        preprocessor.make_feature()
    else:
        print(f"Processed data file {processed_data_csv} already exists.")

    # --- Determine input dimensions and total stocks from processed data file ---
    try:
        data = pd.read_csv(processed_data_csv)
        # Assuming CSV contains date and tic columns plus features
        num_features = data.shape[1] - 2  # Excluding date and tic
        args.enc_in = num_features if args.enc_in is None else args.enc_in
        args.dec_in = num_features if args.dec_in is None else args.dec_in
        args.total_stocks = data['tic'].nunique()
        if (not args.num_stocks) or (args.num_stocks > args.total_stocks):
            args.num_stocks = args.total_stocks

        print(f"Detected {num_features} input features across {args.total_stocks} stocks. Using {args.num_stocks} stocks for training.")
        print(f"Setting enc_in={args.enc_in}, dec_in={args.dec_in}.")
    except Exception as e:
        print(f"Error loading data from {processed_data_csv}: {e}")
        return

    # --- Create a unique experiment identifier and result directory ---
    setting_components = [
        f"{args.model}",
        "DeepAries",
        args.market,
        args.data,
        f"num_stocks({args.num_stocks})",
        f"sl({args.seq_len})",
        f"pl({args.pred_len})"
    ]
    setting = "_".join(setting_components)
    unique_id = uuid.uuid4().hex[:8]
    unique_setting = f"{setting}_{unique_id}"
    result_dir = os.path.join("./results", unique_setting)
    os.makedirs(result_dir, exist_ok=True)

    # --- Initialize logger ---
    global logger
    logger = initialize_logger(result_dir)
    logger.info(f"Dataset root path: {args.root_path}")
    logger.info(f"Dataset file: {args.data_path}")

    # --- GPU Configuration ---
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # --- Set random seed for reproducibility ---
    if args.seed is None:
        args.seed = int(time.time()) % (2 ** 32)
        print(f"No seed provided. Generated random seed: {args.seed}")
    fix_seed(args.seed)

    # --- Run Training or Inference using DeepAries ---
    if args.is_training:
        exp = Exp_DeepAries(args, unique_setting)
        exp.train(unique_setting)
        logger.info(f"DeepAries Backtesting: {setting}")
        exp.backtest(unique_setting)
    else:
        exp = Exp_DeepAries(args, unique_setting)
        logger.info(f"DeepAries Backtesting: {setting}")
        exp.backtest(unique_setting, 1)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
