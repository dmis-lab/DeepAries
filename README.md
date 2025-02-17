# DeepAries

This repository contains the source code for our paper, which introduces **\modelname**—a novel reinforcement learning framework for dynamic portfolio management. Our framework leverages Transformer-based forecasting to extract complex market signals and adapts rebalancing intervals in response to changing market conditions. It has been evaluated across multiple major markets and has shown superior risk-adjusted returns and lower drawdowns compared to traditional fixed-horizon rebalancing strategies.

## Abstract


We propose DeepAries, a novel reinforcement learning framework for portfolio management that dynamically adjusts rebalancing intervals based on prevailing market conditions. Unlike conventional fixed-horizon strategies, DeepAries leverages Transformer-based forecasting to extract complex market signals and employs an adaptive interval selection mechanism—choosing among representative horizons (1, 5, and 20 days)—to balance return pursuit against transaction cost mitigation. Extensive experiments across four major markets (DJ 30, FTSE 100, KOSPI, and CSI 300) demonstrate that our approach achieves superior risk-adjusted returns and lower drawdowns compared to fixed-frequency rebalancing. Furthermore, an interactive demo evaluation on real market data (September 2024 to January 2025) illustrates the practical benefits of adaptive rebalancing in providing timely portfolio updates and empowering investors with more informed decision-making. Overall, DeepAries offers a promising tool for modern, dynamic portfolio management. To enhance accessibility and reproducibility, we provide a live demo of DeepaReis at https://DeepAries.com/

## Features

- **Dynamic Rebalancing:** Adaptive selection among different rebalancing horizons (1, 5, and 20 days) to balance returns with transaction costs.
- **Transformer-Based Forecasting:** Extract complex market signals using state-of-the-art Transformer architectures.
- **Reinforcement Learning:** Incorporates a novel PPO-based reinforcement learning mechanism to dynamically adjust portfolio weights.
- **Multi-Market Evaluation:** Evaluated on DJ 30, FTSE 100, KOSPI, and CSI 300, demonstrating superior risk-adjusted returns.
- **Interactive Demo:** Access our live demo at [DeepAries.com](https://DeepAries.com/) for real-time portfolio management insights.

## Directory Structure


## Requirements

Ensure you have Python 3.7+ installed. The following packages are required:

- torch
- numpy
- pandas
- matplotlib
- scikit-learn
- yfinance

You can install the dependencies with:

```bash
pip install -r requirements.txt

This `README.md` file starts with a brief project overview, followed by the paper abstract, and then details the repository structure, requirements, setup, usage, and additional information. Adjust any sections as needed to fit your project specifics.
