# DeepAries

![teaser](./assets/teaser.png)

This repository contains the source code for our paper, which introduces **DeepAries**—a novel reinforcement learning framework for dynamic portfolio management. Our framework leverages Transformer-based forecasting to extract complex market signals and adapts rebalancing intervals in response to changing market conditions. It has been evaluated across multiple major markets and has shown superior risk-adjusted returns and lower drawdowns compared to traditional fixed-horizon rebalancing strategies.

## Abstract

We propose DeepAries, a novel reinforcement learning framework for portfolio management that dynamically adjusts rebalancing intervals based on prevailing market conditions. Unlike conventional fixed-horizon strategies, DeepAries leverages Transformer-based forecasting to extract complex market signals and employs an adaptive interval selection mechanism—choosing among representative horizons (1, 5, and 20 days)—to balance return pursuit against transaction cost mitigation. Extensive experiments across four major markets (DJ 30, FTSE 100, KOSPI, and CSI 300) demonstrate that our approach achieves superior risk-adjusted returns and lower drawdowns compared to fixed-frequency rebalancing. Furthermore, an interactive demo evaluation on real market data (September 2024 to January 2025) illustrates the practical benefits of adaptive rebalancing in providing timely portfolio updates and empowering investors with more informed decision-making. Overall, DeepAries offers a promising tool for modern, dynamic portfolio management. To enhance accessibility and reproducibility, we provide a live demo of DeepAries at [DeepAries.com](https://DeepAries.com/).

## Features

- **Dynamic Rebalancing:** Adaptive selection among different rebalancing horizons (1, 5, and 20 days) to balance returns with transaction costs.
- **Transformer-Based Forecasting:** Extract complex market signals using state-of-the-art Transformer architectures.
- **Reinforcement Learning:** Incorporates a novel PPO-based reinforcement learning mechanism to dynamically adjust portfolio weights.
- **Multi-Market Evaluation:** Evaluated on DJ 30, FTSE 100, KOSPI, and CSI 300, demonstrating superior risk-adjusted returns.
- **Interactive Demo:** Access our live demo at [DeepAries.com](https://DeepAries.com/) for real-time portfolio management insights.

## Usage

1. **Install Dependencies.**  
   This project requires several Python packages to run. All required packages are listed in the provided `requirements.txt` file.  
   To install these packages, create and activate a virtual environment, then run:
   
       pip install -r requirements.txt

   This will install packages such as `pandas`, `torch`, `yfinance`, and others essential for DeepAries.

2. **Data Preparation.**  
   By default, the project is set up to work with the DJ 30 market. When you run `main.py`, the code will use the provided DJ 30 ticker list to download market data via the yfinance package, store the data locally, and then proceed with the experiments.  
   If you wish to apply DeepAries to a different market, please prepare a ticker file (e.g., `complete_<market>_tickers.csv`) with the list of tickers for that market. Then, adjust the command-line arguments (such as `--market` and `--data`) in `main.py` accordingly before running the experiment.

3. **Run the Experiment.**  
   To start training and evaluation with the default settings (DJ 30 market), run:
   
       python main.py --market dj30 --data general --is_training 1

   Modify these arguments as needed to select other markets or data types.

4. **Pre-trained Models.**  
   Pre-trained models on both the original and open-source datasets are provided. You can perform inference directly using these models without retraining if desired.

## Dataset

The dataset is constructed using publicly available data downloaded via the yfinance package. Our dataset is organized in a three-dimensional format with shape (N, F, T), where:

- **N:** The number of stocks in the market.
- **F:** The number of features per stock (e.g., open, close, volume, etc.).
- **T:** The time window (i.e., a series of time steps representing historical data).

Data preprocessing includes normalization and label generation, ensuring that the data is robust and consistent for both forecasting and reinforcement learning experiments.

## Contributors

<table>
	<tr>
		<th>Name</th>		
		<th>Affiliation</th>
		<th>Email</th>
	</tr>
	<tr>
		<td>Jinkyu Kim</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>no100kill@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Hyunjung Yi</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>ruby3672@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Mogan Gim</td>		
		<td>Department of Biomedical Engineering,<br>Hankuk University of Foreign Studies, Yongin, South Korea</td>
		<td>gimmogan@hufs.ac.kr</td>
	</tr>
   <tr>
		<td>Donghee Choi*</td>		
		<td>School of Computer Science and Engineering, <br>Pusan National University, Busan, South Korea</td>
		<td>dchoi@pusan.ac.kr</td>
	</tr>
	<tr>
		<td>Jaewoo Kang*</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>kangj@korea.ac.kr</td>
	</tr>
</table>

- &ast;: *Corresponding Authors*
