# DeepAries

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

1. **Install dependencies.**  
   Ensure you have:
   - `pandas==1.5.3`
   - `torch==1.11.0`  
   (Other required packages are listed in `requirements.txt`.)

2. **Download and unpack the data.**  
   The public dataset is built using data obtained via [yfinance](https://pypi.org/project/yfinance/). Download the data and unpack it into the `data/` directory.

3. **Run `main.py`.**  
   Select the desired dataset for training and evaluation.

4. **Pre-trained Models.**  
   We provide models trained on the original dataset as well as on open-source data.

## Dataset

### Form

The dataset is constructed using publicly available data from Yahoo Finance (yfinance). Data for various markets is downloaded, and stocks are grouped by prediction dates to form the training, validation, and test sets. Each data sample is of shape **(N, F)**, where:

- **N** - The number of stocks. This can include stocks from any market (e.g., DJ 30, FTSE 100, KOSPI, CSI 300, etc.).
- **F** - The number of features per stock. In our paper, we obtain features from yfinance and apply normalization for consistency and robustness.

You can inspect the data format using the following code snippet:
```python
import pickle

with open('data/original/sample_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)
    print(dl_train.data)  # A pandas DataFrame containing datetime, instrument, and feature columns
```

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
		<td>Department of Metabolism, Digestion and Reproduction,<br>Imperial College London, London, United Kingdom</td>
		<td>donghee.choi@imperial.ac.uk</td>
	</tr>
	<tr>
		<td>Jaewoo Kang*</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>kangj@korea.ac.kr</td>
	</tr>
</table>

- &ast;: *Corresponding Author*
