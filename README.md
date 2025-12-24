# Stock Price Prediction using TensorFlow

A machine learning project for predicting stock prices using TensorFlow and deep learning techniques. This project demonstrates time series forecasting with technical indicators and neural networks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samhitha1504-007/Stock-Price-Prediction_Tensorflow/blob/main/PROJECTIO.ipynb)

## 📋 Overview

This project implements a stock price prediction model using TensorFlow, focusing on HCL Technologies (HCLTECH) stock data. The model uses various technical indicators and time series features to forecast future stock prices.

## ✨ Features

- **Data Preprocessing**
  - Missing value handling with linear interpolation
  - Feature scaling using MinMaxScaler and StandardScaler
  - Time series data transformation

- **Technical Indicators**
  - Moving Averages (MA_20, MA_200)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Return and Volatility calculations

- **Feature Engineering**
  - Lagged features (Previous Close, Previous Volume)
  - Percentage change calculations
  - Rolling window statistics
  - Time series windowing for sequential data

- **Model Architecture**
  - TensorFlow-based deep learning models
  - Time series windowing for sequential predictions
  - Configurable window size and forecast horizon

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow** - Deep learning framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Scikit-learn** - Data preprocessing and scaling

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Samhitha1504-007/Stock-Price-Prediction_Tensorflow.git
cd Stock-Price-Prediction_Tensorflow
```

2. Install required dependencies:
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

## 🚀 Usage

1. **Prepare your data**: Place your stock data CSV file (e.g., `HCLTECH.csv`) in the project directory.

2. **Run the Jupyter Notebook**:
```bash
jupyter notebook PROJECTIO.ipynb
```

3. **Follow the notebook steps**:
   - Load and preprocess data
   - Generate technical indicators
   - Create training windows
   - Train the model
   - Make predictions

## 📊 Data Requirements

The input CSV file should contain the following columns:
- `Date` - Trading date (format: DD-MM-YYYY)
- `Close` - Closing price
- `High` - Highest price
- `Volume` - Trading volume
- Other standard OHLC (Open, High, Low, Close) data

## 🔧 Key Parameters

- `WINDOW_SIZE`: Number of previous days to use for prediction (default: 7)
- `HORIZON`: Number of days ahead to predict (default: 1)
- Moving Average windows: 20 and 200 days
- RSI period: 14 days
- MACD parameters: 12 and 26 day exponential moving averages

## 📈 Model Features

The model uses the following features for prediction:
- Historical closing prices
- Moving averages (20-day and 200-day)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Previous day's close and volume
- Return percentage
- Volatility (30-day rolling standard deviation)

## 📝 Project Structure

```
Stock-Price-Prediction_Tensorflow/
│
├── PROJECTIO.ipynb          # Main Jupyter notebook with complete workflow
├── README.md                # Project documentation
└── HCLTECH.csv             # Sample stock data (user-provided)
```

## 🎯 Results

The project includes:
- Data visualization of stock prices over time
- Technical indicator plots
- Model performance metrics
- Price prediction outputs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Samhitha** - [@Samhitha1504-007](https://github.com/Samhitha1504-007)

Project Link: [https://github.com/Samhitha1504-007/Stock-Price-Prediction_Tensorflow](https://github.com/Samhitha1504-007/Stock-Price-Prediction_Tensorflow)

## 📄 License

This project is open source and available for educational purposes.

## ⚠️ Disclaimer

This project is for educational and research purposes only. Stock price prediction is inherently uncertain, and this model should not be used as the sole basis for investment decisions. Always consult with financial advisors before making investment choices.