# Price Elasticity Estimation via Double Machine Learning

## 📖 Overview
This project estimates the price elasticity of demand for retail products using **Double Machine Learning (DML)**. It addresses the endogeneity of price and high-dimensional confounders (e.g., seasonality, product lifecycle) to provide robust causal inference.

## 📊 Key Results
- **Method**: Hybrid Residual Model (Random Forest for Price + Poisson for Demand).
- **Elasticity**: Estimated an average price elasticity of **-1.05** (Unit Elastic).
- **Validation**: Verified theoretical bias (Attenuation/OVB) through Monte Carlo Simulations.

## 🛠️ Tech Stack
- **Python**: Pandas, NumPy, Scikit-learn, Statsmodels
- **Causal Inference**: Double Machine Learning (DML), Neyman Orthogonality
- **Visualization**: Matplotlib (Binned Scatter Plots)

## 📂 Project Structure
- `data_cleaning.ipynb`: Preprocessing and Feature Engineering (NLP on descriptions, Time features).
- `dml_model.py`: Implementation of Robust DML with Cross-Fitting.
- `simulation.py`: Monte Carlo simulation to verify estimator bias.
- `figs/`: Visualizations of demand curves and sensitivity analysis.

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/YourUsername/retail-price-elasticity-dml.git
