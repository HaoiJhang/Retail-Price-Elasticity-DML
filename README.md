# Price Elasticity Estimation via Double Machine Learning

## 📖 Overview

In microeconomics and modern business decision-making, **Price Elasticity of Demand** is a core metric for measuring market response. It reflects the sensitivity of consumer demand to price changes and directly determines a company's pricing strategy and profitability.

From a management perspective, accurate elasticity estimation provides critical guidance:

- **Inelastic Demand ($|\theta| < 1$):** Moderate price increases can raise total revenue.
- **Elastic Demand ($|\theta| > 1$):** Price reductions (promotions) become an effective means to gain market share and boost revenue.

Therefore, in a dynamic competitive market, accurately capturing the slope of the demand curve is key for manufacturers to maximize returns.

## 🚀 Motivation & Challenges

Although price elasticity is well-defined theoretically, empirical estimation faces severe challenges:

- **Limitations of A/B Testing:** While Randomized Controlled Trials (RCTs) are the ideal evaluation method, displaying differentiated prices to different users in real-world retail scenarios often damages user experience, triggers price discrimination controversies, and weakens brand reputation.
- **Endogeneity in Observational Data:** Causal inference based on historical data is a viable alternative. However, observational data is not randomly assigned. There are complex endogeneity issues between price and demand. Factors such as **seasonality**, **promotional cycles**, and **product quality changes** affect pricing behavior while simultaneously interfering with consumer decisions.
- **Bias in Traditional Models:** Failure to effectively strip away these confounding factors leads to severe bias in traditional statistical regression models, yielding spurious correlations rather than true causal effects.

## 🛠️ Methodology: Double Machine Learning (DML)

To address the challenges above, this project introduces a cutting-edge **Double Machine Learning (DML)** framework designed to extract unbiased price elasticity estimates from high-dimensional, non-linear historical transaction data.

Compared to traditional econometric models, this approach offers significant advantages in two key areas:

1. **High-Dimensional Variable Processing:** Utilizing regularization techniques and feature engineering, the model automatically selects important control variables from massive information sources—including stock codes, date features, text descriptions (NLP), and regional distributions—effectively solving overfitting caused by excessive covariates.
2. **Non-Linear Causal Modeling:** Traditional linear models struggle to capture complex pricing mechanisms. This study combines **Random Forests** (to capture non-linear interactions) and **Poisson Regression** (to handle the discrete count nature of sales data) within the DML framework, achieving more precise prediction and orthogonalization.

## 📊 Key Findings

This project conducts an empirical analysis using the **Kaggle Online Retail Dataset**.

- **Structural Recovery:** Experimental results show that orthogonalized residuals extracted via the DML framework can clearer restore the linear structure of the demand curve.
- **Error Reduction:** The method significantly reduces estimation error compared to baselines.
- **Business Value:** It provides robust quantitative support for enterprises to conduct "Intelligent Pricing" in complex environments.

## 📂 Project Structure

The repository is organized following the logic of the research paper:

- **Section 1: Introduction**
  - Overview of the problem scope and the necessity of DML.
- **Section 2: Theoretical Framework**
  - Constructs the econometric framework for price elasticity.
  - Provides mathematical derivations analyzing the bias mechanisms in **Ordinary Least Squares (OLS)**, **De-meaned (Fixed Effects)** models, and **Naive DML**.
  - Theoretically demonstrates the unbiasedness of the **Robust DML** estimator.
- **Section 3: Simulation Study**
  - Design of a controlled Monte Carlo simulation.
  - Verification of theoretical derivations under known ground-truth elasticity and artificially injected noise.
  - Evaluation of the robustness of different estimators under finite sample conditions.
- **Section 4: Data & Model Specification**
  - Preprocessing flow for the Kaggle retail data.
  - High-dimensional feature engineering based on **NLP** and **Time Series**.
  - Implementation details of the hybrid "**Random Forest + Poisson Regression**" model.
- **Section 5: Empirical Results**
  - Presentation of estimation results and model diagnostics.
  - Reconstruction of the demand curve via **Binned Scatter Plots**.
  - In-depth analysis of elasticity coefficients.
- **Section 6: Conclusion**
  - Summary of findings, managerial implications, and future outlook.

## 💻 Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, Statsmodels, Pandas, NumPy, Matplotlib/Seaborn
- **Core Algorithms:** Random Forest Regressor, Poisson Regressor, Double Machine Learning (DML)
