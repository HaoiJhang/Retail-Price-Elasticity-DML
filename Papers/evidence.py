# %%
# ===== Packages: Baiscs =====

import pandas as pd
import numpy as np

import joblib
import statsmodels.api as sm

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

plt.style.use("classic")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{ebgaramond}",
    }
)


# %%
# 读取数据
df_mdl = pd.read_csv("OnlineRetail_clean.csv")

df_mdl


# %%
def binned_ols(
    df,
    x,
    y,
    n_bins,
    plot=True,
    plot_title="",
    plot_ax=None,
    color="blue",
    label=None,
    **plt_kwargs,
):
    """
    分箱回归并绘图
    """

    df = df.copy()

    # 1. 数据分箱与聚合
    x_bin = x + "_bin"

    df[x_bin] = pd.qcut(df[x], n_bins, duplicates="drop")

    tmp = df.groupby(x_bin, observed=True).agg({x: "mean", y: "mean"}).dropna()

    # 2. OLS 回归拟合
    mdl = sm.OLS(tmp[y], sm.add_constant(tmp[x]))
    res = mdl.fit()

    # 3. 绘图逻辑
    if plot:
        if plot_ax is None:
            fig, plot_ax = plt.subplots(figsize=(6, 4))

        # 处理 alpha 参数
        user_alpha = plt_kwargs.pop("alpha", None)
        scatter_alpha = user_alpha if user_alpha is not None else 0.4
        line_alpha = user_alpha if user_alpha is not None else 1.0

        # A. 绘制散点
        plot_ax.scatter(
            tmp[x], tmp[y], color=color, alpha=scatter_alpha, s=30, **plt_kwargs
        )

        # B. 绘制拟合曲线
        x_pred = tmp[x].sort_values()
        y_pred = res.predict(sm.add_constant(x_pred))

        plot_ax.plot(
            x_pred,
            y_pred,
            color=color,
            linestyle="--",
            linewidth=2,
            label=label,
            alpha=line_alpha,
        )

        if plot_title:
            plot_ax.set_title(plot_title)

    # 4. 清理
    del df[x_bin]
    return res


# %%
# === 绘制 Naive OLS 结果 ======
plt.figure(figsize=(10, 6))  # 创建画布

ols_fit = binned_ols(
    df_mdl,
    x="LnP",
    y="LnQ",
    n_bins=15,
    plot_ax=plt.gca(),  # 使用当前轴
    plot_title="Relationship between LnP and LnQ",
    label="Naive OLS",  # <--- 【关键】必须加这一行
)

# 设置图例在外部
plt.legend(loc="center left", bbox_to_anchor=(1.15, 0.5))


# 保存
plt.grid()
plt.tight_layout()
plt.savefig("observe_messy_relationship_LnP_LnQ.pdf")
plt.show()


# %%
print(ols_fit.summary())


# %%
# ===== 计算 MSE 与 RMSE ======
mse_val = ols_fit.mse_resid

rmse_val = np.sqrt(mse_val)

print(f"Binned MSE: {mse_val}")
print(f"Binned RMSE: {rmse_val}")


# %% [markdown]
# #### subsection 2.2 Possion and Ridge
#
# 最终柏松回归中LnP的回归系数为 -2.87559，Ridge—OLS回归中LnP的回归系数为 -1.79945，尝试下来各个方法得到的结果差异很大。

# %%
# ===== 生成基本的控制变量(特征) =====
feature_generator_basic = ColumnTransformer(
    [
        ("StockCode", OneHotEncoder(), ["StockCode"]),
        ("Date", OneHotEncoder(), ["Date"]),
        ("Country", OneHotEncoder(), ["Country"]),
        ("LnP", "passthrough", ["LnP"]),
    ],
    remainder="drop",
)


# %%
# ===== 构造 Poisson Regression(基本的控制变量) =====
mdl_poisson = Pipeline(
    [
        ("feat_proc", feature_generator_basic),
        (
            "reg",
            linear_model.PoissonRegressor(
                alpha=1e-6,  # l2 penalty strength; manually selected value for minimum interference on LnP-coef (elasticity)
                fit_intercept=False,  # no need, since we have OneHot encodings without drop
                max_iter=100_000,
            ),
        ),
    ],
    verbose=True,
)


# %%
# ===== 构造 Ridge Regression(基本的控制变量) =====
mdl_ridge = Pipeline(
    [
        ("feat_proc", feature_generator_basic),
        (
            "reg",
            linear_model.Ridge(
                alpha=1e-20,  # l2 penalty strength
                fit_intercept=False,
                max_iter=100_000,
            ),
        ),
    ],
    verbose=True,
)


# %%
# ===== Poisson with 基本控制变量的训练 =====
mdl_poisson.fit(
    df_mdl[["LnP", "StockCode", "Date", "Country"]],
    df_mdl[
        "Quantity"
    ],  # Poisson regression has log-link, so LnQ is implicit in loss function
)


# %%
# ===== Ridge with 基本控制变量的训练 =====
mdl_ridge.fit(
    df_mdl[["LnP", "StockCode", "Date", "Country"]],
    df_mdl["LnQ"],  # log-normal
)


# %%
# ===== Poisson 回归的结果参数 =====
print(
    '"Econometrically" estimated elasticity with unit-, time-, and market- controls, using Poisson loss (coef on LnP):'
)

pd.DataFrame(
    {
        "feat": mdl_poisson["feat_proc"].get_feature_names_out(),
        "coef": mdl_poisson["reg"].coef_,
    }
).iloc[-1]


# %%
# ===== Ridge 回归的结果参数 =====
print(
    '"Econometrically" estimated elasticity with unit-, time-, and market- controls, using log-Normal loss (coef on LnP):'
)
pd.DataFrame(
    {
        "feat": mdl_ridge["feat_proc"].get_feature_names_out(),
        "coef": mdl_ridge["reg"].coef_,
    }
).iloc[-1]


# %%
# ===== 计算 MSE 与 RMSE =====

# ==========================================
# 1. 准备数据与真实值 (Ground Truth)
# ==========================================
# 特征矩阵 (两个模型用的是一样的特征)
X = df_mdl[["LnP", "StockCode", "Date", "Country"]]

# 真实值 (必须统一还原到 "原始销量" 才能进行公平对比)
# 假设 df_mdl['LnQ'] 是 log(Quantity)
y_true_original = np.exp(df_mdl["LnQ"])


# ==========================================
# 2. 计算 OLS 模型的 MSE
# ==========================================
# OLS 训练目标是 LnQ，所以预测输出也是 log 尺度
pred_log_ols = mdl_ridge.predict(X)

# 【关键步骤】必须手动指数还原
pred_ols_original = np.exp(pred_log_ols)

# 计算 OLS 在原始尺度下的 MSE
mse_ols = mean_squared_error(y_true_original, pred_ols_original)


# ==========================================
# 3. 计算 Poisson 模型的 MSE
# ==========================================
# Poisson 训练时，y 通常直接输入原始 Quantity (或者 sklearn 内部处理 link function)
# 【关键区别】predict() 方法默认返回 E[y]，也就是预测的“销量数值”
# 不需要手动 np.exp()，sklearn 已经帮你做了 inverse link function
pred_poisson_original = mdl_poisson.predict(X)

# 直接计算 Poisson 在原始尺度下的 MSE
mse_poisson = mean_squared_error(y_true_original, pred_poisson_original)


# ==========================================
# 4. 汇报与打印
# ==========================================
print("----- Model Comparison (Original Scale) -----")
print(f"1. OLS MSE     : {mse_ols:.4f}")
print(f"2. Poisson MSE : {mse_poisson:.4f}")
print("-" * 30)
print(f"1. OLS RMSE     : {np.sqrt(mse_ols):.4f} (件)")
print(f"2. Poisson RMSE : {np.sqrt(mse_poisson):.4f} (件)")

# 简单的判断逻辑
if mse_poisson < mse_ols:
    print("\n[结论]: Poisson 模型在预测实际销量上更准确。")
else:
    print("\n[结论]: OLS 模型表现更好（可能数据分布并不完全符合泊松假设）。")


# %%
# ===== 考虑更多的混杂因素 =====

feature_generator_full = ColumnTransformer(
    [
        (
            "StockCode",
            OneHotEncoder(handle_unknown="ignore"),
            ["StockCode"],
        ),
        (
            "Date",
            OneHotEncoder(),
            [
                "month",
                "DoM",
                "DoW",
            ],
        ),
        (
            "Description",
            CountVectorizer(min_df=0.0025, ngram_range=(1, 3)),
            "Description",
        ),
        ("Country", OneHotEncoder(), ["Country"]),
        (
            "numeric_feats",
            StandardScaler(),
            ["stock_age_days", "sku_avg_p"],
        ),
        ("LnP", "passthrough", ["LnP"]),
    ],
    remainder="drop",
)


# %%
# ===== 构造 Poisson Regression(更多的控制变量) =====

mdl_poisson_full = Pipeline(
    [
        ("feat_proc", feature_generator_full),
        (
            "reg",
            linear_model.PoissonRegressor(
                alpha=1e-6,  # l2 penalty strength; manually selected value for minimum interference on LnP-coef (elasticity)
                fit_intercept=False,  # no need, since we have OneHot encodings without drop
                max_iter=100_000,
            ),
        ),
    ],
    verbose=True,
)


# %%
# ===== 构造 Ridge Regression(更多的控制变量) =====
mdl_ridge_full = Pipeline(
    [
        ("feat_proc", feature_generator_full),
        (
            "reg",
            linear_model.Ridge(
                alpha=1e-20,  # l2 penalty strength
                fit_intercept=False,
                max_iter=100_000,
            ),
        ),
    ],
    verbose=True,
)


# %%
# 定义需要的特征列列表
feature_cols = [
    "LnP",  # Treatment
    "StockCode",
    "Country",  # Confounder (Region FE)
    "Description",  # Confounder (Text Features)
    "month",
    "DoM",
    "DoW",
    "stock_age_days",
    "sku_avg_p",  # Confounders (Numeric)
]


# %%
# ===== Poisson with 更多控制变量的训练 =====
mdl_poisson_full.fit(df_mdl[feature_cols], df_mdl["Quantity"])


# %%
# ===== Poisson 回归的结果参数 =====
print(
    '"Econometrically" estimated elasticity with unit-, time-, and market- controls, using Poisson loss (coef on LnP):'
)

pd.DataFrame(
    {
        "feat": mdl_poisson_full["feat_proc"].get_feature_names_out(),
        "coef": mdl_poisson_full[-1].coef_,
    }
).iloc[-1]


# %%
# ===== Ridge with 基本控制变量的训练 =====
mdl_ridge_full.fit(df_mdl[feature_cols], df_mdl["LnQ"])


# %%
# ===== Ridge 回归的结果参数 =====
print(
    '"Econometrically" estimated elasticity with unit-, time-, and market- controls, using log-Normal loss (coef on LnP):'
)
pd.DataFrame(
    {
        "feat": mdl_ridge["feat_proc"].get_feature_names_out(),
        "coef": mdl_ridge["reg"].coef_,
    }
).iloc[-1]


# %%
# ===== 计算 MSE 与 RMSE =====

# ==========================================
# 1. 准备数据与真实值 (Ground Truth)
# ==========================================
# 特征矩阵
X = df_mdl[feature_cols]

# 真实值
y_true_original = np.exp(df_mdl["LnQ"])


# ==========================================
# 2. 计算 OLS 模型的 MSE
# ==========================================
# OLS 训练目标是 LnQ，所以预测输出也是 log 尺度
pred_log_ols = mdl_ridge_full.predict(X)

pred_ols_original = np.exp(pred_log_ols)

# 计算 OLS 在原始尺度下的 MSE
mse_ols = mean_squared_error(y_true_original, pred_ols_original)


# ==========================================
# 3. 计算 Poisson 模型的 MSE
# ==========================================

pred_poisson_original = mdl_poisson_full.predict(X)

# 直接计算 Poisson 在原始尺度下的 MSE
mse_poisson = mean_squared_error(y_true_original, pred_poisson_original)


# ==========================================
# 4. 汇报与打印
# ==========================================
print("----- Model Comparison (Original Scale) -----")
print(f"1. OLS MSE     : {mse_ols:.4f}")
print(f"2. Poisson MSE : {mse_poisson:.4f}")
print("-" * 30)
print(f"1. OLS RMSE     : {np.sqrt(mse_ols):.4f} (件)")
print(f"2. Poisson RMSE : {np.sqrt(mse_poisson):.4f} (件)")

# 简单的判断逻辑
if mse_poisson < mse_ols:
    print("\n[结论]: Poisson 模型在预测实际销量上更准确。")
else:
    print("\n[结论]: Ridge 模型表现更好（可能数据分布并不完全符合泊松假设）。")


# %%
feats_rf = ColumnTransformer(
    [
        (
            "StockCode",
            OneHotEncoder(handle_unknown="ignore"),
            ["StockCode"],
        ),
        (
            "Date",
            OneHotEncoder(),
            [
                "month",
                "DoM",
                "DoW",
            ],
        ),
        (
            "Description",
            CountVectorizer(min_df=0.0025, ngram_range=(1, 3)),
            "Description",
        ),
        ("Country", OneHotEncoder(), ["Country"]),
        (
            "numeric_feats",
            StandardScaler(),
            ["stock_age_days", "sku_avg_p"],
        ),
    ],
    remainder="drop",
)


# %%
feature_cols = [
    "StockCode",
    "Country",  # Confounder (Region FE)
    "Description",  # Confounder (Text Features)
    "month",
    "DoM",
    "DoW",
    "stock_age_days",
    "sku_avg_p",  # Confounders (Numeric)
]


# %%
model_q = Pipeline(
    [
        ("feat_proc", feats_rf),
        (
            "model_q",
            RandomForestRegressor(
                n_estimators=50, min_samples_leaf=3, n_jobs=-1, verbose=2
            ),
        ),
    ]
)
model_p = Pipeline(
    [
        ("feat_proc", feature_generator_full),
        (
            "model_p",
            RandomForestRegressor(
                n_estimators=50, min_samples_leaf=3, n_jobs=-1, verbose=2
            ),
        ),
    ]
)


# %%
# 1. 训练 Y 模型（用混淆因子预测销量）

# model_q.fit(df_mdl[feature_cols], df_mdl["LnQ"])
# joblib.dump(model_q, "DML_model_q.pkl")


# %%

# 2. 训练 P 模型（用混淆因子预测价格）

# model_p.fit(df_mdl, df_mdl["LnP"])
# joblib.dump(model_p, "DML_model_p.pkl")


# %%

model_q = joblib.load("DML_model_q.pkl")
model_p = joblib.load("DML_model_p.pkl")

# 3. 现在模型已经训练好了，可以进行预测 (计算残差的基础)
q_hat = model_q.predict(df_mdl)  # 预测出的销量基准
p_hat = model_p.predict(df_mdl)  # 预测出的价格基准


# %%
# 计算正交残差
df_mdl = df_mdl.assign(
    dLnP_res=df_mdl["dLnP"] - p_hat,
    dLnQ_res=df_mdl["dLnQ"] - q_hat,
)


# %%
# 3-Stage

df_mdl[["LnP", "LnQ", "dLnP", "dLnQ", "dLnP_res", "dLnQ_res"]].sample(5)


# %%
# 设置画布大小
plt.figure(figsize=(10, 6))

# --- Stage 1: 原始数据 (Raw) ---

fit_raw = binned_ols(
    df_mdl,
    x="LnP",
    y="LnQ",
    n_bins=15,
    plot_ax=plt.gca(),
    color="tab:gray",  # 颜色
    label="Raw Data",  # 图例
    plot_title="Demand Curve Reconstruction",
)

# --- Stage 2: 去均值化 (De-meaned) ---
fit_demean = binned_ols(
    df_mdl,
    x="dLnP",
    y="dLnQ",
    n_bins=15,
    plot_ax=plt.gca(),
    color="tab:blue",
    label="De-meaned (FE only)",
)

# --- Stage 3: DML 残差 (DML Residuals) ---
fit_dml = binned_ols(
    df_mdl,
    x="dLnP_res",
    y="dLnQ_res",
    n_bins=15,
    plot_ax=plt.gca(),
    color="tab:red",
    label="DML Residuals",
)


plt.gca().set(
    xlabel="Log Price (Centered / Residualized)",
    ylabel="Log Quantity (Centered / Residualized)",
)


plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.grid()
plt.tight_layout()
plt.savefig("demand_curve_comparison.pdf")

plt.show()


# %%


# --- 3. 计算并打印 MSE 和 RMSE ---
def report_metrics(stage_name, res, slope_var_name):
    mse = res.mse_resid
    rmse = np.sqrt(mse)
    try:
        elasticity = res.params[slope_var_name]
    except KeyError:
        elasticity = res.params.iloc[-1]

    print(f"[{stage_name}]")
    print(f"  Elasticity (Slope): {elasticity:.4f}")
    print(f"  Binned MSE        : {mse:.4f}")
    print(f"  Binned RMSE       : {rmse:.4f}")
    print("-" * 30)


print("\n=== Model Diagnostics (Binned Fit) ===")
report_metrics("Stage 1: Raw", fit_raw, "LnP")
report_metrics("Stage 2: De-meaned", fit_demean, "dLnP")
report_metrics("Stage 3: DML", fit_dml, "dLnP_res")


# %%
def get_feat_generator_names(gen):
    res = []
    for i, (k, t) in enumerate(gen.named_transformers_.items()):
        if k == "remainder":
            continue

        res += [f"{k}_{n}" for n in t.get_feature_names()]

    return res


# %%
# Plot: 绘制价格模型的重要特征
feat_names = model_p.named_steps["feat_proc"].get_feature_names_out()

feat_imp = pd.DataFrame(
    {
        "feat": feat_names,
        "importance_q": model_q.named_steps["model_q"].feature_importances_,
        "importance_p": model_p.named_steps["model_p"].feature_importances_,
    }
).set_index("feat")

feat_imp.sort_values(by="importance_p").iloc[-5:].plot.barh(
    title="Feature Importances: DML Nuisance Estimators (Price vs Quantity)",
)
plt.tight_layout()
plt.show()
