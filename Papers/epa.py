# %%
# 数据来自 https://www.kaggle.com/vijayuv/onlineretail

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use("classic")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{ebgaramond}",
    }
)


# %%
df = pd.read_csv("OnlineRetail.csv", encoding="cp1252", parse_dates=["InvoiceDate"])
df = df.sort_values(by="InvoiceDate")

len(df)


# %%
df.sample(10)


# %%
# 负数异常数据剔除
df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]


# %%
# 查看同一股票代码的不同产品描述有多少种
df.groupby("StockCode").Description.nunique().sort_values(ascending=False)


# %%
# 剔除异常产品数据
df = df[
    ~df.StockCode.isin(["POST", "DOT", "M", "AMAZONFEE", "BANK CHARGES", "C2", "S"])
]
len(df)


# %%
# 清洗控制变量
df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate)
df["Date"] = pd.to_datetime(df.InvoiceDate.dt.date)
df["revenue"] = df.Quantity * df.UnitPrice


# %%
# 1. 先进行聚合 (为了代码整洁，建议先处理数据)
df_plot = df.groupby("Date").agg(
    {"Quantity": "sum", "revenue": "sum", "InvoiceNo": "nunique"}
)

# 2. 重命名列名 (让图例更漂亮)
df_plot.columns = ["Total Quantity", "Total Revenue", "Number of Orders"]

# 3. 绘图 (获取 ax 对象)
# 注意：Pandas 绘图返回的是左轴 (Primary Axes)
ax = df_plot.plot(
    secondary_y="Number of Orders",  # 指定这一列用右轴
    figsize=(10, 6),  # 加宽画布
    title="Daily Trends: Items, Revenue, and Orders",
)

# 4. 设置 Y 轴标签
ax.set_ylabel("Quantity \& Revenue (Left Axis)")
ax.right_ax.set_ylabel(
    "Number of Orders (Right Axis)"
)  # pandas 把右轴藏在 ax.right_ax 里

# =======================================================
# 5. 【核心修改】合并左右轴的图例
# =======================================================
# 获取左轴的句柄和标签
h1, l1 = ax.get_legend_handles_labels()
# 获取右轴的句柄和标签
h2, l2 = ax.right_ax.get_legend_handles_labels()

# 统一在左轴上绘制图例
ax.legend(
    h1 + h2,
    l1 + l2,
    loc="center left",
    bbox_to_anchor=(1.15, 0.5),
)

# =======================================================
# 6. 保存
# =======================================================
plt.grid()  # 加个网格更好看
plt.tight_layout()
plt.savefig(
    "daily_trends_dual_axis.pdf",
)
plt.show()


# %%
# 剔除异常偏差值
df = (
    df.assign(
        dNormalPrice=lambda d: d.UnitPrice
        / d.groupby("StockCode").UnitPrice.transform("median")
    )
    .pipe(lambda d: d[(d["dNormalPrice"] > 1.0 / 3) & (d["dNormalPrice"] < 3.0)])
    .drop(columns=["dNormalPrice"])
)


# %%
df = df.groupby(["Date", "StockCode", "Country"], as_index=False).agg(
    {"Description": "first", "Quantity": "sum", "revenue": "sum"}
)
df["Description"] = df.groupby("StockCode").Description.transform("first")
df["UnitPrice"] = df["revenue"] / df["Quantity"]


# %%
df["Date"] = pd.to_datetime(df["Date"])

df = df.assign(
    month=lambda d: d.Date.dt.month,
    DoM=lambda d: d.Date.dt.day,
    DoW=lambda d: d.Date.dt.weekday,
    stock_age_days=lambda d: (
        d.Date - d.groupby("StockCode").Date.transform("min")
    ).dt.days,
    sku_avg_p=lambda d: d.groupby("StockCode").UnitPrice.transform(
        "median"
    ),  # Stock Keeping Unit Average Price, 即库存量单位平均价格
)


# %%
def plot_feature_distributions(df):
    """
    绘制 stock_age_days 和 sku_avg_p 的分布图
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # --- Plot 1: Stock Age Days (产品生命周期) ---
    # 直方图 + 核密度估计 (KDE)
    sns.histplot(
        data=df,
        x="stock_age_days",
        bins=30,
        kde=True,
        edgecolor=None,
        ax=axes[0],
    )
    axes[0].set_title("Distribution of Product Life Cycle")
    axes[0].set_xlabel("Days on Shelf (Stock Age)")
    axes[0].set_ylabel("Frequency")
    # --- Plot 2: SKU Median Price (价格分布) ---
    # 注意：零售价格通常是长尾分布，建议使用 Log Scale 才能看清
    sns.histplot(
        data=df,
        x="sku_avg_p",
        bins=30,
        kde=True,
        log_scale=True,  # 【关键】开启对数坐标，防止长尾挤在一起
        edgecolor=None,
        ax=axes[1],
    )
    axes[1].set_title("Distribution of Baseline Price (Log Scale)")
    axes[1].set_xlabel("Median Price (Log Scale)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("feature_distributions.pdf")
    plt.show()


# %%
# 调用函数
plot_feature_distributions(df)


# %%
df.set_index(["Date", "StockCode", "Country"]).sample(5)


# %%
df_mdl = df[(df.groupby("StockCode").UnitPrice.transform("std") > 0)]
df_mdl


# %%
# 将单价和数量取log
df_mdl = df_mdl.assign(
    LnP=np.log(df_mdl["UnitPrice"]),
    LnQ=np.log(df_mdl["Quantity"]),
)


# %%
# De-meaned

df_mdl["dLnP"] = np.log(df_mdl.UnitPrice) - np.log(
    df_mdl.groupby("StockCode").UnitPrice.transform("mean")
)
df_mdl["dLnQ"] = np.log(df_mdl.Quantity) - np.log(
    df_mdl.groupby("StockCode").Quantity.transform("mean")
)


# %%
df_mdl


# %%
# 清洗完毕，数据保存为csv
df_mdl.to_csv("OnlineRetail_clean.csv", index=False)
