# ==========================================
# 1. Simulation: Data Generation
# ==========================================
def generate_simulation_data(
    n_items=50, n_time=100, true_theta=-2.0, noise_level=0.001, seed=42
):
    """Generate synthetic retail data with confounders (Quality & Seasonality)"""
    np.random.seed(seed)
    N = n_items * n_time

    # Confounders: Alpha (Item Fixed Effect), S (Seasonality)
    item_ids = np.repeat(np.arange(n_items), n_time)
    alpha_i = np.random.normal(0, 1, n_items)[item_ids]

    time_ids = np.tile(np.arange(n_time), n_items)
    seasonality = np.sin(time_ids / 10) + np.random.normal(0, 0.2, N)

    # DGP: Price (P) & Demand (Q)
    # P correlates with Alpha & S (Endogeneity)
    ln_P = 1.0 * alpha_i - 0.5 * seasonality + np.random.normal(0, 0.5, N)
    ln_Q = (
        true_theta * ln_P
        + 1.0 * alpha_i
        + 1.0 * seasonality
        + np.random.normal(0, 0.5, N)
    )

    # Simulate First-stage Residuals (with estimation noise)
    p_resid = (ln_P - (1.0 * alpha_i - 0.5 * seasonality)) + np.random.normal(
        0, noise_level, N
    )
    q_resid = (ln_Q - (1.0 * alpha_i + 1.0 * seasonality)) + np.random.normal(
        0, noise_level, N
    )

    return pd.DataFrame(
        {
            "item_id": item_ids,
            "ln_P": ln_P,
            "ln_Q": ln_Q,
            "P_resid": p_resid,
            "Q_resid": q_resid,
        }
    )


# ==========================================
# 2. Visualization: Binned Scatter Plot
# ==========================================
def binned_ols(df, x, y, n_bins=15, ax=None, color="blue", label=None, **kwargs):
    """Non-parametric binned scatter plot with linear fit"""
    df = df.copy()  # Avoid SettingWithCopyWarning
    df[x + "_bin"] = pd.qcut(df[x], n_bins, duplicates="drop")

    # Aggregation (De-noising)
    data = df.groupby(x + "_bin", observed=True)[[x, y]].mean().dropna()

    # OLS Fit on Binned Means
    model = sm.OLS(data[y], sm.add_constant(data[x])).fit()

    if ax:
        # Scatter (Binned Means)
        alpha = kwargs.pop("alpha", 0.4)
        ax.scatter(data[x], data[y], color=color, alpha=alpha, s=30, **kwargs)
        # Linear Fit
        x_pred = np.linspace(data[x].min(), data[x].max(), 100)
        ax.plot(
            x_pred,
            model.predict(sm.add_constant(x_pred)),
            color=color,
            linestyle="--",
            linewidth=2,
            label=label,
        )
    return model


# ==========================================
# 3. Model: High-Dimensional Feature Engineering
# ==========================================
feature_pipeline = ColumnTransformer(
    [
        ("StockCode", OneHotEncoder(handle_unknown="ignore"), ["StockCode"]),
        ("Date", OneHotEncoder(handle_unknown="ignore"), ["month", "DoM", "DoW"]),
        ("NLP", CountVectorizer(min_df=0.0025, ngram_range=(1, 3)), "Description"),
        ("Country", OneHotEncoder(handle_unknown="ignore"), ["Country"]),
        ("Numeric", StandardScaler(), ["stock_age_days", "sku_avg_p"]),
        ("Treatment", "passthrough", ["LnP"]),
    ],
    remainder="drop",
)


# ==========================================
# 4. Inference: Robust DML with Cross-Fitting
# ==========================================
def _estimate_elasticity_binned(t_res, y_res, t_raw, n_bins=15):
    """Calculate elasticity using Binned Means to reduce variance"""
    df = pd.DataFrame({"t": t_res, "y": y_res, "t_raw": t_raw})
    df["bin"] = pd.qcut(df["t"], n_bins, duplicates="drop")
    means = df.groupby("bin", observed=True).mean()

    # Robust DML (Neyman Orthogonal): Cov(T_res, Y_res) / Cov(T_res, T_raw)
    theta_robust = np.dot(means["t"], means["y"]) / np.dot(means["t"], means["t_raw"])
    # Naive DML: Cov(T_res, Y_res) / Var(T_res)
    theta_naive = np.dot(means["t"], means["y"]) / np.dot(means["t"], means["t"])
    return theta_robust, theta_naive


def dml_cross_fitting(df, model_t, model_y, col_t="dLnP", col_y="dLnQ", k=2):
    """Main DML Loop with k-Fold Cross-Fitting"""
    res_robust, res_naive, residuals = [], [], []

    for i, (idx_tr, idx_te) in enumerate(
        KFold(k, shuffle=True, random_state=42).split(df)
    ):
        df_tr, df_te = df.iloc[idx_tr], df.iloc[idx_te].copy()

        # 1. Nuisance Parameter Estimation
        # Note: clone() ensures models are reset for each fold
        model_t.fit(df_tr, df_tr[col_t])
        model_y.fit(df_tr, df_tr[col_y])

        # 2. Orthogonalization (Residual Calculation)
        res_t = df_te[col_t] - model_t.predict(df_te)
        res_y = df_te[col_y] - model_y.predict(df_te)

        # 3. Inference (Binned)
        theta_r, theta_n = _estimate_elasticity_binned(res_t, res_y, df_te[col_t])

        res_robust.append(theta_r)
        res_naive.append(theta_n)
        residuals.append(
            pd.DataFrame({"dLnP_res": res_t, "dLnQ_res": res_y, "Fold": i + 1})
        )

    # Aggregation
    df_res = pd.concat(residuals)
    theta_final = np.nanmean(res_robust)

    # Global RMSE Calculation
    rmse = np.sqrt(
        mean_squared_error(df_res["dLnQ_res"], df_res["dLnP_res"] * theta_final)
    )

    print(
        f"Robust DML: {theta_final:.4f} | Naive DML: {np.nanmean(res_naive):.4f} | RMSE: {rmse:.4f}"
    )
    return {"df_residuals": df_res, "avg_dml_elast": theta_final, "global_rmse": rmse}
