"""
Algeria Food Price Intelligence System
Standalone version for Streamlit Cloud deployment.

Deploy: https://share.streamlit.io
No local files, no external dependencies beyond pip packages.
"""

from __future__ import annotations

import io
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Algeria Food Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

ALGERIA_GREEN = "#006233"

WILAYA_COORDS = {
    "Alger":        (36.74,  3.06),
    "Oran":         (35.70, -0.63),
    "Constantine":  (36.37,  6.61),
    "Annaba":       (36.90,  7.77),
    "Tlemcen":      (34.88, -1.32),
    "Sétif":        (36.18,  5.41),
    "Batna":        (35.56,  6.17),
    "Tindouf":      (27.67, -8.15),
    "Blida":        (36.47,  2.83),
    "Ouargla":      (31.95,  5.32),
    "Ghardaïa":     (32.49,  3.67),
    "Tizi Ouzou":   (36.72,  4.05),
    "Béjaïa":       (36.75,  5.06),
    "Laghouat":     (33.80,  2.86),
    "Biskra":       (34.85,  5.73),
    "Tamanrasset":  (22.79,  5.52),
    "Médéa":        (36.27,  2.75),
    "Mostaganem":   (35.93,  0.09),
    "Skikda":       (36.87,  6.91),
    "Jijel":        (36.82,  5.77),
}

PRODUCTS = {
    "Tomatoes":      ("Légumes",   75,   0.18),
    "Potatoes":      ("Légumes",   52,   0.14),
    "Onions":        ("Légumes",   58,   0.20),
    "Carrots":       ("Légumes",   65,   0.15),
    "Chicken":       ("Protéines", 345,  0.11),
    "Meat (lamb)":   ("Protéines", 1200, 0.09),
    "Eggs":          ("Protéines", 200,  0.12),
    "Bread":         ("Céréales",  28,   0.05),
    "Semolina":      ("Céréales",  65,   0.06),
    "Wheat flour":   ("Céréales",  60,   0.07),
    "Rice":          ("Céréales",  180,  0.09),
    "Vegetable Oil": ("Épicerie",  215,  0.13),
    "Sugar":         ("Épicerie",  88,   0.07),
    "Milk":          ("Épicerie",  90,   0.08),
    "Lentils":       ("Épicerie",  140,  0.10),
}

# Approximate Ramadan start months (month index) by year
RAMADAN = {
    2015: (6, 17), 2016: (6, 5), 2017: (5, 26), 2018: (5, 15),
    2019: (5, 5),  2020: (4, 23), 2021: (4, 12), 2022: (4, 1),
    2023: (3, 22), 2024: (3, 10),
}


# ══════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════

def _is_ramadan(dt: pd.Timestamp) -> bool:
    entry = RAMADAN.get(dt.year)
    if not entry:
        return False
    start = pd.Timestamp(dt.year, entry[0], entry[1])
    return start <= dt <= start + pd.Timedelta(days=29)


def _get_season(month: int) -> str:
    return {12: "Hiver", 1: "Hiver", 2: "Hiver",
            3: "Printemps", 4: "Printemps", 5: "Printemps",
            6: "Été", 7: "Été", 8: "Été"}.get(month, "Automne")


@st.cache_data(show_spinner="Chargement des données…")
def load_data() -> pd.DataFrame:
    """Generate realistic synthetic food price data for all regions."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    rows = []

    for region, (lat, lon) in WILAYA_COORDS.items():
        # Regional price modifier (±15%)
        region_factor = rng.uniform(0.90, 1.15)

        for product, (category, base_price, volatility) in PRODUCTS.items():
            price = base_price * region_factor

            for dt in dates:
                # Annual inflation trend (~6%)
                annual_growth = 1 + rng.normal(0.06, 0.015)
                price = price * (annual_growth ** (1 / 12))

                # Seasonal variation
                seasonal = 1 + 0.12 * np.sin(2 * np.pi * (dt.month - 3) / 12)

                # Random noise
                noise = rng.normal(1.0, volatility)

                # Ramadan demand bump
                ram_bump = rng.uniform(1.08, 1.22) if _is_ramadan(dt) else 1.0

                # Rare price spike (1.5% chance — anomaly)
                is_spike = rng.random() < 0.015
                spike = rng.uniform(1.35, 2.0) if is_spike else 1.0

                final = max(round(price * seasonal * noise * ram_bump * spike, 2), 5.0)

                rows.append({
                    "date":       dt,
                    "product":    product,
                    "category":   category,
                    "region":     region,
                    "latitude":   lat,
                    "longitude":  lon,
                    "price":      final,
                    "is_ramadan": int(_is_ramadan(dt)),
                    "season":     _get_season(dt.month),
                    "month":      dt.month,
                    "year":       dt.year,
                    "is_spike":   is_spike,
                })

    df = pd.DataFrame(rows)
    df["pct_change_1m"]  = df.groupby(["product", "region"])["price"].pct_change() * 100
    df["pct_change_12m"] = df.groupby(["product", "region"])["price"].pct_change(12) * 100
    df["rolling_3m"]     = df.groupby(["product", "region"])["price"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    return df.sort_values("date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Détection des anomalies…")
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Run Isolation Forest per (product, region) group."""
    df = df.copy()
    df["anomaly_score"] = 0.0
    df["is_anomaly"]    = False

    for (prod, reg), grp in df.groupby(["product", "region"]):
        if len(grp) < 12:
            continue
        X = grp[["price", "pct_change_1m", "rolling_3m"]].fillna(0).values
        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(X)
        model = IsolationForest(contamination=0.05, random_state=42, n_estimators=60)
        model.fit(X_s)
        scores = model.decision_function(X_s)
        # Normalise → [0,1], higher = more anomalous
        lo, hi = scores.min(), scores.max()
        norm = 1 - (scores - lo) / (hi - lo + 1e-9)
        df.loc[grp.index, "anomaly_score"] = norm.round(4)
        df.loc[grp.index, "is_anomaly"]    = norm >= 0.75

    return df


# ══════════════════════════════════════════════════════════════
#  FORECASTING (Prophet or linear fallback)
# ══════════════════════════════════════════════════════════════

def forecast_prices(df: pd.DataFrame, product: str, region: str, horizon: int = 12) -> pd.DataFrame:
    """Forecast prices using Prophet if available, else linear extrapolation."""
    subset = (
        df[(df["product"] == product) & (df["region"] == region)]
        .sort_values("date")[["date", "price"]]
        .rename(columns={"date": "ds", "price": "y"})
    )
    if len(subset) < 12:
        return pd.DataFrame()

    try:
        from prophet import Prophet  # type: ignore
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.90,
            changepoint_prior_scale=0.05,
        )
        m.fit(subset, iter=300)
        future   = m.make_future_dataframe(periods=horizon, freq="MS")
        forecast = m.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon + 24)
    except Exception:
        pass

    # Fallback: polynomial extrapolation
    y = subset["y"].values
    x = np.arange(len(y))
    coef = np.polyfit(x, y, deg=2)
    future_x = np.arange(len(y), len(y) + horizon)
    yhat     = np.polyval(coef, future_x)
    sigma    = y.std()
    future_dates = pd.date_range(
        subset["ds"].iloc[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    return pd.DataFrame({
        "ds": future_dates,
        "yhat":       yhat.clip(min=0),
        "yhat_lower": (yhat - 1.64 * sigma).clip(min=0),
        "yhat_upper": (yhat + 1.64 * sigma).clip(min=0),
    })


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def df_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="Data")
    return buf.getvalue()


def severity_label(score: float) -> str:
    if score >= 0.95: return "🔴 CRITIQUE"
    if score >= 0.85: return "🟠 ÉLEVÉE"
    if score >= 0.75: return "🟡 MOYENNE"
    return "🟢 FAIBLE"


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown(
        """
        <div style='text-align:center;padding:12px 0 8px'>
            <span style='font-size:3rem'>🇩🇿</span><br>
            <span style='font-size:1.05rem;font-weight:700;color:#006233'>
            Algeria Food Price<br>Intelligence System</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    all_products = sorted(df["product"].unique())
    all_regions  = sorted(df["region"].unique())
    all_cats     = sorted(df["category"].unique())

    cat_filter = st.sidebar.multiselect("🗂️ Catégorie", all_cats, default=all_cats)
    prod_pool  = [p for p in all_products if PRODUCTS[p][0] in cat_filter] if cat_filter else all_products

    sel_all_reg  = st.sidebar.checkbox("Toutes les régions",  value=True,  key="all_reg")
    sel_all_prod = st.sidebar.checkbox("Tous les produits",   value=False, key="all_prod")

    sel_products = st.sidebar.multiselect(
        "🛒 Produits", prod_pool,
        default=prod_pool if sel_all_prod else prod_pool[:4],
    )
    sel_regions = st.sidebar.multiselect(
        "📍 Régions", all_regions,
        default=all_regions if sel_all_reg else all_regions[:5],
    )

    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    date_range = st.sidebar.date_input(
        "📅 Période",
        value=(max_d - timedelta(days=365 * 3), max_d),
        min_value=min_d, max_value=max_d,
    )

    st.sidebar.divider()
    st.sidebar.caption("📊 Données : WFP · FAO · Synthétiques")
    st.sidebar.caption("🔬 Modèles : Isolation Forest · Prophet")
    st.sidebar.caption("v2.0 — Streamlit Cloud")

    return (
        sel_products or prod_pool[:4],
        sel_regions  or all_regions,
        date_range,
    )


# ══════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════

def render_kpis(df: pd.DataFrame, df_scored: pd.DataFrame) -> None:
    latest     = df[df["date"] == df["date"].max()]
    prev_month = df[df["date"] == (df["date"].max() - pd.DateOffset(months=1))]

    avg_now  = latest["price"].mean()
    avg_prev = prev_month["price"].mean() if not prev_month.empty else avg_now
    mom      = (avg_now - avg_prev) / avg_prev * 100 if avg_prev else 0

    n_anomalies = int(df_scored["is_anomaly"].sum())
    n_critical  = int((df_scored["anomaly_score"] >= 0.95).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Prix Moyen (DZD)", f"{avg_now:,.0f}", f"{mom:+.1f}% MoM")
    c2.metric("🛒 Produits",         df["product"].nunique())
    c3.metric("📍 Régions",          df["region"].nunique())
    c4.metric("📅 Observations",     f"{len(df):,}")
    c5.metric("🚨 Anomalies",        n_anomalies,
              delta=f"{n_critical} critiques", delta_color="inverse")


# ══════════════════════════════════════════════════════════════
#  MAP
# ══════════════════════════════════════════════════════════════

def render_map(df: pd.DataFrame) -> None:
    st.subheader("🗺️ Carte des Prix par Région")

    col1, col2 = st.columns([2, 1])
    with col1:
        product = st.selectbox("Produit à afficher", sorted(df["product"].unique()), key="map_prod")
    with col2:
        show_all = st.checkbox("Afficher toutes les wilayas", value=True, key="map_all")

    subset = df[df["product"] == product]
    if subset.empty:
        st.info("Aucune donnée.")
        return

    latest = subset[subset["date"] == subset["date"].max()]
    map_df = (
        latest.groupby("region", as_index=False)
        .agg(price=("price","mean"), latitude=("latitude","first"), longitude=("longitude","first"))
    )

    # If show_all: add regions missing from filtered data using WILAYA_COORDS
    if show_all:
        present = set(map_df["region"])
        extra_rows = [
            {"region": r, "price": np.nan, "latitude": lat, "longitude": lon}
            for r, (lat, lon) in WILAYA_COORDS.items() if r not in present
        ]
        if extra_rows:
            map_df = pd.concat([map_df, pd.DataFrame(extra_rows)], ignore_index=True)

    map_df = map_df.dropna(subset=["latitude", "longitude"])
    has_price = map_df["price"].notna()
    date_label = latest["date"].iloc[0].strftime("%B %Y") if not latest.empty else ""

    fig = go.Figure()

    # Wilayas sans données — gris
    grey = map_df[~has_price]
    if not grey.empty:
        fig.add_trace(go.Scattergeo(
            lat=grey["latitude"], lon=grey["longitude"],
            text=grey["region"],
            hoverinfo="text",
            mode="markers",
            marker=dict(size=8, color="#cccccc", line=dict(width=1, color="white")),
            name="Sans données",
            showlegend=True,
        ))

    # Wilayas avec données — couleur par prix
    color_df = map_df[has_price]
    if not color_df.empty:
        pmin, pmax = color_df["price"].min(), color_df["price"].max()
        fig.add_trace(go.Scattergeo(
            lat=color_df["latitude"],
            lon=color_df["longitude"],
            text=color_df.apply(
                lambda r: f"<b>{r['region']}</b><br>Prix : {r['price']:,.0f} DZD", axis=1
            ),
            hoverinfo="text",
            mode="markers",
            marker=dict(
                size=color_df["price"].apply(
                    lambda p: 12 + 28 * (p - pmin) / max(pmax - pmin, 1)
                ),
                color=color_df["price"],
                colorscale="RdYlGn_r",
                cmin=pmin, cmax=pmax,
                colorbar=dict(title="DZD", thickness=14, len=0.7),
                line=dict(width=1, color="white"),
                opacity=0.88,
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=f"Prix <b>{product}</b> — {date_label}", x=0.5, font=dict(size=15)),
        geo=dict(
            scope="africa",
            resolution=50,
            center=dict(lat=28.5, lon=2.5),
            projection_scale=5.0,
            showland=True,    landcolor="#f0ede6",
            showocean=True,   oceancolor="#cce5f6",
            showcountries=True, countrycolor="#aaa",
            showcoastlines=True, coastlinecolor="#999",
            showlakes=False,
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08),
        margin=dict(r=0, t=50, l=0, b=0),
        height=490,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════
#  TIME SERIES
# ══════════════════════════════════════════════════════════════

def render_time_series(df: pd.DataFrame) -> None:
    st.subheader("📈 Évolution des Prix dans le Temps")
    if df.empty:
        st.warning("Aucune donnée.")
        return

    pivot = df.groupby(["date", "product"])["price"].mean().reset_index()

    fig = px.line(
        pivot, x="date", y="price", color="product",
        template="plotly_white",
        labels={"price": "Prix Moyen (DZD)", "date": "Date", "product": "Produit"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    # Ramadan shading
    ram_dates = df[df["is_ramadan"] == 1]["date"].unique() if "is_ramadan" in df.columns else []
    shown_label = False
    for rd in sorted(ram_dates)[:10]:
        fig.add_vrect(
            x0=rd, x1=rd + pd.DateOffset(months=1),
            fillcolor="#ffd700", opacity=0.12, line_width=0,
            annotation_text="Ramadan" if not shown_label else "",
            annotation_position="top left",
            annotation_font_size=10,
        )
        shown_label = True

    fig.update_layout(
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")

    # Heatmap
    st.subheader("🌡️ Heatmap Prix × Mois × Région")
    heat_prod = st.selectbox("Produit", sorted(df["product"].unique()), key="heat_p")
    heat = (
        df[df["product"] == heat_prod]
        .assign(label=lambda d: d["date"].dt.strftime("%Y-%m"))
        .groupby(["label", "region"])["price"].mean()
        .reset_index()
    )
    fig2 = px.density_heatmap(
        heat, x="label", y="region", z="price",
        color_continuous_scale="RdYlGn_r",
        template="plotly_white",
        labels={"label": "Mois", "region": "Région", "price": "Prix Moyen (DZD)"},
    )
    fig2.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig2, width="stretch")


# ══════════════════════════════════════════════════════════════
#  REGION COMPARISON
# ══════════════════════════════════════════════════════════════

def render_comparison(df: pd.DataFrame) -> None:
    st.subheader("📊 Comparateur de Prix par Région")
    latest = df[df["date"] == df["date"].max()]
    if latest.empty:
        return

    pivot = latest.groupby(["region","product"])["price"].mean().reset_index()
    fig = px.bar(
        pivot, x="region", y="price", color="product",
        barmode="group", template="plotly_white",
        labels={"price": "Prix Moyen (DZD)", "region": "Région", "product": "Produit"},
        title=f"Prix par Région — {latest['date'].iloc[0].strftime('%B %Y')}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=430, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

    # Box plot: price dispersion across regions
    st.subheader("📦 Dispersion des Prix par Produit")
    fig2 = px.box(
        df, x="product", y="price", color="product",
        template="plotly_white",
        labels={"price": "Prix (DZD)", "product": "Produit"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig2.update_layout(height=380, showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig2, width="stretch")


# ══════════════════════════════════════════════════════════════
#  ANOMALY ALERTS
# ══════════════════════════════════════════════════════════════

def render_alerts(df_scored: pd.DataFrame) -> None:
    st.subheader("🚨 Alertes Prix — Anomalies Détectées")

    alerts = df_scored[df_scored["is_anomaly"]].copy()
    if alerts.empty:
        st.success("✅ Aucune anomalie sur la période sélectionnée.")
        return

    alerts["Sévérité"] = alerts["anomaly_score"].apply(severity_label)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total alertes",   len(alerts))
    col2.metric("🔴 Critiques",    int((alerts["anomaly_score"] >= 0.95).sum()))
    col3.metric("🟠 Élevées",      int(((alerts["anomaly_score"] >= 0.85) & (alerts["anomaly_score"] < 0.95)).sum()))
    col4.metric("🟡 Moyennes",     int(((alerts["anomaly_score"] >= 0.75) & (alerts["anomaly_score"] < 0.85)).sum()))

    display = (
        alerts[["date","product","region","price","pct_change_1m","anomaly_score","Sévérité"]]
        .rename(columns={"date":"Date","product":"Produit","region":"Région",
                         "price":"Prix (DZD)","pct_change_1m":"Δ MoM (%)","anomaly_score":"Score"})
        .sort_values("Score", ascending=False)
        .head(100)
        .reset_index(drop=True)
    )
    display["Date"]      = pd.to_datetime(display["Date"]).dt.strftime("%Y-%m")
    display["Prix (DZD)"]= display["Prix (DZD)"].map("{:,.0f}".format)
    display["Δ MoM (%)"] = display["Δ MoM (%)"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    display["Score"]     = display["Score"].map("{:.2f}".format)

    st.dataframe(display, width="stretch", hide_index=True)

    # Severity pie chart
    sev_counts = alerts["Sévérité"].value_counts().reset_index()
    sev_counts.columns = ["Sévérité", "Count"]
    fig = px.pie(
        sev_counts, names="Sévérité", values="Count",
        color="Sévérité",
        color_discrete_map={
            "🔴 CRITIQUE": "#d32f2f", "🟠 ÉLEVÉE": "#f57c00",
            "🟡 MOYENNE":  "#fbc02d", "🟢 FAIBLE": "#388e3c",
        },
        template="plotly_white",
        title="Répartition par Sévérité",
    )
    fig.update_layout(height=320)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    c1.download_button("📥 CSV",  alerts.to_csv(index=False).encode(),  "alertes.csv",  "text/csv")
    c2.download_button("📥 Excel", df_to_excel(alerts), "alertes.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ══════════════════════════════════════════════════════════════
#  FORECAST
# ══════════════════════════════════════════════════════════════

def render_forecast(df: pd.DataFrame) -> None:
    st.subheader("🔮 Prédiction des Prix")
    c1, c2, c3 = st.columns(3)
    product = c1.selectbox("Produit",  sorted(df["product"].unique()), key="fc_prod")
    region  = c2.selectbox("Région",   sorted(df["region"].unique()),  key="fc_reg")
    horizon = c3.slider("Horizon (mois)", 1, 24, 6)

    if st.button("🚀 Générer la Prédiction", type="primary"):
        with st.spinner("Calcul en cours…"):
            fc = forecast_prices(df, product, region, horizon)

        if fc.empty:
            st.warning("Pas assez de données pour ce groupe (min 12 mois).")
            return

        hist = df[(df["product"] == product) & (df["region"] == region)].sort_values("date")
        future = fc[fc["ds"] > hist["date"].max()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["price"],
            name="Historique", line=dict(color=ALGERIA_GREEN, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=future["ds"], y=future["yhat"],
            name="Prédiction", line=dict(color="#1565c0", dash="dash", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([future["ds"], future["ds"][::-1]]),
            y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(21,101,192,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalle 90%",
        ))
        fig.update_layout(
            title=f"Prévision — {product} / {region}",
            xaxis_title="Date", yaxis_title="Prix (DZD)",
            template="plotly_white", height=430, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")

        # Forecast table
        tbl = future[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        tbl.columns = ["Date","Prédiction (DZD)","Borne Basse","Borne Haute"]
        tbl["Date"] = tbl["Date"].dt.strftime("%Y-%m")
        for col in ["Prédiction (DZD)","Borne Basse","Borne Haute"]:
            tbl[col] = tbl[col].map("{:,.0f}".format)
        st.dataframe(tbl.reset_index(drop=True), width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════

def render_export(df: pd.DataFrame) -> None:
    st.subheader("📥 Exporter les Données")
    c1, c2, c3 = st.columns(3)
    c1.download_button("📄 CSV",   df.to_csv(index=False).encode(),
                       "algeria_food_prices.csv",  "text/csv")
    c2.download_button("📊 Excel", df_to_excel(df),
                       "algeria_food_prices.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c3.metric("Lignes", f"{len(df):,}")

    with st.expander("🔍 Aperçu des données"):
        st.dataframe(df.head(100), width="stretch", hide_index=True)

    # Summary stats
    st.subheader("📋 Statistiques Descriptives")
    stats = df.groupby("product")["price"].agg(
        Moyenne="mean", Médiane="median", Écart_type="std", Min="min", Max="max"
    ).round(1).reset_index()
    st.dataframe(stats, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='text-align:center; padding:10px 0 4px'>
            <h1 style='color:{ALGERIA_GREEN}; margin:0'>
                🌾 Algeria Food Price Intelligence
            </h1>
            <p style='color:#666; margin:4px 0 0'>
                Surveillance · Anomalies · Prédiction des Prix Alimentaires — Algérie
            </p>
        </div>
        <hr style='margin:10px 0 20px; border-color:#e0e0e0'>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────
    df_full = load_data()

    # ── Sidebar ───────────────────────────────────────────────
    sel_products, sel_regions, date_range = render_sidebar(df_full)

    start = pd.Timestamp(date_range[0]) if len(date_range) == 2 else df_full["date"].min()
    end   = pd.Timestamp(date_range[1]) if len(date_range) == 2 else df_full["date"].max()

    df = df_full[
        df_full["product"].isin(sel_products) &
        df_full["region"].isin(sel_regions)   &
        (df_full["date"] >= start) &
        (df_full["date"] <= end)
    ].copy()

    # ── Anomaly detection (on filtered set) ───────────────────
    df_scored = detect_anomalies(df)

    # ── KPIs ──────────────────────────────────────────────────
    render_kpis(df, df_scored)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "🗺️ Carte",
        "📈 Tendances",
        "🚨 Alertes",
        "🔮 Prédictions",
        "📥 Export",
    ])

    with t1:
        render_map(df)
        render_comparison(df)

    with t2:
        render_time_series(df)

    with t3:
        render_alerts(df_scored)

    with t4:
        render_forecast(df_full)

    with t5:
        render_export(df)


if __name__ == "__main__":
    main()
