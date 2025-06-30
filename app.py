#  ─────────────────────────────────────────────────────────────
#  HR ATTRITION INSIGHTS DASHBOARD
#  Streamlit • © 2025 Shaurya (SP Jain GMBA)
#  Purpose: Enterprise-grade analytics for CHRO, HR Director, and ExCo
#  ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="HR Attrition Insights",
    page_icon="📊",
    layout="wide",
)

# ----------------------- DATA INGESTION -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("EA (1).csv")
    return df

df = load_data()

# Label-encode Attrition for quick math
attr_map = {"Yes": 1, "No": 0}
df["AttritionFlag"] = df["Attrition"].map(attr_map)

# ----------------------- SIDEBAR FILTERS ----------------------
st.sidebar.header("🔧 Global Filters")
dept_list  = ["All"] + sorted(df["Department"].unique().tolist())
role_list  = ["All"] + sorted(df["JobRole"].unique().tolist())
gender_list = ["All"] + sorted(df["Gender"].unique().tolist())

dept_sel   = st.sidebar.selectbox("Department", dept_list)
role_sel   = st.sidebar.selectbox("Job Role", role_list)
gender_sel = st.sidebar.selectbox("Gender", gender_list)
age_range  = st.sidebar.slider("Age Range", int(df.Age.min()), int(df.Age.max()),
                               (int(df.Age.min()), int(df.Age.max())))
yrs_range  = st.sidebar.slider("Years at Company", int(df.YearsAtCompany.min()),
                               int(df.YearsAtCompany.max()),
                               (int(df.YearsAtCompany.min()), int(df.YearsAtCompany.max())))

# Apply filters
fdf = df.copy()
if dept_sel != "All":
    fdf = fdf[fdf["Department"] == dept_sel]
if role_sel != "All":
    fdf = fdf[fdf["JobRole"] == role_sel]
if gender_sel != "All":
    fdf = fdf[fdf["Gender"] == gender_sel]

fdf = fdf[(fdf["Age"].between(age_range[0], age_range[1])) &
          (fdf["YearsAtCompany"].between(yrs_range[0], yrs_range[1]))]

# ----------------------- KPI CARDS (1, 2, 3) ------------------
st.title("📊 HR Attrition Insights Dashboard")
st.markdown("Strategic talent analytics—macro to micro—for data-driven HR decisions.")

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Current Headcount", int(fdf.shape[0]))
with kpi2:
    st.metric("Attrition Rate",
              f"{fdf.AttritionFlag.mean():.1%}",
              help="Percentage of employees who left among the filtered cohort")
with kpi3:
    st.metric("Average Monthly Income", f"${fdf.MonthlyIncome.mean():,.0f}")

st.divider()

# ----------------------- TAB LAYOUT ---------------------------
macro_tab, micro_tab, model_tab, data_tab = st.tabs(
    ["🌐 Macro View", "🔎 Micro View", "🤖 ML Model", "🗄 Raw Data"]
)

# ------------- 🌐 MACRO VIEW CHARTS (4-10) --------------------
with macro_tab:
    st.subheader("Organisation-wide Patterns")
    # 4. Attrition by Department
    st.markdown("**Attrition by Department** helps leadership identify at-risk functions.")
    fig1 = px.bar(fdf, x="Department", color="Attrition", barmode="group")
    st.plotly_chart(fig1, use_container_width=True)

    # 5. Attrition by Job Role
    st.markdown("**Attrition by Job Role** spotlights critical roles experiencing churn.")
    fig2 = px.bar(fdf, x="JobRole", color="Attrition", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    # 6. Attrition by Business Travel
    st.markdown("**Impact of Business Travel on Attrition** reveals travel burn-out trends.")
    fig3 = px.histogram(fdf, x="BusinessTravel", color="Attrition")
    st.plotly_chart(fig3, use_container_width=True)

    # 7. Attrition over Age Distribution
    st.markdown("**Age Distribution vs Attrition** uncovers generational turnover.")
    fig4 = px.histogram(fdf, x="Age", color="Attrition", nbins=20, barmode="overlay")
    st.plotly_chart(fig4, use_container_width=True)

    # 8. Attrition vs Years at Company
    st.markdown("**Tenure vs Attrition** highlights early-exit risk periods.")
    fig5 = px.violin(fdf, y="YearsAtCompany", color="Attrition", box=True, points="all")
    st.plotly_chart(fig5, use_container_width=True)

    # 9. Monthly Income Distribution
    st.markdown("**Compensation Distribution** shows pay spread and potential equity gaps.")
    fig6 = px.box(fdf, y="MonthlyIncome", color="Attrition")
    st.plotly_chart(fig6, use_container_width=True)

# ------------- 🔎 MICRO VIEW CHARTS (11-17) -------------------
with micro_tab:
    st.subheader("Slice-and-Dice Analysis")
    # 10. Attrition by Gender
    st.markdown("**Gender vs Attrition** supports DE&I monitoring.")
    fig7 = px.bar(fdf, x="Gender", color="Attrition", barmode="group")
    st.plotly_chart(fig7, use_container_width=True)

    # 11. Attrition by Marital Status
    st.markdown("**Marital Status** correlations with attrition for targeted benefits.")
    fig8 = px.bar(fdf, x="MaritalStatus", color="Attrition", barmode="group")
    st.plotly_chart(fig8, use_container_width=True)

    # 12. Overtime vs Attrition
    st.markdown("**Overtime Workload** impact on employee exits.")
    fig9 = px.pie(fdf, names="OverTime", color="Attrition",
                  title="Attrition Split by Overtime")
    st.plotly_chart(fig9, use_container_width=True)

    # 13. Education Field Heatmap
    st.markdown("**Education Field vs Attrition** to align L&D initiatives.")
    pivot = pd.crosstab(fdf["EducationField"], fdf["AttritionFlag"], normalize="index")
    fig10 = px.imshow(pivot, text_auto=True, color_continuous_scale="Blues",
                      aspect="auto", labels=dict(color="Attrition %"))
    st.plotly_chart(fig10, use_container_width=True)

    # 14. DistanceFromHome vs Attrition
    st.markdown("**Commute Distance Stress Test** – attrition likelihood by distance.")
    fig11 = px.scatter(fdf, x="DistanceFromHome", y="MonthlyIncome",
                       color="Attrition", size="Age", hover_data=["JobRole"])
    st.plotly_chart(fig11, use_container_width=True)

    # 15. Environment Satisfaction
    st.markdown("**Workplace Satisfaction Levels** versus attrition outcomes.")
    fig12 = px.box(fdf, x="EnvironmentSatisfaction", y="Age",
                   color="Attrition", points="all")
    st.plotly_chart(fig12, use_container_width=True)

    # 16. Work-Life Balance
    st.markdown("**Work-Life Balance Scores** in relation to attrition.")
    fig13 = px.histogram(fdf, x="WorkLifeBalance", color="Attrition",
                         barmode="group")
    st.plotly_chart(fig13, use_container_width=True)

    # 17. Training Times Last Year
    st.markdown("**Training Frequency** and its protective effect on retention.")
    fig14 = px.bar(fdf, x="TrainingTimesLastYear", color="Attrition",
                   barmode="group")
    st.plotly_chart(fig14, use_container_width=True)

# ------------- 🤖 SIMPLE ML MODEL (18-19) ---------------------
with model_tab:
    st.subheader("Predictive Signal Exploration")
    st.markdown(
        "Below model is illustrative—**not** production-ready. "
        "It ranks the relative importance of features in predicting attrition."
    )

    # Minimal preprocessing
    model_df = fdf.select_dtypes(include=["int64", "float64"]).copy()
    cat_cols  = fdf.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for c in cat_cols:
        model_df[c] = le.fit_transform(fdf[c])

    X = model_df.drop(columns=["AttritionFlag"])
    y = model_df["AttritionFlag"]

    # 18. RandomForest Feature Importance
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)\
                    .sort_values(ascending=False).head(15)

    st.markdown("**Top Drivers of Attrition (Random Forest Importance)**")
    fig15 = px.bar(importances, orientation="h", labels={"value": "Importance"})
    st.plotly_chart(fig15, use_container_width=True)

    # 19. Simple ROC-AUC (optional)
    # (streamlit display kept minimal to avoid clutter)
    from sklearn.metrics import roc_auc_score
    st.metric("In-sample ROC-AUC", f"{roc_auc_score(y, rf.predict_proba(X)[:,1]):.2f}")

# ------------- 🗄 RAW DATA & PIVOT TOOL (20) ------------------
with data_tab:
    st.subheader("Interactive Data Explorer")
    st.markdown(
        "**Pivot + Filter**—use the sidebar to subset; download to Excel if needed."
    )
    st.dataframe(fdf, use_container_width=True)

    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", csv, "Attrition_subset.csv")

st.success("Dashboard loaded successfully.")
