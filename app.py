import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix

###############################################################################
# HR ATTRITION SUPER‑DASHBOARD – v2.0                                         #
# Covers virtually every dimension & combination in EA.csv                    #
# Author: Shaurya                                                             #
###############################################################################

st.set_page_config(
    page_title="HR Attrition Super‑Dashboard",
    page_icon="📊",
    layout="wide",
)

# --------------------------- DATA LOADING ------------------------------------
@st.cache_data
def load_data(path="EA (1).csv"):
    df = pd.read_csv(path)
    df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

# --------------------------- GLOBAL CONTROLS ---------------------------------
with st.sidebar:
    st.header("🔍 Global Filters")
    dept_opts   = ["All"] + sorted(df.Department.unique())
    role_opts   = ["All"] + sorted(df.JobRole.unique())
    gender_opts = ["All"] + sorted(df.Gender.unique())
    travel_opts = ["All"] + sorted(df.BusinessTravel.unique())

    dept_sel   = st.selectbox("Department", dept_opts)
    role_sel   = st.selectbox("Job Role", role_opts)
    gender_sel = st.selectbox("Gender", gender_opts)
    travel_sel = st.selectbox("Business Travel", travel_opts)

    age_rng  = st.slider("Age Range", int(df.Age.min()), int(df.Age.max()), (25,50))
    tenure_rng = st.slider("Years at Company", int(df.YearsAtCompany.min()),
                            int(df.YearsAtCompany.max()), (0,10))

    st.markdown("---")
    misc_tab = st.expander("ℹ️ About this dashboard")
    with misc_tab:
        st.write("""This upgraded version includes **30+ visualisations** arranged across 6 thematic tabs, plus an auto‑training demo model. Use the **sidebar filters** to slice every chart in real time.""")

# Apply filters
fdf = df.copy()
if dept_sel   != "All": fdf = fdf[fdf.Department==dept_sel]
if role_sel   != "All": fdf = fdf[fdf.JobRole==role_sel]
if gender_sel != "All": fdf = fdf[fdf.Gender==gender_sel]
if travel_sel != "All": fdf = fdf[fdf.BusinessTravel==travel_sel]

fdf = fdf[fdf.Age.between(age_rng[0], age_rng[1]) &
          fdf.YearsAtCompany.between(tenure_rng[0], tenure_rng[1])]

# --------------------------- KPI CARDS ---------------------------------------
st.title("📊 HR Attrition Super‑Dashboard")

k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Employees", fdf.shape[0])
with k2: st.metric("Attrition Rate", f"{fdf.AttritionFlag.mean():.1%}")
with k3: st.metric("Avg Income", f"${fdf.MonthlyIncome.mean():,.0f}")
with k4: st.metric("Avg Age",  f"{fdf.Age.mean():.1f} yrs")

st.divider()

# --------------------------- TAB STRUCTURE -----------------------------------
macro_tab, demo_tab, comp_tab, perf_tab, inter_tab, model_tab, data_tab = st.tabs([
    "🌐 Macro Trends", "👥 Demographics", "💰 Compensation", "🎯 Engagement & Performance",
    "🎛 Interactive Explorer", "🤖 Predictive", "🗄 Raw Data"])

###############################################################################
# 🌐 MACRO TRENDS TAB (Charts 1‑7)
###############################################################################
with macro_tab:
    st.subheader("High‑Level Attrition Overview")

    # 1. Overall attrition pie
    st.caption("▶︎ Company‑wide split between exits vs stays")
    fig = px.pie(fdf, names="Attrition", hole=0.45, title="Overall Attrition Rate")
    st.plotly_chart(fig, use_container_width=True)

    # 2. Attrition by Department bar
    st.caption("▶︎ Department hotspots")
    fig = px.histogram(fdf, x="Department", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Attrition by Job Role
    st.caption("▶︎ Job Role breakdown")
    fig = px.histogram(fdf, x="JobRole", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # 4. Heatmap Dept × JobLevel attrition rates
    st.caption("▶︎ Heatmap – Department vs Job Level")
    heat = pd.crosstab(fdf.Department, fdf.JobLevel, values=fdf.AttritionFlag,
                       aggfunc="mean").fillna(0)
    fig = px.imshow(heat, color_continuous_scale="Reds", text_auto=".1%",
                    labels=dict(color="Attrition%"))
    st.plotly_chart(fig, use_container_width=True)

    # 5. Age distribution hist
    st.caption("▶︎ Age spectrum of stayers vs leavers")
    fig = px.histogram(fdf, x="Age", color="Attrition", nbins=25, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    # 6. YearsAtCompany violin
    st.caption("▶︎ Tenure patterns (attrition risk highest in early years)")
    fig = px.violin(fdf, y="YearsAtCompany", x="Attrition", box=True, points="all")
    st.plotly_chart(fig, use_container_width=True)

    # 7. Business Travel impact
    st.caption("▶︎ Business travel frequency vs attrition")
    fig = px.histogram(fdf, x="BusinessTravel", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 👥 DEMOGRAPHICS TAB (Charts 8‑13)
###############################################################################
with demo_tab:
    st.subheader("Demographic Factors")

    for col in ["Gender", "MaritalStatus", "EducationField", "Education"]:
        st.caption(f"▶︎ Attrition by {col}")
        fig = px.histogram(fdf, x=col, color="Attrition", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    # 12. Parallel categories plot
    st.caption("▶︎ Parallel Categories – Gender × Marital × OverTime × Attrition")
    fig = px.parallel_categories(
        fdf, dimensions=["Gender", "MaritalStatus", "OverTime", "Attrition"],
        color="AttritionFlag", color_continuous_scale=px.colors.sequential.Reds)
    st.plotly_chart(fig, use_container_width=True)

    # 13. Scatter – Age vs Distance, sized by MonthlyIncome
    st.caption("▶︎ Age vs Commute Distance (bubble size = Income)")
    fig = px.scatter(fdf, x="Age", y="DistanceFromHome", size="MonthlyIncome",
                     color="Attrition", hover_data=["JobRole"])
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 💰 COMPENSATION TAB (Charts 14‑19)
###############################################################################
with comp_tab:
    st.subheader("Compensation & Benefits")

    # 14. MonthlyIncome distribution
    st.caption("▶︎ Income distribution")
    fig = px.box(fdf, y="MonthlyIncome", x="Attrition", points="all")
    st.plotly_chart(fig, use_container_width=True)

    # 15. MonthlyRate vs Attrition
    st.caption("▶︎ Monthly Rate vs Attrition")
    fig = px.violin(fdf, y="MonthlyRate", x="Attrition", box=True)
    st.plotly_chart(fig, use_container_width=True)

    # 16. Percent Salary Hike
    st.caption("▶︎ Salary Hike % by Attrition")
    fig = px.histogram(fdf, x="PercentSalaryHike", color="Attrition", nbins=15,
                       barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # 17. StockOptionLevel
    st.caption("▶︎ Stock Option Level vs Attrition")
    fig = px.bar(fdf, x="StockOptionLevel", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # 18. Overtime vs Income scatter
    st.caption("▶︎ Overtime vs Income scatter (hint: burnout + pay)")
    fig = px.scatter(fdf, x="MonthlyIncome", y="OverTime", color="Attrition",
                     size="Age", hover_data=["Department"])
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 🎯 ENGAGEMENT & PERFORMANCE TAB (Charts 20‑26)
###############################################################################
with perf_tab:
    st.subheader("Engagement, Satisfaction & Performance")

    eng_cols = [
        "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
        "RelationshipSatisfaction", "PerformanceRating", "JobInvolvement" ]

    for col in eng_cols:
        st.caption(f"▶︎ {col} vs Attrition")
        fig = px.histogram(fdf, x=col, color="Attrition", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    # 26. TrainingTimesLastYear
    st.caption("▶︎ Training Times Last Year vs Attrition")
    fig = px.box(fdf, x="Attrition", y="TrainingTimesLastYear", points="all")
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 🎛 INTERACTIVE EXPLORER TAB (Charts 27‑29)
###############################################################################
with inter_tab:
    st.subheader("Build‑Your‑Own Comparison")
    st.caption("Select any two categorical variables to cross‑tab attrition rates.")

    cat_cols = df.select_dtypes(include="object").columns.drop("Attrition")
    col1, col2 = st.columns(2)
    with col1:
        cat_x = st.selectbox("X‑axis", cat_cols, index=0)
    with col2:
        cat_y = st.selectbox("Y‑axis", cat_cols, index=1)

    pivot = pd.crosstab(fdf[cat_y], fdf[cat_x], values=fdf["AttritionFlag"],
                        aggfunc="mean").fillna(0)
    fig = px.imshow(pivot, text_auto=".1%", color_continuous_scale="Viridis",
                    labels=dict(color="Attrition %"))
    st.plotly_chart(fig, use_container_width=True)

    # 28. Correlation heatmap numerical vars
    st.caption("▶︎ Correlation heatmap (numericals)")
    num_corr = fdf.select_dtypes(include=["int64","float64"]).corr()
    fig = px.imshow(num_corr, color_continuous_scale="RdBu", text_auto=False)
    st.plotly_chart(fig, use_container_width=True)

    # 29. Pairwise scatter (Age vs MonthlyIncome vs YearsAtCompany)
    st.caption("▶︎ 3‑way scatter – hover to inspect")
    fig = px.scatter_3d(fdf, x="Age", y="YearsAtCompany", z="MonthlyIncome",
                        color="Attrition", hover_data=["JobRole"])
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 🤖 PREDICTIVE TAB (Charts 30‑31)
###############################################################################
with model_tab:
    st.subheader("Illustrative RandomForest Attrition Classifier")
    st.caption("🔥 Prototype only – not tuned for production, but shows feature drivers.")

    # Minimal encoding for categorical variables
    model_df = fdf.copy()
    le = LabelEncoder()
    for col in model_df.select_dtypes(include="object").columns:
        model_df[col] = le.fit_transform(model_df[col])

    X = model_df.drop(columns=["AttritionFlag"])
    y = model_df["AttritionFlag"]
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0)
    rf.fit(X, y)
    pred_prob = rf.predict_proba(X)[:,1]

    auc = roc_auc_score(y, pred_prob)
    st.metric("In‑sample ROC‑AUC", f"{auc:.2f}")

    # 30. Feature importances
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:20]
    fig = px.bar(imp, orientation="h", labels={"value":"Importance"})
    st.plotly_chart(fig, use_container_width=True)

    # 31. Confusion matrix heatmap
    st.caption("▶︎ Confusion Matrix (Cut‑off 0.5)")
    cm = confusion_matrix(y, rf.predict(X))
    cm_df = pd.DataFrame(cm, index=["Stay","Leave"], columns=["Pred Stay","Pred Leave"])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 🗄 RAW DATA TAB (Chart 32)
###############################################################################
with data_tab:
    st.subheader("Data Table & Export")
    st.dataframe(fdf, use_container_width=True)
    st.download_button("Download current view", fdf.to_csv(index=False).encode(),
                       "Filtered_Attrition.csv")

# FIN
st.success("✨ Dashboard rendered with 32 visuals. Explore away!")
