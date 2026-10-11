from pathlib import Path
from io import BytesIO
import zipfile
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SmartStudy AI | Learning Analytics", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1450px;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important;}
.hero {padding: 1.35rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#12304A,#176B87); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2.2rem;}
.hero p {color:#DDEEF4; margin:.45rem 0 0 0; font-size:1rem;}
.metric-card {border:1px solid #D9E4EA; border-radius:14px; padding:1rem;}
.small-note {color:#617282; font-size:.88rem;}
</style>
<div class="hero">
  <h1>🎓 SmartStudy AI</h1>
  <p>Student Performance Analytics & Grade Prediction · Machine Learning project · SDG 4: Quality Education</p>
</div>
""", unsafe_allow_html=True)

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
# Fallback mirrors of the same UCI dataset. GitHub mirrors use comma delimiters;
# the original UCI files use semicolons. The loader detects either format.
MIRRORS = {
    "Mathematics": "https://raw.githubusercontent.com/sarthak-srivastava/Student-Performance-Dataset/master/student-mat.csv",
    "Portuguese": "https://raw.githubusercontent.com/sarthak-srivastava/Student-Performance-Dataset/master/student-por.csv",
}
LOCAL_MAT = Path(__file__).parent / "student-mat.csv"
LOCAL_POR = Path(__file__).parent / "student-por.csv"


def read_student_csv(file_obj):
    # sep=None detects comma-separated mirrors and semicolon-separated UCI files.
    return pd.read_csv(file_obj, sep=None, engine="python")


@st.cache_data(show_spinner=False)
def load_data(subject):
    local = LOCAL_MAT if subject == "Mathematics" else LOCAL_POR
    if local.exists():
        return read_student_csv(local), "Local UCI dataset file"

    filename = "student-mat.csv" if subject == "Mathematics" else "student-por.csv"
    errors = []
    # First try the official UCI ZIP archive.
    try:
        response = requests.get(DATA_URL, timeout=20, headers={"User-Agent": "SmartStudyAI/1.0"})
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            matches = [name for name in zf.namelist()
                       if Path(name.replace("\\", "/")).name.lower() == filename.lower()]
            if matches:
                with zf.open(matches[0]) as f:
                    return read_student_csv(f), "UCI Machine Learning Repository (official ZIP)"
            errors.append("Official ZIP did not contain the expected CSV filename")
    except Exception as exc:
        errors.append(f"UCI ZIP: {exc}")

    # If the ZIP endpoint is unavailable/changed, use a mirror of the same public dataset.
    try:
        response = requests.get(MIRRORS[subject], timeout=20, headers={"User-Agent": "SmartStudyAI/1.0"})
        response.raise_for_status()
        df = read_student_csv(BytesIO(response.content))
        if "G3" not in df.columns or len(df.columns) < 30:
            raise ValueError("Downloaded file did not look like the expected student dataset")
        return df, "UCI Student Performance dataset (public mirror; original dataset attribution retained)"
    except Exception as exc:
        errors.append(f"Dataset mirror: {exc}")

    raise RuntimeError("Could not load the dataset. " + " | ".join(errors) +
                       ". You can manually download the CSV and place it beside app.py.")

# Main-screen subject selector replaces the sidebar for a cleaner dashboard layout.
subject_col, source_col = st.columns([1, 2.4], gap="large")
with subject_col:
    st.markdown("#### 📚 Choose dataset subject")
    subject = st.selectbox("Subject", ["Mathematics", "Portuguese"], index=0, label_visibility="collapsed")
with source_col:
    st.markdown("#### Dataset & model notes")
    st.markdown("[UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance) · Public research dataset from two Portuguese schools — not a government dataset.")
    st.caption("Prediction model excludes G1/G2 to reduce target leakage; estimates are for learning and demonstration only.")
st.markdown("---")

try:
    df, source_status = load_data(subject)
except Exception as exc:
    st.error("SmartStudy AI could not load the student dataset.")
    st.code(str(exc))
    st.markdown("Download the official UCI ZIP from the UCI dataset link above, extract `student-mat.csv` and `student-por.csv`, and place the CSV files beside `app.py`. Then refresh the app.")
    st.stop()

features = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob",
            "reason","guardian","traveltime","studytime","failures","schoolsup","famsup",
            "paid","activities","nursery","higher","internet","romantic","famrel","freetime",
            "goout","Dalc","Walc","health","absences"]
target = "G3"
df["G3"] = pd.to_numeric(df["G3"], errors="coerce")
df = df.dropna(subset=["G3"]).copy()
df["Grade band"] = pd.cut(df["G3"], bins=[-1,9,13,20], labels=["Needs support (0–9)","Developing (10–13)","Strong (14–20)"])
n = len(df)
avg = df["G3"].mean()
pass_rate = (df["G3"] >= 10).mean() * 100
fail_rate = (df["G3"] < 10).mean() * 100

tab_overview, tab_explore, tab_predict, tab_model, tab_about = st.tabs(
    ["📊 Overview", "🔎 Explore data", "🎯 Grade predictor", "🧠 Model evaluation", "ℹ️ About & ethics"]
)

with tab_overview:
    st.caption(f"Data source: {source_status} · Selected subject: {subject} · UCI public research dataset")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Student records", f"{n:,}")
    c2.metric("Average final grade", f"{avg:.2f} / 20")
    c3.metric("Grade ≥ 10", f"{pass_rate:.1f}%")
    c4.metric("Grade < 10", f"{fail_rate:.1f}%")
    left,right = st.columns([1.15,1])
    with left:
        counts = df["G3"].value_counts().sort_index().rename_axis("Final grade").reset_index(name="Students")
        fig = px.bar(counts, x="Final grade", y="Students", title="Final grade distribution", color="Students",
                     color_continuous_scale="Teal", template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        band = df["Grade band"].value_counts().reindex(["Needs support (0–9)","Developing (10–13)","Strong (14–20)"]).fillna(0).reset_index()
        band.columns = ["Grade band","Students"]
        fig2 = px.pie(band, names="Grade band", values="Students", hole=.55, title="Performance bands",
                      color="Grade band", color_discrete_sequence=["#E07A5F","#F2CC8F","#2A9D8F"], template="plotly_white")
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(margin=dict(l=10,r=10,t=55,b=10), legend_title_text="")
        st.plotly_chart(fig2, width="stretch")
    st.info("Grade bands are descriptive groupings for this demo, not official pass/fail decisions. The dataset reflects Portuguese schools and may not generalize to other countries or institutions.")

with tab_explore:
    st.subheader("Explore relationships in the dataset")
    a,b = st.columns(2)
    with a:
        x_choice = st.selectbox("Compare final grade against", ["studytime","absences","failures","age","G1","G2"], index=0)
        fig = px.box(df, x=x_choice, y="G3", points="outliers", color_discrete_sequence=["#176B87"],
                     title=f"Final grade vs {x_choice}", template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    with b:
        group_choice = st.selectbox("Group average by", ["studytime","failures","sex","schoolsup","internet","higher"], index=0)
        grouped = df.groupby(group_choice, observed=False)["G3"].mean().reset_index()
        grouped[group_choice] = grouped[group_choice].astype(str)
        fig = px.bar(grouped, x=group_choice, y="G3", text_auto=".2f", title=f"Average final grade by {group_choice}",
                     color="G3", color_continuous_scale="Teal", template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Average final grade (0–20)")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Dataset preview")
    st.dataframe(df.drop(columns=["Grade band"]).head(15), width="stretch", hide_index=True)
    st.download_button("⬇️ Download selected dataset as CSV", df.drop(columns=["Grade band"]).to_csv(index=False).encode("utf-8"),
                       file_name=f"uci_student_performance_{subject.lower()}.csv", mime="text/csv")

with tab_predict:
    st.subheader("Estimate a final grade (0–20)")
    st.write("Enter a sample student profile. This is an educational estimate, not a formal assessment.")
    col1,col2,col3 = st.columns(3)
    with col1:
        school = st.selectbox("School", sorted(df["school"].unique()), key="p_school")
        sex = st.selectbox("Sex (as recorded in dataset)", sorted(df["sex"].unique()), key="p_sex")
        age = st.slider("Age", int(df["age"].min()), int(df["age"].max()), 17)
        address = st.selectbox("Home address type", sorted(df["address"].unique()), key="p_address")
        studytime = st.select_slider("Weekly study time category", options=[1,2,3,4], value=2,
                                     format_func=lambda x: {1:"<2 hours",2:"2–5 hours",3:"5–10 hours",4:">10 hours"}[x])
        failures = st.select_slider("Past class failures", options=[0,1,2,3], value=0)
        absences = st.slider("School absences", 0, max(30, int(df["absences"].max())), 2)
    with col2:
        schoolsup = st.selectbox("Extra school support?", ["yes","no"], index=1)
        famsup = st.selectbox("Family educational support?", ["yes","no"], index=0)
        paid = st.selectbox("Extra paid classes?", ["yes","no"], index=1)
        activities = st.selectbox("Extracurricular activities?", ["yes","no"], index=0)
        internet = st.selectbox("Internet access at home?", ["yes","no"], index=0)
        higher = st.selectbox("Wants higher education?", ["yes","no"], index=0)
        nursery = st.selectbox("Attended nursery?", ["yes","no"], index=0)
        romantic = st.selectbox("In a relationship?", ["yes","no"], index=1)
    with col3:
        famsize = st.selectbox("Family size", sorted(df["famsize"].unique()), key="p_famsize")
        Pstatus = st.selectbox("Parents living together?", sorted(df["Pstatus"].unique()), key="p_pstatus")
        Medu = st.slider("Mother's education (0–4)", 0, 4, 3)
        Fedu = st.slider("Father's education (0–4)", 0, 4, 2)
        traveltime = st.select_slider("Travel time category", options=[1,2,3,4], value=2)
        famrel = st.slider("Family relationship quality (1–5)", 1, 5, 4)
        freetime = st.slider("Free time after school (1–5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1–5)", 1, 5, 3)
    # Modal categorical values are used for some less central fields so all required features are supplied.
    def mode_value(col):
        return df[col].mode(dropna=True).iloc[0]
    row = {
        "school":school, "sex":sex, "age":age, "address":address, "famsize":famsize, "Pstatus":Pstatus,
        "Medu":Medu, "Fedu":Fedu, "Mjob":mode_value("Mjob"), "Fjob":mode_value("Fjob"),
        "reason":mode_value("reason"), "guardian":mode_value("guardian"), "traveltime":traveltime,
        "studytime":studytime, "failures":failures, "schoolsup":schoolsup, "famsup":famsup,
        "paid":paid, "activities":activities, "nursery":nursery, "higher":higher, "internet":internet,
        "romantic":romantic, "famrel":famrel, "freetime":freetime, "goout":goout,
        "Dalc":int(df["Dalc"].median()), "Walc":int(df["Walc"].median()), "health":int(df["health"].median()),
        "absences":absences
    }
    @st.cache_resource
    def train_model(data_hash, subject_name):
        # data_hash participates in Streamlit's cache key
        data = df.copy()
        X = data[features]
        y = data[target]
        categorical = X.select_dtypes(include=["object"]).columns.tolist()
        numeric = [c for c in X.columns if c not in categorical]
        pre = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
        ])
        reg = Pipeline([("preprocess", pre), ("model", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2))])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
        reg.fit(Xtr, ytr)
        preds = reg.predict(Xte)
        metrics = {"MAE": mean_absolute_error(yte,preds), "RMSE": float(np.sqrt(mean_squared_error(yte,preds))),
                   "R2": r2_score(yte,preds), "y_test": yte, "preds": preds}
        return reg, metrics
    reg, metrics = train_model(f"{len(df)}-{subject}-{float(df['G3'].mean()):.3f}", subject)
    if st.button("✨ Predict estimated final grade", type="primary"):
        sample = pd.DataFrame([row])[features]
        estimate = float(np.clip(reg.predict(sample)[0], 0, 20))
        st.markdown("### Prediction result")
        x,y,z = st.columns([1,1,2])
        x.metric("Estimated final grade", f"{estimate:.1f} / 20")
        y.metric("Approx. percentage", f"{estimate*5:.0f}%")
        if estimate < 10:
            z.warning("This estimate falls below 10/20 in this dataset's grade scale. Consider supportive academic discussion; this is not a label or decision.")
        elif estimate < 14:
            z.info("This estimate is in the middle grade range. A study plan and teacher feedback may be useful.")
        else:
            z.success("This estimate is in a higher grade range. Encourage consistent learning habits.")
        st.caption("The estimate is based on a public dataset from Portuguese schools and may be inaccurate for an individual or a different education system.")
    else:
        st.info("Set the profile above, then click **Predict estimated final grade**.")

with tab_model:
    st.subheader("Held-out model evaluation")
    st.caption("Random Forest regression trained without G1/G2 to reduce target leakage and focus on earlier indicators.")
    m1,m2,m3 = st.columns(3)
    m1.metric("MAE (lower is better)", f"{metrics['MAE']:.2f} grade points")
    m2.metric("RMSE (lower is better)", f"{metrics['RMSE']:.2f} grade points")
    m3.metric("R² (closer to 1 is better)", f"{metrics['R2']:.3f}")
    actual = np.array(metrics["y_test"])
    pred = np.array(metrics["preds"])
    scatter = pd.DataFrame({"Actual final grade":actual, "Predicted grade":pred})
    fig = px.scatter(scatter, x="Actual final grade", y="Predicted grade", opacity=.72,
                     title="Actual vs predicted grades on test split", template="plotly_white",
                     color_discrete_sequence=["#176B87"])
    fig.add_shape(type="line", x0=0, y0=0, x1=20, y1=20, line=dict(dash="dash", color="#E07A5F"))
    fig.update_layout(xaxis_range=[0,20], yaxis_range=[0,20])
    st.plotly_chart(fig, width="stretch")
    residual = pd.DataFrame({"Prediction error (actual − predicted)": actual-pred})
    fig2 = px.histogram(residual, x="Prediction error (actual − predicted)", nbins=24,
                        title="Prediction error distribution", template="plotly_white",
                        color_discrete_sequence=["#2A9D8F"])
    st.plotly_chart(fig2, width="stretch")
    st.warning("These metrics describe performance on a held-out split of this dataset only. They do not guarantee performance on new schools or individual students.")

with tab_about:
    st.subheader("Dataset, citation and responsible use")
    st.markdown("""
**Dataset:** Student Performance, UCI Machine Learning Repository (ID 320). The repository describes student achievement data from two Portuguese schools, with mathematics and Portuguese-language files, 649 records in the combined repository, and 33 attributes. The selected subject file has its own row count.

**Citation:** P. Cortez and A. M. G. Silva (2008), *Using data mining to predict secondary school student performance*, 5th Annual Future Business Technology Conference. Dataset DOI: [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). License: CC BY 4.0.

**Important limitations**
- This is a public research dataset, **not a government dataset**, and it represents Portuguese schools, not Indian students.
- Final grades are historical outcomes; estimates should not be used for grading, admissions, discipline, or other high-stakes decisions.
- Demographic and family-context variables can encode social bias. A real deployment needs a clear purpose, privacy safeguards, fairness review, and educator oversight.
- This prototype excludes first- and second-period grades (G1/G2) from predictors because the UCI documentation warns that they are strongly correlated with the final grade G3. This reduces leakage but may reduce predictive performance.
""")
    st.markdown("[Open official UCI dataset page](https://archive.ics.uci.edu/dataset/320/student%2Bperformance)")
    st.markdown("[Scikit-learn documentation](https://scikit-learn.org/stable/) · [Streamlit documentation](https://docs.streamlit.io/) · [Plotly documentation](https://plotly.com/python/)")

st.markdown("---")
st.markdown("<p class='small-note'>SmartStudy AI · Internship learning prototype · Outputs are estimates for demonstration and must be reviewed by a human.</p>", unsafe_allow_html=True)
