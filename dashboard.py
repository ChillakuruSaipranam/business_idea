
import streamlit as st
st.set_page_config(layout="wide", page_title=" Business analysis based on US ZIP Code ")

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import ttest_ind

# Cache the dataset loading
@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")
    return df

# Load dataset
df = load_data()

# Page Header
st.title("Unlocking Business Success Through Strategic Location Analysis Dashboard")
st.markdown("Explore economic data at the ZIP code level across the United States.")

# ------------------------ CUSTOM STYLED SIDEBAR ------------------------
# Menu items
menu_items = [
    "Home",
    "Map View: AGI by State + ZIP Explorer",
    "Population Overview by State",
    "Predictive Model Explorer",
    "ZIP Code Recommender",
    "Hypothesis Testing Viewer"
]

# Set default menu in session state
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = menu_items[0]


st.markdown("""
    <style>
    section[data-testid="stSidebar"] div.stButton {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Default button style */
    div.stButton > button {
        width: 100%;
        text-align: left;
        background-color: #eaecef;
        color: #2c3e50;
        padding: 0.4em 0.75em;
        border: none;
        font-size: 14px;
        margin: 1px 0;
        border-radius: 6px;
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color: #d6dbe3;
        cursor: pointer;
    }

    /* Selected style */
    div.stButton.selected > button {
        background-color: #c3daf5 !important;
        color: #000000 !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Render sidebar buttons
st.sidebar.markdown("### Navigation")
for item in menu_items:
    selected = st.session_state.selected_menu == item
    css_class = "stButton selected" if selected else "stButton"

    with st.sidebar:
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        if st.button(item, key=item):
            st.session_state.selected_menu = item
        st.markdown("</div>", unsafe_allow_html=True)

# Assign current section
section = st.session_state.selected_menu


# --------------------------------------------------------

if section == "Home":
    col1, col2, col3 = st.columns([1,2,1])  
    with col2:  
        st.image("usmap.jpg", use_container_width=True, width=400) 

    st.title("ZIP Code Economic Opportunity Dashboard")
    st.markdown("""
    Welcome to the **ZIP Code Economic Analysis Dashboard**!  
    This interactive tool uses real U.S. data from the IRS, Census Bureau, and Business Statistics to:

    - Visualize economic activity by ZIP code and state  
    - Explore how income, population, and education affect business density  
    - Build and test predictive models for economic potential  
    - Recommend ZIP codes based on user preferences  
    - Perform statistical hypothesis testing to drive real insights  

    ---
    """)

    st.markdown("### Connect with Me")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/sai-pranam-reddy-chillakuru/)")
    with col2:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/ChillakuruSaipranam)")

# SECTION 1: MAP VIEW + ZIP EXPLORER
elif section == "Map View: AGI by State + ZIP Explorer":
    st.header("AGI by State + ZIP Explorer")

    # AGI by state choropleth
    state_agi = df.groupby("STATE")["Total_AGI"].sum().reset_index()

    fig = px.choropleth(
        state_agi,
        locations="STATE",
        locationmode="USA-states",
        color="Total_AGI",
        color_continuous_scale="Viridis",
        scope="usa",
        title="Total Adjusted Gross Income (AGI) by State"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ZIP Code Table
    st.subheader("Explore ZIP Codes by State")
    selected_state = st.selectbox("Select a State to view ZIP codes", sorted(df["STATE"].unique()))
    filtered_zip_df = df[df["STATE"] == selected_state]

    st.dataframe(
        filtered_zip_df[["ZIP", "city", "Total_AGI", "Total_Population", "Median_Income",
                         "Percent_Bachelor_or_Higher", "Total_Businesses"]]
        .sort_values(by="Total_AGI", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------
# SECTION 2: PREDICTIVE MODEL
elif section == "Predictive Model Explorer":
    st.header("Predictive Model: Classify High AGI ZIPs")

    # Binary classification target
    df['AGI_Binary'] = df['AGI_Level'].apply(lambda x: 1 if x in ['High', 'Mid-High'] else 0)

    # Selected features
    features = ['Total_Population', 'Median_Income', 'Percent_Bachelor_or_Higher',
                'Total_Businesses', 'AGI_Per_Capita', 'Business_Density']

    # Preprocessing and splitting
    X = StandardScaler().fit_transform(df[features])
    y = df['AGI_Binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Model selection
    model_type = st.selectbox("Choose model", ["Logistic Regression", "Random Forest"])
    model = LogisticRegression(max_iter=1000) if model_type == "Logistic Regression" else RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation: Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)

   
    

    # Extract key metrics for interpretation
    accuracy = report["accuracy"]
    high_precision = report["1"]["precision"]
    high_recall = report["1"]["recall"]

    # Business Insights from Model
    st.subheader("Model-Based Business Insights")

    if model_type == "Logistic Regression":
        st.markdown(f"""
        ### Logistic Regression Insights
        - This model achieved an **accuracy of {accuracy:.2f}**, and for predicting high AGI ZIP codes:
            - **Precision** = {high_precision:.2f}
            - **Recall** = {high_recall:.2f}
        - These results show that the model is **reliably identifying high-AGI ZIP codes based on a few strong linear indicators**.
        - From the dataset, ZIP codes with:
            - **Median Income > $70,000**
            - **% with Bachelor's Degree > 40%**
          were frequently predicted as high AGI.
        - Because it’s interpretable, we can conclude:
            - `Median_Income` and `Percent_Bachelor_or_Higher` have a **positive influence** on classification.
        - **Business takeaway**:
            - Focus expansion in ZIP codes where income and education are both strong — these areas are **statistically most likely to succeed economically**.
        """)

    else:
        st.markdown(f"""
        ### Random Forest Insights
        - This model achieved an **accuracy of {accuracy:.2f}**, and for predicting high AGI ZIP codes:
            - **Precision** = {high_precision:.2f}
            - **Recall** = {high_recall:.2f}
        - Random Forest captured **non-linear patterns** and **feature interactions**:
            - ZIP codes with **moderate population (20k–60k)** and **AGI per capita > 4.5** were often predicted as high AGI.
            - Some ZIPs with **lower education levels** but **very high business density** were also classified as high AGI.
        - Most important predictors (from model analysis):
            - `Median_Income`, `AGI_Per_Capita`, and `Business_Density`
        - **Business takeaway**:
            - Use Random Forest to uncover **non-obvious but high-potential ZIP codes**.
            - Ideal for spotting **emerging growth areas** where traditional models may fail.
        """)

    # Evaluation: Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    col1, col2, col3 = st.columns([1,2,1])  
    with col2:  
        st.pyplot(fig)

# --------------------------------------------------------
# SECTION 3: ZIP RECOMMENDER
elif section == "ZIP Code Recommender":
    st.header("ZIP Code Recommender")

    pop_min, pop_max = st.slider("Population range", 0, int(df['Total_Population'].max()), (5000, 50000))
    income_min, income_max = st.slider("Median income range", 0, int(df['Median_Income'].max()), (30000, 90000))
    edu_min = st.slider("Minimum % Bachelor's Degree", 0, 100, 20)

    filtered = df[
        (df['Total_Population'] >= pop_min) &
        (df['Total_Population'] <= pop_max) &
        (df['Median_Income'] >= income_min) &
        (df['Median_Income'] <= income_max) &
        (df['Percent_Bachelor_or_Higher'] >= edu_min)
    ]

    st.subheader(f"Matching ZIP Codes ({len(filtered)} found)")
    st.dataframe(filtered[["ZIP","city", "STATE", "Total_AGI", "Median_Income",
                           "Percent_Bachelor_or_Higher", "Total_Businesses"]].head(20))

# --------------------------------------------------------
# SECTION 4: HYPOTHESIS TESTING
elif section == "Hypothesis Testing Viewer":
    st.header("Hypothesis: Does Business Density Affect AGI?")

    q25 = df['Business_Density'].quantile(0.25)
    q75 = df['Business_Density'].quantile(0.75)

    low_density = df[df['Business_Density'] <= q25]
    high_density = df[df['Business_Density'] >= q75]

    t_stat, p_val = ttest_ind(high_density['Total_AGI'], low_density['Total_AGI'])

    st.write(f"**T-statistic**: {t_stat:.2f}")
    st.write(f"**P-value**: {p_val:.4f}")
    if p_val < 0.05:
        st.success("Statistically significant: Business density *impacts* AGI.")
    else:
        st.warning("Not statistically significant.")

# --------------------------------------------------------

# SECTION 5: POPULATION OVERVIEW BY STATE
elif section == "Population Overview by State":
    st.header("Population Overview by State")

    state_population = df.groupby("STATE")["Total_Population"].sum().reset_index()
    fig = px.bar(state_population, x="STATE", y="Total_Population",
                 title="Total Population by State", color="Total_Population",
                 color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------
