import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Customer Journey System", layout="wide")
st.title("(Customer Journey System)")

# --- 1. Data Loading & Cleaning ---
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data_all.xltx")
        # Ensure correct column names if they differ
        column_map = {
            'Region': 'Country',
            'Solution': 'solution',
            'Action Type': 'types'
        }
        df = df.rename(columns=column_map)
        
        # Basic Cleaning
        required_cols = ['Country', 'solution', 'types', 'activity_date', 'account_id', 'opportunity_stage']
        # Filter only if columns exist
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df.dropna(subset=existing_cols)
        
        # Standardize opportunity_stage
        df['is_won'] = df['opportunity_stage'].apply(lambda x: 1 if str(x).lower() == 'won' else 0)
        
        return df
    except Exception as e:
        return None

df = load_data()

if df is None:
    st.error("Error loading 'data_all.xltx'. Please ensure the file exists and has the correct format.")
    st.stop()

# --- 2. Journey & Path Processing ---
@st.cache_data
def process_journeys(data):
    # Sort by date
    data = data.sort_values(by=['account_id', 'activity_date'])
    
    # Group by account to get paths
    # We create a string representation of the path: "Email -> Call -> Meeting"
    paths = data.groupby(['account_id', 'Country', 'solution', 'is_won'])['types'].apply(lambda x: ' -> '.join(x)).reset_index()
    paths.rename(columns={'types': 'path'}, inplace=True)
    return paths

journeys_df = process_journeys(df)

# --- 3. Decision Tree Analysis (Feature Importance) ---
def train_dt_for_importance(data_subset):
    if data_subset.empty:
        return pd.Series()
        
    # Pivot to get count of each action type per account
    # We need a matrix: Rows=Accounts, Cols=Action Types, Values=Count
    # Filter data_subset to only include Won/Lost (ignore others for training if needed, but here we use all with is_won)
    
    # We need to aggregate raw actions for the subset accounts
    subset_accounts = data_subset['account_id'].unique()
    raw_subset = df[df['account_id'].isin(subset_accounts)]
    
    X = pd.crosstab(raw_subset['account_id'], raw_subset['types'])
    # Align y (outcome)
    y = raw_subset.groupby('account_id')['is_won'].max() # If any won, it's a win
    
    # Align indices
    X = X.loc[y.index]
    
    if len(y.unique()) < 2:
        return pd.Series(dtype=float) # Cannot train if only one class
        
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X, y)
    
    importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importance

# --- Sidebar Inputs ---
st.sidebar.header("Filter Options")
all_countries = sorted(df['Country'].unique().astype(str))
all_solutions = sorted(df['solution'].unique().astype(str))

selected_country = st.sidebar.selectbox("Select Country", all_countries)
selected_solution = st.sidebar.selectbox("Select Solution", all_solutions)

# --- Main Layout ---

# A. Top 5 Paths
st.header("1. Top 5 Customer Journey Paths")
st.markdown("Most frequent sequences of actions for the selected segment.")

filtered_journeys = journeys_df[
    (journeys_df['Country'] == selected_country) & 
    (journeys_df['solution'] == selected_solution)
]

if not filtered_journeys.empty:
    top_paths = filtered_journeys['path'].value_counts().head(5)
    st.table(top_paths.reset_index(name='Count').rename(columns={'index': 'Path'}))
else:
    st.info("No journeys found for this selection.")

st.divider()

# B. Action Recommendations (DT Based)
st.header("2. Top 4 Next Best Actions")

# Calculate Feature Importances for different contexts
# 1. By Country
subset_country = df[df['Country'] == selected_country]
imp_country = train_dt_for_importance(subset_country)

# 2. By Solution
subset_solution = df[df['solution'] == selected_solution]
imp_solution = train_dt_for_importance(subset_solution)

# 3. By Country & Solution
subset_both = df[(df['Country'] == selected_country) & (df['solution'] == selected_solution)]
imp_both = train_dt_for_importance(subset_both)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(f"By Country: {selected_country}")
    if not imp_country.empty:
        st.write(imp_country.head(4))
    else:
        st.write("Insufficient data.")

with col2:
    st.subheader(f"By Solution: {selected_solution}")
    if not imp_solution.empty:
        st.write(imp_solution.head(4))
    else:
        st.write("Insufficient data.")

with col3:
    st.subheader("By Both")
    if not imp_both.empty:
        st.write(imp_both.head(4))
    else:
        st.write("Insufficient data.")

st.divider()

# C. Dynamic Weighting & "Add Action" System
st.header("3. Dynamic Action Weights & Simulation")

# Initialize Session State for Weights if not exists
if 'action_weights' not in st.session_state:
    # Default weights - can be customized
    unique_actions = df['types'].unique()
    st.session_state['action_weights'] = {action: 0.5 for action in unique_actions}

# Display Current Top Recommendations based on Weighted Importance
# Base importance comes from the "Both" model (most specific)
if not imp_both.empty:
    base_scores = imp_both
else:
    base_scores = imp_country # Fallback

# Calculate Weighted Scores
weighted_scores = {}
for action, score in base_scores.items():
    current_weight = st.session_state['action_weights'].get(action, 0.5)
    weighted_scores[action] = score * current_weight

sorted_weighted_actions = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:4]

st.subheader("Recommended Next Actions (Weighted)")
st.caption("Based on Model Importance × Current Dynamic Weight")

cols = st.columns(4)
for i, (action, score) in enumerate(sorted_weighted_actions):
    with cols[i]:
        st.metric(label=action, value=f"{score:.3f}", delta=f"W: {st.session_state['action_weights'].get(action, 0.5):.2f}")

# Add Action Interface
st.markdown("### Add Action to Journey")
col_input, col_btn = st.columns([3, 1])

with col_input:
    action_to_add = st.selectbox("Select Action to Add", sorted(df['types'].unique()))

with col_btn:
    st.write("") # Spacer
    st.write("")
    if st.button("Add Action & Update Weights"):
        # Formula: new weight = base weight * (1 - last touch weight)
        # Assuming "base weight" is the weight BEFORE this update.
        # Assuming "last touch weight" is the CURRENT weight of the action being added.
        
        current_w = st.session_state['action_weights'].get(action_to_add, 0.5)
        new_w = current_w * (1.0 - current_w)
        
        # Update state
        st.session_state['action_weights'][action_to_add] = new_w
        
        st.success(f"Added '{action_to_add}'. New weight: {new_w:.4f}")
        st.rerun()

st.divider()

# D. Best Trip to Win
st.header("4. Best Trip Prediction (Winning Path)")

# Logic: Look at Account with 'Won' status in the selected segment.
# Find the Top Path among Winners.
winning_journeys = filtered_journeys[filtered_journeys['is_won'] == 1]

if not winning_journeys.empty:
    best_trip = winning_journeys['path'].value_counts().idxmax()
    count = winning_journeys['path'].value_counts().max()
    st.success(f"🏆 Best Winning Path for {selected_country} / {selected_solution}")
    st.code(best_trip)
    st.write(f"Frequency in Winning Deals: {count}")
    
    # Visualize flow?
    # Simple step visualization
    steps = best_trip.split(" -> ")
    if len(steps) > 1:
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = list(set(steps)),
              color = "blue"
            ),
            link = dict(
              source = [list(set(steps)).index(steps[i]) for i in range(len(steps)-1)],
              target = [list(set(steps)).index(steps[i+1]) for i in range(len(steps)-1)],
              value = [1] * (len(steps)-1)
          ))])
        fig.update_layout(title_text="Best Path Flow", font_size=10, height=300)
        st.plotly_chart(fig)
else:
    st.warning("No winning journeys found for this segment yet.")
