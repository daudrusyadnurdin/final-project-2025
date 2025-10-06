#-----------------------------------------------------------------------------------------------------------------
# Call library, esp. streamlit, pandas, numpy, etc
#-----------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np #number

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

#-----------------------------------------------------------------------------------------------------------------
# --- Page configuration
#-----------------------------------------------------------------------------------------------------------------
st.set_page_config(
   page_title="Telco Customer Churn Analysis", # @browser
   page_icon="📊",
   layout="wide",
   initial_sidebar_state="expanded"
)

#-----------------------------------------------------------------------------------------------------------------
# Retrieve the data: Telco_customer_churn.csv @github
#-----------------------------------------------------------------------------------------------------------------
@st.cache_data # to enhance performance
def load_data():
    url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/data/Telco_customer_churn.csv"
    return pd.read_csv(url)

df = load_data()

# Special case: handling data type conversion
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

#-----------------------------------------------------------------------------------------------------------------
# App banner and title & description
#-----------------------------------------------------------------------------------------------------------------

# Gambar header dari GitHub (RAW URL)
header_image_url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/assets/telco-business.jpg"
st.image(header_image_url, use_container_width=True)

st.title("🔄 Telco Customer Churn Analysis")
st.markdown("""
            The telecommunications industry is highly competitive, making customer churn a particular concern, 
            given the increasingly high cost of acquiring new customers. 
            Companies strive to reduce this churn rate as much as possible, thereby increasing cost efficiency. 
            The following is a simulation of a churn analysis for a telecommunications company in California, USA.
""")

#-----------------------------------------------------------------------------------------------------------------
# Filtering parameter
#-----------------------------------------------------------------------------------------------------------------
# Sidebar filters
st.sidebar.header("Filtering parameters")

# Geographic filters
flt_city = st.sidebar.multiselect("City", 
                                    options=sorted(df['City'].dropna().unique())
                                 )

# Demographic filters
flt_gender = st.sidebar.multiselect("Gender", 
                                       options=df['Gender'].unique()
                                    )
flt_sr_citizen = st.sidebar.multiselect("Senior Citizen", 
                                          options=df['Senior Citizen'].unique()
                                       )

# Tenure months range
min_tenure, max_tenure = int(df['Tenure Months'].min()), int(df['Tenure Months'].max())
rg_tenure = st.sidebar.slider(
    "Tenure Months",
    min_value=min_tenure,
    max_value=max_tenure,
    value=(min_tenure, max_tenure)
)

# Monthly charges range
min_monchrg, max_monchrg = float(df['Monthly Charges'].min()), float(df['Monthly Charges'].max())
rg_monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=min_monchrg,
    max_value=max_monchrg,
    value=(min_monchrg, max_monchrg)
)

#-----------------------------------------------------------------------------------------------------------------
# Select the data based on filtering parameters
#-----------------------------------------------------------------------------------------------------------------
df_selected = df.copy()

# Apply multiselect filters only if a selection has been made for that filter
# City, Gender & Senior Citizen
if flt_city:
   df_selected = df_selected[df_selected["City"].isin(flt_city)]
if flt_gender:
   df_selected = df_selected[df_selected["Gender"].isin(flt_gender)]
if flt_sr_citizen:
   df_selected = df_selected[df_selected["Senior Citizen"].isin(flt_sr_citizen)]

# Always apply the slider filter
# Tenure
df_selected = df_selected[
   (df_selected["Tenure Months"] >= rg_tenure[0]) &
   (df_selected["Tenure Months"] <= rg_tenure[1])
]
# Monthly Charges
df_selected = df_selected[
   (df_selected["Monthly Charges"] >= rg_monthly_charges[0]) &
   (df_selected["Monthly Charges"] <= rg_monthly_charges[1])
]

# Display error message if no data is selected
if df_selected.empty:
   st.warning("No data available for the selected filters. Please adjust your selection.")
   st.stop() # Halts the app execution

#-----------------------------------------------------------------------------------------------------------------
# Dashboard & reporting
#-----------------------------------------------------------------------------------------------------------------
#-------------------------------------
# KPI
#-------------------------------------
st.subheader("🔢 Key Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

total_customers = len(df_selected)
churn_customers = df_selected['Churn Label'].value_counts().get('Yes', 0)
total_revenue = df_selected['Total Charges'].sum()
churn_revenue = df_selected[df_selected['Churn Label'] == 'Yes']['Total Charges'].sum()

with col1:
    st.metric("Total Customers", f"{total_customers:,}")
with col2:
    st.metric("Total Churn", f"{churn_customers:,}")
with col3:
    st.metric("Churn Rate", f"{(churn_customers/total_customers)*100:.1f}%")
with col4:
    st.metric("Total Revenue (k US$)", f"${total_revenue/1000:,.2f}")
with col5:
    st.metric("Revenue at Risk (k US$)", f"${churn_revenue/1000:,.2f}")
with col6:
    st.metric("% Revenue at Risk", f"{(churn_revenue/total_revenue)*100:.2f}%")

st.markdown("---")

#------------------------------------------
# Churn reason: based on customer
#------------------------------------------
st.subheader("💔 Why our customer churn?")
# --- Buat Figure ---
fig, ax = plt.subplots(figsize=(10, 6))  

# --- Data ---
df_cr = (
    df_selected.groupby("Churn Reason")["Churn Value"]
    .sum()
    .reset_index()
    .sort_values(by="Churn Value")  # urut dari kecil ke besar
)

# --- Buat daftar warna dinamis ---
n = len(df_cr)
# default semua lightgrey
colors = ['#E2E2E2'] * n
# ganti 3 terakhir (nilai tertinggi) jadi lightsalmon, salmon, tomato
top_colors = ["tomato", "salmon", "lightsalmon"]
for i, c in enumerate(top_colors, start=1):
    if i <= n:  # antisipasi jika data < 3
        colors[-i] = c

# --- Plot Horizontal Bar ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(df_cr["Churn Reason"], df_cr["Churn Value"], color=colors)

ax.bar_label(ax.containers[0], padding=3)

# --- Judul & Label ---
ax.set_ylabel("Churn Reason")
ax.set_xlabel("Total Churn")

# --- Grid & Style ---
ax.grid(axis="x", ls="--", color="lavender")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# --- Tampilkan di Streamlit ---
st.pyplot(fig)

st.markdown("---")

#-------------------------
# set columns
#-------------------------
col_custom = 'black'
col1, col2 = st.columns(2)

with col1:
    #-----------------------------------------------------------
    # Top 10 - Total Charges
    #-----------------------------------------------------------
    st.subheader("💰 Top 10 most productive cities...")

    # --- Data (contoh sesuai kode Anda) ---
    df_chrg_chrn10 = (df_selected.groupby('City')[['Total Charges', 'Churn Value']].sum().reset_index()
                      .sort_values(by='Total Charges', ascending=False).head(10).reset_index(drop=True)
                    )
    df_barh = df_chrg_chrn10.sort_values(by='Total Charges')

    # --- Figure & Axis ---
    fig, ax = plt.subplots(figsize=(10, 6)) 

    # --- Plot Horizontal Bar ---
    colors = ['#E2E2E2'] * len(df_barh)
    # colors[-3:] = ['lightgreen', 'limegreen', 'green'] --> original ver @ assignment day 13
    colors[-3:] = ['#deebf7', '#6baed6', '#08519c']
    ax.barh(
        df_barh['City'],
        df_barh['Total Charges'] / 1000,  # dibagi 1000 sesuai label
        color=colors
    )
    ax.tick_params(axis='x', labelsize=15)   # angka pada sumbu X
    ax.tick_params(axis='y', labelsize=15)   # nama kota pada sumbu Y

    # --- Judul & Label ---
    ax.set_ylabel("City", fontsize=16, color=col_custom)
    ax.set_xlabel("Total Charges (x1000 US$)", fontsize=16, color=col_custom)

    # --- Grid & Style ---
    ax.grid(axis='x', ls='--', color='lavender')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Annotasi Bar ---
    ax.bar_label(ax.containers[0], fmt='%.2f', padding=3, fontsize=15)

    plt.tight_layout()

    # --- Tampilkan di Streamlit ---
    st.pyplot(fig)

with col2:
    #-----------------------
    #TOP 10 - TOTAL CUSTOMER
    #-----------------------
    st.subheader("👥 Top 10 most customers cities...")

    fig, ax = plt.subplots(figsize=(10, 6)) 

    df_barh = (df_selected.groupby("City")['CustomerID'].count().reset_index()
               .sort_values(by='CustomerID', ascending=False).head(10)
               .sort_values(by='CustomerID')
                )
    colors = ['#E2E2E2'] * len(df_barh)

    # colors[-3:] = ['lightgreen', 'limegreen', 'green'] --> original version @Assignment Day 13
    colors[-3:] = ['#deebf7', '#6baed6', '#08519c']
    ax.barh(  df_barh['City'],
                df_barh['CustomerID'],
                color=colors
            )
    ax.tick_params(axis='x', labelsize=15)   # angka pada sumbu X
    ax.tick_params(axis='y', labelsize=15)   # nama kota pada sumbu Y

    ax.set_ylabel("City", fontsize=16, color=col_custom)
    ax.set_xlabel("Total Customer", fontsize=16, color=col_custom)

    ax.grid(axis='x', ls='--', color='lavender')

    #annotate bars
    ax.bar_label(ax.containers[0], padding=3, fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

#-------------------------
# set new columms
#-------------------------
col3, col4 = st.columns(2)

with col3:
    #-------------------
    #TOP 10 - TOTAL CURN
    #-------------------
    st.subheader("💔 Top 10 most churn cities...")

    fig, ax = plt.subplots(figsize=(10, 6)) 
    df_barh = (df_selected
                .groupby("City")['Churn Value'].sum()
                .reset_index()
                .sort_values(by='Churn Value', ascending=False)
                .head(10)
                .sort_values(by='Churn Value')
                )
    n = len(df_barh)
    colors = ['#E2E2E2'] * n
    colors[-3:] = ["lightsalmon", "salmon", "tomato"]
    ax.barh(  df_barh['City'],
              df_barh['Churn Value'],
              color=colors
            )
    ax.tick_params(axis='x', labelsize=15)   # angka pada sumbu X
    ax.tick_params(axis='y', labelsize=15)   # nama kota pada sumbu Y

    ax.set_ylabel("City", fontsize=16, color=col_custom)
    ax.set_xlabel("Total Churn", fontsize=16, color=col_custom)

    ax.grid(axis='x', ls='--', color='lavender')

    #annotate bars
    ax.bar_label(ax.containers[0], padding=3, fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

with col4:
    #-------------------
    #TOP 10 - % CURN
    #-------------------
    st.subheader("")

    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    # Hitung dulu total customer untuk di-merged dengan top-10 churn, yang sudah dihitung sebelumnya
    # informasi ini hanya pelengkap saja.
    df_barh2 = (df.groupby("City")['CustomerID'].count().reset_index()
                .sort_values(by='CustomerID', ascending=False) #.head(10) BUG!!
                .sort_values(by='CustomerID')
                )
    # Join/merge berdasarkan kolom City
    # df_barh sudah dihitung sebelumnya, karena grafik ini kelanjutan dari grafik sebelumnya
    df_merge = pd.merge(
        df_barh,
        df_barh2,
        on="City",     # kolom kunci
        how="inner"    # mengikuti yang 10 
    )

    # urutkan kembali
    # Buat kolom persentase churn
    df_merge['Churn %'] = df_merge['Churn Value'] / df_merge['CustomerID'] * 100

    # Urut berdasarkan kolom baru
    df_merge = df_merge.sort_values(by='Churn %', ascending=True)

    n = len(df_merge)
    colors = ['#E2E2E2'] * n
    colors[-3:] = ["lightsalmon", "salmon", "tomato"]
    ax.barh(  df_merge['City'],
            #   df_sorted['Churn Value']/df_merge['CustomerID']*100,
              df_merge['Churn %'],
              color=colors
            )
    #annotate bars
    ax.bar_label(ax.containers[0], padding=3, fontsize=15, fmt='%.2f%%')

    ax.tick_params(axis='x', labelsize=15)   # angka pada sumbu X
    ax.tick_params(axis='y', labelsize=15)   # nama kota pada sumbu Y

    ax.set_ylabel("City", fontsize=16, color=col_custom)
    ax.set_xlabel("%age Churn", fontsize=16, color=col_custom)

    ax.grid(axis='x', ls='--', color='lavender')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

st.markdown("---")

#-----------------------------------------------------------------------------------------------------------------
# CLTV vs Churn analysis
#-----------------------------------------------------------------------------------------------------------------
st.subheader("📊 Will our high level CLTV customers churn?")
st.markdown("""
    Analyzing customer lifetime value relationship with churn behavior.
""")

# Create CLTV segments
q1 = df_selected['CLTV'].quantile(0.33)
q2 = df_selected['CLTV'].quantile(0.66)

def segment(x):
    if x >= 5000:      # contoh batas nilai CLTV dalam $
        return 'Platinum'
    elif x >= 2500:
        return 'Gold'
    elif x >= 1000:
        return 'Silver'
    else:
        return 'Bronze'

df_selected['CLTV Segment'] = df_selected['CLTV'].apply(segment)

# Main analysis
segment_analysis = df_selected.groupby('CLTV Segment').agg({
                                                                'CustomerID'     : 'count',
                                                                'Churn Value'    : 'sum',
                                                                'CLTV'           : 'mean',
                                                                'Monthly Charges': 'mean',
                                                                'Tenure Months'  : 'mean'
                                                            }).reset_index()

segment_analysis['Churn Rate (%)'] = (segment_analysis['Churn Value'] / segment_analysis['CustomerID']) * 100
segment_analysis['Avg CLTV'] = segment_analysis['CLTV']

# Order segments
segment_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
segment_analysis['CLTV Segment'] = pd.Categorical(
                                                    segment_analysis['CLTV Segment'], 
                                                    categories=segment_order, 
                                                    ordered=True
                                                )
segment_analysis = segment_analysis.sort_values('CLTV Segment')


# colors = ['#CD7F32', 'silver', 'gold', '#E2E2E2'] 
colors = ['silver', 'gold', '#E2E2E2'] 

# Row 1: Main charts
col1, col2 = st.columns(2)

with col1:
    # Churn Rate by CLTV Segment - Plotly
    fig1 = px.bar(
                    segment_analysis,
                    x='CLTV Segment',
                    y='Churn Rate (%)',
                    title='<b>Churn Rate by CLTV Segment</b>',
                    color='CLTV Segment',
                    color_discrete_sequence=colors,
                    text_auto='.1f'
                )
    fig1.update_layout(
                        xaxis_title="CLTV Segment",
                        yaxis_title="Churn Rate (%)",
                        showlegend=False
                    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Customer Distribution - Plotly
    fig2 = px.pie(
                    segment_analysis,
                    values='CustomerID',
                    names='CLTV Segment',
                    title='<b>Customer Distribution by CLTV Segment</b>',
                    color='CLTV Segment',
                    color_discrete_sequence=colors
                )
    
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

# Row 2: Additional metrics
col3, col4 = st.columns(2)

with col3:
    # Average CLTV by Segment
    fig3 = px.bar(
                    segment_analysis,
                    x='CLTV Segment',
                    y='Avg CLTV',
                    title='<b>Average CLTV by Segment</b>',
                    color='CLTV Segment',
                    color_discrete_sequence=colors,
                    text_auto='.0f'
                )
    fig3.update_layout(
                    xaxis_title="CLTV Segment",
                    yaxis_title="Average CLTV ($)",
                    showlegend=False
                )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # Monthly Charges vs Tenure
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
                        name='Avg Monthly Charges',
                        x=segment_analysis['CLTV Segment'],
                        y=segment_analysis['Monthly Charges'],
                        marker_color=colors,
                        text=segment_analysis['Monthly Charges'].round(1),
                        textposition='auto'
                    ))
    fig4.add_trace(go.Scatter(
                        name='Avg Tenure (Months)',
                        x=segment_analysis['CLTV Segment'],
                        y=segment_analysis['Tenure Months'],
                        mode='lines+markers+text',
                        line=dict(color='#6baed6', width=3),
                        marker=dict(size=10),
                        text=segment_analysis['Tenure Months'].round(1),
                        textposition='top center',
                        yaxis='y2'
                    ))
   
    fig4.update_layout(
                        title='<b>Monthly Charges & Tenure by Segment</b>',
                        xaxis_title="CLTV Segment",
                        yaxis_title="Monthly Charges ($)",
                        yaxis2=dict(
                            title="Tenure (Months)",
                            overlaying='y',
                            side='right'
                        ),
                        showlegend=True
                    )
    st.plotly_chart(fig4, use_container_width=True)

# Defisi CLTV segmen
st.markdown("**CLTV segmen Definition:** ")
st.markdown("""
            - CLTV ≥ 5000        : 'Platinum'
            - 2500 ≤ CLTV < 5000 : 'Gold' 
            - 1000 ≤ CLTV < 2500 : 'Silver'
            - CLTV < 1000        : 'Bronze' 
            """)

st.markdown("---")

st.markdown("""
#### General Conclusions
- A churn rate of 26.5% (assuming a year) is quite high for a telco business in the US. 
  According to several sources, the normal annual churn rate ranges from 15-25% (source: CustomerGauge), with an average of 22%. 
  Therefore, this company's churn rate is considered high.
- Financially, it results in a loss of revenue of approximately 17.8% in the future, potentially losing nearly one-fifth of revenue 
  if churn is not addressed. 
  Furthermore, if a customer acquisition program is implemented and the costs are high, 
  this will make it difficult for the company to maintain financial stability (revenue).
- Mitigation opportunities include focusing on customer retention programs, which require strategies such as 
  loyalty programs, upselling, proactive support, and service improvements to be prioritized.
- In short, this telco company is currently facing quite high churn and the potential for significant revenue losses. 
  Customer retention must be a primary focus to maintain revenue stability
"""
)
st.markdown("---")

#-----------------------------------------------------------------------------------------------------------------
# --- Display of raw data
#-----------------------------------------------------------------------------------------------------------------
with st.expander("View of Raw Data"):
   st.dataframe(df)
   st.markdown(f"**Data Dimensions:** {df.shape[0]} rows, {df.shape[1]} columns")


#-----------------------------------------------------------------------------------------------------------------
# Some additional information of this work
#-----------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.write("Data Source: [Telco Customer Churn Dataset](https://github.com/daudrusyadnurdin/marketing-analysis)")
st.write("") 
st.write("""
        This assignment was created with reference to the previous assignment, namely:
        **Day 13 - Fundamentals of Data Visualization**, with several modifications adapted to the environment in Streamlit.
        Based on the dataset's structure, no date information was found. 
        This data represents a snapshot of customer status (churn/non-churn) at a telco company in California, USA.
         """)

#---The End-------------------------------------------------------------------------------------------------------
