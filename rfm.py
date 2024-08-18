import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

st.set_page_config(layout="wide")

st.title("RFM Analysis and Visualization Dashboard")

st.markdown("""
This dashboard presents a detailed RFM (Recency, Frequency, Monetary) analysis of customer data. 
Upload your CSV file containing customer purchase data to get started.
""")

st.sidebar.header("About RFM Analysis")
st.sidebar.markdown("""
RFM (Recency, Frequency, Monetary) analysis is a marketing technique used to determine quantitatively which customers are the best ones by examining how recently a customer has purchased (Recency), how often they purchase (Frequency), and how much the customer spends (Monetary).

Your CSV file should include the following columns:
1. Customer ID
2. Purchase Date
3. Purchase Amount


Developed by Neo Mauizan
Example CSV File can be download here https://bit.ly/DummyDataset
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display raw data
    st.subheader("Raw Data Preview")
    st.write(df.head())
    
    # Column mapping
    st.subheader("Column Mapping")
    st.write("Please map your columns to the required RFM fields:")
    
    customer_id_col = st.selectbox("Select Customer ID column", df.columns)
    date_col = st.selectbox("Select Purchase Date column", df.columns)
    amount_col = st.selectbox("Select Purchase Amount column", df.columns)
    
    # Display sample dates
    st.write("Sample dates from your data:", df[date_col].head())
    
    # Date parsing options
    date_parse_method = st.radio("Choose date parsing method:", 
                                 ("Specify format", "Auto-detect format"))
    
    if date_parse_method == "Specify format":
        date_format = st.text_input("Enter the date format (e.g., '%Y-%m-%d' for YYYY-MM-DD)", "%Y-%m-%d")
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format).dt.date
        except ValueError as e:
            st.error(f"Error parsing dates: {str(e)}")
            st.error("Please check your date format and try again.")
            st.stop()
    else:
        try:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True).dt.date
        except ValueError as e:
            st.error(f"Error auto-detecting date format: {str(e)}")
            st.error("Please try specifying the date format manually.")
            st.stop()
    
    # Verify date parsing
    st.write("Parsed dates:", df[date_col].head())
    
    # RFM Calculation
    current_date = df[date_col].max()
    rfm_df = df.groupby(customer_id_col).agg({
        date_col: lambda x: (current_date - x.max()).days,
        customer_id_col: 'count',
        amount_col: 'sum'
    })

    rfm_df.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Display RFM summary statistics
    st.subheader("RFM Summary Statistics")
    st.write(rfm_df.describe())
    
    # Scoring
    # For Frequency
    freq_values = sorted(rfm_df['Frequency'].unique())
    if len(freq_values) <= 4:
        freq_labels = range(1, len(freq_values) + 1)
        rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'], bins=[-np.inf] + freq_values, labels=freq_labels, include_lowest=True)
    else:
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'], q=4, labels=[1, 2, 3, 4], duplicates='drop')

    # For Recency and Monetary
    for col in ['Recency', 'Monetary']:
        col_name = col[0] + "_Score"
        try:
            rfm_df[col_name] = pd.qcut(rfm_df[col], q=4, labels=[4, 3, 2, 1], duplicates='drop')
        except ValueError:
            st.warning(f"Could not create 4 equal-sized bins for {col}. Using custom binning instead.")
            if col == 'Recency':
                bins = [0, 30, 90, 180, float('inf')]
                labels = [4, 3, 2, 1]
            else:  # Monetary
                monetary_median = rfm_df[col].median()
                bins = [0, monetary_median/2, monetary_median, monetary_median*2, float('inf')]
                labels = [1, 2, 3, 4]
            rfm_df[col_name] = pd.cut(rfm_df[col], bins=bins, labels=labels, include_lowest=True)

    # Display score distributions
    st.subheader("RFM Score Distribution")
    for col in ['R_Score', 'F_Score', 'M_Score']:
        st.write(f"{col} distribution:")
        score_distribution = rfm_df[col].value_counts().sort_index()
        score_df = pd.DataFrame({col: score_distribution.index, 'Count': score_distribution.values})
        st.write(score_df)

    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(int) + rfm_df['F_Score'].astype(int) + rfm_df['M_Score'].astype(int)

    # Customer Segmentation
    def segment_customers(row):
        if row['RFM_Score'] >= 9:
            return 'Best Customers'
        elif (row['R_Score'] == 4) and (row['F_Score'].astype(int) + row['M_Score'].astype(int) >= 4):
            return 'Lost Best Customers'
        elif (row['R_Score'] >= 3) and (row['F_Score'].astype(int) + row['M_Score'].astype(int) >= 3):
            return 'Loyal Customers'
        elif (row['R_Score'] >= 3) and (row['F_Score'].astype(int) + row['M_Score'].astype(int) <= 2):
            return 'New Customers'
        return 'Lost Cheap Customers'

    rfm_df['Customer_Segment'] = rfm_df.apply(segment_customers, axis=1)

    # Data Validation
    st.subheader("Data Validation")
    st.write("Total customers:", len(rfm_df))
    st.write("Customers per segment:")
    st.write(rfm_df['Customer_Segment'].value_counts())

    # Visualizations
    st.header("RFM Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RFM Scores Distribution")
        fig_hist = px.histogram(rfm_df, x='RFM_Score', nbins=20, 
                                title="Distribution of RFM Scores",
                                labels={'RFM_Score': 'RFM Score', 'count': 'Number of Customers'})
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Customer Segmentation")
        fig_pie = px.pie(rfm_df, names='Customer_Segment', title="Customer Segments")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.header("Detailed RFM Analysis")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("RFM Heatmap")
        rfm_heatmap = rfm_df.pivot_table(index='R_Score', columns='F_Score', values='Monetary', aggfunc='mean')
        fig_heatmap, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(rfm_heatmap, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5, ax=ax)
        plt.title("RFM Heatmap: Average Monetary Value")
        plt.xlabel("Frequency Score")
        plt.ylabel("Recency Score")
        st.pyplot(fig_heatmap)

    with col4:
        st.subheader("RFM 3D Scatter Plot")
        fig_3d = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', 
                               color='Customer_Segment', hover_name=rfm_df.index,
                               labels={'Recency': 'Recency (days)', 'Frequency': 'Frequency (count)', 'Monetary': 'Monetary (total spend)'},
                               title="3D RFM Visualization")
        st.plotly_chart(fig_3d, use_container_width=True)

    st.header("Customer Insights and Recommendations")

    st.markdown("""
    Based on the RFM analysis, here are some key insights and recommendations:

    1. **Best Customers**: Focus on retaining them through personalized offers and exclusive benefits.
    2. **Lost Best Customers**: Reach out with targeted win-back campaigns.
    3. **Loyal Customers**: Encourage increased purchase frequency or try new products.
    4. **New Customers**: Convert them into loyal ones with excellent service and relevant recommendations.
    5. **Lost Cheap Customers**: Analyze cost-effectiveness of re-engagement efforts.
    """)

    if st.checkbox("Show Raw RFM Data"):
        st.subheader("Raw RFM Data")
        st.write(rfm_df)

else:
    st.info("Please upload a CSV file to begin the RFM analysis.")
