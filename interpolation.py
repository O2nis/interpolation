import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import pchip_interpolate
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="TimeSeries Resampling Tool",
    page_icon="⏱️",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data
def load_data(file):
    """Loads CSV or Excel file."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def process_data(_df, start_datetime, input_freq_str, output_freq_str, selected_cols):
    """
    Performs resampling and interpolation.
    Wrapped in @st.cache_data to prevent re-calculating on download.
    """
    # Mapping frequency strings
    freq_map = {
        'seconds': 'S',
        'minute': 'T',
        '15 minutes': '15T',
        'hourly': 'H'
    }
    
    in_freq = freq_map.get(input_freq_str, 'T')
    out_freq = freq_map.get(output_freq_str, 'H')
    
    # 1. Construct Original Dataframe with Datetime Index
    # Create index based on start time and row count
    original_index = pd.date_range(start=start_datetime, periods=len(_df), freq=in_freq)
    df_orig = _df.copy()
    df_orig.index = original_index
    
    # Ensure numeric
    for col in selected_cols:
        df_orig[col] = pd.to_numeric(df_orig[col], errors='coerce')

    # 2. Create Output Range
    full_range = pd.date_range(start=df_orig.index.min(), end=df_orig.index.max(), freq=out_freq)
    
    results = {}
    
    # 3. Interpolate
    for method in ['Linear', 'PCHIP']:
        df_resampled = pd.DataFrame(index=full_range)
        
        for col in selected_cols:
            series = df_orig[col].dropna()
            
            if len(series) < 2:
                df_resampled[col] = np.nan
                continue

            if method == 'Linear':
                temp = series.reindex(full_range)
                df_resampled[col] = temp.interpolate(method='linear')
                
            elif method == 'PCHIP':
                x_original = series.index.astype(np.int64)
                y_original = series.values
                x_new = full_range.astype(np.int64)
                y_pchip = pchip_interpolate(x_original, y_original, x_new)
                df_resampled[col] = y_pchip
        
        results[method] = df_resampled

    return df_orig, results

# --- Streamlit UI ---

st.title("⏱️ TimeSeries Resampling & Interpolation")
st.markdown("""
Upload a file. The app generates a timeline from your selected Start Date.
Visualize a **specific 24-hour period** to compare interpolation accuracy.
""")

# 1. Sidebar Configuration
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        
        # --- Timeline Settings ---
        st.sidebar.subheader("1. Timeline Definition")
        default_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        start_date = st.sidebar.date_input("Start Date (Row 1)", value=default_start.date())
        start_time = st.sidebar.time_input("Start Time (Row 1)", value=default_start.time())
        start_datetime = datetime.combine(start_date, start_time)
        
        # --- Frequency Settings ---
        st.sidebar.subheader("2. Frequency Settings")
        timestep_options = ["seconds", "minute", "15 minutes", "hourly"]
        
        input_timestep = st.sidebar.selectbox("Input Frequency (File Timestep)", timestep_options, index=2)
        output_timestep = st.sidebar.selectbox("Output Target Frequency", timestep_options, index=3)
        
        # --- Column Selection ---
        st.sidebar.subheader("3. Data Columns")
        all_cols = df.columns.tolist()
        selected_cols = st.sidebar.multiselect("Select Columns", all_cols, default=all_cols)
        
        # --- Process Button ---
        process_btn = st.sidebar.button("Elaborate Data")
        
        if process_btn or 'processed_data' in st.session_state:
            # Check if we need to process or if we can use session state (though caching handles the heavy lifting)
            
            # Perform processing (Cached)
            df_orig, resampled_results = process_data(df, start_datetime, input_timestep, output_timestep, selected_cols)
            
            # Store in session state to avoid re-running logic if inputs haven't changed (though caching is the main fix)
            st.session_state['processed_data'] = resampled_results
            st.session_state['orig_data'] = df_orig
            
            st.success("Processing Complete! (Cached)")
            
            # --- Visualization Section ---
            st.header("24-Hour Analysis Comparison")
            
            # Determine Min/Max date available
            min_date = df_orig.index.min().date()
            max_date = df_orig.index.max().date()
            
            # Widget to select the specific 24h period
            col1, col2 = st.columns([2, 1])
            with col1:
                analysis_date = st.date_input(
                    "Select a Day to Analyze (24h Window)", 
                    value=min_date, 
                    min_value=min_date, 
                    max_value=max_date
                )
            
            with col2:
                st.write("")
                st.write(f"**Date Range:** {min_date} to {max_date}")
            
            # Filter data for the selected day
            start_of_day = datetime.combine(analysis_date, datetime.min.time())
            end_of_day = datetime.combine(analysis_date, datetime.max.time())
            
            # We need to reindex Original data to Output Frequency to compare apples to apples on the chart
            # (Or we just plot markers for original and lines for resampled)
            # Let's plot the resampled data primarily, and the original as a reference if it falls on the same grid,
            # or just plot the original values as scatter points to show where data came from.
            
            for col in selected_cols:
                st.subheader(f"Analysis for: {col}")
                
                # Create a DataFrame for the specific day containing all methods
                # Note: Original data might not have points for every output timestep, so we handle it carefully.
                
                # 1. Get Resampled Data for this day
                df_linear_day = resampled_results['Linear'].loc[start_of_day:end_of_day]
                df_pchip_day = resampled_results['PCHIP'].loc[start_of_day:end_of_day]
                
                # 2. Get Original Data for this day (Scatter)
                mask = (df_orig.index >= start_of_day) & (df_orig.index <= end_of_day)
                df_orig_day = df_orig.loc[mask]
                
                # 3. Calculate Daily Average Metrics for this day
                avg_orig = df_orig_day[col].mean()
                avg_linear = df_linear_day[col].mean()
                avg_pchip = df_pchip_day[col].mean()
                
                # Display Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Original Daily Avg", f"{avg_orig:.2f}")
                m2.metric("Linear Daily Avg", f"{avg_linear:.2f}", delta=f"{avg_linear - avg_orig:.2f}")
                m3.metric("PCHIP Daily Avg", f"{avg_pchip:.2f}", delta=f"{avg_pchip - avg_orig:.2f}")
                
                # 4. Plotting
                fig = go.Figure()
                
                # Plot Original as Scatter (Markers only)
                fig.add_trace(go.Scatter(
                    x=df_orig_day.index, y=df_orig_day[col], 
                    mode='markers', name='Original Data', 
                    marker=dict(color='black', size=6)
                ))
                
                # Plot Linear
                fig.add_trace(go.Scatter(
                    x=df_linear_day.index, y=df_linear_day[col], 
                    mode='lines', name='Linear Interp', 
                    line=dict(color='blue')
                ))
                
                # Plot PCHIP
                fig.add_trace(go.Scatter(
                    x=df_pchip_day.index, y=df_pchip_day[col], 
                    mode='lines', name='PCHIP Interp', 
                    line=dict(color='red', dash='dot')
                ))
                
                fig.update_layout(
                    title=f"24-Hour Profile: {col}", 
                    xaxis_title="Time", 
                    yaxis_title="Value",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # --- Output 2: Full Resampled Data Preview ---
            with st.expander("Preview Full Resampled Data (PCHIP)"):
                st.dataframe(resampled_results['PCHIP'])
            
            # --- Output 3: CSV Download ---
            st.header("Download Results")
            
            # Prepare CSV (Combined)
            # We only include Linear and PCHIP in download to keep file size reasonable, 
            # or we include Original if reindexed. Let's stick to resampled results.
            download_list = []
            for method, data in resampled_results.items():
                df_dl = data.copy()
                df_dl.columns = [f"{c} ({method})" for c in df_dl.columns]
                download_list.append(df_dl)
            
            final_download = pd.concat(download_list, axis=1)
            final_download.index.name = "Timestamp"
            final_download.reset_index(inplace=True)
            
            # Convert to CSV buffer
            buffer = BytesIO()
            final_download.to_csv(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                label="Download Interpolated Data (CSV)",
                data=buffer,
                file_name=f"interpolated_data_{analysis_date}.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Please upload a file via the sidebar to begin.")
