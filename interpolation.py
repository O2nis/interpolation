import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("Interpolation App")

    # Step 1: File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.warning("Please upload a CSV file with at least one column named 'Load'.")
        return

    # Frequency options
    freq_input_map = {
        "1 hour": "H",
        "30 minutes": "30min",
        "15 minutes": "15min",
        "10 minutes": "10min",
    }
    freq_output_map = {
        "15 minutes": "15min",
        "10 minutes": "10min",
        "1 minute": "T",
    }

    # Step 2: Choose input frequency
    st.header("Select Input Time Step")
    selected_input_freq_label = st.selectbox(
        "Input frequency:",
        list(freq_input_map.keys()),
        index=0
    )
    selected_input_freq = freq_input_map[selected_input_freq_label]

    # Step 3: Choose output frequency
    st.header("Select Output Time Step")
    selected_output_freq_label = st.selectbox(
        "Output frequency:",
        list(freq_output_map.keys()),
        index=0
    )
    selected_output_freq = freq_output_map[selected_output_freq_label]

    # Read data into DataFrame
    data = pd.read_csv(uploaded_file)

    # Quick validation: ensure 'Load' column exists
    if 'Load' not in data.columns:
        st.error("Your CSV must contain a column named 'Load'. Please try again.")
        return

    # STEP 4: Interpret the input data according to the selected input frequency.
    #         We'll assume the CSV rows represent consecutive samples at the given frequency.
    #         Create a DateTimeIndex from a fixed start time, set it as the DataFrame index.
    n_rows = len(data)
    start_time = pd.to_datetime("2020-01-01 00:00:00")
    input_index = pd.date_range(start=start_time, periods=n_rows, freq=selected_input_freq)
    data.index = input_index

    # Now reindex to the selected output frequency
    output_index = pd.date_range(start=data.index[0], end=data.index[-1], freq=selected_output_freq)
    data_reindexed = data.reindex(output_index)

    # Perform interpolations
    linear_interpolated = data_reindexed.interpolate(method='linear')
    spline_interpolated = data_reindexed.interpolate(method='spline', order=3)
    pchip_interpolated = data_reindexed.interpolate(method='pchip')

    # Save interpolated data to CSV with filenames that reflect the chosen output freq
    linear_file_name = f"linear_interpolated_{selected_output_freq_label.replace(' ', '')}.csv"
    spline_file_name = f"spline_interpolated_{selected_output_freq_label.replace(' ', '')}.csv"
    pchip_file_name = f"pchip_interpolated_{selected_output_freq_label.replace(' ', '')}.csv"

    # Provide downloads
    st.header("Download Interpolated CSVs")
    st.download_button(
        label=f"Download {linear_file_name}",
        data=linear_interpolated.to_csv(),
        file_name=linear_file_name,
        mime="text/csv"
    )
    st.download_button(
        label=f"Download {spline_file_name}",
        data=spline_interpolated.to_csv(),
        file_name=spline_file_name,
        mime="text/csv"
    )
    st.download_button(
        label=f"Download {pchip_file_name}",
        data=pchip_interpolated.to_csv(),
        file_name=pchip_file_name,
        mime="text/csv"
    )

    # STEP 5: Compute differences
    difference_spline_linear = spline_interpolated['Load'] - linear_interpolated['Load']
    difference_pchip_linear = pchip_interpolated['Load'] - linear_interpolated['Load']
    difference_spline_pchip = spline_interpolated['Load'] - pchip_interpolated['Load']

    # STEP 6: Plot original vs interpolated in the first 24 hours
    st.header("Charts for the First 24 Hours")
    zoom_end = start_time + pd.Timedelta(hours=24)

    # Slice the data for the first 24 hours
    data_zoom = data.loc[start_time:zoom_end]
    linear_zoom = linear_interpolated.loc[start_time:zoom_end]
    spline_zoom = spline_interpolated.loc[start_time:zoom_end]
    pchip_zoom = pchip_interpolated.loc[start_time:zoom_end]

    # Plot original data and interpolations
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(data_zoom.index, data_zoom['Load'], 'o', label='Original Data', markersize=4)
    ax1.plot(linear_zoom.index, linear_zoom['Load'], label='Linear Interpolation')
    ax1.plot(spline_zoom.index, spline_zoom['Load'], label='Spline Interpolation')
    ax1.plot(pchip_zoom.index, pchip_zoom['Load'], label='PCHIP Interpolation')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load (MW)')
    ax1.set_title('Comparison of Interpolation Methods (First 24 Hours)')
    ax1.legend()
    st.pyplot(fig1)

    # Plot the differences
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(difference_spline_linear.loc[start_time:zoom_end].index,
             difference_spline_linear.loc[start_time:zoom_end],
             label='Spline - Linear')
    ax2.plot(difference_pchip_linear.loc[start_time:zoom_end].index,
             difference_pchip_linear.loc[start_time:zoom_end],
             label='PCHIP - Linear')
    ax2.plot(difference_spline_pchip.loc[start_time:zoom_end].index,
             difference_spline_pchip.loc[start_time:zoom_end],
             label='Spline - PCHIP')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Load Difference (MW)')
    ax2.set_title('Differences Between Interpolation Methods (First 24 Hours)')
    ax2.legend()
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
