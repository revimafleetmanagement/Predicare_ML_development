#----------------------------------------------------------------------------- LIBRARY IMPORTATION----------------------------------------------------
import streamlit as st
st.set_page_config(layout='wide')
import src.Streamlit_functions as Streamlit_functions
import src.config as config
import os
import sys
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------- DATABASE EXPORTATION AND UPDATE --------------------------------------------------------------
DATA_BASE_PATH=r'data_base\raw_data_for_ML_analysis.pkl'
ML_MODEL=r'PrediCare_Ensemble_Voting_Classifier'
@st.cache_data
def load_data():
    if os.path.exists(DATA_BASE_PATH) and os.path.exists(ML_MODEL):
        df=pd.read_pickle(DATA_BASE_PATH)
        with open(ML_MODEL, "rb") as f:
            loaded_package = pickle.load(f)
        return df[(df['MES_EGT_CORRECTED']>450) & (df['TOTAL_GENLOAD']>140)],loaded_package
    else:
        print(f"one listed file below is missing. Please ensure that all files exist to make streamlit dashboard works properly.\n 1- {DATA_BASE_PATH}\n 2- {ML_MODEL}")
        sys.exit()

data_base,loaded_package=load_data()
#------------------------------------------------------ PLOTLY PLOTTING WITH STREAMLIT ------------------------------------------------------------------------------------

@st.fragment
def scatter_section(df):
    all_param=['MES_EGT_SEL','MES_FUEL_CMD', 'MES_FUEL_DP', 'MES_FUEL_P',
               'MES_FUEL_PMP','MES_FUEL_T', 'MES_GENLD_L', 'MES_GENLD_R',
               'MES_OIL_P1', 'MES_OIL_P2','MES_OIL_P3', 'MES_OIL_P4',
               'MES_OIL_QTY', 'MES_OIL_T', 'MES_PAMB','MES_TAMB',
               'MES_WFCALC', 'MES_WFCMD','MES_EGT_CORRECTED', 'MES_EGT_MARGIN',
               'MES_WF_CORRECTED','TOTAL_GENLOAD', 'Ratio_HRS_CYC',
               'SMA_10_MES_EGT_CORRECTED','SMA_10_MES_TAMB', 'SMA_10_MES_EGT_MARGIN',
               'SMA_10_MES_WF_CORRECTED','SMA_10_MES_OIL_P1', 'SMA_10_MES_OIL_P2',
               'SMA_10_MES_OIL_P3','SMA_10_MES_OIL_P4', 'SMA_10_MES_FUEL_P','SMA_10_MES_FUEL_DP',
               'SMA_10_MES_FUEL_CMD', 'SMA_10_MES_WFCMD', 'SMA_10_MES_FUEL_PMP',
               'SMA_30_MES_EGT_CORRECTED', 'SMA_30_MES_TAMB', 'SMA_30_MES_EGT_MARGIN',
               'SMA_30_MES_WF_CORRECTED', 'SMA_30_MES_OIL_P1', 'SMA_30_MES_OIL_P2',
               'SMA_30_MES_OIL_P3', 'SMA_30_MES_OIL_P4', 'SMA_30_MES_FUEL_P',
               'SMA_30_MES_FUEL_DP', 'SMA_30_MES_FUEL_CMD', 'SMA_30_MES_WFCMD',
               'SMA_30_MES_FUEL_PMP', 'MED_30_MES_EGT_CORRECTED', 'MED_30_MES_TAMB',
               'MED_30_MES_EGT_MARGIN', 'MED_30_MES_WF_CORRECTED', 'MED_30_MES_OIL_P1',
               'MED_30_MES_OIL_P2', 'MED_30_MES_OIL_P3', 'MED_30_MES_OIL_P4',
               'MED_30_MES_FUEL_P', 'MED_30_MES_FUEL_DP', 'MED_30_MES_FUEL_CMD',
               'MED_30_MES_WFCMD', 'MED_30_MES_FUEL_PMP']
    col1,col2,col3=st.columns(3)
    with col1:
        parameter=st.selectbox("Parameter",[config.PARAMETERS_DICT[k]['name'] for k in all_param],key="param")
    with col2:
        sma_window=st.selectbox("SMA Window",[30,40,50,60,70,80,90,100],key="sma")
    with col3:
        reg=st.selectbox('FWOT',data_base['Reg'].unique().tolist(),key="reg")
    airline=data_base[data_base.Reg==reg]['Airline'].unique()[0]
    airline=config.AIRLINE_DICT[airline] if airline else ""
    df_filtered=df
    param_wilco_dict={v['name']:k for k,v in config.PARAMETERS_DICT.items()}
    data=Streamlit_functions.compute_data_for_plotly(
        df=df_filtered,AC_ident=reg,col_name='Reg',
        parameter_name=param_wilco_dict[parameter],window_length=sma_window,bins=100,limit_distribution=2,smoothness=100
    )

    # KPI metrics row
    current_sma=data['y_main_sma'].iloc[-1]
    three_month_mean=data['y_hlines_three_months']
    fleet_mean=data['secondary_mean']
    unit=config.PARAMETERS_DICT[param_wilco_dict[parameter]]['unit']
    k1,k2,k3=st.columns(3)
    with k1:
        st.metric(
            f"Current {parameter} - moving Average {sma_window}",
            f"{current_sma:.2f}{unit}" if pd.notna(current_sma) else "N/A"
        )
    with k2:
        delta_str=f"{(three_month_mean-fleet_mean):+.2f} vs {airline} fleet Average" if pd.notna(three_month_mean) else None
        st.metric(
            f"{parameter} - 3Months Average",
            f"{three_month_mean:.2f}{unit}" if pd.notna(three_month_mean) else "N/A",
            delta=delta_str,
            delta_color="inverse"
        )
    with k3:
        st.metric(
            f"{airline} Fleet Average",
            f"{fleet_mean:.2f}{unit}" if pd.notna(fleet_mean) else "N/A"
        )

    # Sparse data guard
    if data['y_upper_boundaries'].isna().all():
        st.warning(
            f"Boundary band unavailable: {reg} has fewer records than the smoothing window "
            f"({sma_window * 100} rows required). Try a smaller SMA Window."
        )

    fig=Streamlit_functions.plot_graph_plotly_side_by_side(data)
    st.plotly_chart(fig,width='stretch')

    # Download button
    fwot_csv=df_filtered[df_filtered['Reg']==reg].to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download {reg} data as CSV",
        data=fwot_csv,
        file_name=f"{reg}_data.csv",
        mime="text/csv"
    )

@st.fragment
def heatmap_section(df):
    existing_airlines={config.AIRLINE_DICT[k]:k for k in data_base['Airline'].unique()}
    existing_airlines['All']="All"
    col4,col_airline=st.columns(2)
    with col4:
        freq_label=st.selectbox(
            "HeatMap Timestamp",
            options=["Weekly","Daily","Monthly","bi-Monthly","Quarter"],
            key="freq"
        )
    with col_airline:
        heatmap_airline=st.selectbox(
            "Airline",
            options=list(existing_airlines.keys()),
            key="heatmap_airline"
        )
    freq={
        "Daily":"D",
        "Weekly":"W",
        "Monthly":"M",
        "bi-Monthly":"2M",
        "Quarter":"3M"
    }[freq_label]
    selected_code=existing_airlines[heatmap_airline]
    if selected_code=="All":
        db_filtered=df
    else:
        db_filtered=df[df['Airline']==selected_code]
    mapping_categories = loaded_package["mapping_categories"]
    heatmap_df=Streamlit_functions.compute_heatmap_data(data_base=db_filtered,freq=freq,_ml_model=loaded_package)
    heatmap_fig=Streamlit_functions.plot_heatmap_plotly(heatmap_df=heatmap_df,freq=freq,map_categories=mapping_categories)
    st.plotly_chart(heatmap_fig,width='stretch')

@st.fragment
def histogram_section(df):
    existing_airlines={config.AIRLINE_DICT[k]:k for k in data_base['Airline'].unique()}
    existing_airlines['All']="All"
    col5,=st.columns(1)
    with col5:
        select_airline=st.selectbox(
            "Select Airline",
            options=list(existing_airlines.keys()),
            key="airline"
        )
    selected_code=existing_airlines[select_airline]
    hist_ratio_hrs_cyc=Streamlit_functions.plot_histogram_ratio_hrs_cyc(
        df=df,
        selected_airline=selected_code,
        base_alpha=0.10,
        highlight_airline=0.8,
        nbins=100
    )
    st.plotly_chart(hist_ratio_hrs_cyc,width='stretch')

@st.fragment
def pollutants_section(df):
    existing_airlines={config.AIRLINE_DICT[k]:k for k in data_base['Airline'].unique()}
    existing_airlines['All']="All"
    col_airline,=st.columns(1)
    with col_airline:
        select_airline=st.selectbox(
            "Select Airline",
            options=list(existing_airlines.keys()),
            key="pollutant_airline"
        )
    selected_code=existing_airlines[select_airline]
    pollutants_fig=Streamlit_functions.plot_pollutants_box(
        df=df,
        selected_airline=selected_code
    )
    st.plotly_chart(pollutants_fig,width='stretch')

min_date=pd.to_datetime(data_base['Date'].min()).date()
max_date=pd.to_datetime(data_base['Date'].max()).date()
default_start=max(min_date,(pd.Timestamp(max_date)-pd.DateOffset(years=1)).date())
date_range=st.date_input(
    "Date Range",
    value=(default_start,max_date),
    min_value=min_date,
    max_value=max_date,
    key="global_date_range"
)
if isinstance(date_range,(list,tuple)) and len(date_range)==2:
    start,end=pd.Timestamp(date_range[0]).tz_localize('UTC'),pd.Timestamp(date_range[1]).tz_localize('UTC')
    df_filtered=data_base[data_base['Date'].between(start,end)]
else:
    df_filtered=data_base

scatter_section(df_filtered)
heatmap_section(df_filtered[df_filtered['Reason_for_removal'].isnull()])
# heatmap_section(df_filtered)
pollutants_section(df_filtered)
histogram_section(df_filtered)
