# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:58:14 2021

@author: uy308417
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.compose import ReducedRegressionForecaster
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting import plot_series

from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import ForecastingGridSearchCV

from sklearn.metrics import mean_squared_error
import math

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import AutoARIMA ## requires larger history
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster

#from sktime.forecasting.arima import ARIMA
#from sktime.forecasting.fbprophet import Prophet

from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.ets import AutoETS

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import LinearSVR

import streamlit as st
import six
import os
import base64
import sys
sys.modules['sklearn.externals.six'] = six
import time
st.set_page_config(layout="wide")
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("ULS Forecaster")
st.title("Advanced Time Series Forecasting Tool")


col1,col2 = st.beta_columns(2)
with col1:
    st.subheader('Example csv file format to upload:')
    url ='https://raw.githubusercontent.com/frnietz/MLTS_multi/main/MultiProductTSMLexample.csv'
    df_sample=pd.read_csv(url)
    st.write(df_sample)
    st.text("Upload sales history in exactly given format including date column")
    st.text("Date column should be in date format")

with col2:
    st.subheader('About ULS Forecaster:')
    st.write("Generate sales forecast for multiple products with statistical and machine learning algorithms.  \n"
    "To utilize all models, your data should have at least 36 data points.  \n"
    "For smaller data sets of sales history, tool will use Exponential Smoothing  \n"
 
    )

st.write("**Statistical models:** Naive, Theta, Exponential Smoothing, TBATS. **Machine learning models:** Linear Regression, K-Neighbors, Random Forest, Gradient Boosting, Extreme Gradient Boosting, Support Vector Machine, Extra Trees  \n"
 )

uploaded_file = st.sidebar.file_uploader("Upload a file in csv format", type=("csv"))
st.sidebar.title("Upload Your Sales History")


def load_data(file):
    df = pd.read_csv(file)
    df['date']=pd.to_datetime(df['date'])
    df2 =df.drop(['date'], axis=1)
    df2=df2.replace(0, 0.01)
    df2['total']=df2.sum(axis=1)
    return df, df2


if uploaded_file is not None:
    df, df2 = load_data(uploaded_file)
    # prepare models
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('RF', RandomForestRegressor()))
    models.append(('GB', GradientBoostingRegressor()))
    models.append(('XGBoost', XGBRegressor(verbosity = 0)))
    models.append(('SVM', LinearSVR()))
    models.append(('Extra Trees', ExtraTreesRegressor()))
    models.append(('Naive', NaiveForecaster(strategy="mean", sp=12)))
    models.append(('Theta', ThetaForecaster(sp=12)))
    models.append(('Exp_Smoothing', ExponentialSmoothing(trend="add", seasonal="multiplicative", sp=12)))
    #models.append(('Exp_Smoothing', ExponentialSmoothing(trend="add", seasonal="additive", sp=3)))
    models.append(('TBATS', TBATS(sp=12, use_trend=True, use_box_cox=False)))
    
    forecast_horizon = st.sidebar.slider(label = 'Forecast Length (months)',min_value = 3, max_value = 36, value = 12)
    window_length = st.sidebar.slider(label = 'Sliding Window Length ',min_value = 1, value = 12)
    
    if df2.shape[0]<36:
        regressor = 'Exp_Smoothing_Small'
    else:
        # evaluate each model in turn
        results1 = []
        names = []
        dn_forecast = []
        dn_test =[]

        for name, model in models:
            if name == 'LR' or name == 'KNN' or name == 'RF' or name == 'GB' or name == 'XGBoost' or name == 'SVM' or name == 'Extra Trees':
                forecaster = ReducedRegressionForecaster(regressor=model, window_length=window_length,strategy='recursive')
            elif name == 'Exp_Smoothing_Small':
                pass
            else:
                forecaster = model
            y = df2['total'].reset_index(drop=True)
            y_train, y_test = temporal_train_test_split(y, test_size = 0.33)
            fh = np.arange(y_test.shape[0]) + 1
            forecaster.fit(y_train)
            y_pred = forecaster.predict(fh)
            dn_forecast.append(y_pred)
            dn_test.append(y_test)
            accuracy_results = mean_squared_error(y_test,y_pred,squared=False)
            results1.append(accuracy_results)
            names.append(name)
            msg = "%s: %.0f " % (name, accuracy_results.mean())
            #print(msg)
        #plot algorithm comparison
        fig, ax = plt.subplots(figsize=(15,5))
        ax.scatter(names,results1)
        ax.set_title('Algorithm Performance Comparison (Minimum RMSE)')
        ax.set_ylabel('RMSE')
        ax.set_xticklabels(names)
        st.pyplot(fig)
        #plt.show()

        res = {names[i]: results1[i] for i in range(len(names))}

        regressor=min(res, key=res.get)
        st.write(regressor+' is the best performing model with minimum RMSE')
    
    def select_regressor(selection):
        regressors = {
        'LR': LinearRegression(),
        'KNN': KNeighborsRegressor(),
        'RF': RandomForestRegressor(),
        'GB': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(verbosity = 0),
        'SVM': LinearSVR(),
        'Extra Trees': ExtraTreesRegressor(),
        'Naive' : NaiveForecaster(strategy="last", sp=12),
        'Theta': ThetaForecaster(sp=12),
        'Exp_Smoothing': ExponentialSmoothing(trend="add", seasonal="multiplicative", sp=12),
        'Exp_Smoothing_Small': ExponentialSmoothing(trend="add", seasonal="additive", sp=3),
        'TBATS': TBATS(sp=12, use_trend=True, use_box_cox=False)
         }

        return regressors[selection]
    
    
    def calculate_forecast(df_, regressor, forecast_horizon, window_length):
        df = df_.copy()
        new_forecast = []
        if regressor == 'Naive' or regressor == 'Theta' or regressor == 'Exp_Smoothing' or regressor == 'TBATS':
            regressor = select_regressor(regressor)
            forecaster = regressor
        elif regressor == regressor == 'Exp_Smoothing_Small':
            forecaster = ExponentialSmoothing(trend="add", seasonal="additive", sp=3)
        else:
            regressor = select_regressor(regressor)
            forecaster = ReducedRegressionForecaster(regressor = regressor, window_length = window_length, strategy='recursive')
        for i in df.columns :
            y = df.iloc[:,df.columns.get_loc(i)].reset_index(drop=True)
            fh = np.arange(forecast_horizon) + 1
            forecaster.fit(y, fh=fh)
            y_pred = forecaster.predict(fh)
            new_forecast.append(y_pred)
        new_forecast = pd.concat(new_forecast, axis=1)
        new_forecast.columns=df.columns.tolist()
        return new_forecast

    
    def calculate_smape(df_, regressor, forecast_horizon, window_length):
        df = df_.copy()
        dn_forecast = []
        dn_test =[]
        results = []
        if regressor == 'Naive' or regressor == 'Theta' or regressor == 'Exp_Smoothing' or regressor == 'TBATS' or regressor == 'Exp_Smoothing_Small':
            regressor = select_regressor(regressor)
            forecaster = regressor
        elif regressor == regressor == 'Exp_Smoothing_Small':
            forecaster = ExponentialSmoothing(trend="add", seasonal="additive", sp=3)
        else:
            regressor = select_regressor(regressor)
            forecaster = ReducedRegressionForecaster(regressor = regressor, window_length = window_length, strategy='recursive')
        for i in df.columns:
            y = df.iloc[:,df.columns.get_loc(i)].reset_index(drop=True)
            y_train, y_test = temporal_train_test_split(y, test_size = 0.33)
            fh = np.arange(y_test.shape[0]) + 1
            forecaster.fit(y_train, fh=fh)
            y_pred = forecaster.predict(fh)
            dn_forecast.append(y_pred)
            dn_test.append(y_test)
        dn_forecast = pd.concat(dn_forecast, axis=1)
        dn_test = pd.concat(dn_test, axis=1)
        dn_forecast.columns=dn_test.columns.tolist()
        
        
        fig, ax = plt.subplots(1, 1,figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.001)
        #fig.suptitle('last 12 months actual vs forecast')

        for column in dn_test:
            results.append(round(100*smape_loss(dn_forecast[column],dn_test[column]),1))
        
        ax.plot(dn_forecast['total'],'o-',color='orange' ,label="predicted")
        ax.plot(dn_test['total'], 'o-',color='blue',label="actual")
        ax.set_title('Testing the performance: actual vs forecast')
        ax.legend()
        st.pyplot(fig)
        #plt.show()
        return pd.DataFrame(results).set_index(dn_test.columns)
    
    
    smape = calculate_smape(df2,regressor, forecast_horizon, window_length)
    df_forecast = calculate_forecast(df2, regressor, forecast_horizon, window_length)
    df_forecast[df_forecast < 0] = 0
    
    date = df['date'][0]
    periods = df.shape[0] + forecast_horizon
    date_index = pd.date_range(date, periods=periods, freq='MS')
    date_index = date_index.date
    actual_and_forecast = [df2, df_forecast]
    combined = pd.concat(actual_and_forecast)

    combined.set_index(date_index,inplace=True)
    combined_transpose = combined.T
    smape.rename( columns={0:'sMAPE %'}, inplace=True )
    smape['accuracy %'] = smape['sMAPE %'].apply(lambda x: 0 if ((100- x)<0) else 100- x)
    
    last_of_all = [smape,combined_transpose]
    final = pd.concat(last_of_all, axis=1)
    st.markdown('**Accuracy(%)** achieved for total value with model is: ')
    st.write(smape.at['total', 'accuracy %'])
    st.text('A snapshot of accuracy, sales history and forecast: ')
    st.write(final.tail())
    

    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.plot(df2['total'],label="actual")
    ax.plot(df_forecast['total'],label="forecast")
    ax.set_title('Total Volume History and Forecast' )
    legend = ax.legend()
    st.write(fig)
    csv = final.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="actual_and_forecast.csv">Download CSV File</a>()'
    st.markdown(href, unsafe_allow_html=True)

else:
    st.write("Contact us for further support or more information")
