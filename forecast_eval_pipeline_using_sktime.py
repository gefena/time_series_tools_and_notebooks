#############################################################
# forecast evaluation pipeline using sktime
#############################################################
# Methodolgy
# =========================
# - Choose mininmal train period
# - Go over all wanted models
#   - walk_forward_validation
#     - go over time series in intervals similar to production
#     - split to train and test, where train is for the known history and test is for the unknown
#     - predict future (using some specific method) and collect predictions or errors
#   - Calculate error over all test period and show results for each forecast method
#
# New Models
# ===============
# New models can be introduced using classes
# Examples in custom models section
#############################################################
# Imports
#############################################################

import time

import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['font.size'] = '14'

import numpy as np
import scipy as sp

import pandas as pd
import seaborn as sns

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error
from math import sqrt

from warnings import simplefilter

#import numpy as np
#import pandas as pd

#from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    #ReducedRegressionForecaster,
    RecursiveRegressionForecaster,
    TransformedTargetForecaster,
)

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series

simplefilter("ignore", FutureWarning)
#%matplotlib inline

from sktime.forecasting.ets import AutoETS
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS

from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.fbprophet import Prophet

import ruptures as rpt
import changefinder



#############################################################
# Functions
#  Links: https://github.com/alan-turing-institute/sktime/blob/master/examples/01_forecasting.ipynb
#
#############################################################

def load_series_from_file(filename, col_date, col):
    data = pd.read_csv(filename)
    print("data.shape:", data.shape)
    #data.head(5)
    
    #col_date = "date"
    data[col_date] = pd.to_datetime(data[col_date])
    data = data.sort_values(by=col_date)
    data = data.set_index(col_date)
    #data.head(4)
    
    series = data[col]
    return(series)

def get_outlier_limits_iqr(series, chosen_quantiles=[0.25,0.75]):
    series_quantile = series.quantile(chosen_quantiles)
    series_quantile_array = series_quantile.values      
    iqr = max(series_quantile_array) - min(series_quantile_array)
    limits = series_quantile_array + (1.5*iqr) * np.array([-1,1])
    return(limits)

def remove_max_outliers(series):
    series = series.copy()
    limits = get_outlier_limits_iqr(series, [0.02,0.98])
    series.loc[series > max(limits)] = np.nan
    return(series)

def resample_series_using_mean(series, time_agg):   
    if time_agg in ['1H', 'H', '1D', 'D']:
        series = series.resample(time_agg).mean()
    elif time_agg in ['1W', 'W', '1M', 'M', '1Q', 'Q']:    
        series = series.resample(time_agg, kind='period', convention='start').mean()
    return(series)    

#def calc_periods_for_new_frequency(periods, new_frequency):
#    # when original frequency is hours
#    divide_factor_dict = {'1D':24, 'D':24, '1W':168, 'W':168,'1M':720,'M':720}
#    new_periods = int(periods/divide_factor_dict[new_frequency]) 
#    return(new_periods)

def calc_periods_for_new_frequency(periods, original_frequency, new_frequency):
    # when original frequency is hours
    divide_factor_dict = {('H','1D'):24, ('H','D'):24, ('H','1W'):168, ('H','W'):168, 
                          ('H','1M'):720, ('H','M'):720}
    new_periods = int(periods/divide_factor_dict[(original_frequency, new_frequency)]) 
    return(new_periods)

def preprocess_series(series, original_frequency, fill_na_flag=True, remove_max_outliers_flag=True, percentage_data=100,
                     resample_flag=False, new_frequency=''):
    
    if remove_max_outliers_flag:
        series = remove_max_outliers(series) # Removing "crazy" outliers
    if percentage_data < 100:    
        series_original_size = series.shape[0]
        series_custom_size = int((percentage_data/100)*series_original_size)
        #series = series.tail(series_custom_size)
        series = series.iloc[-series_custom_size:]

    series = series.asfreq(freq=original_frequency)
    #series.asfreq(freq=original_frequency)
    
    if resample_flag and original_frequency!=new_frequency and new_frequency!='':
        series = resample_series_using_mean(series, new_frequency)
    
    series_index = series.index
    if new_frequency=="strip_frequency":
        series = pd.Series(series.values)
        
    if fill_na_flag:
        # Consider using: sktime.transformations.series.impute.Imputer
        #series = series.fillna(method="ffill").fillna(method="bfill")
        series = series.interpolate(method='linear').fillna(method="ffill").fillna(method="bfill")
        
    return(series, series_index)    


def code_transform_for_eval(code):
    if '\n' not in code:
        return(code)
    code_as_list = code.split('\n')
    code_as_list_filtered = []
    for i in range(len(code_as_list)):
        line = code_as_list[i]
        if len(line)>0 and '#' not in line:
            code_as_list_filtered.append(line.strip())

    new_code = ''.join(code_as_list_filtered)        
    return(new_code)

def step_split_train_test(series, start_train_size, num_steps, step_size, forecast_horizon):
    # Split to train and test according to the number of steps we did
    train_size = start_train_size + num_steps * step_size
    test_size = forecast_horizon
    current_series_size = train_size + forecast_horizon
    #current_series = series.head(current_series_size)
    current_series = series.iloc[0:current_series_size]
    series_train, series_test = temporal_train_test_split(current_series, test_size=test_size)
    #plot_series(series_train, series_test, labels=["series_train", "series_test"])
    #print(series_train.shape[0], series_test.shape[0])
    return(series_train, series_test)


def forcaster_fit_and_predict(forcaster_name, series_train, series_test, fh, models_dict, loss_metric="smape", return_prediction=False):
    
    if forcaster_name in models_dict:
        code_for_eval = code_transform_for_eval(models_dict[forcaster_name])
        forecaster = eval(code_for_eval)
    else:
        print("[ERROR] forcaster_name not in models_dict")
        return()

    # forecaster fit
    # =====================================
    #print(series_train)
    forecaster.fit(series_train)
    
    # forecaster predict
    # =====================================
    if forcaster_name=="Prophet": 
        series_pred = forecaster.predict(fh.to_relative(cutoff=series_train.index[-1]), return_pred_int=False)  
    else:
        #print(fh)
        series_pred = forecaster.predict(fh)
        #plot_series(series_train, series_test, series_pred, labels=["series_train", "series_test", "series_pred"])
     
    # Collect errors
    # =====================================
    if loss_metric=="smape":
        current_smape_loss = smape_loss(series_pred, series_test)
        error_series = series_pred - series_test
    
    if return_prediction:
        return(current_smape_loss, series_pred)
    else:
        return(current_smape_loss, error_series)
    
def forcaster_fit_and_predict_future(forcaster_name, series_train, fh, models_dict):
    
    if forcaster_name in models_dict:
        code_for_eval = code_transform_for_eval(models_dict[forcaster_name])
        forecaster = eval(code_for_eval)
    else:
        print("[ERROR] forcaster_name not in models_dict")
        return()

    # forecaster fit
    # =====================================
    #print(series_train)
    forecaster.fit(series_train)
    
    # forecaster predict
    # =====================================
    if forcaster_name=="Prophet": 
        series_pred = forecaster.predict(fh.to_relative(cutoff=series_train.index[-1]), return_pred_int=False)  
    else:
        series_pred = forecaster.predict(fh)
        #plot_series(series_train, series_test, series_pred, labels=["series_train", "series_test", "series_pred"])
    
    return(series_pred)

def get_horizon_object_for_unknown_future(series, horizon_size, freq_to_set):
    if freq_to_set=='H' or freq_to_set=='D' or freq_to_set=='W':
            future_index = pd.date_range(start=series.index[-1], periods=horizon_size, freq=freq_to_set) #[1:]  
    elif freq_to_set=='strip_frequency':
        future_index = list(range(series.index[-1]+1, series.index[-1] + 1 + horizon_size))
    else:
        #print("using period_range")
        future_index = pd.period_range(start=series.index[-1], periods=horizon_size, freq=freq_to_set)
        #future_index = pd.PeriodIndex(future_index, freq=freq_to_set)
    fh_future = ForecastingHorizon(future_index, is_relative=False)
    return(fh_future)

def get_horizon_object_for_test_period(series_test, freq_to_set):
    if freq_to_set=='H' or freq_to_set=='D' or freq_to_set=='W' or freq_to_set=='strip_frequency': # or freq_to_set=='B'
        fh = ForecastingHorizon(series_test.index, is_relative=False)
    else:    
        future_index = pd.period_range(start=series_test.index[0], periods=series_test.shape[0], freq=freq_to_set)
        #future_index = pd.PeriodIndex(future_index, freq=freq_to_set)
        fh = ForecastingHorizon(future_index, is_relative=False)
    return(fh)

# Basic step
def walk_forward_split_train_predict(series, start_train_size, max_train_size, frequency, forecast_horizon, step_size, model_to_use, models_dict, loss_metric="smape"):
    number_of_steps = int((series.shape[0] - start_train_size - forecast_horizon) / step_size)
    print("number_of_steps:", number_of_steps)

    model_loss_per_step = []
    model_error_series_per_step = []
    for num_steps in range(number_of_steps+1):
        #print("num_steps:", num_steps)
        if start_train_size + (num_steps * step_size) + forecast_horizon > series.shape[0]:
            print("[Warning] Too many steps")
            break

        # Split to train and test according to the number of steps we did
        series_train, series_test = step_split_train_test(series, start_train_size, num_steps, step_size, forecast_horizon)  
        
        if max_train_size >= start_train_size and max_train_size < series_train.shape[0]:
            #series_train = series_train.tail(max_train_size) 
            series_train = series_train.iloc[-max_train_size:]

        # Generating Forecast horizon
        #fh = ForecastingHorizon(series_test.index, is_relative=False)
        fh = get_horizon_object_for_test_period(series_test, frequency)

        # Using sktime last value for prediction
        current_smape_loss, error_series = forcaster_fit_and_predict(model_to_use, series_train, series_test, fh, models_dict, loss_metric)

        model_loss_per_step.append(current_smape_loss)
        model_error_series_per_step.append(error_series.values)

    return(model_loss_per_step, model_error_series_per_step)


def eval_pipeline(series, models_dict, models_to_use_list, loss_metric, start_train_size, max_train_size, forecast_horizon, step_size,
                 original_frequency, new_frequency):
    
    print("series size:", series.shape[0])
    
    if new_frequency != original_frequency and new_frequency != 'strip_frequency':
        start_train_size = calc_periods_for_new_frequency(start_train_size, original_frequency, new_frequency)
        max_train_size = calc_periods_for_new_frequency(max_train_size, original_frequency, new_frequency)
        forecast_horizon = calc_periods_for_new_frequency(forecast_horizon, original_frequency, new_frequency)
        step_size = calc_periods_for_new_frequency(step_size, original_frequency, new_frequency)
        print("new step size:", step_size)
    

    #series = series.fillna(method='ffill')
    models_result = dict()
    models_error_series = dict()
    for i in range(len(models_to_use_list)):
        t0 = time.time()

        model_to_use = models_to_use_list[i]
        try:
            model_loss_per_step, model_error_series_per_step = walk_forward_split_train_predict(series, start_train_size, 
                                                                   max_train_size, new_frequency, forecast_horizon, step_size, model_to_use, models_dict, loss_metric)
            models_result[model_to_use] = model_loss_per_step
            models_error_series[model_to_use] = model_error_series_per_step
            print(model_to_use, np.median(model_loss_per_step))
        except:
            print("[Error] running " + model_to_use)

        elapsed_time = time.time() - t0
        print("[exp msg] elapsed time for process: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))) 

    result_df = pd.DataFrame.from_dict(models_result)
    return(result_df, models_error_series)


def get_top_model(result_df):
    top_k = 3
    quantile_to_use = 0.90
    top_models = result_df.median().sort_values().head(top_k).index
    #top_model = result_df[list(top_models)].std().sort_values().head(1).index[0] # Using sd to choose the top one
    top_model = result_df[list(top_models)].quantile([quantile_to_use]).transpose().sort_values(by=quantile_to_use).head(1).index[0] # using the min chosen-quantile for the top one
    return(top_model)

def plot_series_train_test_pred(series_train, series_test, series_pred, reindex_flag=False, original_series_index=None):
    if reindex_flag:
        series_train = reindex_series(series_train, original_series_index[series_train.index])
        series_test = reindex_series(series_test, original_series_index[series_test.index])
        series_pred = reindex_series(series_pred, original_series_index[series_pred.index])
    
    series_train.plot(figsize=(20,4), label='series_train', marker='o', ms=4)
    series_test.plot(label='series_test', marker='o', ms=4)
    series_pred.plot(label='series_pred', marker='o', ms=4)
    plt.legend()
    plt.show()
    
def plot_series_train_pred(series_train, series_pred, prediction_interval_flag=False, pred_int_df=None, reindex_flag=False, original_series_index=None, frequency=None):
    if reindex_flag:
        series_train = reindex_series(series_train, original_series_index[series_train.index])
        future_index = pd.date_range(start=series_train.index[-1], periods=series_pred.shape[0], freq=frequency)
        series_pred = reindex_series(series_pred, future_index)
    
    series_train.plot(figsize=(20,4), label='series_train', marker='o', ms=4)
    series_pred.plot(label='series_pred', marker='o', ms=4)
    if prediction_interval_flag:
        #pred_int_df = get_prediction_interval_dataframe_from_experiment(models_error_series, model_to_use, series_pred)
        plt.fill_between(pred_int_df["lower"].index, pred_int_df["lower"], pred_int_df["upper"],
                    where=pred_int_df["upper"] >= pred_int_df["lower"],
                    facecolor='green', alpha=0.2, interpolate=True, label='pred interval')
    plt.legend()
    plt.show()    

##############################################################
# changepoints / breakpoints in time series
##############################################################

def find_changepoints_for_time_series(series, modeltype="binary", number_breakpoints=10, plot_flag=True, plot_with_dates=False, show_time_flag=False):
    
    #RUPTURES PACKAGE
    #points=np.array(series)
    points=series.values
    title=""
    
    t0 = time.time()
    if modeltype=="binary":
        title="Change Point Detection: Binary Segmentation Search Method"
        model="l2"
        changepoint_model = rpt.Binseg(model=model).fit(points)
        result = changepoint_model.predict(n_bkps=number_breakpoints)
    if modeltype=="pelt":
        title="Change Point Detection: Pelt Search Method"
        model="rbf"
        changepoint_model = rpt.Pelt(model=model).fit(points)
        result = changepoint_model.predict(pen=10)    
    if modeltype=="window":
        title="Change Point Detection: Window-Based Search Method"
        model="l2"
        changepoint_model = rpt.Window(width=40, model=model).fit(points)
        result = changepoint_model.predict(n_bkps=number_breakpoints)  
    if modeltype=="Dynamic":
        title="Change Point Detection: Dynamic Programming Search Method"
        model="l1"
        changepoint_model = rpt.Dynp(model=model, min_size=3, jump=5).fit(points)
        result = changepoint_model.predict(n_bkps=number_breakpoints)
    if modeltype=="online":
        # CHANGEFINDER PACKAGE
        title="Simulates the working of finding changepoints in online fashion"
        cf = changefinder.ChangeFinder()
        scores = [cf.update(p) for p in points]
        result = (-np.array(scores)).argsort()[:number_breakpoints]
        result = sorted(list(result))
        if series.shape[0] not in result:
            result.append(series.shape[0])
    
    if 0 not in result:
        new_result = [0]
        new_result.extend(result)
        result=new_result

    if show_time_flag:
        elapsed_time = time.time() - t0
        print("[exp msg] elapsed time for process: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))) 
    
    if plot_flag:
        if not plot_with_dates:
            rpt.display(points, result, figsize=(18, 6))
            plt.title(title)
            plt.show()
        else:    
            series.plot(figsize=(18, 6))
            plt.title(title)
            for i in range(len(result)-1):  
                if i%2==0:
                    current_color='xkcd:salmon'
                else:
                    current_color='xkcd:sky blue'        
                #plt.fill_between(series.index[result[i]:result[i+1]], series.max(), color=current_color, alpha=0.3)
                plt.fill_between(series.index[result[i]:result[i+1]], y1=series.max()*1.1, y2=series.min()*0.9, color=current_color, alpha=0.3)
            plt.show()    
                
    return(result)  

##############################################################
# prediction interval
# Link: https://otexts.com/fpp2/prediction-intervals.html
##############################################################

def get_interval_error(y, yhat):
    #interval = 1.15 * stdev # 1.96 * stdev
    interval = 1.28 * (y - yhat).std() # 1.96 * stdev / 1.15 * stdev
    #print(interval)
    return(interval)

def get_prediction_intervals_dataframe(series_pred, interval_error):
    horizon_size = series_pred.shape[0]
    interval_horizon = interval_error * (np.array(list(range(1,horizon_size+1)))**0.5)
    pred_int = pd.DataFrame()
    pred_int["lower"] = series_pred #- interval_horizon #interval
    pred_int["lower"] = pred_int["lower"] - interval_horizon
    pred_int["upper"] = series_pred #+ interval_horizon #interval
    pred_int["upper"] = pred_int["upper"] + interval_horizon
    return(pred_int)

def get_prediction_interval_using_test_period(series, test_size, forecaster):
    series_train, series_test = temporal_train_test_split(series, test_size=test_size)
    forecaster.fit(series_train)
    fh = ForecastingHorizon(series_test.index, is_relative=False)
    series_pred = forecaster.predict(fh)
    interval_error = get_interval_error(series_test, series_pred)
    return(interval_error)

def get_prediction_interval_dataframe_from_experiment(models_error_series, model_to_use, series_pred):
    errors_df = pd.DataFrame.from_records(models_error_series[model_to_use]) 
    #errors_std_per_horizon_step = errors_df.std()
    errors_std_first = errors_df[0].std()
    interval_error = 1.28 * errors_std_first
    pred_int_df = get_prediction_intervals_dataframe(series_pred, interval_error)
    return(pred_int_df)

##############################################################
# utils
##############################################################

def reindex_series(series, new_index):
    df_series = pd.DataFrame(series.values, index=new_index)
    new_series = df_series[0] 
    return(new_series)

##############################################################
# custom models
##############################################################

class PolynomialTrendForecasterWithChangepoints:
    def __init__(self, degree, modeltype="binary", number_breakpoints=5, min_period=100):
        # for examlpe: PolynomialTrendForecasterWithChangepoints(degree=1, modeltype="binary", number_breakpoints=5, min_period=90)
        self.degree = degree
        self.forecaster = PolynomialTrendForecaster(degree=self.degree)    
        self.modeltype = modeltype
        self.number_breakpoints = number_breakpoints
        self.min_period = min_period
        
    def get_chagnepoint_where_period_longer_than_min(self, chagnepoint_results, series, min_period):
        series_size = series.shape[0]
        for i in range(len(chagnepoint_results)):
            idx = len(chagnepoint_results) - 1 - i
            if series_size - chagnepoint_results[idx] >= min_period:
                break
        return(chagnepoint_results[idx])    
      
    def fit(self, series_for_fit):         
        #modeltype="binary" # more options in function
        #number_breakpoints=5
        plot_flag=False
        plot_with_dates=False #True
        show_time_flag=False
        result = find_changepoints_for_time_series(series_for_fit, self.modeltype, self.number_breakpoints, plot_flag, plot_with_dates, show_time_flag)

        min_period = self.min_period
        changepoint_chosen = self.get_chagnepoint_where_period_longer_than_min(result, series_for_fit, min_period)
        #print(changepoint_chosen)
        final_series_for_fit = series_for_fit.iloc[changepoint_chosen:]

        self.forecaster.fit(final_series_for_fit)
        
    def predict(self, fh):  
        series_pred = self.forecaster.predict(fh)
        return(series_pred)








