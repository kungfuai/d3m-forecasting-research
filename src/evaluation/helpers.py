import numpy as np

def get_seasonal_errors(past_datas, freq: str = 'M'):

    if freq == 'W':
        seasonality = 1
    elif freq == 'M':
        seasonality = 12

    y_ts = [
        past_data[:-seasonality]
        for past_data in past_datas
    ]
    y_tms = [
        past_data[seasonality:]
        for past_data in past_datas
    ]
    seasonal_maes = [
        np.mean(abs(y_t - y_tm), axis = 0).T 
        for y_t, y_tm in zip(
            y_ts, y_tms
        )
    ]
    return seasonal_maes

def get_mase(target, forecast, seasonal_error):
    flag = seasonal_error == 0
    return (np.mean(np.abs(target - forecast), axis=0) * (1 - flag)) / (
        seasonal_error + flag
    )