import pandas as pd
import numpy as np
import ta

def get_weights_floored(d, num_k, floor=1e-3):
    r"""Calculate weights ($w$) for each lag ($k$) through
    $w_k = -w_{k-1} \frac{d - k + 1}{k}$ provided weight above a minimum value
    (floor) for the weights to prevent computation of weights for the entire
    time series.

    Args:
        d (int): differencing value.
        num_k (int): number of lags (typically length of timeseries) to calculate w.
        floor (float): minimum value for the weights for computational efficiency.
    """
    w_k = np.array([1])
    k = 1

    while k < num_k:
        w_k_latest = -w_k[-1] * ((d - k + 1)) / k
        if abs(w_k_latest) <= floor:
            break

        w_k = np.append(w_k, w_k_latest)

        k += 1

    w_k = w_k.reshape(-1, 1)

    return w_k


def frac_diff(df, d=0.8, floor=1e-5):
    r"""Fractionally difference time series via CPU.

    Args:
        df (pd.DataFrame): dataframe of raw time series values.
        d (float): differencing value from 0 to 1 where > 1 has no FD.
        floor (float): minimum value of weights, ignoring anything smaller.
    """
    # Get weights window
    weights = get_weights_floored(d=d, num_k=len(df), floor=floor)
    weights_window_size = len(weights)

    # Reverse weights
    weights = weights[::-1]

    # Blank fractionally differenced series to be filled
    df_fd = []

    # Slide window of time series, to calculated fractionally differenced values
    # per window
    for idx in range(weights_window_size, df.shape[0]):
        # Dot product of weights and original values
        # to get fractionally differenced values
        df_fd.append(np.dot(weights.T, df.iloc[idx - weights_window_size:idx]).item())

    # Return FD values and weights
    df_fd = pd.DataFrame(df_fd)

    return df_fd

diff = lambda x, y: x - y
abs_diff = lambda x, y: abs(x - y)
get_hod = lambda x: x.dt.hour
get_dow = lambda x: x.dt.dayofweek
get_woy = lambda x: x.dt.week

indicators = [
    ('RSI', ta.rsi, ['Close']),
    ('MFI', ta.money_flow_index, ['High', 'Low', 'Close', 'Volume']),
    ('TSI', ta.tsi, ['Close']),
    ('UO', ta.uo, ['High', 'Low', 'Close']),
    ('AO', ta.ao, ['High', 'Close']),
    ('MACDDI', ta.macd_diff, ['Close']),
    ('VIP', ta.vortex_indicator_pos, ['High', 'Low', 'Close']),
    ('VIN', ta.vortex_indicator_neg, ['High', 'Low', 'Close']),
    ('VIDIF', abs_diff, ['VIP', 'VIN']),
    ('TRIX', ta.trix, ['Close']),
    ('MI', ta.mass_index, ['High', 'Low']),
    ('CCI', ta.cci, ['High', 'Low', 'Close']),
    ('DPO', ta.dpo, ['Close']),
    ('KST', ta.kst, ['Close']),
    ('KSTS', ta.kst_sig, ['Close']),
    ('KSTDI', diff, ['KST', 'KSTS']),
    ('ARU', ta.aroon_up, ['Close']),
    ('ARD', ta.aroon_down, ['Close']),
    ('ARI', diff, ['ARU', 'ARD']),
    ('BBH', ta.bollinger_hband, ['Close']),
    ('BBL', ta.bollinger_lband, ['Close']),
    ('BBM', ta.bollinger_mavg, ['Close']),
    ('BBHI', ta.bollinger_hband_indicator, ['Close']),
    ('BBLI', ta.bollinger_lband_indicator, ['Close']),
    ('KCHI', ta.keltner_channel_hband_indicator, ['High', 'Low', 'Close']),
    ('KCLI', ta.keltner_channel_lband_indicator, ['High', 'Low', 'Close']),
    ('DCHI', ta.donchian_channel_hband_indicator, ['Close']),
    ('DCLI', ta.donchian_channel_lband_indicator, ['Close']),
    ('ADI', ta.acc_dist_index, ['High', 'Low', 'Close', 'Volume']),
    ('CMF', ta.chaikin_money_flow, ['High', 'Low', 'Close', 'Volume']),
    ('FI', ta.force_index, ['Close', 'Volume']),
    ('EM', ta.ease_of_movement, ['High', 'Low', 'Close', 'Volume']),
    ('VPT', ta.volume_price_trend, ['Close', 'Volume']),
    ('NVI', ta.negative_volume_index, ['Close', 'Volume']),
    ('DR', ta.daily_return, ['Close']),
    ('DLR', ta.daily_log_return, ['Close'])
]

#skip_list = ['ARU', 'ARD', 'VIP', 'VIN', 'KST', 'KSTS']
skip_list = []


def add_indicators(df, d=0.5, floor=1e-5):
    output = df
    generated_features = []

    # uncomment if you want to use additional features - otherwise only OHCLV data is used
    """
    for name, f, arg_names in indicators:
        wrapper = lambda func, args: func(*args)
        args = [df[arg_name] for arg_name in arg_names]
        output[name] = wrapper(f, args)

    generated_features = [i[0] for i in indicators]
    generated_features = list(np.setdiff1d(generated_features, skip_list))
    """
    output['CFD'] = np.nan
    cfd = frac_diff(pd.DataFrame(df['Close']), d, floor)
    offset = (len(df) - len(cfd))
    output.loc[offset:, ['CFD']] = cfd[0].values

    output['OFD'] = np.nan
    ofd = frac_diff(pd.DataFrame(df['Open']), d, floor)
    output.loc[offset:, ['OFD']] = ofd[0].values

    output['HFD'] = np.nan

    hfd = frac_diff(pd.DataFrame(df['High']), d, floor)
    output.loc[offset:, ['HFD']] = hfd[0].values

    output['LFD'] = np.nan
    lfd = frac_diff(pd.DataFrame(df['Low']), d, floor)
    output.loc[offset:, ['LFD']] = lfd[0].values

    output['VFD'] = pd.DataFrame(df['Volume'])

    #output['MOH'] = df['Date'].dt.minute
    #output['HOD'] = df['Date'].dt.hour
    #output['DOW'] = df['Date'].dt.dayofweek

    output['open_ratio'] = 0.5
    output['open_pl_ratio'] = 0.5

    generated_features.append('CFD')
    generated_features.append('OFD')
    generated_features.append('HFD')
    generated_features.append('LFD')
    generated_features.append('VFD')
    #generated_features.append('MOH')
    #generated_features.append('HOD')
    #generated_features.append('DOW')
    generated_features.append('open_ratio')
    generated_features.append('open_pl_ratio')

    #result_data = output[200:].reset_index(drop=True)

    return output, generated_features
