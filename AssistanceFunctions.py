from sklearn.preprocessing import MinMaxScaler
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing
import os
from sklearn.metrics import classification_report
from datetime import time

class DeepLearning:
    @staticmethod
    def format_data(vals):
        """This function formats the train test splits
        Normalization and reshaping take place here
        NB: this function is specifically made for the mnist dataset"""
        # Note that the pixel values (are valuess from 0-255 ) showing the intensity of the grayscale
        # Normalization takes place for efficiency of the model and to reduce errors because of gradient decent

        temp = []
        for split in vals:
            print(split.shape)
            instance = (split.reshape((-1,28*28)))# change to 1D array
            temp.append(instance.astype("float32")) #Normalization

        print("Data Format complete")
        return temp
    @staticmethod
    def indicator_calc(price_action, data_requirements, indicator_type, periods):
        types = None
        import pandas_ta as ta  # Optional dependency for legacy indicator utilities.
        indicator_func= getattr(ta, indicator_type)
        indicators = pd.DataFrame()

        if data_requirements.lower() == 'hlc':##high low close
            for value in periods:
                indicators[f'{indicator_type}_{value}'] = indicator_func(price_action['high'], price_action['low'], price_action['close'], window=value)

        elif data_requirements.lower() == 'c':
            for value in periods:
                indicators[f'{indicator_type}_{value}'] = indicator_func(price_action['close'], window=value)

        elif data_requirements.lower() == 'hl':
            for value in periods:
                indicators[f'{indicator_type}_{value}'] = indicator_func(price_action['high'],price_action['low'], window=value)

        elif data_requirements.lower() == 'ohlc':
            for value in periods:
                try:
                    indicators[f'{indicator_type}_{value}'] = indicator_func(price_action['open'],price_action['high'],price_action['low'],price_action['close'], window=value)

                except:
                    indicators[f'{indicator_type}_{value}'] = indicator_func(price_action['close']) #Candlestick patterns
                    print('Candlesticks')# to be removed

        return indicators

    @staticmethod
    def indicator_calc_others(price_action,type,data_requirements,parameter_dict_list):
        'Parameters should be an array of dict values that are nested'
        import pandas_ta as ta  # Optional dependency for legacy indicator utilities.
        indicator_func, indicators = getattr(ta, type), pd.DataFrame()
        if data_requirements == "c":
            for arguments in parameter_dict_list:
                kwargs = arguments
                indicators[f'{type}_{arguments}']= indicator_func(price_action['close'],**kwargs)

        else:
            print(f'Too many variables to use {type}')

        return indicators


    @staticmethod
    def timeframe(data, timeframe_desired):
        """
        This function takes in 1MINUTE data and converts it to the desired timeframe
        LIMITED FUNCTIONALITY (Only yto change to 1hr - for now )
        :param data: The price action data
        :param timeframe_desired: format (1H)
        :return:
        IF PROBLEM ARISES IT IS DUE TO THE WEEKEND DATA SKIPS
        """
        new_timeframe = []
        if timeframe_desired == '1H':
            for index,(dt,open,high,low,close) in enumerate(data.itertuples(index=False)):
                closeindex = 59 + index
                max_high, min_low = 0,0

                if dt.minute == 0 :
                    # take the index then add 60 to it then get the max high, low , take the close of the last one, and open of the first one
                    max_high = max(data[index:59 + index].high)
                    min_low = min(data[index:59 + index].low)

                    new_timeframe.append([dt,open,max_high,min_low,data.iloc[closeindex,4]])

        return pd.DataFrame(new_timeframe, columns=data.columns)

    @staticmethod
    def chunks_creation(data):
        '''
        This functions creates sub-divisions of the data input
        Its Main purpose is to be used in conjunction with a parallelization library
        Chunks == no of total available cores on system ( will be a nested array )
        :return:
        '''
        chunk_size = round(len(data) / multiprocessing.cpu_count())
        nested_df, slice_index = [], 0

        for limit in range(1, len(data) + 1):
            if slice_index == 0 : # first iteration
                nested_df.append(data[:chunk_size])
                slice_index += chunk_size
                continue

            elif limit == multiprocessing.cpu_count(): # total number of chunks created ( stop the program )
                break

            else:
                nested_df.append(data[slice_index:slice_index + chunk_size])
                slice_index += chunk_size


        return nested_df
    @staticmethod
    def merge_csv_files(directory_path):
        # List to store dataframes
        df_list = []

        # Loop through the files in the directory
        for file in os.listdir(directory_path):
            if file.endswith(".csv"):  # Only process CSV files
                file_path = os.path.join(directory_path, file)
                df = pd.read_csv(file_path)
                df.columns = ['date','time', 'open','high','low','close','volume']
                df_list.append(df)

        # Concatenate all dataframes in the list
        merged_df = pd.concat(df_list, ignore_index=True)

        return merged_df

    @staticmethod
    def scale_ohlc_sequences(sequences,thresh_max=1.5):
        """
        Scale nested OHLC sequences such that their values fall within a specified range by adding or subtracting an offset.
            ENSURE THE MAX OF THE WHOLE DATASET IS LESS THAN THE (THRESH_MAX)

        Args:
            sequences (list of list of floats): Nested list containing OHLC sequences. Each sublist represents a sequence.
            target_max (float): The maximum value of the target range. Default is 2.0.

        Returns:
            list of list of floats: Scaled OHLC sequences and label for the sequence
        """

        scaled_sequences = []
        for label, timestep in sequences:
            timestep = np.array(timestep)

            price_max = np.max(timestep)

            temp_arithmetic = thresh_max - price_max

            new_sequence = timestep + temp_arithmetic
            scaled_sequences.append([label,[new_sequence]])

        return scaled_sequences
    @staticmethod
    def price_gen(sl, tp, prices, look_forward, sequence_length, heikin_ashi):
        scaler = MinMaxScaler(feature_range=(-1, 1))

        logged_df = prices[['open', 'high', 'low', 'close']]
        log_returns = np.log(logged_df / logged_df.shift(1))
        log_returns[['open', 'high', 'low', 'close']] = scaler.fit_transform(log_returns[['open', 'high', 'low', 'close']])

        # replace the nan values on the first row with the values of the second
        log_returns.iloc[0] = log_returns.iloc[1]

        log_returns = pd.concat([log_returns, prices.drop(['open', 'high', 'low', 'close'], axis=1)], axis=1)

        del  logged_df

        data = []  # store the nested arrays
        analysis = []
        ashi = []

        #find index position of the 'close' column
        close_index = list(prices.columns).index('close')

        # errors arise when the end has reached(if using the parallel)
        for index,value in enumerate(prices.values):

                try:
                    index_position = index  # get loc has time complexity of 0(1) or 0( log n )
                    future_prices = prices.iloc[index_position : index_position + look_forward]  # iloc is used because index is datetime

                except Exception as e:
                    # price instance is close to the end
                    if len(prices) - index_position < look_forward:
                        break
                    else:
                        raise Exception(f'Unexpected error!\n Error msg: {e}')

                entry_price = prices.iloc[index_position , close_index] # Represents the closing price

                #After the entry calc ( the candles for the entry are to be removed ) so we say that the entry candle == next candle ( the ones infront of it )
                if len(future_prices) < look_forward:
                    print('Below Future prices had problems resulting in a SKIP')
                    print(future_prices)
                    continue
                # future_prices.iloc[0] = future_prices.iloc[1]

                #check the date (High spreads during the off season
                current_time = future_prices.index[0].time()
                end_time = time(21, 0)
                start_time = time(6, 0)

                if current_time>=end_time or current_time<=start_time:
                    continue

                future_prices.iloc[0] = future_prices.iloc[1]

                #Trading logic
                sl_buy, tp_buy = entry_price - sl, entry_price + tp
                sl_sell, tp_sell = entry_price + sl, entry_price - tp

                # vectorized application
                buy_not_stopped = (future_prices['low'] > sl_buy).all()
                buy_take = (future_prices['high'] > tp_buy).any()
                sell_not_stopped = (future_prices['high'] < sl_sell).all()
                sell_take = (future_prices['low'] < tp_sell).any()

                # Trade outcome logic
                if buy_take and buy_not_stopped:
                    hit_type = 0 # for a buy
                elif sell_take and  sell_not_stopped:
                    hit_type = 1 # for a sell
                else:
                    hit_type = 2 #for NO TRADE



                # Instance creation ( SEQUENTIAL )
                start_index = index_position  - sequence_length
                if start_index <= 0 :
                    continue

                end_index = index_position + 1


                instance = log_returns.iloc[start_index: end_index]


                instance = instance[['open','high', 'low', 'close']].to_numpy() # takes the prev n instances ( the look back data )


                data.append([hit_type, instance])

                instance_df = prices.iloc[start_index: end_index]  # takes the prev n instances ( the look back data )

                ta_columns = ['atr_7','atr_14','atr_21', 'ema_21', 'ema_50', 'ema_90', 'ema_180']
                instance = instance_df[ta_columns]

                instance = instance_df[ta_columns].to_numpy()

                analysis.append(instance)


                if len(instance) == 0:
                    raise Exception("Instance value is Zero!")

        return analysis,data, ashi

    @staticmethod
    def label_encode(data):
        from tensorflow.keras.utils import to_categorical
        encoder = to_categorical(data, num_classes=3)
        return encoder


    # Define custom metric functions, and loss function ( run on RUNPODD 4090 instance )
    @staticmethod
    def custom_loss_function(y_pred, y_true):
        '''This custom loss maximizes the Precision of ( buy and sell ) while factoring in the recall '''
        import tensorflow as tf
        alpha, beta = 0.7, 0.3 #Weights to be used


        if tf.executing_eagerly():  # Check if eager execution is enabled
            y_true = y_true.numpy()
            y_pred = y_pred.numpy()

        y_pred = tf.make_ndarray(y_pred,y_true)
        print(y_pred)
        print(y_true)
        quit()
        ypred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)

        print(ypred, y_true)

        #Classification Report
        class_report = classification_report(y_true,y_pred, output_dict=True, zero_division=0.0)
        a, b = None, None

        recall_sell, recall_buy = class_report['1']['recall'], class_report['0']['recall']
        precision_sell, precison_buy = class_report['1']['precision'], class_report['0']['precision']
        precision_noTRD, recall_noTRD = class_report['2']['precision'], class_report['2']['recall']


        a = 1 - (precison_buy * alpha ) +  (recall_buy * beta)
        b = 1 - (precision_sell * alpha ) + (recall_sell * beta)
        c = 1 - (precision_noTRD) + recall_noTRD #figure out what do  with this value

        # selecting between eager and graph computation ( reason fo the wierd input to the loss function )

        return a + b + c




    @staticmethod
    def shuffle(x1, x2, y, random_state=None):

        # Ensure all arrays have the same length along the first dimension
        assert x1.shape[0] == x2.shape[0] == y.shape[0], "All input arrays must have the same number of samples."

        # Generate a random permutation of indices
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(x1.shape[0])

        # Shuffle all arrays using the same indices
        x1_shuffled = x1[indices]
        x2_shuffled = x2[indices]
        y_shuffled = y[indices]

        return x1_shuffled, x2_shuffled, y_shuffled
    @staticmethod
    def scale(price):
        df = pd.DataFrame(price, columns=['label', 'price'])
        data = df['price']

            #create chunks
        chunk_size = round(len(data) / multiprocessing.cpu_count())
        nested_df, slice_index = [], 0

        for limit in range(1, len(data) + 1):
            if slice_index == 0:  # first iteration
                nested_df.append(data[:chunk_size])
                slice_index += chunk_size

            else:
                if (slice_index+chunk_size) > len(data):
                    nested_df.append(data[slice_index:])
                else:
                    nested_df.append(data[slice_index:slice_index + chunk_size])
                    slice_index += chunk_size


        def scale_func(nested_price):
            scaler = MinMaxScaler(feature_range=(-2, 2))
            scaled = []
            for index,timestep in enumerate(nested_price):
                temp_df = pd.DataFrame(timestep, columns=['open', 'high','low', 'close'])

                log_returns = np.log(temp_df / temp_df.shift(1))

                log_returns[['open', 'high', 'low', 'close']] = scaler.fit_transform(log_returns[['open', 'high', 'low', 'close']])

                #replace the nan values on the first row with the values oof the second
                log_returns.iloc[0] = log_returns.iloc[1]
                array = log_returns.to_numpy()
                scaled.append([array])
            return scaled

        processed_chunks = Parallel(n_jobs=-1)(delayed(scale_func)(chunk) for chunk in nested_df)

        return processed_chunks

    @staticmethod
    def format_test(x1,x2, y, timesteps, features_x1, features_x2):
        #flatten
        x1 = np.concatenate(x1).flatten()
        x1 = x1.reshape(len(y), timesteps, features_x1)

        x2 = np.concatenate(x2).flatten()
        x2 = x2.reshape(len(y), timesteps, features_x2)

        return x1, x2


    @staticmethod
    def loss_counter(ytrue, ypred):
        # should only focus if its a buy or a sell
        counter = 0
        max_counter = 0

        for pred_index, true in enumerate(ytrue):
            if ypred[pred_index] == 2: continue  # if prediction was no trade no loss needs to be counted
            if ypred[pred_index] != true:
                counter += 1
                max_counter = max(max_counter, counter)

            else:
                counter = 0

        return max_counter
    @staticmethod
    def price_gen_tech(sl, tp, prices, look_forward, sequence_length, ashi_price):

        label = []
        analysis = []
        ashi =[]
        #find index position of the 'close' column
        close_index = list(prices.columns).index('close')

        # errors arise when the end has reached(if using the parallel)
        for index,value in enumerate(prices.values, start=0):

                try:
                    index_position = index  # get loc has time complexity of 0(1) or 0( log n )
                    future_prices = prices.iloc[index_position : index_position + look_forward]  # iloc is used because index is datetime

                except Exception as e:
                    # price instance is close to the end
                    if len(prices) - index_position < look_forward:
                        break
                    else:
                        raise Exception(f'Unexpected error!\n Error msg: {e}')

                entry_price = prices.iloc[index_position , close_index] # Represents the closing price

                #After the entry calc ( the candles for the entry are to be removed ) so we say that the entry candle == next candle ( the ones infront of it )
                if len(future_prices) < look_forward:
                    print('Below Future prices had problems resulting in a SKIP')
                    print(future_prices)
                    continue


                #check the date (High spreads during the off season
                current_time = future_prices.index[0].time()
                end_time = time(21, 0)
                start_time = time(6, 0)

                if current_time >= end_time or current_time <= start_time:
                    continue

                temp_current =  future_prices.iloc[0]
                future_prices.iloc[0] = future_prices.iloc[1]

                #Trading logic
                sl_buy, tp_buy = entry_price - sl, entry_price + tp
                sl_sell, tp_sell = entry_price + sl, entry_price - tp

                # # vectorized application
                # buy_not_stopped = (future_prices['low'] > sl_buy).all()
                # buy_take = (future_prices['high'] > tp_buy).any()
                # sell_not_stopped = (future_prices['high'] < sl_sell).all()
                # sell_take = (future_prices['low'] < tp_sell).any()
                #
                #
                # # Trade outcome logic
                # if buy_take and buy_not_stopped:
                #     hit_type = 0 # for a buy
                # elif sell_take and  sell_not_stopped:
                #     hit_type = 1 # for a sell
                # else:
                #     hit_type = 2 # for NO TRADE



                #New label creator
                for index, instance in enumerate(future_prices.itertuples()):
                    #for a buy position
                    if instance.high > tp_buy  : # tp was hit
                        prev_lows = future_prices.iloc[:index + 1] # to include the last value
                        if (prev_lows['low'] > sl_buy).all():
                            hit_type = 0

                    elif instance.low < tp_sell:
                        prev_highs = future_prices.iloc[:index + 1]
                        if (prev_highs['high'] < sl_sell ).all():
                            hit_type = 1

                    #No trade took Place
                    else:
                        hit_type = 2


                # Instance creation ( SEQUENTIAL )
                start_index = index_position  - sequence_length
                if start_index <= 0 :
                    continue

                end_index = index_position + 1

                ashi_instance = ashi_price.iloc[start_index:end_index]
                ashi.append(ashi_instance.to_numpy())

                instance_df = prices.iloc[start_index: end_index]  # takes the prev n instances ( the look back data )

                ta_columns = ['atr_7','atr_14', 'ema_12', 'ema_21']

                # Change back the last index
                instance_df.iloc[sequence_length] = temp_current

                instance = instance_df[ta_columns].to_numpy()

                analysis.append(instance)
                label.append(hit_type)

                #for heikin ashi
                if len(instance) == 0:
                    raise Exception("Instance value is Zero!")

        return analysis, label, ashi


    @staticmethod
    def heiken_ashi(price):
        actual_price = price.copy()
        actual_price.reset_index(inplace= True)
        heikin_ashi  = pd.DataFrame()

        heikin_ashi['datetime'] = actual_price['datetime']

        # Calculate Heikin-Ashi close (vectorized)
        heikin_ashi['H_close'] = (actual_price['high'] + actual_price['low'] + actual_price['close'] + actual_price['open']) / 4

        # Initialize the first Heikin-Ashi open price as the average of the first actual open and close
        heikin_ashi['H_open'] = 0.0  # Temporary column for initialization
        heikin_ashi.iloc[0, heikin_ashi.columns.get_loc('H_open')] = (actual_price['open'].iloc[0] +  actual_price['close'].iloc[0]) / 2 #first value

        # Calculate open prices using a loop
        for index in range(1, len(actual_price)):
            heikin_ashi.iloc[index, heikin_ashi.columns.get_loc('H_open')] = (heikin_ashi['H_open'].iloc[index - 1] + heikin_ashi['H_close'].iloc[index - 1]) / 2

        # Calculate Heikin-Ashi high and low (vectorized)
        heikin_ashi['H_high'] = actual_price[['high', 'open', 'close']].max(axis=1)
        heikin_ashi['H_low'] = actual_price[['low', 'open', 'close']].min(axis=1)

        heikin_ashi.loc[heikin_ashi['H_high'] < heikin_ashi['H_open'], 'H_high'] = heikin_ashi['H_open']

        # Adjust HA_Low for bullish candles where HA_Low > HA_Open
        heikin_ashi.loc[heikin_ashi['H_low'] > heikin_ashi['H_open'], 'H_low'] = heikin_ashi['H_open']

        #creation of classes ( strong bull )
        heikin_ashi['strong_bull'] = (heikin_ashi['H_low'] == heikin_ashi['H_open']).astype(int)
        heikin_ashi['strong_bear'] = (heikin_ashi['H_high'] == heikin_ashi['H_open']).astype(int)
        heikin_ashi['neutral'] = ((heikin_ashi['H_low'] != heikin_ashi['H_open']) & (  heikin_ashi['H_high'] != heikin_ashi['H_open'])).astype(int)

        # heikin_ashi.drop('datetime', inplace=True, axis=1)


        return heikin_ashi

    @staticmethod
    def price_gen_live(sl, tp, prices, look_forward, sequence_length, ashi_price):
        '''This function is to be used to for simulation trading to ensure all is working correctly and results can be recreated
        this function only runs once
        The input is to be 23
        '''


        # find index position of the 'close' column
        close_index = list(prices.columns).index('close')

        index_position = 50 # the instance cut off ( when prediction is to start ) -- try to modify

        future_prices = prices.iloc[index_position : index_position + (look_forward)] # add 1 cause the last instance is usually not included ( so add an extra instance in the end)

        entry_price = prices.iloc[index_position, close_index]  # Represents the closing price

        temp_current = future_prices.iloc[0]

        future_prices.iloc[0] = future_prices.iloc[1]

        # Trading logic
        sl_buy, tp_buy = entry_price - sl, entry_price + tp
        sl_sell, tp_sell = entry_price + sl, entry_price - tp

        hit_type = 2
        for index, instance in enumerate(future_prices.itertuples()):
            # for a buy position
            if instance.high > tp_buy:  # tp was hit
                prev_lows = future_prices.iloc[:index + 1]  # to include the last value
                if (prev_lows['low'] > sl_buy).all():
                    hit_type = 0

            elif instance.low < tp_sell:
                prev_highs = future_prices.iloc[:index + 1]
                if (prev_highs['high'] < sl_sell).all():
                    hit_type = 1

            # No trade took Place
            else:
                hit_type = 2

        start_index = 0
        end_index = index_position + 1 # factor in the last -- possible problem

        #ashi instance creation
        ashi = (ashi_price.iloc[start_index:end_index]).to_numpy()

        instance_df = prices.iloc[start_index: end_index]  # takes the prev n instances ( the look back data )
        ta_columns = ['atr_7', 'atr_14', 'ema_12', 'ema_21']

        #Change back the last index
        instance_df.iloc[50] = temp_current

        analysis = instance_df[ta_columns].to_numpy()

        return analysis, hit_type, ashi



    @staticmethod
    def feature_creator(prices, scaler=None, config=None):
        """Use the persisted scaler and schema instead of the legacy feature layout."""
        from pipeline import inference_inputs
        if scaler is None or config is None:
            raise ValueError('Load a versioned model bundle and provide its scaler and config')
        if 'datetime' not in prices.columns:
            prices = prices.reset_index()
        inputs = inference_inputs(prices, scaler, config)
        return inputs['TA'], inputs['OHLC']

    @staticmethod
    def price_gen_V2(sl, tp, prices, look_forward, sequence_length, ta_columns):


        label = []
        analysis = []
        #find index position of the 'close' column
        close_index = list(prices.columns).index('close')

        print(prices)
        # errors arise when the end has reached(if using the parallel)
        for index,value in enumerate(prices.values, start=0):

                try:
                    index_position = index  # get loc has time complexity of 0(1) or 0( log n )
                    future_prices = prices.iloc[index_position : index_position + look_forward]  # iloc is used because index is datetime

                except Exception as e:
                    # price instance is close to the end
                    if len(prices) - index_position < look_forward:
                        break
                    else:
                        raise Exception(f'Unexpected error!\n Error msg: {e}')

                entry_price = prices.iloc[index_position , close_index] # Represents the closing price

                #After the entry calc ( the candles for the entry are to be removed ) so we say that the entry candle == next candle ( the ones infront of it )
                if len(future_prices) < look_forward:
                    print('Below Future prices had problems resulting in a SKIP')
                    print(future_prices)
                    continue


                #check the date (High spreads during the off season
                # current_time = future_prices.index[0].time()
                # end_time = time(21, 0)
                # start_time = time(6, 0)
                #
                # if current_time >= end_time or current_time <= start_time:
                #     continue

                temp_current =  future_prices.iloc[0]
                future_prices.iloc[0] = future_prices.iloc[1]

                #Trading logic
                sl_buy, tp_buy = entry_price - sl, entry_price + tp
                sl_sell, tp_sell = entry_price + sl, entry_price - tp

                #New label creator
                for index, instance in enumerate(future_prices.itertuples()):
                    #for a buy position
                    if instance.high > tp_buy  : # tp was hit
                        prev_lows = future_prices.iloc[:index + 1] # to include the last value
                        if (prev_lows['low'] > sl_buy).all():
                            hit_type = 0

                    elif instance.low < tp_sell:
                        prev_highs = future_prices.iloc[:index + 1]
                        if (prev_highs['high'] < sl_sell ).all():
                            hit_type = 1

                    #No trade took Place
                    else:
                        hit_type = 2

                # Instance creation ( SEQUENTIAL )
                start_index = index_position  - sequence_length
                if start_index <= 0 :
                    continue

                end_index = index_position + 1

                instance_df = prices.iloc[start_index: end_index]  # takes the prev n instances ( the look back data )

                # Change back the last index
                instance_df.iloc[sequence_length] = temp_current

                instance = instance_df[ta_columns].to_numpy()

                analysis.append(instance)
                label.append(hit_type)

                #for heikin ashi
                if len(instance) == 0:
                    raise Exception("Instance value is Zero!")

        return analysis, label


    @staticmethod
    def price_gen_V3(price, forward_length, timestep_length, indicator_columns,repeat):

        close_index = list(price.columns).index('close') # find index position of the 'close' column

        bbands_trades_short = price[(price['high'] > price['BBU_10_2.0']) & (price['ema_200'] > price['close'])]
        bbands_trades_long = price[(price['low'] < price['BBL_10_2.0']) & (price['ema_200'] < price['close'])]



        index_to_keep = []
        labels = []
        indicators = []

        def trade_results(indexes, instances, type):

            close_index = list(instances.columns).index('close')
            high_index = list(instances.columns).index('high')
            low_index = list(instances.columns).index('low')

            #removal of indexes close together
            for start, i in enumerate(indexes):
                #for the first iteration
                if start == 0:
                    index_to_keep.append(i)
                    continue
                last_index = index_to_keep[-1]
                if last_index + 3 >= i:
                    continue
                else:
                    index_to_keep.append(i)

            instances.reset_index(inplace=True, drop=False)
            filtered_instances = instances[instances['index'].isin(index_to_keep)]
            filtered_instances = filtered_instances.drop('index', axis=1)#remove before looping


            #To increase total number of instances
            if repeat == True:
                filtered_instances = instances.sample(frac=1, random_state=42)
                print(filtered_instances)
                quit()


            #CHECK CORESPONDING COLUMNS BEFORE ANY CALC
            for trd_exec_index, instance in zip(index_to_keep,filtered_instances.values): # mismatch here
                entry_price = instance[close_index]
                stop_level, tp_level = 0, 0
                if type == 'buy':
                    sl_dif = entry_price - instance[low_index]

                    if sl_dif >= 0.0015:
                        stop_level = 0.0010
                    elif sl_dif <= 0.0005:
                        stop_level = 0.0005
                    else:
                        stop_level = sl_dif

                    tp_level = (stop_level * 1.5) + entry_price
                    stop_level = entry_price - stop_level
                else:
                    # Sell trades
                    sl_dif = instance[high_index] - entry_price

                    if sl_dif >= 0.0015:
                        stop_level = 0.0010
                    elif sl_dif <= 0.0005:
                        stop_level = 0.0005
                    else:
                        stop_level = sl_dif

                    tp_level = entry_price - (stop_level * 1.5)
                    stop_level = entry_price + stop_level


                #check which level was hit first ( sl or tp ) -- intrgt the indicators here
                try:
                    future_prices = price[trd_exec_index + 1: trd_exec_index + forward_length] # plus one is to avoid the current one

                except IndexError:
                    print('No more Future prices available')
                    break
                except Exception as e:
                    raise Exception(f"Error occured : {e}")

                if type == 'buy':
                    for index, instance_val in enumerate(future_prices.itertuples()):
                        if instance_val.high > tp_level  : # tp was hit
                            prev = future_prices.iloc[:index] # to include the last value

                            if (prev['low'] > stop_level).all():
                                hit_type = 0 #sucessful buy  trade
                                break
                            else:
                                hit_type= 2 #sl level was hit ( loss )
                                break
                        else:
                            hit_type = 2  # sl level was hit ( loss )


                elif type == 'sell':
                    for index, instance_val in enumerate(future_prices.itertuples()):
                        if instance_val.low < tp_level  : # tp was hit
                            prev = future_prices.iloc[:index + 1] # to include the last value
                            if (prev['high'] < stop_level).all():
                                hit_type = 1
                                break
                            else:
                                hit_type= 2 # sl level was hit
                                break
                        else:
                            hit_type = 2

                if trd_exec_index <= timestep_length:
                    continue

                elif trd_exec_index + forward_length >= len(price):
                    break # end point has been achieved


                # Creating time steps and indicators ( should be two inputs) PREVENT POSSIBLE OCC. OF DATA LEAKAGE
                prev_price = price[trd_exec_index - timestep_length : trd_exec_index + 1 ] #include the current one ( the one that broke the bbands )

                temp_df = prev_price[indicator_columns]

                #create the timestep and append it
                temp_reshape = temp_df.values.flatten().reshape(len(temp_df), len(indicator_columns))

                try:
                    labels.append(hit_type)# save label

                except UnboundLocalError:
                    print('local Error ( refrenced before assignment) ')
                    quit()
                except Exception as e:
                    print(e)

                indicators.append(temp_reshape)

        # another ex for look forward
        trade_results(bbands_trades_long.index, bbands_trades_long,'buy')
        trade_results(bbands_trades_short.index, bbands_trades_short, 'sell')
        return labels, indicators


    @staticmethod
    def label_creator(sl,tp, price,look_forward, look_back, columns_wanted):
        from pipeline import label_windows
        return label_windows(price=price, sl=sl, tp=tp, look_forward=look_forward,
                             look_back=look_back, columns_wanted=columns_wanted)

    @staticmethod
    def price_gen_V4(price, forward_length, timestep_length, indicator_columns, repeat):

        close_index = list(price.columns).index('close')  # find index position of the 'close' column

        bbands_trades_short = price[(price['high'] > price['BBU_20_2.0']) & (price['ema_200'] > price['close'])]
        bbands_trades_long = price[(price['low'] < price['BBL_20_2.0']) & (price['ema_200'] < price['close'])]

        index_to_keep = []
        labels = []
        indicators = []

        def trade_results(indexes, instances, type):

            close_index = list(instances.columns).index('close')
            high_index = list(instances.columns).index('high')
            low_index = list(instances.columns).index('low')

            # removal of indexes close together
            for start, i in enumerate(indexes):
                # for the first iteration
                if start == 0:
                    index_to_keep.append(i)
                    continue
                last_index = index_to_keep[-1]
                if last_index + 3 >= i:
                    continue
                else:
                    index_to_keep.append(i)

            instances.reset_index(inplace=True, drop=False)
            filtered_instances = instances[instances['index'].isin(index_to_keep)]
            filtered_instances = filtered_instances.drop('index', axis=1)  # remove before looping

            # To increase total number of instances
            if repeat == True:
                filtered_instances = instances.sample(frac=1, random_state=42)
                print(filtered_instances)
                quit()

            # CHECK CORESPONDING COLUMNS BEFORE ANY CALC
            for trd_exec_index, instance in zip(index_to_keep, filtered_instances.values):  # mismatch here
                entry_price = instance[close_index]
                stop_level, tp_level = 0, 0
                if type == 'buy':
                    sl_dif = entry_price - instance[low_index]

                    if sl_dif >= 0.0015:
                        stop_level = 0.0010
                    elif sl_dif <= 0.0005:
                        stop_level = 0.0005
                    else:
                        stop_level = sl_dif

                    tp_level = (stop_level * 1) + entry_price
                    stop_level = entry_price - stop_level
                else:
                    # Sell trades
                    sl_dif = instance[high_index] - entry_price

                    if sl_dif >= 0.0015:
                        stop_level = 0.0010
                    elif sl_dif <= 0.0005:
                        stop_level = 0.0005
                    else:
                        stop_level = sl_dif

                    tp_level = entry_price - (stop_level * 1)
                    stop_level = entry_price + stop_level

                # check which level was hit first ( sl or tp ) -- intrgt the indicators here
                try:
                    future_prices = price[
                                    trd_exec_index + 1: trd_exec_index + forward_length]  # plus one is to avoid the current one

                except IndexError:
                    print('No more Future prices available')
                    break
                except Exception as e:
                    raise Exception(f"Error occured : {e}")

                if type == 'buy':
                    for index, instance_val in enumerate(future_prices.itertuples()):
                        if instance_val.high > tp_level:  # tp was hit
                            prev = future_prices.iloc[:index]  # to include the last value

                            if (prev['low'] > stop_level).all():
                                hit_type = 0  # sucessful buy  trade
                                break
                            else:
                                hit_type = 2  # sl level was hit ( loss )
                                break
                        else:
                            hit_type = 2  # sl level was hit ( loss )


                elif type == 'sell':
                    for index, instance_val in enumerate(future_prices.itertuples()):
                        if instance_val.low < tp_level:  # tp was hit
                            prev = future_prices.iloc[:index + 1]  # to include the last value
                            if (prev['high'] < stop_level).all():
                                hit_type = 1
                                break
                            else:
                                hit_type = 2  # sl level was hit
                                break
                        else:
                            hit_type = 2

                if trd_exec_index <= timestep_length:
                    continue

                elif trd_exec_index + forward_length >= len(price):
                    break  # end point has been achieved

                # Creating time steps and indicators ( should be two inputs) PREVENT POSSIBLE OCC. OF DATA LEAKAGE
                prev_price = price[ trd_exec_index - timestep_length: trd_exec_index + 1 ]  # include the current one ( the one that broke the bbands )

                temp_df = prev_price[indicator_columns]

                # create the timestep and append it
                temp_reshape = temp_df.values.flatten().reshape(len(temp_df), len(indicator_columns))

                try:
                    labels.append(hit_type)  # save label

                except UnboundLocalError:
                    print('local Error ( refrenced before assignment) ')
                    quit()
                except Exception as e:
                    print(e)

                indicators.append(temp_reshape)












