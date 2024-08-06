import keras
from keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

import asyncio
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

class LSTM():
    def __init__(self, ticker_dict) -> None:
        """
        TICKER
        - SYMBOL
        - HISTORY
        - QUARTERLY INCOME STMT
        - QUARTERLY CASHFLOW
        """
        self.ticker = ticker_dict

    def _create_sequences(self, data, seq_length):
        data_array = []
        for i in range(len(data) - seq_length + 1):
            seq = data[i:i+seq_length]
            data_array.append(seq)
        return np.array(data_array)
    
    def _start_log(self) -> None:
        pass

    async def _evaluate_model_for_history(self,
                                          lstm_sizes=[25, 50],
                                          learning_rates=[0.001, 0.005],
                                          epochs_to_test=[50, 100],
                                          seq_lengths=[5, 10],
                                          optimizers=[Adam]) -> int:

        def _model_def(lstm_neurons: int, seq_length: int) -> keras.Sequential:
            model = keras.Sequential([
                keras.layers.LSTM(lstm_neurons, return_sequences=True, 
                                  input_shape=(seq_length, self.ticker['stock_history'].shape[1])),
                keras.layers.LSTM(lstm_neurons, return_sequences=True),
                keras.layers.LSTM(lstm_neurons),
                keras.layers.Dense(20, activation='relu'),
                keras.layers.Dense(1)
            ])
            return model

        best_score = float('inf')
        total_iterations = len(epochs_to_test) * len(optimizers) * len(learning_rates) * len(lstm_sizes) * len(seq_lengths)

        scaler = StandardScaler()

        async def _get_score(model: keras.Sequential, epochs: int, 
                             train_data: np.array, train_labels: np.array,
                             val_data: np.array, val_labels: np.array) -> float:
            
            history = model.fit(train_data, train_labels, epochs=epochs,
                                validation_data=(val_data, val_labels),
                                verbose=0)  
                                
            val_loss = min(history.history['val_loss'])
            return val_loss

        tasks = []
        with tqdm(total=total_iterations, desc=f"Evaluation of LSTM for {self.ticker['stock_symbol']}") as pbar:
            for seq_length in seq_lengths:
                stock_history = self.ticker['stock_history'].values
                X = self._create_sequences(scaler.fit_transform(stock_history), seq_length)
                y = self.ticker['stock_history']['Close'].shift(-50).dropna().values
                X = X[:len(y)]
                
                for epochs in epochs_to_test:
                    for optimizer in optimizers:
                        for learning_rate in learning_rates:
                            for lstm_neurons in lstm_sizes:
                                opt = optimizer(learning_rate=learning_rate)
                                
                                model = _model_def(lstm_neurons, seq_length)
                                model.compile(optimizer=opt, loss='mean_squared_error')

                                # Split data into train and validation sets
                                train_size = int(len(X) * 0.8)
                                train_data, val_data = X[:train_size], X[train_size:]
                                train_labels, val_labels = y[:train_size], y[train_size:]

                                tasks.append(
                                    asyncio.create_task(
                                        _get_score(model, epochs, train_data, train_labels, val_data, val_labels)
                                    )
                                )

                                pbar.update(1)
        
        best_score = min(await asyncio.gather(*tasks))
        return best_score

    def _evaluate_model_for_income(self) -> int:
        pass

    def _evaluate_model_for_cashflow(self) -> int:
        pass

    def _evaluate_model_for_total_score(self) -> int:
        pass

    async def evaluate(self) -> int:
        hist_ew = await self._evaluate_model_for_history()
        # income_ew = self._evaluate_model_for_income()
        # cflow_ew = self._evaluate_model_for_cashflow()
        # t = self._evaluate_model_for_total_score()

        return hist_ew
        # return (hist_ew + income_ew + cflow_ew + t * 3) / 6
