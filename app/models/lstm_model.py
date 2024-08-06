import keras
from keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import warnings

from tqdm import tqdm

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

    def _evaluate_model_for_history(self,
                                lstm_sizes = [10, 25, 50, 100, 150],
                                learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1],
                                epochs_to_test = [50, 100, 150],
                                seq_lengths = [5, 10, 20, 30, 40],
                                optimizers = [
                                    Adam,
                                    RMSprop,
                                    SGD
                                ]
                                ) -> int:

        def _model_def(lstm_neurons: int) -> keras.Sequential:
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
        
        total_iterations = len(epochs_to_test)*len(optimizers)*len(learning_rates)*len(lstm_sizes)*len(seq_lengths)

        with tqdm(total=total_iterations, desc=f"Evaulation of LSTM for {self.ticker['stock_symbol']}") as pbar:
            for seq_length in seq_lengths:
                stock_history = self.ticker['stock_history'].values
                X = self._create_sequences(stock_history, seq_length)
                y = self.ticker['stock_history']['Close'].shift(-50).dropna()
                
                for epochs in epochs_to_test:
                    for optimizer in optimizers:
                        for learning_rate in learning_rates:
                            for lstm_neurons in lstm_sizes:
                                opt = optimizer(learning_rate=learning_rate)
                                
                                model = _model_def(lstm_neurons)
                                model.compile(optimizer=opt, loss='mean_squared_error')

                                # Split data into train and validation sets
                                train_size = int(len(X) * 0.8)
                                train_data, val_data = X[:train_size], X[train_size:]
                                train_labels, val_labels = y[:train_size], y[train_size:]

                                history = model.fit(train_data, train_labels, epochs=epochs,
                                                    validation_data=(val_data, val_labels),
                                                    verbose=0)  # Set verbose to 0 to disable logging
                                
                                val_loss = min(history.history['val_loss'])
                                if val_loss < best_score:
                                    best_score = val_loss
                                
                                pbar.update(1)

        return best_score

    def _evaluate_model_for_income(self) -> int:
        pass

    def _evaluate_model_for_cashflow(self) -> int:
        pass

    def _evaluate_model_for_total_score(self) -> int:
        pass

    def evaluate(self) -> int:

        hist_ew = self._evaluate_model_for_history()
        #income_ew = self._evaluate_model_for_income()
        #cflow_ew = self._evaluate_model_for_cashflow()

        #t = self._evaluate_model_for_total_score()

        return hist_ew
        #return (hist_ew + income_ew + cflow_ew + t * 3)/6
