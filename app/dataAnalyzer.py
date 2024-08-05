from datasetCreator import DatasetCreator
from models.lstm_model import LSTM

class analyzer:
    def __init__(self) -> None:
        dcreator = DatasetCreator()
        self.data = dcreator.get_datasets()
    
    def perform_mdoel_creation(self) -> None:
        for ticker in self.data:
            lstm = LSTM(ticker)
            print(lstm.evaluate())

# Debug ONLY
if __name__ == '__main__':
    analyzer = analyzer()
    analyzer.perform_mdoel_creation()