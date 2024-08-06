from datasetCreator import DatasetCreator
from models.lstm_model import LSTM

from termcolor import colored

class analyzer:
    def __init__(self) -> None:
        print(colored("INITIALIZING DATASET CREATOR", "red"))
        dcreator = DatasetCreator()
        print(colored("SUCCESFUL", "red"))
        print(colored("DOWNLOADING DATA", "red"))
        self.data = dcreator.get_datasets()
        print(colored("SUCCESFUL", "red"))
    
    def perform_mdoel_creation(self) -> None:
        print(colored("TRAINING MODELS", "green"))
        for ticker in self.data:
            lstm = LSTM(ticker)
            print(lstm.evaluate())

# Debug ONLY
if __name__ == '__main__':
    analyzer = analyzer()
    print(colored("TESTING MODEL CREATION", "green"))
    analyzer.perform_mdoel_creation()
    print(colored("SUCCESFUL", "green"))