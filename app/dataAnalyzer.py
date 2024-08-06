from datasetCreator import DatasetCreator
from models.lstm_model import LSTM

from termcolor import colored

import asyncio

class Analyzer:
    def __init__(self) -> None:
        print(colored("INITIALIZING DATASET CREATOR", "red"))
        self.dcreator = DatasetCreator()
        print(colored("SUCCESSFUL", "red"))
        print(colored("DOWNLOADING DATA", "red"))
        self.data = self.dcreator.get_datasets()
        print(colored("SUCCESSFUL", "red"))

    async def _run_model(self, ticker) -> None:
        lstm = LSTM(ticker)
        result = await lstm.evaluate()
        print(colored(f"Model evaluation for {ticker['stock_symbol']} completed with score: {result}", "yellow"))

    async def perform_model_creation(self) -> None:
        print(colored("TRAINING MODELS", "green"))
        tasks = []
        for ticker in self.data:
            tasks.append(asyncio.create_task(self._run_model(ticker)))
        
        await asyncio.gather(*tasks)
        print(colored("ALL MODELS TRAINED SUCCESSFULLY", "green"))

# Debug ONLY
if __name__ == '__main__':
    analyzer = Analyzer()
    print(colored("TESTING MODEL CREATION", "green"))
    
    # Running the asynchronous function in the event loop
    asyncio.run(analyzer.perform_model_creation())
