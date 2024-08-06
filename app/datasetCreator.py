import pandas as pd
import os
import json
import warnings
from dataLoader import Loader

warnings.simplefilter(action='ignore', category=FutureWarning)

class DatasetCreator:
    def __init__(self) -> None:
        # Load available stocks
        self.aviable_stocks = pd.read_csv('app_files/AviableStocksSymbols.csv')
        self.aviable_stocks['Symbol'] = self.aviable_stocks['Symbol'].astype(str).fillna('')
        self.tickers_request = self.aviable_stocks['Symbol'].to_list()

        # Check and update tickers from cache
        self._check_cached_tickers()

        # Initialize DataLoader if there are new tickers to fetch
        if self.tickers_request:
            self.DataLoader = Loader(' '.join(self.tickers_request))
            self._collect_tickers(pass_request=False)
        else:
            self._collect_tickers(pass_request=True)

        # Cache the collected tickers
        self._cache_tickers()

    def _cache_tickers(self, data_cache_path='app_files/data_cache/tickers_cache.json') -> None:
        try:
            with open(data_cache_path, "r") as file:
                data = json.load(file) if file.read().strip() else []
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        # Get new tickers
        collected_tickers = self.DataLoader.get_tickers() if hasattr(self, 'DataLoader') else []
        to_dump = []

        for ticker in collected_tickers:
            json_ticker = {}
            for e in ticker:
                try:
                    json_ticker[e] = ticker[e].to_json() if isinstance(ticker[e], pd.DataFrame) else ticker[e]
                except:
                    json_ticker[e] = ticker[e]
            to_dump.append(json_ticker)

        data.extend(to_dump)

        with open(data_cache_path, "w") as file:
            json.dump(data, file, indent=4)

    def _check_cached_tickers(self, data_cache_path='app_files/data_cache/tickers_cache.json') -> None:
        if os.path.exists(data_cache_path):
            with open(data_cache_path, "r") as file:
                try:
                    cached_tickers = json.load(file)
                except json.JSONDecodeError:
                    cached_tickers = []

                tickers_symbols = [ticker['stock_symbol'] for ticker in cached_tickers]
                self.tickers_request = [ticker for ticker in self.tickers_request if ticker not in tickers_symbols]

    def _collect_tickers(self, cache_path='app_files/data_cache/tickers_cache.json', pass_request=False) -> None:
        try:
            with open(cache_path, 'r') as file:
                cache_load = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            cache_load = []

        for ticker in cache_load:
            for e in ticker:
                try:
                    ticker[e] = pd.read_json(ticker[e]) if isinstance(ticker[e], str) else ticker[e]
                except:
                    pass

        if pass_request and hasattr(self, 'DataLoader'):
            DataLoader_data = self.DataLoader.get_tickers()
            self.tickers_collection = DataLoader_data + cache_load
        else:
            self.tickers_collection = cache_load

    def get_datasets(self) -> list[dict]:
        return self.tickers_collection

# Debug ONLY
# if __name__ == '__main__':
#     dc = DatasetCreator()