import yfinance
import json
from tqdm import tqdm
import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class Loader:
    def _get_prohibited_branches(self,
                                 pes_path='app_files/prohibited_economic_sectors.json'
                                ) -> list[str]:
        return json.load(open(pes_path, 'r'))
    
    #def _is_valid_branch(self, ticker) -> bool:
    #    #print(ticker)
    #    #return ticker.info['sector'] not in self.prohibited_branches
    #    pass

    def _start_log(self,
                    logs_folder_path='app_files/logs/'
                   ) -> None:
        
        now = datetime.datetime.now()
        # Format the time as hhmmss
        formatted_time = now.strftime("%H%M%S")
        self.log_file = open(logs_folder_path + 'data_load_log_' + formatted_time + '.log', 'w')

    def __init__(self, tickers_req: list[str]) -> None:
        tickers = yfinance.Tickers(tickers_req)
        #print(tickers)
        self.filtered_tickers = []

        self.prohibited_branches = self._get_prohibited_branches()
        #print(self.prohibited_branches)

        self._start_log()

        for ticker_id in tqdm(tickers_req.split(), desc="Processing tickers"):
            #print(tickers.tickers['MSFT'].info['sector'])
            try:
                #   OLD
                #print(self._is_valid_branch(tickers.tickers[ticker_id]))
                #if self._is_valid_branch(tickers.tickers[ticker_id.upper()]) and '^' not in list(ticker_id):
                #print(tickers.tickers[ticker_id.upper()])

                ticker = tickers.tickers[ticker_id.upper()]
                to_dump = {
                    'stock_symbol' : ticker_id,
                    'stock_history' : ticker.history(period="max"),
                    'quarterly_income_stmt' : ticker.quarterly_income_stmt,
                    'quarterly_cashflow' : ticker.quarterly_cashflow
                }
                
                #print(to_dump)
                self.filtered_tickers.append(to_dump)

                self.log_file.write(f'SUCCESFUL FETCH: {ticker_id.upper()}' + '\n')
            except Exception as e:
                self.log_file.write(f'UNSUCCESFUL FETCH: {ticker_id.upper()}' + '\n')
                #print(f"Error fetching data for {ticker_id}: {e}")

    def get_tickers(self) -> dict:
        return self.filtered_tickers

# Debug ONLY
#if __name__ == '__main__':
#    tickers = 'msft aapl goog'
#    loader = Loader(tickers)
#    print(loader.get_tickers())