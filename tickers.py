import pandas as pd
import requests
import io
import FinanceDataReader as fdr

def get_krx_tickers(market='ALL'):
    """
    FinanceDataReader 및 위키피디아에서 상장 종목 목록을 다운로드하여 야후 파이낸스 티커 포맷으로 변환합니다.
    market: 'ALL' (전체), 'KOSPI' (코스피), 'KOSDAQ' (코스닥), 'S&P 500', 'NASDAQ 100'
    """
    tickers_list = []
    
    # 1. KOSPI
    if market in ['ALL', 'KOSPI']:
        try:
            df_kospi = fdr.StockListing('KOSPI')
            if not df_kospi.empty:
                code_col = [col for col in df_kospi.columns if col.lower() == 'code']
                name_col = [col for col in df_kospi.columns if col.lower() == 'name']
                if code_col and name_col:
                    df_result = pd.DataFrame()
                    df_result['회사명'] = df_kospi[name_col[0]]
                    df_result['ticker'] = df_kospi[code_col[0]].astype(str).str.zfill(6) + ".KS"
                    df_result['시장'] = 'KOSPI'
                    tickers_list.append(df_result)
        except Exception as e:
            print(f"KOSPI 종목 수집 실패: {e}")
            
    # 2. KOSDAQ
    if market in ['ALL', 'KOSDAQ']:
        try:
            df_kosdaq = fdr.StockListing('KOSDAQ')
            if not df_kosdaq.empty:
                code_col = [col for col in df_kosdaq.columns if col.lower() == 'code']
                name_col = [col for col in df_kosdaq.columns if col.lower() == 'name']
                if code_col and name_col:
                    df_result = pd.DataFrame()
                    df_result['회사명'] = df_kosdaq[name_col[0]]
                    df_result['ticker'] = df_kosdaq[code_col[0]].astype(str).str.zfill(6) + ".KQ"
                    df_result['시장'] = 'KOSDAQ'
                    tickers_list.append(df_result)
        except Exception as e:
            print(f"KOSDAQ 종목 수집 실패: {e}")
            
    # 3. S&P 500 (US)
    if market == 'S&P 500':
        try:
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(sp500_url, headers=headers)
            df_sp500 = pd.read_html(io.StringIO(response.text))[0]
            
            df_result = pd.DataFrame()
            df_result['회사명'] = df_sp500['Security']
            # yfinance 호환을 위해 . 대신 - 사용 (예: BRK.B -> BRK-B)
            df_result['ticker'] = df_sp500['Symbol'].astype(str).str.replace('.', '-', regex=False)
            df_result['시장'] = 'S&P 500'
            tickers_list.append(df_result)
        except Exception as e:
            print(f"S&P 500 수집 실패: {e}")
            
    # 4. NASDAQ 100 (US)
    if market == 'NASDAQ 100':
        try:
            nasdaq_url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(nasdaq_url, headers=headers)
            dfs = pd.read_html(io.StringIO(response.text))

            
            df_result = None
            for df in dfs:
                if 'Ticker' in df.columns and 'Company' in df.columns:
                    df_result = pd.DataFrame()
                    df_result['회사명'] = df['Company']
                    df_result['ticker'] = df['Ticker'].astype(str).str.replace('.', '-', regex=False)
                    df_result['시장'] = 'NASDAQ 100'
                    break
                    
            if df_result is not None:
                tickers_list.append(df_result)
        except Exception as e:
            print(f"NASDAQ 100 수집 실패: {e}")
            
    if not tickers_list:
        return pd.DataFrame(columns=['회사명', 'ticker', '시장'])
        
    df_result = pd.concat(tickers_list, ignore_index=True)
    
    # 한국 주식인 경우만 우선주/스팩 필터링 적용
    kr_mask = df_result['ticker'].str.endswith('.KS') | df_result['ticker'].str.endswith('.KQ')
    
    df_kr = df_result[kr_mask]
    df_us = df_result[~kr_mask]
    
    df_kr = df_kr[~df_kr['회사명'].str.endswith('우')]
    df_kr = df_kr[~df_kr['회사명'].str.endswith('우B')]
    df_kr = df_kr[~df_kr['회사명'].str.contains('스팩')]
    df_kr = df_kr[~df_kr['회사명'].str.contains('제1호')]
    
    df_result = pd.concat([df_kr, df_us], ignore_index=True)
    df_result = df_result.reset_index(drop=True)
    
    return df_result


if __name__ == "__main__":
    # 테스트 실행
    print("종목 목록 수집 테스트 중...")
    df = get_krx_tickers('ALL')
    print(df.head())
    print(f"총 수집된 종목 수: {len(df)}")
