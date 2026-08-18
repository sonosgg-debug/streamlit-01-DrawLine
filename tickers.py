import pandas as pd
import requests
import io

def get_krx_tickers(market='ALL'):
    """
    KIND(한국거래소)에서 상장 종목 목록을 다운로드하여 야후 파이낸스 티커 포맷으로 변환합니다.
    market: 'ALL' (전체), 'KOSPI' (코스피), 'KOSDAQ' (코스닥)
    """
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    tickers_list = []
    
    # 1. KOSPI (stockMkt)
    if market in ['ALL', 'KOSPI']:
        try:
            # KIND에서 코스피 종목 다운로드
            response = requests.get(url + "&marketType=stockMkt")
            df_kospi = pd.read_html(io.StringIO(response.text))[0]
            df_kospi['종목코드'] = df_kospi['종목코드'].astype(str).str.zfill(6)
            df_kospi['ticker'] = df_kospi['종목코드'] + ".KS"
            df_kospi['시장'] = 'KOSPI'
            tickers_list.append(df_kospi[['회사명', 'ticker', '시장']])
        except Exception as e:
            print(f"KOSPI 종목 수집 실패: {e}")
            
    # 2. KOSDAQ (kosdaqMkt)
    if market in ['ALL', 'KOSDAQ']:
        try:
            # KIND에서 코스닥 종목 다운로드
            response = requests.get(url + "&marketType=kosdaqMkt")
            df_kosdaq = pd.read_html(io.StringIO(response.text))[0]
            df_kosdaq['종목코드'] = df_kosdaq['종목코드'].astype(str).str.zfill(6)
            df_kosdaq['ticker'] = df_kosdaq['종목코드'] + ".KQ"
            df_kosdaq['시장'] = 'KOSDAQ'
            tickers_list.append(df_kosdaq[['회사명', 'ticker', '시장']])
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
