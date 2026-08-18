import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import linprog
from scipy.signal import find_peaks
import concurrent.futures
import time

def fit_upper_trendline(high_prices):
    """
    최근 고가(High) 데이터를 바탕으로 상단 저항선(Downtrend/Horizontal line)을 
    선형 프로그래밍(Linear Programming)을 이용해 피팅합니다.
    - 모든 가격이 이 선 아래 또는 선상에 위치하도록 하며,
    - 마지막 시점의 선 가격(또는 전체 오차의 합)을 최소화하도록 유도합니다.
    """
    n = len(high_prices)
    if n < 5:
        return None, None, None
        
    x = np.arange(n)
    
    # 변수: [a, b] (y = ax + b)
    # 목적함수: 마지막 날의 추세선 값 (a * (n-1) + b) 최소화
    c = [n - 1, 1]
    
    # 제약조건: a * i + b >= high_prices[i] => -i * a - b <= -high_prices[i]
    A = []
    b_ub = []
    for i in range(n):
        A.append([-i, -1])
        b_ub.append(-high_prices[i])
        
    # a <= 0 (하향 또는 평행 저항선), b >= 0
    bounds = [(-np.inf, 0), (0, np.inf)]
    
    res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        a, b = res.x
        trendline = a * x + b
        return trendline, a, b
    else:
        return None, None, None

def screen_single_stock(ticker, name, df, lookback_period=40, vol_ratio_thresh=1.5, apply_trend_template=True, breakout_window=2):
    """
    개별 주식 데이터프레임을 받아 데이비드 라이언 스크리닝 조건을 검증합니다.
    """
    if df is None or len(df) < 220: # 200일 MA + 20일 우상향 확인을 위한 최소 영업일수
        return None
        
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    ma50 = close.rolling(window=50).mean()
    ma150 = close.rolling(window=150).mean()
    ma200 = close.rolling(window=200).mean()
    
    curr_price = close.iloc[-1]
    curr_ma50 = ma50.iloc[-1]
    curr_ma150 = ma150.iloc[-1]
    curr_ma200 = ma200.iloc[-1]
    ma200_20ago = ma200.iloc[-20] if len(ma200) > 20 else curr_ma200
    
    high_52w = high.iloc[-250:].max()
    low_52w = low.iloc[-250:].min()
    
    cond1 = curr_price > curr_ma150 and curr_price > curr_ma200
    cond2 = curr_ma150 > curr_ma200
    cond3 = curr_ma200 > ma200_20ago
    cond4 = curr_ma50 > curr_ma150 and curr_ma50 > curr_ma200
    cond5 = curr_price > curr_ma50
    cond6 = curr_price >= (low_52w * 1.25)
    cond7 = curr_price >= (high_52w * 0.75)
    
    if apply_trend_template:
        if not (cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7):
            return None

        
    is_breakout = False
    breakout_date = None
    breakout_vol = 0
    breakout_idx_offset = None
    
    for i in range(1, breakout_window + 1):
        start_idx = -lookback_period - i
        end_idx = -i
        
        if len(df) < abs(start_idx):
            continue
            
        fit_highs = high.iloc[start_idx:end_idx].values
        fit_closes = close.iloc[start_idx:end_idx].values
        
        trendline_fit, a, b = fit_upper_trendline(fit_highs)
        if trendline_fit is None:
            continue
            
        extrapolated_trend_val = a * lookback_period + b
        prev_trend_val = trendline_fit[-1]
        
        cond_breakout = (close.iloc[-i] > extrapolated_trend_val) and (close.iloc[-i-1] <= prev_trend_val)
        
        if cond_breakout:
            still_above = True
            for j in range(1, i):
                extrapolated_j = a * (lookback_period + (i - j)) + b
                if close.iloc[-j] <= extrapolated_j:
                    still_above = False
                    break
            
            if still_above:
                is_breakout = True
                breakout_date = df.index[-i].strftime('%Y-%m-%d')
                breakout_vol = volume.iloc[-i]
                breakout_idx_offset = i
                break
                
    if not is_breakout:
        return None
        
    avg_vol_20 = volume.iloc[-20-lookback_period-breakout_idx_offset:-lookback_period-breakout_idx_offset].mean() 
    if avg_vol_20 == 0 or np.isnan(avg_vol_20):
        avg_vol_20 = volume.iloc[-20:].mean()
        
    vol_ratio = breakout_vol / avg_vol_20 if avg_vol_20 > 0 else 0
    
    if vol_ratio < vol_ratio_thresh:
        return None

        
    breakout_date = df.index[-breakout_idx_offset].strftime('%Y-%m-%d')



    
    return {
        'ticker': ticker,
        'name': name,
        'price': curr_price,
        'ma50': curr_ma50,
        'ma150': curr_ma150,
        'ma200': curr_ma200,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'vol_ratio': vol_ratio,
        'breakout_date': breakout_date,
        'trend_slope': a,
        'trend_intercept': b
    }

def run_screener(tickers_df, lookback_period=40, vol_ratio_thresh=1.5, chunk_size=50):
    """
    전체 상장종목에 대해 yfinance 데이터를 수집하고 스크리닝을 수행합니다.
    """
    results = []
    tickers = tickers_df['ticker'].tolist()
    name_map = dict(zip(tickers_df['ticker'], tickers_df['회사명']))
    
    total_tickers = len(tickers)
    chunks = [tickers[i:i + chunk_size] for i in range(0, total_tickers, chunk_size)]
    
    print(f"스크리닝 시작: 총 {total_tickers}개 종목, {len(chunks)}개 청크로 나누어 데이터 다운로드 진행")
    
    start_time = time.time()
    
    for idx, chunk in enumerate(chunks):
        print(f"청크 처리 중 [{idx+1}/{len(chunks)}] (종목수: {len(chunk)})...")
        try:
            # yfinance를 통해 멀티 티커 1년치 데이터 일괄 다운로드
            data = yf.download(chunk, period="1y", group_by="ticker", progress=False)
            
            # 단일 종목 다운로드인 경우 구조 조정
            if len(chunk) == 1:
                ticker = chunk[0]
                df_single = data
                if not df_single.empty:
                    # MultiIndex가 아니면 직접 처리
                    res = screen_single_stock(ticker, name_map[ticker], df_single, lookback_period, vol_ratio_thresh)
                    if res:
                        results.append(res)
                continue
                
            # 여러 종목 다운로드인 경우
            for ticker in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        ticker_level = 'Ticker' if 'Ticker' in data.columns.names else 1
                        tickers_in_data = data.columns.get_level_values(ticker_level).unique()
                        if ticker not in tickers_in_data:
                            continue
                        df_single = data.xs(ticker, level=ticker_level, axis=1).dropna(subset=['Close', 'High', 'Low', 'Volume'])
                    else:
                        df_single = data.dropna(subset=['Close', 'High', 'Low', 'Volume'])
                        
                    if len(df_single) < 100:
                        continue
                        
                    res = screen_single_stock(ticker, name_map[ticker], df_single, lookback_period, vol_ratio_thresh)
                    if res:
                        results.append(res)
                except Exception as e:
                    # 특정 종목 오류 발생 시 건너뜀
                    continue

                    
        except Exception as e:
            print(f"청크 다운로드 실패: {e}")
            
        # API 과부하 방지를 위한 짧은 딜레이
        time.sleep(0.5)
        
    print(f"스크리닝 완료! 소요 시간: {time.time() - start_time:.2f}초. 탐지된 종목 수: {len(results)}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    from tickers import get_krx_tickers
    print("스크리너 유닛 테스트 실행")
    # 샘플 테스트로 50개 종목만 테스트 진행
    tickers_df = get_krx_tickers('KOSPI').head(50)
    res_df = run_screener(tickers_df)
    print(res_df)
