import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime, timezone, timedelta
from scipy.optimize import linprog
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

KST = timezone(timedelta(hours=9))

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
    best_a, best_b = None, None
    
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
                best_a, best_b = a, b
                break
                
    if not is_breakout or breakout_idx_offset is None:
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
        'trend_slope': best_a,
        'trend_intercept': best_b
    }


def _process_single_stock(
    ticker: str,
    name: str,
    start_date: str,
    lookback_period: int,
    vol_ratio_thresh: float,
    apply_trend_template: bool,
    breakout_window: int,
    df_cached: pd.DataFrame = None
):
    """
    단일 종목의 데이터를 수집하고 데이비드 라이언 상단 추세선 돌파 스크리닝을 수행하는 통합 워커 함수.
    """
    try:
        if df_cached is not None:
            df = df_cached
        else:
            code = ticker.split('.')[0]
            df = fdr.DataReader(code, start_date)

        if df is None or len(df) < 220:
            return None

        # 미체결 당일 더미 행(Volume=0) 방지 처리
        if len(df) > 1 and df['Volume'].iloc[-1] == 0:
            df = df.iloc[:-1]

        # 거래정지나 최근 5영업일 거래량 전무 종목 제외
        recent_vol = df['Volume'].iloc[-5:].sum()
        if pd.isna(recent_vol) or recent_vol <= 0:
            return None

        df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if len(df) < 220:
            return None

        return screen_single_stock(
            ticker=ticker,
            name=name,
            df=df,
            lookback_period=lookback_period,
            vol_ratio_thresh=vol_ratio_thresh,
            apply_trend_template=apply_trend_template,
            breakout_window=breakout_window
        )
    except Exception:
        return None


def run_screening_task(
    tickers_df: pd.DataFrame,
    lookback_period: int = 40,
    vol_ratio_thresh: float = 1.5,
    apply_trend_template: bool = True,
    breakout_window: int = 3,
    max_workers: int = 24,
    progress_callback = None
) -> pd.DataFrame:
    """
    App-20 초고속 멀티스레딩 엔진 방식을 적용하여
    전체 유니버스 종목의 데이터 수집과 추세선 돌파 판정을 완전 병렬로 수행합니다.
    """
    if tickers_df is None or tickers_df.empty:
        return pd.DataFrame()

    total_stocks = len(tickers_df)
    results = []
    completed_count = 0

    start_date = (datetime.now(KST) - timedelta(days=730)).strftime('%Y-%m-%d')

    kr_mask = tickers_df['ticker'].str.endswith('.KS') | tickers_df['ticker'].str.endswith('.KQ')
    df_kr = tickers_df[kr_mask].copy()
    df_us = tickers_df[~kr_mask].copy()

    # 1. 한국 주식: FinanceDataReader 기반 초고속 멀티스레딩 원스톱 수집 & 분석
    if not df_kr.empty:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(
                    _process_single_stock,
                    row['ticker'],
                    row['회사명'],
                    start_date,
                    lookback_period,
                    vol_ratio_thresh,
                    apply_trend_template,
                    breakout_window
                ): (row['ticker'], row['회사명'])
                for _, row in df_kr.iterrows()
            }

            for future in as_completed(future_to_stock):
                ticker, name = future_to_stock[future]
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_stocks, name)
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception:
                    pass

    # 2. 미국 주식: yfinance 일괄 다운로드 후 멀티스레드 병렬 분석
    if not df_us.empty:
        us_tickers = df_us['ticker'].tolist()
        name_map = dict(zip(df_us['ticker'], df_us['회사명']))
        try:
            data = yf.download(us_tickers, period="2y", group_by="ticker", progress=False, timeout=20)
            us_stock_dfs = {}
            for t in us_tickers:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        ticker_level = 'Ticker' if 'Ticker' in data.columns.names else 1
                        tickers_in_data = data.columns.get_level_values(ticker_level).unique()
                        if t not in tickers_in_data:
                            continue
                        df_single = data.xs(t, level=ticker_level, axis=1).dropna(subset=['Close', 'High', 'Low', 'Volume'])
                    else:
                        df_single = data.dropna(subset=['Close', 'High', 'Low', 'Volume'])

                    if df_single.index.tz is not None:
                        df_single.index = df_single.index.tz_localize(None)

                    if len(df_single) >= 220:
                        us_stock_dfs[t] = df_single
                except Exception:
                    continue

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_us = {
                    executor.submit(
                        _process_single_stock,
                        t,
                        name_map[t],
                        start_date,
                        lookback_period,
                        vol_ratio_thresh,
                        apply_trend_template,
                        breakout_window,
                        us_stock_dfs[t]
                    ): (t, name_map[t])
                    for t in us_stock_dfs
                }

                for future in as_completed(future_to_us):
                    t, name = future_to_us[future]
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_stocks, name)
                    try:
                        res = future.result()
                        if res is not None:
                            results.append(res)
                    except Exception:
                        pass
        except Exception as e:
            print(f"미국 주식 다운로드 중 오류: {e}")

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    # 거래량 비율 기준 내림차순 정렬
    if 'vol_ratio' in df_res.columns:
        df_res = df_res.sort_values(by='vol_ratio', ascending=False).reset_index(drop=True)

    return df_res


def run_screener(tickers_df, lookback_period=40, vol_ratio_thresh=1.5, chunk_size=50, max_workers=24):
    """하위 호환성을 위한 래퍼 함수"""
    return run_screening_task(
        tickers_df=tickers_df,
        lookback_period=lookback_period,
        vol_ratio_thresh=vol_ratio_thresh,
        apply_trend_template=True,
        breakout_window=3,
        max_workers=max_workers
    )
