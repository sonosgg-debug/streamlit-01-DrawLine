import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timezone, timedelta
import FinanceDataReader as fdr

KST = timezone(timedelta(hours=9))

def is_valid_screening_stock(code: str, name: str) -> bool:
    """
    기술적 분석 스크리닝에 부적합한 노이즈 종목(우선주, 스팩, ETF/ETN, 정리매매, 리츠 등)을 걸러냅니다.
    """
    code_str = str(code).strip()
    name_str = str(name).strip()

    # 1. 우선주 배제 (보통 끝자리 '0'이 아니거나 명칭에 '우' 포함)
    if not code_str.endswith('0'):
        return False
    if name_str.endswith('우') or ('우B' in name_str) or ('우C' in name_str):
        return False

    # 2. 스팩(SPAC) 배제
    if '스팩' in name_str or '제1호' in name_str or '제2호' in name_str:
        return False

    # 3. ETF / ETN 배제 (KODEX, TIGER, KBSTAR, ACE, SOL, HANARO 등)
    etf_prefixes = ['KODEX', 'TIGER', 'KBSTAR', 'ACE', 'SOL', 'HANARO', 'KOSEF', 'ARIRANG', 'PLUS', 'TIMEFOLIO']
    for prefix in etf_prefixes:
        if name_str.startswith(prefix):
            return False

    # 4. 리츠, 인프라투융자회사 등 일반 제조업/서비스업과 차트 속성이 다른 종목 배제
    if name_str.endswith('리츠') or '투융자' in name_str:
        return False

    return True


_CACHED_FALLBACK_MARCAP = None


def get_fallback_marcap_map() -> dict:
    """
    FinanceDataReader의 KRX 당일자 캐시 파일이 장중/장마감 전이라 시가총액(Marcap)이 결측치(NaN)인 경우,
    최근 10영업일을 역순으로 탐색하여 시가총액이 유효하게 존재하는 가장 최근 거래일의 데이터를 로드하고
    {종목코드: 시가총액} 매핑 딕셔너리를 반환합니다.
    """
    global _CACHED_FALLBACK_MARCAP
    if _CACHED_FALLBACK_MARCAP is not None and len(_CACHED_FALLBACK_MARCAP) > 0:
        return _CACHED_FALLBACK_MARCAP

    base_url = "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing/krx/"
    today = datetime.now(KST)

    for days_back in range(0, 11):
        target_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"{base_url}{target_date}.csv"
        try:
            df = pd.read_csv(url, dtype={"Code": str, "ISU_SRT_CD": str}, low_memory=False)
            code_col = "Code" if "Code" in df.columns else "ISU_SRT_CD"
            marcap_col = None
            for col in ["Marcap", "MKTCAP", "시가총액"]:
                if col in df.columns:
                    marcap_col = col
                    break

            if marcap_col and code_col in df.columns:
                valid_series = pd.to_numeric(df[marcap_col], errors="coerce")
                if valid_series.notna().sum() > 500:
                    df["clean_code"] = df[code_col].astype(str).str.zfill(6)
                    df["clean_marcap"] = valid_series.fillna(0)
                    marcap_dict = dict(zip(df["clean_code"], df["clean_marcap"]))
                    _CACHED_FALLBACK_MARCAP = marcap_dict
                    return _CACHED_FALLBACK_MARCAP
        except Exception:
            continue

    return {}


def get_krx_tickers(market='KOSPI', scope='top500', min_marcap_eok=0) -> pd.DataFrame:
    """
    FinanceDataReader 및 위키피디아에서 상장 종목 목록을 다운로드하여 야후 파이낸스 티커 포맷으로 변환합니다.
    - market: 'KOSPI', 'KOSDAQ', 'ALL', 'S&P 500', 'NASDAQ 100'
    - scope: 'top300', 'top500', 'top1000', 'all'
    - min_marcap_eok: 최소 시가총액 (단위: 억원)
    """
    clean_market = market.upper().strip()

    # 1. 미국 주식 (S&P 500, NASDAQ 100)
    if 'S&P' in clean_market or '500' in clean_market:
        try:
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(sp500_url, headers=headers, timeout=15)
            df_sp500 = pd.read_html(io.StringIO(response.text))[0]
            
            df_result = pd.DataFrame()
            df_result['회사명'] = df_sp500['Security']
            df_result['ticker'] = df_sp500['Symbol'].astype(str).str.replace('.', '-', regex=False)
            df_result['시장'] = 'S&P 500'
            df_result['시가총액_억원'] = 0
            if scope == 'top300':
                df_result = df_result.head(300)
            return df_result.reset_index(drop=True)
        except Exception as e:
            print(f"S&P 500 수집 실패: {e}")
            return pd.DataFrame(columns=['회사명', 'ticker', '시장', '시가총액_억원'])

    if 'NASDAQ' in clean_market or '100' in clean_market:
        try:
            nasdaq_url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(nasdaq_url, headers=headers, timeout=15)
            dfs = pd.read_html(io.StringIO(response.text))
            
            df_result = None
            for df in dfs:
                if 'Ticker' in df.columns and 'Company' in df.columns:
                    df_result = pd.DataFrame()
                    df_result['회사명'] = df['Company']
                    df_result['ticker'] = df['Ticker'].astype(str).str.replace('.', '-', regex=False)
                    df_result['시장'] = 'NASDAQ 100'
                    df_result['시가총액_억원'] = 0
                    break
            if df_result is not None:
                return df_result.reset_index(drop=True)
        except Exception as e:
            print(f"NASDAQ 100 수집 실패: {e}")
            return pd.DataFrame(columns=['회사명', 'ticker', '시장', '시가총액_억원'])

    # 2. 한국 주식 (KOSPI, KOSDAQ, ALL)
    target_markets = []
    if 'ALL' in clean_market or ('KOSPI' in clean_market and 'KOSDAQ' in clean_market):
        target_markets = ['KOSPI', 'KOSDAQ']
    elif 'KOSDAQ' in clean_market or '코스닥' in market:
        target_markets = ['KOSDAQ']
    else:
        target_markets = ['KOSPI']

    dfs_kr = []
    for m in target_markets:
        try:
            df_listing = fdr.StockListing(m)
        except Exception as e:
            print(f"StockListing({m}) 실패: {e}")
            df_all = fdr.StockListing('KRX')
            target_id = 'STK' if m == 'KOSPI' else 'KSQ'
            if 'MarketId' in df_all.columns:
                df_listing = df_all[df_all['MarketId'] == target_id].copy()
            elif 'Market' in df_all.columns:
                df_listing = df_all[df_all['Market'].str.upper() == m].copy()
            else:
                df_listing = df_all.copy()

        # 컬럼 표준화
        if 'Code' not in df_listing.columns and 'Symbol' in df_listing.columns:
            df_listing['Code'] = df_listing['Symbol']
        if 'Code' not in df_listing.columns and 'ISU_SRT_CD' in df_listing.columns:
            df_listing['Code'] = df_listing['ISU_SRT_CD']

        df_listing['Code'] = df_listing['Code'].astype(str).str.zfill(6)
        suffix = ".KS" if m == 'KOSPI' else ".KQ"
        df_listing['ticker'] = df_listing['Code'] + suffix
        df_listing['시장'] = m

        # 노이즈 종목 필터링
        valid_mask = df_listing.apply(lambda r: is_valid_screening_stock(r['Code'], r['Name']), axis=1)
        df_filtered = df_listing[valid_mask].copy()

        # 시가총액 확보
        marcap_col = None
        for col in ['Marcap', 'MKTCAP', '시가총액', 'MarketCap']:
            if col in df_filtered.columns:
                marcap_col = col
                break

        if marcap_col:
            df_filtered['Marcap_Num'] = pd.to_numeric(df_filtered[marcap_col], errors='coerce').fillna(0)
        else:
            df_filtered['Marcap_Num'] = 0

        # 결측 시 폴백 적용
        if (df_filtered['Marcap_Num'] > 0).sum() < max(10, len(df_filtered) * 0.5):
            fallback_map = get_fallback_marcap_map()
            if fallback_map:
                fallback_series = df_filtered['Code'].map(fallback_map).fillna(0)
                df_filtered['Marcap_Num'] = np.where(
                    df_filtered['Marcap_Num'] > 0,
                    df_filtered['Marcap_Num'],
                    fallback_series
                )

        dfs_kr.append(df_filtered)

    if not dfs_kr:
        return pd.DataFrame(columns=['회사명', 'ticker', '시장', '시가총액_억원'])

    df_combined = pd.concat(dfs_kr, ignore_index=True)
    
    # 시가총액 기준 내림차순 정렬
    df_sorted = df_combined.sort_values(by='Marcap_Num', ascending=False).reset_index(drop=True)

    # 최소 시가총액 필터 (단위: 억원)
    if min_marcap_eok > 0 and (df_sorted['Marcap_Num'] > 0).any():
        min_won = min_marcap_eok * 100_000_000
        df_sorted = df_sorted[df_sorted['Marcap_Num'] >= min_won].reset_index(drop=True)

    # 대상 범위 지정 (scope)
    if scope == 'top300':
        df_result = df_sorted.head(300)
    elif scope == 'top500':
        df_result = df_sorted.head(500)
    elif scope == 'top1000':
        df_result = df_sorted.head(1000)
    else:  # 'all'
        df_result = df_sorted

    df_final = pd.DataFrame()
    df_final['회사명'] = df_result['Name']
    df_final['ticker'] = df_result['ticker']
    df_final['시장'] = df_result['시장']
    df_final['시가총액_억원'] = (df_result['Marcap_Num'] / 100_000_000).round().astype(int)

    return df_final.reset_index(drop=True)


def get_latest_expected_trading_day(target_date: str = None) -> str:
    """
    가장 최근 거래 완료된 실제 영업일 YYYY-MM-DD 반환.
    - target_date가 전달된 경우: 해당 날짜 기준 (또는 직전 영업일)
    - target_date가 없는 경우: KST 기준 15:45 이전이거나 오늘이 주말/새벽이면 직전 마감 거래일 반환
    """
    now_kst = datetime.now(KST)
    if target_date:
        try:
            clean_date = str(target_date).replace('-', '')
            dt = datetime.strptime(clean_date, "%Y%m%d").replace(tzinfo=KST)
        except Exception:
            dt = now_kst
    else:
        dt = now_kst

    # 평일 15:45 이후에만 당일 종가 확정
    if dt.weekday() < 5 and (dt.hour > 15 or (dt.hour == 15 and dt.minute >= 45)):
        return dt.strftime("%Y-%m-%d")

    # 장전, 새벽, 주말: 직전 마감 거래일 산출
    if dt.weekday() == 0:    # 월요일 장전 -> 지난주 금요일 (3일 전)
        days_back = 3
    elif dt.weekday() == 6:  # 일요일 -> 지난주 금요일 (2일 전)
        days_back = 2
    elif dt.weekday() == 5:  # 토요일 -> 지난주 금요일 (1일 전)
        days_back = 1
    else:                    # 화~금 장전/새벽 -> 전일 (1일 전)
        days_back = 1

    return (dt - timedelta(days=days_back)).strftime("%Y-%m-%d")


if __name__ == "__main__":
    print("종목 목록 수집 테스트 중...")
    df = get_krx_tickers('KOSPI', scope='top500')
    print(df.head())
    print(f"총 수집된 종목 수: {len(df)}")
