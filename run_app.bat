@echo off
title David Ryan Just Draw the Line Stock Screener
cd /d "%~dp0"

echo =====================================================================
echo   David Ryan "Just Draw the Line" Stock Screener
echo =====================================================================
echo.
echo [알림] Streamlit 앱 서버를 시작합니다...
echo [알림] 서버 구동이 완료되면 웹 브라우저가 자동으로 열립니다.
echo.
echo ※ 앱을 종료하려면 이 창을 닫거나 콘솔에서 [Ctrl + C]를 누르세요.
echo ---------------------------------------------------------------------
echo.

python -m streamlit run app.py --server.headless false

if errorlevel 1 (
    echo.
    echo [오류] 앱 실행 중 문제가 발생했습니다.
    echo 파이썬 환경 또는 패키지 설치(pip install -r requirements.txt)를 확인하세요.
    echo.
    pause
)
