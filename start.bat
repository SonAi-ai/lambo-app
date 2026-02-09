@echo off
title Lambo czy Karton? - Uruchamianie...
echo ========================================================
echo      ODPALAM SYSTEM ANALIZY DLA PAWLA...
echo      Sprawdzam, czy stac nas na Lambo...
echo ========================================================
echo.

:: Tutaj jest zmiana - zamiast "streamlit run" dajemy "python -m streamlit run"
python -m streamlit run market_app.py

pause