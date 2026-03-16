@echo off
TITLE Lambo Start System
echo ========================================================
echo      Wykryto Launcher 'py'. Naprawianie srodowiska...
echo ========================================================
echo.

:: 1. INSTALACJA BRAKUJACYCH ELEMENTOW
echo 🔧 Instaluje biblioteki (requests, streamlit, itp.)...
py -m pip install -r requirements.txt

:: 2. URUCHOMIENIE PROGRAMU GŁÓWNEGO (Streamlit)
echo.
echo 🚀 Odpalam Lambo Dashboard...
py -m streamlit run market_app.py

:: 3. W razie bledow nie zamykaj okna
pause