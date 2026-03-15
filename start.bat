@echo off
TITLE Lambo Start System
echo ========================================================
echo      Wykryto Launcher 'py'. Naprawianie srodowiska...
echo ========================================================
echo.

:: 1. INSTALACJA BRAKUJACYCH ELEMENTOW (Kluczowy krok, ktorego zabraklo)
echo 🔧 Instaluje biblioteki (requests, streamlit, itp.)...
py -m pip install -r requirements.txt

:: 2. URUCHOMIENIE SKANERA TELEGRAM (Zwiad przed odpaleniem apki)
echo.
echo 📡 Uruchamiam Skaner Rynku (Wysylanie raportu na Telegram)...
py lambo_cron_bot.py

:: 3. URUCHOMIENIE PROGRAMU GŁÓWNEGO (Streamlit)
echo.
echo 🚀 Skaner zakonczyl prace. Odpalam Lambo Dashboard...
py -m streamlit run market_app.py

:: 4. W razie bledow nie zamykaj okna
pause