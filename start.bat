@echo off
TITLE Lambo czy Karton - Launcher
echo ========================================================
echo      🚀 STARTUJE SYSTEM LAMBO CZY KARTON...
echo      Czekaj, sprawdzam czy masz wszystkie czesci...
echo ========================================================
echo.

:: 1. Najpierw instalujemy biblioteki (jesli ich nie ma)
echo 🔧 Sprawdzanie bibliotek (moze chwile potrwac)...
pip install -r requirements.txt

:: 2. Jesli instalacja sie uda (lub juz sa), odpalamy program
echo.
echo 📈 Uruchamianie Aplikacji...
echo.
streamlit run market_app.py

:: 3. Pauza na koniec, zeby okno nie zniknelo w razie bledu
echo.
echo ========================================================
echo      JESLI WIDZISZ BLAD POWYZEJ:
echo      Zrob zdjecie i wyslij do Pawla.
echo ========================================================
pause