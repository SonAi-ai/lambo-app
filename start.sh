#!/bin/bash
echo "========================================================"
echo "     Wykryto Launcher Linux/Mac. Przygotowanie..."
echo "========================================================"
echo ""

# 1. INSTALACJA BIBLIOTEK
echo "🔧 Instaluje biblioteki (requests, streamlit, itp.)..."
python3 -m pip install -r requirements.txt

# 2. URUCHOMIENIE PROGRAMU GŁÓWNEGO (Streamlit)
echo ""
echo "🚀 Odpalam Lambo Dashboard..."
python3 -m streamlit run market_app.py