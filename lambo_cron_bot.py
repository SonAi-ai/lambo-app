import yfinance as yf
import pandas as pd
import json
import requests
import os
import warnings

warnings.filterwarnings('ignore')

class LamboGroupScanner:
    """Samodzielny skrypt CRON do skanowania rynku pod kątem MACRO oraz SWING TRADE"""
    
    def __init__(self, bot_token, group_chat_id):
        self.bot_token = bot_token
        self.group_chat_id = group_chat_id
        self.portfolio_file = "lambo_portfolio.json"
        
        # --- GLOBALNA LISTA VIP (Naprawiony błąd składni przy XTB.WA) ---
        self.vip_tickers = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'XLM-USD',
            'NVDA', 'TSLA', 'PLTR', 'MSTR', 'COIN', 'LUNR', 'DELL', 'UEC', 'FSLR', 'PATH', 'GEVO', 'BA', 'VRT', 'ADSK', 'SYK', 'LITE', 'PTC', 'ISRG',
            'SYM', 'RTX', 'CCJ', 'OKLO', 'SMR', 'GPP.WA', 'ENPH', 'JSW.WA', 'NKE', 'VOW3.DE', 'VLO', 'DAR', 'BA.L', 'LDO.MI', 'DFEN', 'HO.PA', 'UNH',
            'AMD', 'UBER', 'SNOW', 'XTB.WA', 'NBIS', 'MP', 'CRWD', 'IONQ', 'KTOS', 'AMZN', 'S', 'REMX', 'LYC.AX', 'ILU.AX', 'ARR.AX', 'NTU.AX', 
            'TMRC', 'UUUU', 'UURAF', 'ARE.TO', 'METL', 
            'CCJ', 'NXE', 'DNN', 'ALB', 'SQM', 'SGML', 'LAC', 'SLI',
            'CREG', 'FLNC', 'QS', 'ENVX', 'ENOV', 'LIT', 'LTH',
            'PLS.AX', 'MIN.AX', 'CHPT'
        ]

    def get_portfolio_tickers(self):
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [v["ticker"] for k, v in data.items()]
            except Exception:
                pass
        return []

    def analyze_macro(self, ticker):
        """Skaner MACRO (10 lat, interwał miesięczny) - Szuka punktów zwrotnych na lata"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="10y", interval="1mo")
            
            if hist.empty or len(hist) < 15:
                return None
                
            hist = hist.dropna(subset=['Close'])
            close = hist['Close']
            vol = hist['Volume']
            
            curr_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            
            sma_10 = float(close.rolling(window=10).mean().iloc[-1])
            sma_10_prev = float(close.rolling(window=10).mean().iloc[-2]) if len(close) > 10 else curr_price
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            curr_rsi = float(rsi.iloc[-1])
            
            obv = [0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]: obv.append(obv[-1] + vol.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]: obv.append(obv[-1] - vol.iloc[i])
                else: obv.append(obv[-1])
            hist['OBV'] = obv
            
            lookback = min(24, len(close))
            recent_hist = hist.tail(lookback)
            macro_high = recent_hist['High'].max()
            macro_low = recent_hist['Low'].min()
            
            if macro_high > macro_low:
                range_position = ((curr_price - macro_low) / (macro_high - macro_low)) * 100
            else:
                range_position = 50.0

            signal = None
            
            if len(rsi) >= 20 and range_position > 90:
                past_rsi_max = rsi.iloc[-20:-4].max()
                recent_rsi_max = rsi.iloc[-4:].max() 
                if recent_rsi_max < past_rsi_max and curr_rsi > 55:
                    signal = f"☠️ FALA 5 + DYWERGENCJA (Cena przy makro szczycie. RSI: {curr_rsi:.1f}. Śmierdzi korektą!)"
            
            if not signal and range_position < 35:
                obv_now = hist['OBV'].iloc[-1]
                obv_3m_ago = hist['OBV'].iloc[-4] if len(hist) >= 4 else hist['OBV'].iloc[0]
                if obv_now > obv_3m_ago:
                    signal = f"🏦 FALA 1 / AKUMULACJA (Stabilny dołek {range_position:.1f}%. Wieloryby ładują wora!)"
                    
            if not signal and curr_price < sma_10 and prev_price >= sma_10_prev and range_position < 60:
                signal = "🚨 BESSA ALERT (Utrata 10-miesięcznej średniej wsparcia)"
                
            if not signal and range_position >= 40 and range_position <= 80 and curr_price < sma_10 and curr_rsi < 55:
                signal = "🌊 FALA 4 (Makro korekta / Chłodzenie wskaźników przed rajdem)"
                
            if not signal and curr_price > sma_10 and prev_price <= sma_10_prev and curr_rsi > 50:
                signal = "🚀 FALA 3 WYBICIE (Przebicie 10-miesięcznej średniej w górę!)"

            if signal:
                return f"<b>{ticker}</b> | ${curr_price:.2f}\n└ <i>{signal}</i>"
            return None
        except Exception:
            return None

    def analyze_swing(self, ticker):
        """Skaner TRADE/SWING (6 miesięcy, interwał dzienny) - Szuka szybkich okazji na teraz"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo", interval="1d")
            
            if hist.empty or len(hist) < 25:
                return None
                
            close = hist['Close'].dropna()
            vol = hist['Volume'].dropna()
            
            curr_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            
            sma_20 = float(close.rolling(window=20).mean().iloc[-1])
            sma_20_prev = float(close.rolling(window=20).mean().iloc[-2])
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs)).iloc[-1])
            
            avg_vol_20 = vol.rolling(window=20).mean().iloc[-1]
            curr_vol = vol.iloc[-1]
            vol_spike = curr_vol > (avg_vol_20 * 1.5) # Wolumen o 50% większy niż średnia
            
            signal = None
            
            # 1. Agresywne Wybicie w górę z dużym wolumenem
            if curr_price > sma_20 and prev_price <= sma_20_prev and vol_spike:
                signal = "⚡ LONG: Wybicie w górę (Złamanie SMA 20 + Skok Wolumenu)"
                
            # 2. Kupowanie Krwi (Szybkie odbicie z wyprzedania)
            elif curr_price > prev_price and rsi < 35 and vol_spike:
                signal = "🩸 LONG: Odbicie od dna (RSI wyprzedane, wchodzi duży kapitał)"
                
            # 3. Short / Realizacja Zysków (Utrata wsparcia)
            elif curr_price < sma_20 and prev_price >= sma_20_prev and rsi > 50:
                signal = "🔪 SHORT / UWAGA: Cena przełamała linię obrony (SMA 20) w dół"

            if signal:
                return f"<b>{ticker}</b> | ${curr_price:.2f} | RSI: {rsi:.0f}\n└ <i>{signal}</i>"
            return None
        except Exception:
            return None

    def run_scan(self):
        """Uruchamia pełne skanowanie rynku i wysyła podzielony raport"""
        portfolio_tickers = self.get_portfolio_tickers()
        all_targets = list(set(self.vip_tickers + portfolio_tickers))
        
        macro_alerts = []
        swing_alerts = []
        
        for ticker in all_targets:
            # Skanujemy jedną spółkę dwoma radarami
            res_macro = self.analyze_macro(ticker)
            if res_macro: macro_alerts.append(res_macro)
                
            res_swing = self.analyze_swing(ticker)
            if res_swing: swing_alerts.append(res_swing)
        
        # Formatowanie wiadomości końcowej
        message = ""
        if macro_alerts or swing_alerts:
            message += "🤖 <b>LAMBO RADAR (Raport Automatyczny)</b> 📡\nPrzeskanowałem rynek. Oto co widzę:\n\n"
            
            if macro_alerts:
                message += "🛡️ <b>WIZJA MACRO (HODL na lata):</b>\n"
                message += "\n".join(macro_alerts) + "\n\n"
                
            if swing_alerts:
                message += "⚡ <b>SETUPY TRADE / SWING (Akcja na teraz):</b>\n"
                message += "\n".join(swing_alerts)
                
            self.send_telegram_alert(message)

    def send_telegram_alert(self, message):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.group_chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            response = requests.post(url, data=payload)
            if response.status_code != 200:
                print(f"Błąd Telegram API: {response.text}")
        except Exception as e:
            print(f"Błąd wysyłania na Telegram: {e}")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    
    TOKEN = os.getenv("TG_TOKEN")
    GROUP_ID = "-1003705938612" 
    
    if not TOKEN:
        print("BŁĄD KRYTYCZNY: Nie znaleziono tokena. Utwórz plik .env z wartością TG_TOKEN!")
    else:
        scanner = LamboGroupScanner(TOKEN, GROUP_ID)
        scanner.run_scan()
