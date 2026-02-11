import requests
import uuid
import streamlit.components.v1 as components
import streamlit as st
import matplotlib
matplotlib.use('Agg') # <--- KLUCZOWE: Musi byc przed pyplot i seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import csv
import json
import feedparser
import time
import pandas_datareader.data as web
from textblob import TextBlob
from datetime import datetime, timedelta
from math import pi

# --- KONFIGURACJA WERSJI ---
APP_VERSION = "1.0"  # Zmień na 1.1, 1.2 itd. jak dodasz coś nowego
# ---------------------------

# --- OBSŁUGA PROPHET ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Lambo czy Karton? (v46 Stable)", layout="wide", page_icon="🕸️")

# --- KLASA GŁÓWNA ---
class MarketProbabilityIndex:
    def __init__(self):
        self.filename = "market_log.csv"
        self.wallet_file = "wallet.json"
        self.lookback_period = "max"
        self.assets = {
            'sp500': '^GSPC', 'gold': 'GC=F', 'silver': 'SI=F',
            'copper': 'HG=F', 'bonds_10y': '^TNX', 'oil': 'CL=F',
            'vix': '^VIX', 'dxy': 'DX-Y.NYB',
            'btc': 'BTC-USD', 'eth': 'ETH-USD', 'sol': 'SOL-USD',
            'bnb': 'BNB-USD', 'xrp': 'XRP-USD', 'ada': 'ADA-USD', 'doge': 'DOGE-USD',
            'nasdaq': '^IXIC',
            'hyg': 'HYG',
            'ibit': 'IBIT', 
            'etha': 'ETHA'
        }
        # NOWOSC: SEKTORY GOSPODARKI (ETF)
        self.sectors = {
            'Technologia (XLK)': 'XLK',
            'Finanse (XLF)': 'XLF',
            'Zdrowie (XLV)': 'XLV',
            'Energia (XLE)': 'XLE',
            'Konsumpcyjne (XLY)': 'XLY', # Discretionary
            'Defensywne (XLP)': 'XLP',   # Staples
            'Przemysl (XLI)': 'XLI',
            'Komunikacja (XLC)': 'XLC',
            'Materialy (XLB)': 'XLB',
            'Nieruchomosci (XLRE)': 'XLRE',
            'Uzytecznosc (XLU)': 'XLU'
        }
        self.rss_feeds = {
            'crypto': 'https://cointelegraph.com/rss',
            'economy': 'https://finance.yahoo.com/news/rssindex'
        }
        
        # Init Session State
        if 'theme_mode' not in st.session_state:
            st.session_state['theme_mode'] = 'dark'
        
        # Init Lazy Load State
        if 'active_lazy_chart' not in st.session_state:
            st.session_state['active_lazy_chart'] = None

    def load_wallet(self):
        if os.path.exists(self.wallet_file):
            try:
                with open(self.wallet_file, 'r') as f:
                    return json.load(f)
            except:
                return {'btc': 0.0, 'eth': 0.0, 'sol': 0.0}
        return {'btc': 0.0, 'eth': 0.0, 'sol': 0.0}

    def save_wallet_callback(self):
        data = {
            'btc': st.session_state.get('user_btc_input', 0.0),
            'eth': st.session_state.get('user_eth_input', 0.0),
            'sol': st.session_state.get('user_sol_input', 0.0)
        }
        with open(self.wallet_file, 'w') as f:
            json.dump(data, f)

    def toggle_theme(self):
        if st.session_state['theme_mode'] == 'dark':
            st.session_state['theme_mode'] = 'light'
        else:
            st.session_state['theme_mode'] = 'dark'
            
    def get_theme_colors(self):
        if st.session_state['theme_mode'] == 'dark':
            return {
                'bg': '#0e1117',
                'sidebar_bg': '#262730', 
                'text': 'white',
                'grid': 'white',
                'grid_alpha': 0.1,
                'bull': '#00ff00',
                'bear': '#ff0000',
                'accent': 'cyan',
                'sns_style': 'darkgrid',
                'table_header_text': 'black',
                'input_bg': '#0e1117',
                'progress_bg': '#333333'
            }
        else:
            return {
                'bg': '#ffffff',
                'sidebar_bg': '#f0f2f6',
                'text': 'black',
                'grid': 'black',
                'grid_alpha': 0.1,
                'bull': '#008000',
                'bear': '#d60000',
                'accent': 'blue',
                'sns_style': 'whitegrid',
                'table_header_text': 'white',
                'input_bg': '#ffffff',
                'progress_bg': '#e0e0e0'
            }

    @st.cache_data(ttl=3600) 
    def get_market_data(_self):
        try:
            tickers = list(_self.assets.values()) # Tylko glowne aktywa
            df = yf.download(tickers, period="max", interval="1d", progress=False)['Close']
            vol_df = yf.download([_self.assets['ibit'], _self.assets['etha']], period="1y", interval="1d", progress=False)['Volume']
            
            if df.empty: return None, None
            
            inv_assets = {v: k for k, v in _self.assets.items()}
            df.rename(columns=inv_assets, inplace=True)
            df.ffill(inplace=True)
            
            if not vol_df.empty:
                mapping = {_self.assets['ibit']: 'ibit_vol', _self.assets['etha']: 'etha_vol'}
                vol_df.rename(columns=mapping, inplace=True)
            
            return df, vol_df
        except Exception as e:
            st.error(f"Blad danych: {e}")
            return None, None

    # --- HELPERS (Te brakowalo!) ---
    def normalize(self, v, h): return (h < v).mean()
    
    def calc_rsi(self, s, p=14):
        d = s.diff()
        g = (d.where(d>0,0)).rolling(p).mean()
        l = (-d.where(d<0,0)).rolling(p).mean()
        return 100 - (100/(1+(g/l)))

    def calculate_cycle_metrics(self, df):
        btc_series = df['btc'].dropna()
        sma_200w = btc_series.rolling(window=1400).mean() # ~200 weeks
        mvrv_proxy = btc_series / sma_200w
        copper = df['copper'].dropna(); gold = df['gold'].dropna()
        common_index = copper.index.intersection(gold.index)
        ism_raw = (copper.loc[common_index] / gold.loc[common_index]) * 1000
        ism_proxy = ism_raw.rolling(window=50).mean()
        return mvrv_proxy, ism_proxy, sma_200w

    # --- POPRAWKA: MINI CHART (SAFE MODE) ---
    def create_mini_chart(self, data, label, color_hex, fill=True):
        t = self.get_theme_colors()
        # FIX: Uzywamy plt.figure() zamiast subplots, zeby uniknac konfliktu watkow
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(111)
        
        ax.plot(data.index, data, color=color_hex, linewidth=1.5)
        if fill:
            ax.fill_between(data.index, data, alpha=0.2, color=color_hex)
        
        fig.patch.set_facecolor(t['bg'])
        ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text'], labelsize=8)
        ax.spines['bottom'].set_color(t['text'])
        ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(label, color=t['text'], fontsize=10)
        ax.grid(False)
        
        return fig 

    def plot_sector_momentum(self, df):
        t = self.get_theme_colors()
        period = 30
        if len(df) < period: return None
        
        sector_perf = {}
        for name, ticker in self.sectors.items():
            if name in df.columns:
                start = df[name].iloc[-period]
                end = df[name].iloc[-1]
                perf = (end - start) / start * 100
                sector_perf[name] = perf
        
        sorted_perf = dict(sorted(sector_perf.items(), key=lambda item: item[1], reverse=True))
        
        fig = plt.figure(figsize=(10, 6)) # FIX: Safe Mode
        ax = fig.add_subplot(111)
        
        names = list(sorted_perf.keys())
        values = list(sorted_perf.values())
        colors = [t['bull'] if v > 0 else t['bear'] for v in values]
        
        bars = ax.barh(names, values, color=colors, alpha=0.8)
        ax.set_title("ROTACJA KAPITALU (Sektory USA - 30 Dni)", fontsize=16, color=t['text'])
        ax.set_xlabel('Zmiana (%)', color=t['text'])
        ax.axvline(0, color=t['text'], linestyle='-', linewidth=1)
        
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.5 if width > 0 else width - 2.5
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:+.1f}%', va='center', color=t['text'], fontsize=9, fontweight='bold')

        fig.patch.set_facecolor(t['bg'])
        ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text'], axis='y')
        ax.spines['bottom'].set_color(t['text'])
        ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis() 
        
        return fig

    # --- BRAKUJĄCA METODA: DANE SEKTOROWE ---
    def get_lazy_sector_data(self):
        """Pobiera dane sektorowe na zadanie (Lazy Load)."""
        try:
            tickers = list(self.sectors.values())
            # Pobieramy tylko ostatni rok
            df = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
            
            if df.empty: return None
            
            inv_sectors = {v: k for k, v in self.sectors.items()}
            df.rename(columns=inv_sectors, inplace=True)
            df.ffill(inplace=True)
            return df
        except: return None

    # --- METODA 1: GLOBAL LIQUIDITY ---
    def get_lazy_liquidity_data(self):
        proxies = {'USA': 'SPY', 'Europe': 'VGK', 'China': 'MCHI', 'Japan': 'EWJ', 'BTC': 'BTC-USD'}
        try:
            df = yf.download(list(proxies.values()), period="2y", interval="1d", progress=False)['Close']
        except: return None
        if df.empty: return None
        df.ffill(inplace=True); df.dropna(inplace=True)
        if len(df) > 0:
            df_norm = df / df.iloc[0] * 100
            market_cols = [c for c in df_norm.columns if c != 'BTC-USD']
            if market_cols: df_norm['Global_Liquidity_Index'] = df_norm[market_cols].mean(axis=1)
            else: df_norm['Global_Liquidity_Index'] = 0
            return df_norm
        return None

    def plot_global_liquidity_chart(self, df):
        if df is None or df.empty or 'Global_Liquidity_Index' not in df.columns: return None
        t = self.get_theme_colors()
        
        fig = plt.figure(figsize=(10, 6)) # FIX: Safe Mode
        ax1 = fig.add_subplot(111)
        
        ax1.plot(df.index, df['Global_Liquidity_Index'], color='#00e5ff', linewidth=2.5, label='Global Liquidity (Proxy)')
        y_min = df['Global_Liquidity_Index'].min()
        y_max = df['Global_Liquidity_Index'].max()
        ax1.fill_between(df.index, df['Global_Liquidity_Index'], y_min - (y_max-y_min)*0.1, color='#00e5ff', alpha=0.15)
        
        ax1.set_ylabel('Global Liquidity Index', color='#00e5ff', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#00e5ff', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['BTC-USD'], color='orange', linestyle='--', linewidth=1.5, label='Bitcoin', alpha=0.8)
        ax2.set_ylabel('Bitcoin ($)', color='orange', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='orange', colors=t['text'])
        
        ax1.set_title("GLOBAL LIQUIDITY vs BITCOIN", fontsize=16, color=t['text'], fontweight='bold')
        ax1.grid(True, alpha=0.15, color=t['grid'])
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
        return fig

    # --- CREDIT IMPULSE ---
    def get_lazy_credit_impulse(self):
        tickers = ['HYG', 'IEF', 'KBE', 'EUFN', 'FXI', 'BTC-USD']
        try: df = yf.download(tickers, period="3y", interval="1d", progress=False)['Close']
        except: return None
        if df.empty: return None
        df.ffill(inplace=True); df.dropna(inplace=True)
        df['US_Spread_Proxy'] = df['HYG'] / df['IEF']
        norm = pd.DataFrame()
        norm['US_Credit'] = df['US_Spread_Proxy'] / df['US_Spread_Proxy'].iloc[0] * 100
        norm['US_Banks'] = df['KBE'] / df['KBE'].iloc[0] * 100
        norm['EU_Banks'] = df['EUFN'] / df['EUFN'].iloc[0] * 100
        norm['CN_Stimulus'] = df['FXI'] / df['FXI'].iloc[0] * 100
        norm['Global_Credit_Impulse'] = ((norm['US_Credit'] * 0.4) + (norm['US_Banks'] * 0.2) + (norm['EU_Banks'] * 0.2) + (norm['CN_Stimulus'] * 0.2))
        norm['BTC'] = df['BTC-USD']
        return norm

    def plot_credit_impulse_chart(self, df):
        if df is None or df.empty: return None
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        
        ax1.plot(df.index, df['Global_Credit_Impulse'], color='#bfff00', linewidth=2.5, label='Global Credit Impulse (Proxy)')
        y_min = df['Global_Credit_Impulse'].min()
        ax1.fill_between(df.index, df['Global_Credit_Impulse'], y_min, color='#bfff00', alpha=0.1)
        ax1.set_ylabel('Latwosc Kredytowa (Index)', color='#bfff00', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#bfff00', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['BTC'], color='white', linestyle='--', linewidth=1.5, label='Bitcoin', alpha=0.9)
        ax2.set_ylabel('Bitcoin ($)', color='white', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='white', colors=t['text'])
        
        ax1.set_title("GLOBALNA AKCJA KREDYTOWA vs BTC", fontsize=16, color=t['text'], fontweight='bold')
        ax1.grid(True, alpha=0.15, color=t['grid'])
        
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
        return fig

    # --- ALL COMPONENTS ---
    def get_lazy_all_components_data(self):
        tickers = ['HYG', 'IEF', 'KBE', 'EUFN', 'FXI', 'BTC-USD']
        try: df = yf.download(tickers, period="2y", interval="1d", progress=False)['Close']
        except: return None
        if df.empty: return None
        df.ffill(inplace=True); df.dropna(inplace=True)
        if len(df) > 0:
            df['Credit_Spread_Ratio'] = df['HYG'] / df['IEF']
            plot_data = pd.DataFrame()
            plot_data['Risk_Ratio (HYG/IEF)'] = df['Credit_Spread_Ratio']
            plot_data['USA Banks (KBE)'] = df['KBE']
            plot_data['EU Banks (EUFN)'] = df['EUFN']
            plot_data['China (FXI)'] = df['FXI']
            if 'BTC-USD' in df.columns: plot_data['Bitcoin'] = df['BTC-USD']
            df_norm = plot_data / plot_data.iloc[0] * 100
            return df_norm
        return None

    def plot_all_components_chart(self, df):
        if df is None or df.empty: return None
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        lines_config = {
            'Bitcoin': {'color': 'orange', 'width': 2.5, 'style': '--'},
            'Risk_Ratio (HYG/IEF)': {'color': '#00ff00', 'width': 2, 'style': '-'},
            'USA Banks (KBE)': {'color': '#00d4ff', 'width': 1.5, 'style': '-'},
            'EU Banks (EUFN)': {'color': '#0055ff', 'width': 1.5, 'style': '-'},
            'China (FXI)': {'color': '#ff0055', 'width': 1.5, 'style': '-'}
        }
        for col in df.columns:
            cfg = lines_config.get(col, {'color': 'gray', 'width': 1, 'style': '-'})
            ax.plot(df.index, df[col], color=cfg['color'], linewidth=cfg['width'], linestyle=cfg['style'], label=col)
            last_val = df[col].iloc[-1]
            ax.text(df.index[-1], last_val, f" {last_val:.0f}%", color=cfg['color'], fontsize=9, va='center', fontweight='bold')

        ax.set_ylabel('Zmiana Wartosci (Start = 100%)', color=t['text'])
        ax.tick_params(axis='y', labelcolor=t['text'], colors=t['text']); ax.tick_params(axis='x', colors=t['text'])
        ax.axhline(100, color=t['text'], linestyle=':', alpha=0.3)
        ax.set_title("WYSGIG AKTYWOW (Risk-On vs Risk-Off)", fontsize=16, color=t['text'], fontweight='bold')
        ax.grid(True, alpha=0.15, color=t['grid'])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        return fig

    # --- BANK STIMULUS ---
    def get_lazy_bank_stimulus_data(self):
        tickers = ['KBE', 'SHY', 'BTC-USD']
        try: df = yf.download(tickers, period="2y", interval="1d", progress=False)['Close']
        except: return None
        if df.empty: return None
        df.ffill(inplace=True); df.dropna(inplace=True)
        if len(df) > 0:
            df_norm = df / df.iloc[0] * 100
            df_norm['Stimulus_Probability'] = df_norm['SHY'] / df_norm['KBE']
            df_norm['Stimulus_Probability'] = df_norm['Stimulus_Probability'] / df_norm['Stimulus_Probability'].iloc[0] * 100
            if 'BTC-USD' in df_norm.columns: df_norm['BTC'] = df_norm['BTC-USD']
            return df_norm
        return None

    def plot_bank_stimulus_chart(self, df):
        if df is None or df.empty: return None
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        
        ax1.plot(df.index, df['KBE'], color='#ff4b4b', linewidth=2, label='Kondycja Bankow (KBE)')
        ax1.set_ylabel('Kondycja Bankow (Start=100)', color='#ff4b4b', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#ff4b4b', colors=t['text']); ax1.tick_params(axis='x', colors=t['text'])
        
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['Stimulus_Probability'], color='#00e5ff', linestyle='-.', linewidth=2, label='Prawdopodobienstwo Ratunku')
        ax2.set_ylabel('Szansa na Stymulus', color='#00e5ff', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#00e5ff', colors=t['text'])
        ax2.fill_between(df.index, df['Stimulus_Probability'], df['Stimulus_Probability'].min(), color='#00e5ff', alpha=0.1)

        ax3 = ax1.twinx(); ax3.spines['right'].set_position(('outward', 60)) 
        ax3.plot(df.index, df['BTC'], color='orange', linestyle='--', linewidth=1.5, label='Bitcoin (Reakcja)')
        ax3.set_ylabel('Bitcoin (Start=100)', color='orange', fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='orange', colors=t['text'])
        ax3.spines['top'].set_visible(False); ax3.spines['left'].set_visible(False); ax3.spines['bottom'].set_visible(False)

        ax1.set_title("BANKI vs STYMULUS vs BITCOIN", fontsize=16, color=t['text'], fontweight='bold')
        ax1.grid(True, alpha=0.15, color=t['grid'])
        
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
        fig.subplots_adjust(right=0.85)
        return fig

    # --- ORACLE ---
    @st.cache_data(ttl=3600)
    def get_oracle_prediction(_self, df, days=30):
        if not PROPHET_AVAILABLE: return None
        data = df[['btc']].copy()
        mvrv, ism, _ = _self.calculate_cycle_metrics(df)
        data['mvrv'] = mvrv; data['ism'] = ism
        data = data.reset_index(); date_col = data.columns[0]
        data.rename(columns={date_col: 'ds', 'btc': 'y'}, inplace=True)
        data['ds'] = data['ds'].dt.tz_localize(None); data.dropna(inplace=True)
        train_data = data.tail(1460)
        if len(train_data) < 365: return None 
        m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.5)
        m.add_regressor('mvrv'); m.add_regressor('ism')
        m.fit(train_data)
        future = m.make_future_dataframe(periods=days)
        last_mvrv = train_data['mvrv'].iloc[-1]; last_ism = train_data['ism'].iloc[-1]
        future = pd.merge(future, data[['ds', 'mvrv', 'ism']], on='ds', how='left')
        future['mvrv'] = future['mvrv'].fillna(last_mvrv); future['ism'] = future['ism'].fillna(last_ism)
        return m.predict(future)

    def plot_oracle_forecast(self, forecast):
        if forecast is None: return None
        t = self.get_theme_colors() 
        viz_data = forecast.tail(90)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        today_date = viz_data['ds'].iloc[-31] 
        ax.plot(viz_data['ds'], viz_data['yhat'], color='#9d00ff', linewidth=3, label='Wyrocznia (Fundamenty)', linestyle='-')
        ax.fill_between(viz_data['ds'], viz_data['yhat_lower'], viz_data['yhat_upper'], color='#9d00ff', alpha=0.15)
        ax.axvline(x=today_date, color=t['text'], linestyle=':', label='TERAZ')
        ax.set_title(f"WYROCZNIA AI (Logika: MVRV + ISM)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_ylabel('Cena BTC ($)', color=t['text'])
        ax.legend(facecolor=t['bg'], labelcolor=t['text'], loc='upper left')
        ax.grid(True, alpha=0.15, color=t['grid'])
        ax.text(viz_data['ds'].iloc[0], viz_data['yhat'].min(), "Model ignoruje pory roku.\nBazuje na: Czy jest tanio (MVRV) i czy gospodarka rośnie (ISM)?", color=t['text'], fontsize=8, alpha=0.7, va='bottom')
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        return fig

    # --- ZEUS ---
    @st.cache_data(ttl=3600)
    def get_zeus_prediction(_self, df, days=1200):
        target_cycle_years = 4.95; standard_cycle_years = 4.0
        stretch_factor = target_cycle_years / standard_cycle_years
        prev_bottom = pd.Timestamp("2018-12-15"); curr_bottom = pd.Timestamp("2022-11-21") 
        hist_data = df.loc[prev_bottom:"2022-11-01", 'btc'].copy()
        if hist_data.empty: return None
        fractal_data = []
        base_price_current = df.loc[curr_bottom:, 'btc'].iloc[0] if curr_bottom in df.index else df['btc'].iloc[-1]
        base_price_history = hist_data.iloc[0]
        for date, price in hist_data.items():
            delta_days = (date - prev_bottom).days
            new_delta = delta_days * stretch_factor
            new_date = curr_bottom + pd.Timedelta(days=new_delta)
            roi = price / base_price_history
            projected_price = base_price_current * roi
            fractal_data.append({'ds': new_date, 'yhat': projected_price})
        fractal_df = pd.DataFrame(fractal_data); fractal_df = fractal_df.set_index('ds').sort_index()
        today = pd.Timestamp.now().normalize()
        if fractal_df.empty: return None
        try:
            idx = fractal_df.index.get_indexer([today], method='nearest')[0]
            price_fractal_today = fractal_df.iloc[idx]['yhat']
        except: return None
        price_real_today = df['btc'].iloc[-1]
        correction_ratio = price_real_today / price_fractal_today
        fractal_df['yhat'] = fractal_df['yhat'] * correction_ratio
        viz_start = today - pd.Timedelta(days=800); viz_end = today + pd.Timedelta(days=days)
        forecast = fractal_df.loc[viz_start:viz_end].reset_index()
        forecast['yhat_lower'] = forecast['yhat'] * 0.90; forecast['yhat_upper'] = forecast['yhat'] * 1.10
        forecast['stretch_factor'] = stretch_factor
        mvrv, ism, _ = _self.calculate_cycle_metrics(df)
        forecast['mvrv'] = mvrv.iloc[-1]; forecast['ism'] = ism.iloc[-1]
        return forecast

    def plot_zeus_forecast(self, forecast, df):
        if forecast is None or forecast.empty: return None
        t = self.get_theme_colors() 
        stretch = forecast['stretch_factor'].iloc[0]
        curr_bottom = pd.Timestamp("2022-11-21")
        real_data = df.loc[curr_bottom:].copy()
        peak_idx = forecast['yhat'].idxmax(); peak_date = forecast.loc[peak_idx, 'ds']
        peak_price = forecast.loc[peak_idx, 'yhat']; peak_str = peak_date.strftime('%Y-%m-%d')
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        today_date = pd.Timestamp.now().normalize()
        
        ax.plot(forecast['ds'], forecast['yhat'], color='#ffd700', linewidth=1, alpha=0.3, linestyle='--')
        ax.plot(real_data.index, real_data['btc'], color='#ffffff', linewidth=3, zorder=5)
        future_forecast = forecast[forecast['ds'] >= today_date]
        ax.plot(future_forecast['ds'], future_forecast['yhat'], color='#ffd700', linewidth=3, zorder=4)
        ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='#ffd700', alpha=0.1)
        ax.axvline(x=today_date, color=t['text'], linestyle=':')
        ax.scatter([peak_date], [peak_price], color='red', s=120, zorder=6, edgecolors='white')
        ax.annotate(f"CEL (4.95 Lat)\n{peak_str}\n${peak_price:,.0f}", xy=(peak_date, peak_price), xytext=(peak_date, peak_price * 1.15), arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red', fontweight='bold', ha='center', bbox=dict(facecolor=t['bg'], alpha=0.8, edgecolor='red'))
        title = f"⚡ ZEUS: Rzeczywistość vs Model (4.95 Lat)"
        subtitle = f"Biała: Realny Rynek | Żółta: Projekcja Przyszłości\nWspółczynnik Czasu: x{stretch:.4f}"
        ax.set_title(title, fontsize=16, color=t['text'], fontweight='bold', loc='left')
        ax.text(forecast['ds'].iloc[0], forecast['yhat'].max(), subtitle, fontsize=9, color=t['text'], bbox=dict(facecolor=t['bg'], alpha=0.7))
        ax.grid(True, alpha=0.15, color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        return fig

    # --- CRONOS ---
    @st.cache_data(ttl=3600)
    def get_cronos_prediction(_self, df, days=1200):
        target_cycle_years = 4.8; standard_cycle_years = 4.0; stretch_factor = target_cycle_years / standard_cycle_years
        prev_bottom = pd.Timestamp("2015-01-14"); curr_bottom = pd.Timestamp("2022-11-21")
        hist_data = df.loc[prev_bottom:"2018-01-30", 'btc'].copy()
        if hist_data.empty: return None
        fractal_data = []
        base_price_current = df.loc[curr_bottom:, 'btc'].iloc[0] if curr_bottom in df.index else df['btc'].iloc[-1]
        base_price_history = hist_data.iloc[0]
        for date, price in hist_data.items():
            delta_days = (date - prev_bottom).days
            new_delta = delta_days * stretch_factor
            new_date = curr_bottom + pd.Timedelta(days=new_delta)
            roi = price / base_price_history
            projected_price = base_price_current * roi
            fractal_data.append({'ds': new_date, 'yhat': projected_price})
        fractal_df = pd.DataFrame(fractal_data); fractal_df = fractal_df.set_index('ds').sort_index()
        today = pd.Timestamp.now().normalize()
        if fractal_df.empty: return None
        try:
            idx = fractal_df.index.get_indexer([today], method='nearest')[0]
            price_fractal_today = fractal_df.iloc[idx]['yhat']
        except: return None
        price_real_today = df['btc'].iloc[-1]
        correction_ratio = price_real_today / price_fractal_today
        fractal_df['yhat'] = fractal_df['yhat'] * correction_ratio
        viz_start = today - pd.Timedelta(days=1300); viz_end = today + pd.Timedelta(days=days)
        forecast = fractal_df.loc[viz_start:viz_end].reset_index()
        forecast['yhat_lower'] = forecast['yhat'] * 0.80; forecast['yhat_upper'] = forecast['yhat'] * 1.20
        forecast['stretch_factor'] = stretch_factor
        mvrv, ism, _ = _self.calculate_cycle_metrics(df)
        forecast['mvrv'] = mvrv.iloc[-1]; forecast['ism'] = ism.iloc[-1]
        return forecast

    def plot_cronos_forecast(self, forecast, df):
        if forecast is None or forecast.empty: return None
        t = self.get_theme_colors() 
        stretch = forecast['stretch_factor'].iloc[0]
        curr_bottom = pd.Timestamp("2022-11-21")
        real_data = df.loc[curr_bottom:].copy()
        peak_idx = forecast['yhat'].idxmax(); peak_date = forecast.loc[peak_idx, 'ds']
        peak_price = forecast.loc[peak_idx, 'yhat']; peak_str = peak_date.strftime('%Y-%m-%d')
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        today_date = pd.Timestamp.now().normalize()
        
        ax.plot(forecast['ds'], forecast['yhat'], color='#00ffff', linewidth=1, alpha=0.3, linestyle='--')
        ax.plot(real_data.index, real_data['btc'], color='#ffffff', linewidth=3, zorder=5)
        future_forecast = forecast[forecast['ds'] >= today_date]
        ax.plot(future_forecast['ds'], future_forecast['yhat'], color='#00ffff', linewidth=3, zorder=4)
        ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='#00ffff', alpha=0.1)
        ax.axvline(x=today_date, color=t['text'], linestyle=':')
        ax.scatter([peak_date], [peak_price], color='white', s=120, zorder=6, edgecolors='#00ffff', linewidth=2)
        ax.annotate(f"CEL (4.8 Lat)\n{peak_str}\n${peak_price:,.0f}", xy=(peak_date, peak_price), xytext=(peak_date, peak_price * 1.15), arrowprops=dict(facecolor='#00ffff', shrink=0.05), fontsize=10, color='#00ffff', fontweight='bold', ha='center', bbox=dict(facecolor=t['bg'], alpha=0.8, edgecolor='#00ffff'))
        title = f"⏳ CRONOS: Rzeczywistość vs Model (4.8 Lat)"
        subtitle = f"Biała: Realny Rynek | Błękitna: Projekcja Przyszłości\nWspółczynnik Czasu: x{stretch:.2f}"
        ax.set_title(title, fontsize=16, color=t['text'], fontweight='bold', loc='left')
        ax.text(forecast['ds'].iloc[0], forecast['yhat'].max(), subtitle, fontsize=9, color=t['text'], bbox=dict(facecolor=t['bg'], alpha=0.7))
        ax.grid(True, alpha=0.15, color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        return fig

    # --- CYKLICZNE PORÓWNANIE ---
    @st.cache_data(ttl=3600)
    def get_cycle_comparison_data(_self, df):
        cycles = {'2011 (Genesis)': pd.Timestamp("2011-11-18"), '2015 (Legendary)': pd.Timestamp("2015-01-14"), '2019 (Last)': pd.Timestamp("2018-12-15"), '2023 (Current)': pd.Timestamp("2022-11-21")}
        comparison_data = {}
        genesis_points = [("2011-11-18", 2.05), ("2012-01-01", 5.2), ("2012-06-01", 6.7), ("2012-08-15", 13.5), ("2012-11-01", 10.8), ("2013-01-01", 13.3), ("2013-02-01", 20.4), ("2013-03-01", 33.0), ("2013-04-09", 230.0), ("2013-04-16", 68.0), ("2013-07-05", 65.0), ("2013-10-01", 130.0), ("2013-11-01", 210.0), ("2013-11-30", 1130.0), ("2013-12-18", 520.0), ("2014-01-05", 950.0), ("2014-04-10", 360.0), ("2014-06-01", 660.0), ("2014-11-01", 320.0), ("2015-01-14", 170.0)]
        gen_df = pd.DataFrame(genesis_points, columns=['date', 'price']); gen_df['date'] = pd.to_datetime(gen_df['date']); gen_df = gen_df.set_index('date')
        full_idx = pd.date_range(start=gen_df.index.min(), end=gen_df.index.max(), freq='D'); gen_daily = gen_df.reindex(full_idx)
        gen_daily['price'] = gen_daily['price'].interpolate(method='pchip'); noise = np.random.normal(0, 0.02, size=len(gen_daily)); gen_daily['price'] = gen_daily['price'] * (1 + noise)
        days_passed = range(len(gen_daily)); base_price = gen_daily['price'].iloc[0]; roi = gen_daily['price'].values / base_price
        comparison_data['2011 (Genesis)'] = pd.DataFrame({'days': days_passed, 'roi': roi, 'price': gen_daily['price'].values})
        for name, start_date in cycles.items():
            if name == '2011 (Genesis)': continue 
            if start_date < df.index[0]: continue 
            cycle_slice = df.loc[start_date:].iloc[:1460]['btc'].copy()
            if cycle_slice.empty: continue
            days_passed = range(len(cycle_slice)); base_price = cycle_slice.iloc[0]; roi = cycle_slice.values / base_price
            comparison_data[name] = pd.DataFrame({'days': days_passed, 'roi': roi, 'price': cycle_slice.values})
        return comparison_data

    def plot_cycle_comparison(self, comparison_data):
        if not comparison_data: return None
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        colors = {'2011 (Genesis)': '#8a2be2', '2015 (Legendary)': '#00ffff', '2019 (Last)': '#ffd700', '2023 (Current)': '#ffffff'}
        for name, data in comparison_data.items():
            color = colors.get(name, 'gray'); linewidth = 3 if 'Current' in name else 1.5; alpha = 1.0 if 'Current' in name else 0.6
            ax.plot(data['days'], data['roi'], label=name, color=color, linewidth=linewidth, alpha=alpha)
            if 'Current' in name:
                last_day = data['days'].iloc[-1]; last_roi = data['roi'].iloc[-1]; current_price = data['price'].iloc[-1]
                ax.scatter([last_day], [last_roi], color='red', s=80, zorder=10, edgecolors='white')
                ax.annotate(f"DZIŚ (Dzień {last_day})\n${current_price:,.0f} ({last_roi:.1f}x)", xy=(last_day, last_roi), xytext=(last_day + 50, last_roi), color='white', fontsize=9, fontweight='bold', arrowprops=dict(arrowstyle="->", color='white'))
        ax.set_yscale('log')
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter()); ax.set_yticks([1, 2, 5, 10, 20, 50, 100]); ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_title("Wojna Cykli: Porównanie ROI (Od Dna)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Dni od Dna Cyklu', color=t['text']); ax.set_ylabel('ROI (Mnożnik Ceny) - Skala Log', color=t['text'])
        ax.legend(facecolor=t['bg'], labelcolor=t['text']); ax.grid(True, alpha=0.15, color=t['grid'], which='both')
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        return fig

    # --- SENTIMENT: LONG vs SHORT (WERSJA MACRO-AWARE - NAJBARDZIEJ REALNA) ---
    @st.cache_data(ttl=3600)
    def get_sentiment_structure(_self, df):
        """
        Wersja 'Macro-Aware':
        Symuluje decyzje na podstawie Ceny, Wolumenu ORAZ Ryzyka Globalnego (VIX).
        To najlepsze przybliżenie rzeczywistości na darmowych danych.
        """
        # Sprawdzamy czy mamy potrzebne kolumny
        required = ['btc', 'vix']
        if not all(col in df.columns for col in required) or len(df) < 200: 
            return pd.DataFrame()
        
        # --- DANE ---
        btc = df['btc']
        vix = df['vix']
        
        # Ostatnie wartości
        last_price = btc.iloc[-1]
        last_vix = vix.iloc[-1]
        
        # Wskaźniki BTC
        sma200 = btc.rolling(200).mean().iloc[-1]
        dist_sma = last_price / sma200 
        
        mom_30d = last_price / btc.iloc[-30]
        
        # Wolumen (jeśli dostępny, używamy jako mnożnika siły)
        # Jeśli brak wolumenu w danych, przyjmujemy neutralny mnożnik 1.0
        if 'volume' in df.columns: # Uwaga: yfinance często daje volume w osobnej ramce, tu zakładamy uproszczenie
             # W tej klasie df ma tylko 'Close', więc pomijamy wolumen w tej konkretnej metodzie
             # aby nie komplikować kodu błędami indeksowania. 
             # Skupimy się na VIX jako głównym ulepszeniu.
             pass

        # RSI (Emocje)
        delta = btc.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        sentiment_data = []

        # --- A. ASSET MANAGERS (Instytucje) ---
        # Logika: "Smart Money" patrzy na Trend (SMA) i Ryzyko (VIX).
        # BAZA: Od 85% do 99% (bo są structural long).
        base_long = np.interp(dist_sma, [0.8, 1.0, 1.4], [85, 92, 99])
        
        # KOREKTA VIX (To jest ten "Realizm"):
        # Jeśli VIX jest wysoki (strach na rynku), instytucje redukują ryzyko (Long spada).
        # VIX 12 (Spokój) -> +0% (Bez zmian)
        # VIX 30 (Panika) -> -15% (Redukcja ekspozycji)
        vix_penalty = np.interp(last_vix, [12, 30], [0, 15])
        
        am_long = np.clip(base_long - vix_penalty, 60, 100) # Nie pozwalamy spaść poniżej 60%
        sentiment_data.append({'Group': 'Asset Mgr', 'Long': am_long, 'Short': 100 - am_long})

        # --- B. LEVERAGED FUNDS (Spekulanci) ---
        # Logika: Agresywni. Spadek ceny o 5% to dla nich sygnał do ataku (Short).
        # Momentum < 0.95 (-5%) -> Long 20% (Short 80%)
        lf_long = np.interp(mom_30d, [0.95, 1.00, 1.10], [20, 50, 85])
        sentiment_data.append({'Group': 'Lev Funds', 'Long': lf_long, 'Short': 100 - lf_long})

        # --- C. DEALER (Market Makers) ---
        sentiment_data.append({'Group': 'Dealer', 'Long': 52, 'Short': 48})

        # --- D. OTHER REPORTABLES ---
        # Mniejsi gracze, reagują na RSI i VIX.
        other_base = np.interp(rsi, [30, 50, 70], [45, 60, 75])
        # Jak jest panika na VIX, też uciekają
        other_long = other_base - (vix_penalty * 0.5) 
        sentiment_data.append({'Group': 'Other', 'Long': other_long, 'Short': 100 - other_long})

        # --- E. RETAIL (Ulica) ---
        # Czyste emocje (RSI).
        ret_long = np.interp(rsi, [20, 50, 80], [25, 50, 80])
        sentiment_data.append({'Group': 'Retail', 'Long': ret_long, 'Short': 100 - ret_long})
        
        return pd.DataFrame(sentiment_data)

    # --- WIZUALIZACJA: LONG/SHORT BARS (SAFE MODE) ---
    def plot_sentiment_breakdown(self, sent_df):
        if sent_df is None or sent_df.empty: return None
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 5)) # FIX
        ax = fig.add_subplot(111)
        
        groups = sent_df['Group']; longs = sent_df['Long']; shorts = sent_df['Short']
        y_pos = range(len(groups)); bar_height = 0.6
        ax.barh(y_pos, longs, height=bar_height, color='#00C853', label='LONG', edgecolor=t['bg'])
        ax.barh(y_pos, shorts, height=bar_height, left=longs, color='#FF3D00', label='SHORT', edgecolor=t['bg'])
        for i, (l, s) in enumerate(zip(longs, shorts)):
            ax.text(l / 2, i, f"{l:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
            ax.text(l + (s / 2), i, f"{s:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax.set_yticks(y_pos); ax.set_yticklabels(groups, fontsize=11, fontweight='bold', color=t['text']); ax.invert_yaxis() 
        ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color=t['text'])
        ax.set_title("Sentyment Rynku: Kto gra na co?", fontsize=16, color=t['text'], fontweight='bold', loc='left', pad=25)
        ax.text(0, 1.02, "Szacowana ekspozycja (Model 6M)", transform=ax.transAxes, fontsize=10, color=t['text'], alpha=0.7)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, labelcolor=t['text'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', colors=t['text'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        return fig

    # --- WIZUALIZACJA: DONUT SNAPSHOT (WERSJA "CRYPTO-NATIVE" - WIĘCEJ LEWARÓW) ---
    def plot_sentiment_donut_snapshot(self, df):
        """
        Generuje strukturę rynku.
        ZMIANA: Zwiększony udział Leveraged Funds, aby oddać spekulacyjny charakter krypto.
        """
        if df is None or 'btc' not in df.columns: return None
        
        # 1. DANE
        price = df['btc']
        if len(price) > 30:
            volatility = price.pct_change().rolling(30).std().iloc[-1] * 100
            sma200 = price.rolling(200).mean().iloc[-1]
            last_price = price.iloc[-1]
            prev_price = price.iloc[-30]
        else: return None

        # 2. NOWE WAGI BAZOWE (Więcej spekulacji)
        # Asset Mgr (HODL): 30% (Było 35%)
        # Lev Funds (Spekula): 25% (Było 20%) -> Startujemy z wyższego pułapu
        weights = {'Asset Mgr': 30, 'Lev Funds': 25, 'Dealer': 15, 'Other': 20, 'Retail': 10}
        
        # 3. MODYFIKATORY
        
        # Hossa (Cena > SMA200) -> Kapitał płynie do ETFów (Asset Mgr)
        if last_price > sma200:
            weights['Asset Mgr'] += 10
            weights['Retail'] += 5
        else:
            # Bessa -> Instytucje uciekają, zostają spekulanci grający shorty
            weights['Lev Funds'] += 10 
            
        # Zmienność (To jest klucz do Twojego pytania)
        # Jeśli rynek szaleje (>2.5% zmienności), Lewary przejmują kontrolę
        if volatility > 2.5:
            weights['Lev Funds'] += 20 # DUŻY BOOST DLA LEWARÓW
            weights['Asset Mgr'] -= 10 # Instytucje nie lubią ryzyka
            weights['Retail'] += 5     # Ulica lubi hazard
        else:
            # Cisza na rynku -> Lewary nudzą się i wychodzą
            weights['Asset Mgr'] += 5
            
        # FOMO
        if (last_price / prev_price) - 1 > 0.20:
            weights['Retail'] += 15
            weights['Lev Funds'] += 5

        # Normalizacja do 100%
        total_weight = sum(weights.values())
        final_sizes = [(v / total_weight) * 100 for v in weights.values()]
        labels = list(weights.keys())
        
        # Szacunek OI (Bez zmian)
        avg_vol = df.filter(like='volume').iloc[-1].mean() if 'volume' in df.columns else 0
        if avg_vol == 0: estimated_oi = 240000 
        else: estimated_oi = int(avg_vol / last_price * 18) # Zwiększony mnożnik (18x) bo więcej lewarów
        if estimated_oi < 1000 or estimated_oi > 20000000: estimated_oi = 245000

        # RYSOWANIE
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        colors = ['#ff5722', '#00bcd4', '#ffeb3b', '#9c27b0', '#4caf50']
        
        wedges, texts, autotexts = ax.pie(
            final_sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            startangle=90, pctdistance=0.8,
            textprops={'color': t['text'], 'fontsize': 9, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': t['bg'], 'linewidth': 2, 'antialiased': True}
        )
        for text in texts: text.set_visible(False)
        centre_circle = plt.Circle((0,0), 0.60, fc=t['bg']); ax.add_artist(centre_circle)
        
        ax.text(0, 0.1, "SZACUNKOWE OI", ha='center', fontsize=8, color=t['text'], alpha=0.7)
        ax.text(0, -0.1, f"{estimated_oi:,.0f} BTC".replace(",", " "), ha='center', fontsize=14, color=t['text'], fontweight='bold')
        ax.text(-1.3, 1.2, "Migawka Rynku", fontsize=16, color=t['text'], fontweight='bold')
        ax.text(-1.3, 1.05, "Struktura kapitału (Crypto-Native)", fontsize=9, color=t['text'], alpha=0.7)
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, labelcolor=t['text'])
        
        ax.axis('equal')
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        return fig

    # --- HELPERS ---
    def get_crypto_fear_greed(self):
        try:
            response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=3)
            data = response.json()
            return int(data['data'][0]['value'])
        except: return 50

    # --- ORACLE (AI MODEL) - Z CACHOWANIEM ---
    @st.cache_data(ttl=3600, show_spinner=False) # <--- TO JEST KLUCZ DO SZYBKOŚCI
    def get_ai_prediction(_self, df, days=30):
        """
        Liczy prognozę AI. Dzięki @st.cache_data robi to tylko RAZ na godzinę.
        """
        if not PROPHET_AVAILABLE: return None
        
        # Przygotowanie danych
        data = df[['btc']].reset_index()
        data.columns = ['ds', 'y']
        
        # Usuwamy strefę czasową (wymagane przez Prophet)
        data['ds'] = data['ds'].dt.tz_localize(None)
        
        # Bierzemy ostatnie 2 lata do nauki (szybciej niż cała historia)
        train_data = data.tail(730) 
        
        # Konfiguracja modelu
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        
        # Trenowanie (To jest ten moment "cmdstanpy", który trwa długo)
        m.fit(train_data)
        
        # Prognoza
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        
        return forecast

    def plot_ai_forecast(self, forecast):
        if forecast is None: return None
        t = self.get_theme_colors() 
        viz_data = forecast.tail(120); fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(111)
        today_date = viz_data['ds'].iloc[-31] 
        ax.plot(viz_data['ds'], viz_data['yhat'], color=t['accent'], label='AI Trend', linestyle='--')
        ax.fill_between(viz_data['ds'], viz_data['yhat_lower'], viz_data['yhat_upper'], color=t['accent'], alpha=0.2, label='Obszar Niepewnosci')
        ax.axvline(x=today_date, color=t['text'], linestyle=':', label='TERAZ')
        ax.set_title(f"SZKLANA KULA AI (Prognoza 30 dni)", fontsize=16, color=t['text'])
        ax.legend(facecolor=t['bg'], labelcolor=t['text']); ax.grid(True, alpha=t['grid_alpha'], color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        return fig

    def plot_relative_strength_radar(self, df):
        alts = ['eth', 'sol', 'bnb', 'xrp', 'ada', 'doge']; period = 30
        if len(df) < period: return None
        t = self.get_theme_colors(); performance = {}
        btc_start = df['btc'].iloc[-period]; btc_end = df['btc'].iloc[-1]; btc_perf = (btc_end - btc_start) / btc_start * 100
        for alt in alts:
            if alt in df.columns:
                start = df[alt].iloc[-period]; end = df[alt].iloc[-1]; perf = (end - start) / start * 100
                rel_strength = perf - btc_perf; performance[alt.upper()] = rel_strength
        sorted_perf = dict(sorted(performance.items(), key=lambda item: item[1], reverse=True))
        fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(111)
        coins = list(sorted_perf.keys()); values = list(sorted_perf.values()); colors = [t['bull'] if v > 0 else t['bear'] for v in values] 
        bars = ax.barh(coins, values, color=colors, alpha=0.7)
        ax.axvline(x=0, color=t['text'], linestyle='-', linewidth=2, label='Bitcoin (Baza)')
        ax.set_title(f"RADAR SILY (30 Dni vs Bitcoin)", fontsize=16, color=t['text']); ax.set_xlabel('Przewaga nad BTC (%)', color=t['text'])
        for bar in bars:
            width = bar.get_width(); label_x_pos = width + 1 if width > 0 else width - 5
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:+.1f}%', va='center', color=t['text'], fontweight='bold')
        ax.grid(True, alpha=t['grid_alpha'], axis='x', color=t['grid']); fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text'], axis='y', labelcolor=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.invert_yaxis() 
        return fig

    def get_fair_value_analysis(self, df):
        btc = df['btc'].dropna(); btc = btc[btc > 0]
        days = np.arange(1, len(btc) + 1); log_days = np.log(days); log_price = np.log(btc.values)
        slope, intercept = np.polyfit(log_days, log_price, 1); fair_value_log = intercept + slope * log_days
        fair_value = np.exp(fair_value_log); fair_value_series = pd.Series(fair_value, index=btc.index)
        current_price = btc.iloc[-1]; current_fair = fair_value[-1]; deviation = (current_price - current_fair) / current_fair * 100
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.plot(btc.index, btc, color=t['text'], linewidth=1, label='Cena Rynkowa', alpha=0.8)
        ax.plot(btc.index, fair_value_series, color='#ff00ff', linewidth=2, linestyle='--', label='Godziwa Cena (Power Law)')
        ax.fill_between(btc.index, btc, fair_value_series, where=(btc < fair_value_series), color='green', alpha=0.2, label='Niedowartosciowanie (OKAZJA)')
        ax.fill_between(btc.index, btc, fair_value_series, where=(btc > fair_value_series), color='red', alpha=0.2, label='Przewartosciowanie (BANKA)')
        ax.set_title(f"GODZIWA CENA (Power Law Model)", fontsize=16, color=t['text']); ax.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        ax.grid(True, alpha=t['grid_alpha'], which='both', color=t['grid']); fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        return fig, current_fair, deviation

    def get_market_matrix(self, df, mvrv, ism, fair_value_dev, news_score):
        rsi = self.calc_rsi(df['btc']).iloc[-1]; score_tech = np.clip(rsi, 0, 100)
        mvrv_val = mvrv.iloc[-1]; score_onchain = np.clip((mvrv_val / 3.5) * 100, 0, 100)
        ism_val = ism.iloc[-1]; score_macro = np.clip(ism_val * 60, 0, 100)
        score_value = np.clip(fair_value_dev + 50, 0, 100); score_sent = np.clip(news_score * 100, 0, 100)
        categories = ['Technika (RSI)', 'On-Chain (MVRV)', 'Makro (ISM)', 'Wycena (Trend)', 'Sentyment (AI)']
        values = [score_tech, score_onchain, score_macro, score_value, score_sent]; values += values[:1]
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]; angles += angles[:1]
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, color=t['bull'], linewidth=2, linestyle='solid'); ax.fill(angles, values, color=t['bull'], alpha=0.4)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, color=t['text'], fontsize=11, fontweight='bold')
        ax.set_yticks([20, 40, 60, 80, 100]); ax.set_yticklabels(["20","40","60","80","100"], color="gray", fontsize=8)
        ax.set_ylim(0, 100); ax.set_facecolor(t['bg']); fig.patch.set_facecolor(t['bg'])
        ax.spines['polar'].set_color(t['text']); ax.grid(color=t['grid'], alpha=t['grid_alpha'])
        avg_score = np.mean([score_tech, score_onchain, score_macro, score_value, score_sent])
        status = "SILNY" if avg_score > 60 else "NEUTRALNY" if avg_score > 40 else "SLABY"
        ax.set_title(f"MATRYCA RYNKU: {status} ({avg_score:.0f}/100)", color=t['text'], fontsize=15, pad=20)
        return fig

    def plot_gaussian_channel(self, df):
        data = df['btc'].tail(730).copy(); window = 20
        basis = data.rolling(window=window).mean(); std_dev = data.rolling(window=window).std()
        upper_2 = basis + (2 * std_dev); lower_2 = basis - (2 * std_dev); upper_3 = basis + (3 * std_dev); lower_3 = basis - (3 * std_dev)
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(111)
        ax.plot(data.index, data, color=t['text'], linewidth=1.5, label='Cena BTC'); ax.plot(basis.index, basis, color='orange', linestyle='--', linewidth=1, label='Srodek')
        ax.fill_between(data.index, lower_2, upper_2, color='green', alpha=0.1, label='Strefa Normalna')
        ax.fill_between(data.index, upper_2, upper_3, color='red', alpha=0.2, label='Ekstremum Gorne')
        ax.fill_between(data.index, lower_3, lower_2, color='blue', alpha=0.2, label='Ekstremum Dolne')
        ax.plot(upper_3.index, upper_3, color='red', linewidth=0.5); ax.plot(lower_3.index, lower_3, color='cyan', linewidth=0.5)
        ax.set_title("KANAL GAUSSA: Statystyka", fontsize=16, color=t['text']); ax.grid(False)
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        return fig

    def plot_fibonacci_chart(self, df):
        btc = df['btc'].tail(365); high_price = btc.max(); low_price = btc.min(); diff = high_price - low_price
        levels = {0.0:(high_price,"Szczyt",'red'), 0.236:(high_price-0.236*diff,"0.236",'orange'), 0.382:(high_price-0.382*diff,"0.382",'orange'), 0.5:(high_price-0.5*diff,"0.5",'gray'), 0.618:(high_price-0.618*diff,"GOLDEN POCKET",'#00ff00'), 0.786:(high_price-0.786*diff,"0.786",'cyan'), 1.0:(low_price,"Dolek",'blue')}
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(111)
        ax.plot(btc.index, btc, color='lightgray' if t['bg']=='#0e1117' else 'gray', linewidth=1.5, label='Cena BTC')
        for level, (price, label, color) in levels.items():
            linewidth = 2 if level == 0.618 else 1; linestyle = '-' if level in [0,1,0.618] else '--'
            ax.axhline(y=price, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
            ax.text(btc.index[0], price, f"{label}: ${price:,.0f}", color=color, fontsize=9, va='bottom', fontweight='bold')
        golden_top = levels[0.618][0]; golden_bot = levels[0.786][0]
        ax.fill_between(btc.index, golden_top, golden_bot, color='green', alpha=0.1)
        ax.set_title("GEOMETRIA RYNKU: Poziomy Fibonacciego", fontsize=16, color=t['text']); ax.set_ylabel('Cena ($)', color=t['text']); ax.grid(False)
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        return fig, levels[0.618][0]

    def plot_history_truth(self, df, mvrv, ism):
        days_back = 365 * 8; btc = df['btc'].tail(days_back); mvrv_cut = mvrv.tail(days_back); ism_cut = ism.tail(days_back)
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7)); ax1 = fig.add_subplot(111)
        ax1.set_yscale('log'); ax1.plot(btc.index, btc, color=t['text'], alpha=0.3, linewidth=1, label='Cena BTC (Log)')
        ax1.set_ylabel('Cena BTC (Log)', color=t['text']); ax1.tick_params(axis='y', labelcolor=t['text'], colors=t['text']); ax1.tick_params(axis='x', colors=t['text'])
        ax2 = ax1.twinx(); ax2.plot(mvrv_cut.index, mvrv_cut, color='#ff9900', linewidth=2, label='MVRV')
        ax2.plot(ism_cut.index, ism_cut, color='#d900ff', linewidth=2, alpha=0.8, label='ISM Proxy')
        ax2.set_ylabel('Wskazniki', color=t['text']); ax2.tick_params(axis='y', labelcolor=t['text'], colors=t['text'])
        halvings = [datetime(2016, 7, 9), datetime(2020, 5, 11), datetime(2024, 4, 19)]
        for date in halvings:
            if date >= btc.index[0]: ax1.axvline(x=date, color=t['bull'], linestyle='--', linewidth=1); ax1.text(date, btc.min(), " HALVING", color=t['bull'], rotation=90, va='bottom', fontsize=8)
        ax2.axhline(y=1.0, color=t['text'], linestyle='--', linewidth=2.5, alpha=0.9, label='MVRV DNO (1.0)')
        ax2.text(btc.index[0], 1.05, " POZIOM OKAZJI (1.0)", color=t['text'], fontsize=10, fontweight='bold', ha='left')
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        ax1.set_title("HISTORIA PRAWDY", fontsize=16, color=t['text'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg']); ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        return fig

    def plot_etf_tracker(self, df, vol_df, asset_key, etf_name):
        if asset_key not in df.columns: return None
        price = df[asset_key].tail(180); t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6)); ax1 = fig.add_subplot(111)
        ax1.plot(price.index, price, color=t['text'], linewidth=2, label=f'{etf_name} Cena')
        ax1.set_ylabel('Cena ($)', color=t['text']); ax1.tick_params(axis='y', labelcolor=t['text'], colors=t['text']); ax1.tick_params(axis='x', colors=t['text'])
        if vol_df is not None:
            target_col = f"{asset_key}_vol"
            if target_col not in vol_df.columns: target_col = self.assets[asset_key] 
            if target_col in vol_df.columns:
                vol = vol_df[target_col].tail(180); ax2 = ax1.twinx()
                ax2.bar(vol.index, vol, color=t['accent'], alpha=0.3, label='Wolumen')
                ax2.set_ylabel('Wolumen', color=t['accent']); ax2.tick_params(axis='y', labelcolor=t['accent'], colors=t['text']); ax2.grid(False)
        ax1.set_title(f"SLAD WIELORYBA: {etf_name}", fontsize=16, color=t['text'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg']); ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        return fig

    def plot_liquidity_chart(self, df):
        period = 730; hyg = df['hyg'].tail(period); btc = df['btc'].tail(period)
        hyg_norm = hyg / hyg.iloc[0] * 100; btc_norm = btc / btc.iloc[0] * 100
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6)); ax1 = fig.add_subplot(111); color = t['bull'] 
        ax1.plot(hyg_norm.index, hyg_norm, color=color, linewidth=2, label='Plynnosc (HYG)')
        ax1.set_ylabel('Plynnosc', color=color); ax1.tick_params(axis='y', labelcolor=color, colors=t['text']); ax1.tick_params(axis='x', colors=t['text'])
        ax2 = ax1.twinx(); color = t['text']
        ax2.plot(btc_norm.index, btc_norm, color=color, linestyle='--', alpha=0.6, label='Bitcoin')
        ax2.tick_params(axis='y', labelcolor=color, colors=t['text'])
        ax1.set_title("PALIWO RAKIETOWE", fontsize=16, color=t['text'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        return fig

    def plot_rainbow_chart(self, df):
        btc = df['btc'].dropna(); btc = btc[btc > 0]; df_rainbow = pd.DataFrame({'price': btc})
        x = np.log(np.arange(len(df_rainbow)) + 1); y = np.log(df_rainbow['price'].values)      
        coeffs = np.polyfit(x, y, 1); fitted_y = np.polyval(coeffs, x)
        bands = {'Bubble':(fitted_y+1.5,'red'),'FOMO':(fitted_y+1.0,'orange'),'HODL':(fitted_y+0.0,'yellow'),'Cheap':(fitted_y-1.0,'green'),'Fire Sale':(fitted_y-1.8,'blue')}
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(111); prev_y = None
        ax.plot(df_rainbow.index, np.log(df_rainbow['price']), color=t['text'], linewidth=1.5, zorder=10)
        for label, (y_vals, color) in bands.items():
            ax.plot(df_rainbow.index, y_vals, color=color, alpha=0.3, linewidth=0.5)
            if prev_y is not None: ax.fill_between(df_rainbow.index, prev_y, y_vals, color=color, alpha=0.1)
            prev_y = y_vals
            ax.text(df_rainbow.index[-1], y_vals[-1], f" {label}", color=color, fontsize=8)
        ax.set_title("BITCOIN RAINBOW CHART", fontsize=16, color=t['text']); ax.grid(True, alpha=t['grid_alpha'], color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        return fig

    def plot_cycle_clock(self, mvrv, ism_proxy):
        lookback = 365; common_idx = mvrv.index.intersection(ism_proxy.index)
        y_data = mvrv.loc[common_idx].tail(lookback); x_data = ism_proxy.loc[common_idx].tail(lookback)
        cur_x = x_data.iloc[-1]; cur_y = y_data.iloc[-1]
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111)
        ax.plot(x_data, y_data, color='gray', alpha=0.3, linewidth=1)
        ax.scatter(cur_x, cur_y, color=t['bull'], s=200, zorder=5, edgecolors=t['text'])
        ax.axhline(y=1.6, color=t['text'], linestyle='--', alpha=0.3); ax.axvline(x=x_data.mean(), color=t['text'], linestyle='--', alpha=0.3)
        ax.text(x_data.min(), y_data.min(), "RECESJA (Tanio)", color='red', fontsize=10)
        ax.text(x_data.max(), y_data.max(), "EUPHORIA (Drogo)", color='lime', ha='right', fontsize=10)
        ax.set_title("ZEGAR CYKLU", fontsize=16, color=t['text']); ax.grid(True, alpha=0.2, color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        return fig

    def plot_altseason_chart(self, df):
        eth_btc = df['eth'] / df['btc']; sma_50 = eth_btc.rolling(window=50).mean()
        data = eth_btc.tail(365); sma_data = sma_50.tail(365)
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 5)); ax = fig.add_subplot(111)
        ax.plot(data.index, data, color=t['accent'], linewidth=2, label='ETH/BTC')
        ax.plot(sma_data.index, sma_data, color='orange', linestyle='--', label='Trend')
        ax.fill_between(data.index, data, sma_data, where=(data > sma_data), color='green', alpha=0.1, label="ALTSEASON")
        ax.fill_between(data.index, data, sma_data, where=(data <= sma_data), color='red', alpha=0.1, label="BITCOIN SEASON")
        ax.set_title("ALTCOIN SEASON INDEX", fontsize=16, color=t['text']); ax.legend(facecolor=t['bg'], labelcolor=t['text']); ax.grid(True, alpha=t['grid_alpha'], color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text']); ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        return fig

    # --- FIX: KORELACJA (BEZPIECZNA DLA STREAMLIT) ---
    def plot_correlation_heatmap(self, df):
        # Bierzemy ostatnie 60 dni
        recent_data = df.tail(60).copy()
        corr_matrix = recent_data.corr()
        
        t = self.get_theme_colors()
        
        # Tworzymy figurę
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Rysujemy mapę ciepła na konkretnej osi (ax=ax)
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap='RdBu', 
            vmin=-1, 
            vmax=1, 
            center=0, 
            ax=ax, 
            linewidths=0.5,
            cbar_kws={'ticks': [-1, 0, 1]} # Upraszczamy pasek legendy
        )
        
        ax.set_title("Macierz Korelacji (60 dni)", fontsize=14, color=t['text'])
        
        # Kolorowanie etykiet
        ax.tick_params(axis='x', colors=t['text'], rotation=45)
        ax.tick_params(axis='y', colors=t['text'], rotation=0)
        
        # Kolorowanie paska legendy (colorbar)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=t['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=t['text'])
        
        fig.patch.set_facecolor(t['bg'])
        ax.set_facecolor(t['bg'])
        
        return fig

    def get_sniper_signals(self, df):
        targets = ['btc', 'eth', 'sol', 'sp500', 'gold', 'silver']
        signals = []
        for asset in targets:
            series = df[asset].dropna()
            if len(series) < 200: continue
            curr_price = series.iloc[-1]; sma_50 = series.rolling(50).mean().iloc[-1]; sma_200 = series.rolling(200).mean().iloc[-1]; rsi = self.calc_rsi(series).iloc[-1]
            trend_icon = "🟢 WZROSTOWY" if curr_price > sma_200 else "🔴 SPADKOWY"
            cross_icon = "✨ GOLDEN CROSS" if sma_50 > sma_200 else "☠️ DEATH CROSS"
            if rsi < 30: rsi_stat = "🥶 WYPRZEDANIE"
            elif rsi > 70: rsi_stat = "🔥 PRZEGRZANIE"
            else: rsi_stat = "Neutralnie"
            signals.append({"Aktywo": asset.upper(), "Cena": f"{curr_price:,.2f}", "Trend (200 SMA)": trend_icon, "Sygnal (Krzyz)": cross_icon, "RSI (14)": f"{rsi:.1f} | {rsi_stat}"})
        return pd.DataFrame(signals)

    def check_anomalies(self, df):
        alerts = []
        last = df.iloc[-1]; prev = df.iloc[-2]
        btc_ch = (last['btc'] - prev['btc']) / prev['btc'] * 100
        if btc_ch < -5: alerts.append((f"🚨 KRASZ BITCOINA! -{abs(btc_ch):.2f}%", f"Cena BTC spadla gwaltownie o {abs(btc_ch):.2f}% w jeden dzien. Sprawdz poziomy wsparcia."))
        elif btc_ch > 5: alerts.append((f"🚀 BITCOIN POMPUJE! +{btc_ch:.2f}%", f"Silny wzrost o {btc_ch:.2f}%. Uwazaj na FOMO, ale trend jest silny."))
        vix_ch = (last['vix'] - prev['vix']) / prev['vix'] * 100
        if vix_ch > 10: alerts.append((f"⚠️ PANIKA (VIX)! +{vix_ch:.2f}%", f"**Indeks Strachu (VIX) wzrosl o {vix_ch:.2f}%!**\n\n📉 **Co to znaczy?**\nVIX mierzy oczekiwana zmiennosc na gieldzie S&P 500. Tak gwaltowny skok oznacza, ze inwestorzy panicznie kupuja zabezpieczenia (opcje PUT), spodziewajac sie spadkow.\n\n🧐 **Jak to czytac?**\n1. **Krew na ulicach:** Rynek jest w trybie 'Risk-Off' (ucieczka od ryzyka). Krypto moze oberwac rykoszetem.\n2. **Szansa na dolek:** Statystycznie, gdy VIX wystrzela w kosmos, czesto jestesmy blisko lokalnego dolka cenowego. Jak mowia klasycy: *'Kupuj, gdy leje sie krew'*, ale zachowaj ostroznosc."))
        dxy_ch = (last['dxy'] - prev['dxy']) / prev['dxy'] * 100
        if dxy_ch > 1: alerts.append((f"💵 SILNY DOLAR! +{dxy_ch:.2f}%", "Dolar drozeje. Zazwyczaj, gdy Dolar jest silny, aktywa ryzykowne (Krypto/Akcje) radza sobie slabiej."))
        return alerts

    def analyze_news_sentiment(self, category):
        url = self.rss_feeds.get(category)
        if not url: return 0.5, []
        try:
            feed = feedparser.parse(url); headlines = []; total = 0; count = 0
            for e in feed.entries[:5]:
                p = TextBlob(e.title).sentiment.polarity
                headlines.append((e.title, p, e.link)); total += p; count += 1
            if count == 0: return 0.5, []
            return np.clip(0.5 + (total/count * 0.5), 0, 1), headlines
        except: return 0.5, []

    def normalize(self, v, h): return (h < v).mean()
    def calc_rsi(self, s, p=14):
        d = s.diff(); g = (d.where(d>0,0)).rolling(p).mean(); l = (-d.where(d<0,0)).rolling(p).mean()
        return 100 - (100/(1+(g/l)))

    def get_historical_indices(self, df):
        h = df.copy(); window = 500
        h['sp500_rank'] = h['sp500'].rolling(window=window).rank(pct=True)
        h['copper_rank'] = h['copper'].rolling(window=window).rank(pct=True)
        h['bonds_rank'] = 1.0 - h['bonds_10y'].rolling(window=window).rank(pct=True)
        h['vix_rank'] = 1.0 - h['vix'].rolling(window=window).rank(pct=True)
        h['dxy_rank'] = 1.0 - h['dxy'].rolling(window=window).rank(pct=True)
        h['gold_rsi'] = 1.0 - (self.calc_rsi(h['gold']) / 100)
        h['Eco_Index'] = ((h['sp500_rank'] * 0.20) + (h['copper_rank'] * 0.15) + (h['bonds_rank'] * 0.10) + (h['vix_rank'] * 0.15) + (h['dxy_rank'] * 0.15) + (0.5 * 0.15) + (h['gold_rsi'] * 0.10))
        sma_btc = h['btc'].rolling(window=200).mean(); dist = (h['btc'] - sma_btc) / sma_btc
        h['btc_dist_score'] = (0.5 + dist).clip(0, 1)
        h['dxy_rank_cry'] = 1.0 - h['dxy'].rolling(window=window).rank(pct=True)
        h['eth_rsi_score'] = self.calc_rsi(h['eth']) / 100; h['btc_rsi_score'] = self.calc_rsi(h['btc']) / 100
        approx_fng = h['vix_rank']
        h['Cry_Index'] = ((h['btc_dist_score'] * 0.30) + (approx_fng * 0.15) + (h['dxy_rank_cry'] * 0.10) + (0.5 * 0.15) + (h['eth_rsi_score'] * 0.15) + (h['btc_rsi_score'] * 0.15))
        h['Eco_Index'] = h['Eco_Index'].rolling(window=7).mean(); h['Cry_Index'] = h['Cry_Index'].rolling(window=7).mean()
        return h[['Eco_Index', 'Cry_Index']]

    def analyze_economy(self, df, news):
        s, w = [], []; rec = df.tail(500)
        s.append(self.normalize(df['sp500'].iloc[-1], rec['sp500'])); w.append(0.20)
        s.append(self.normalize(df['copper'].iloc[-1], rec['copper'])); w.append(0.15)
        s.append(1.0 - self.normalize(df['bonds_10y'].iloc[-1], rec['bonds_10y'])); w.append(0.10)
        s.append(1.0 - self.normalize(df['vix'].iloc[-1], rec['vix'])); w.append(0.15)
        s.append(1.0 - self.normalize(df['dxy'].iloc[-1], rec['dxy'])); w.append(0.15)
        s.append(news); w.append(0.15)
        s.append(1.0 - (self.calc_rsi(df['gold']).iloc[-1]/100)); w.append(0.10)
        return np.average(s, weights=w)

    def analyze_crypto(self, df, fng, news):
        s, w = [], []; sma = df['btc'].rolling(200).mean().iloc[-1]
        dist = (df['btc'].iloc[-1] - sma)/sma
        s.append(np.clip(0.5 + dist, 0, 1)); w.append(0.30)
        s.append(fng/100); w.append(0.15)
        s.append(1.0 - self.normalize(df['dxy'].iloc[-1], df['dxy'].tail(500))); w.append(0.10)
        s.append(news); w.append(0.15)
        s.append(self.calc_rsi(df['eth']).iloc[-1]/100); w.append(0.15)
        s.append(self.calc_rsi(df['btc']).iloc[-1]/100); w.append(0.15)
        return np.average(s, weights=w)

    def save_log(self, eco, cry, btc, dxy, fng, news):
        exists = os.path.isfile(self.filename)
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            if not exists: wr.writerow(["Data","Czas","Eco","Crypto","BTC","DXY","F&G","News"])
            now = datetime.now()
            wr.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), round(eco,4), round(cry,4), round(btc,2), round(dxy,2), fng, round(news,4)])

    # --- FIX: LIKWIDACJE (OSTATECZNA NAPRAWA STRUKTURY DANYCH) ---
    def get_liquidation_proxy(self, df):
        """
        Szacuje likwidacje. 
        Wersja z 'brutalnym' czyszczeniem MultiIndexu, który powodował błędy.
        """
        try:
            # Pobieramy dane OHLC
            btc_ohlc = yf.download('BTC-USD', period="1y", interval="1d", progress=False)
            
            if btc_ohlc.empty: return None

            # 1. NAPRAWA STRUKTURY (FLATTENING)
            # Jeśli kolumny to MultiIndex (np. ('Close', 'BTC-USD')), spłaszczamy je
            if isinstance(btc_ohlc.columns, pd.MultiIndex):
                # Odrzucamy poziom z nazwą tickera ('BTC-USD'), zostawiamy tylko 'Open', 'High' itp.
                btc_ohlc.columns = btc_ohlc.columns.droplevel(1) 
                # Jeśli powyższe nie zadziała w Twojej wersji, alternatywa:
                # btc_ohlc.columns = [c[0] for c in btc_ohlc.columns]

            # Upewniamy się, że to są czyste dane numeryczne
            btc_ohlc = btc_ohlc.astype(float)

            data = pd.DataFrame(index=btc_ohlc.index)
            
            # 2. OBLICZENIA (Teraz bezpieczne)
            # Min/Max z Open i Close dla każdej świecy
            min_oc = btc_ohlc[['Open', 'Close']].min(axis=1)
            max_oc = btc_ohlc[['Open', 'Close']].max(axis=1)
            
            # Obliczamy długość knotów
            data['Lower_Wick'] = min_oc - btc_ohlc['Low']
            data['Upper_Wick'] = btc_ohlc['High'] - max_oc
            data['Volume'] = btc_ohlc['Volume']
            
            # 3. WSKAŹNIK "REKT"
            # (Wielkość Knota * Wolumen) / 1 miliard
            # Używamy .fillna(0) na wypadek błędów dzielenia/braków
            data['Long_Rekt'] = ((data['Lower_Wick'] * data['Volume']) / 1e9).fillna(0)
            data['Short_Rekt'] = ((data['Upper_Wick'] * data['Volume']) / 1e9).fillna(0)
            
            return data
            
        except Exception as e:
            print(f"Błąd likwidacji: {e}") # Wypisz w konsoli, ale nie wywalaj programu
            return None

    def plot_liquidation_radar(self, liq_df):
        if liq_df is None or liq_df.empty: return None
        t = self.get_theme_colors()
        
        # Bierzemy ostatnie 60 dni
        subset = liq_df.tail(60)
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Rysujemy Long Rekt (Czerwone słupki w dół - bo to spadki)
        ax.bar(subset.index, -subset['Long_Rekt'], color='#ff0055', label='Likwidacja Longow (Krach)', alpha=0.8)
        
        # Rysujemy Short Rekt (Zielone słupki w górę - bo to pompy)
        ax.bar(subset.index, subset['Short_Rekt'], color='#00ff55', label='Likwidacja Shortow (Squeeze)', alpha=0.8)
        
        ax.axhline(0, color=t['text'], linewidth=1)
        
        ax.set_title("RADAR LIKWIDACJI (Proxy: Knoty * Wolumen)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_ylabel('Sila Likwidacji (Indeks)', color=t['text'])
        
        # Legenda
        ax.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        # Dodajemy adnotację edukacyjną
        ax.text(subset.index[0], subset['Short_Rekt'].max(), 
                "To nie są dane z giełdy (niedostępne w Yahoo).\nTo analiza 'Wicks & Volume' - pokazuje gdzie bolało.", 
                color=t['text'], fontsize=9, alpha=0.7, va='top')

        fig.patch.set_facecolor(t['bg'])
        ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text'])
        ax.spines['bottom'].set_color(t['text'])
        ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return fig

    # --- NOWOŚĆ: MAYER MULTIPLE (OBLICZENIA) ---
    def get_mayer_multiple_data(self, df):
        """
        Oblicza wskaźnik Mayer Multiple (Cena / 200 SMA) oraz wstęgi.
        """
        if df is None or 'btc' not in df.columns: return None
        
        # Kopiujemy dane (ostatnie 4 lata wystarczą do analizy cyklu)
        data = df[['btc']].tail(1460).copy()
        
        # Obliczenia
        data['SMA_200'] = data['btc'].rolling(window=200).mean()
        data['Mayer_Multiple'] = data['btc'] / data['SMA_200']
        
        # Wstęgi (Bands)
        data['Band_Bubble'] = data['SMA_200'] * 2.4  # Szczyt bańki
        data['Band_Fair'] = data['SMA_200'] * 1.0    # Średnia
        data['Band_Cheap'] = data['SMA_200'] * 0.8   # Tanio
        data['Band_Fire_Sale'] = data['SMA_200'] * 0.6 # Dno generacyjne
        
        return data

    # --- NOWOŚĆ: MAYER MULTIPLE (WIZUALIZACJA) ---
    def plot_mayer_multiple_bands(self, mayer_df):
        """
        Rysuje Mayer Multiple Bands na podstawie obliczonych danych.
        """
        if mayer_df is None or mayer_df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Wyciągamy dane dla wygody
        price = mayer_df['btc']
        bubble = mayer_df['Band_Bubble']
        cheap = mayer_df['Band_Cheap']
        fire = mayer_df['Band_Fire_Sale']
        
        # 1. Rysujemy Cenę (Logarytmicznie, żeby lepiej widzieć cykle)
        ax.semilogy(price.index, price, color='white', linewidth=1.5, label='Cena BTC', zorder=5)
        
        # 2. Rysujemy Pasma
        ax.plot(bubble.index, bubble, color='#ff0055', linestyle='--', linewidth=1, label='Szczyt Bańki (2.4x)')
        ax.plot(mayer_df.index, mayer_df['Band_Fair'], color='gray', linestyle=':', linewidth=1, label='Fair Value (1.0x)')
        ax.plot(cheap.index, cheap, color='#00ff55', linestyle='--', linewidth=1, label='Strefa Okazji (0.8x)')
        
        # 3. Wypełnienia (Strefy decyzyjne)
        # Strefa Sprzedaży (Powyżej 2.4)
        ax.fill_between(mayer_df.index, bubble, bubble * 1.3, color='#ff0055', alpha=0.2)
        # Strefa Zakupu (Poniżej 0.8)
        ax.fill_between(mayer_df.index, fire, cheap, color='#00ff55', alpha=0.2)
        
        # 4. Status Aktualny
        current_mm = mayer_df['Mayer_Multiple'].iloc[-1]
        
        if current_mm < 0.8: status = "OKAZJA ŻYCIA (Kupuj!)"
        elif current_mm < 1.1: status = "TANIO / FAIR"
        elif current_mm < 2.4: status = "HOSSA (Trzymaj)"
        else: status = "BAŃKA (Sprzedawaj!)"
        
        ax.set_title(f"MAYER MULTIPLE: {current_mm:.2f}x | {status}", fontsize=16, color=t['text'], fontweight='bold')
        
        # Kosmetyka
        ax.legend(facecolor=t['bg'], labelcolor=t['text'], loc='upper left')
        ax.grid(True, alpha=0.15, color=t['grid'])
        
        fig.patch.set_facecolor(t['bg'])
        ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text'])
        ax.spines['bottom'].set_color(t['text'])
        ax.spines['left'].set_color(t['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return fig

    # --- UPDATE: VOLATILITY SQUEEZE DATA (Wyższy Próg / Czułość) ---
    def get_volatility_squeeze_data(self, df):
        """
        Oblicza Wstęgi Bollingera.
        ZMIANA: Podniesiony próg czułości (min. 0.10), żeby łapać więcej sygnałów.
        """
        if df is None or 'btc' not in df.columns: return None, 0
        
        data = df[['btc']].tail(400).copy() # Nieco więcej danych dla płynności
        
        # 1. Obliczenie Wstęg Bollingera
        data['SMA_20'] = data['btc'].rolling(window=20).mean()
        data['StdDev'] = data['btc'].rolling(window=20).std()
        data['Upper'] = data['SMA_20'] + (data['StdDev'] * 2)
        data['Lower'] = data['SMA_20'] - (data['StdDev'] * 2)
        
        # 2. Obliczenie Szerokości (BB Width)
        data['BB_Width'] = (data['Upper'] - data['Lower']) / data['SMA_20']
        
        # 3. Wyznaczenie Progu Wybuchu (PODNIESIONY)
        # Obliczamy 20-ty percentyl (zamiast 10-tego), żeby strefa była szersza
        dynamic_threshold = data['BB_Width'].tail(180).quantile(0.20)
        
        # --- FIX: WYMUSZENIE MINIMUM 0.10 ---
        # Jeśli matematyka wyliczy 0.07, my i tak wymusimy 0.10.
        threshold = max(dynamic_threshold, 0.10)
        
        return data, threshold

    # --- UPDATE: VOLATILITY SQUEEZE (Wersja z Ceną BTC) ---
    def plot_volatility_squeeze(self, data, threshold):
        """
        Rysuje Szerokość Wstęgi ORAZ Cenę BTC na drugiej osi.
        Dzięki temu widać kontekst (czy sprężyna naciąga się na szczycie czy na dnie).
        """
        if data is None or data.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        
        # --- OŚ 1 (LEWA): SQUEEZE (Niebieska linia) ---
        current_width = data['BB_Width'].iloc[-1]
        
        # Rysujemy linię zmienności
        line1, = ax1.plot(data.index, data['BB_Width'], color='#00e5ff', linewidth=2, label='Naciąg Sprężyny (Lewa Oś)')
        
        # Rysujemy Próg (Czerwona przerywana)
        line2 = ax1.axhline(threshold, color='#ff0055', linestyle='--', linewidth=2, label=f'Próg Wybuchu ({threshold:.3f})')
        
        # Wypełnienie strefy wybuchu
        ax1.fill_between(data.index, 0, threshold, color='#ff0055', alpha=0.15)
        
        # Opis osi lewej
        ax1.set_ylabel('Szerokość Wstęgi', color='#00e5ff', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#00e5ff', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        # --- OŚ 2 (PRAWA): CENA BTC (Złota linia) ---
        ax2 = ax1.twinx() # Tworzymy bliźniaczą oś X
        
        # Rysujemy cenę (nieco cieńszą/delikatniejszą, żeby nie zasłaniała wskaźnika)
        line3, = ax2.plot(data.index, data['btc'], color='#ffd700', linewidth=1, alpha=0.6, linestyle='-', label='Cena BTC (Prawa Oś)')
        
        # Opis osi prawej
        ax2.set_ylabel('Cena BTC ($)', color='#ffd700', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#ffd700', colors=t['text'])
        
        # Ukrywamy górną oś dla czystości
        ax2.spines['top'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text'])
        ax1.spines['left'].set_color(t['text'])
        ax2.spines['right'].set_color(t['text']) # Prawa krawędź widoczna dla ceny
        ax2.spines['left'].set_visible(False)

        # --- LOGIKA STATUSU (BEZ ZMIAN) ---
        if current_width <= threshold:
            status = "⚠️ SQUEEZE! (Sprężyna naciągnięta)"
            status_color = '#ff0055'
            # Kropka alarmowa na końcu linii zmienności
            ax1.scatter(data.index[-1], current_width, s=150, color='red', zorder=10, edgecolors='white')
        else:
            status = "Luz (Czekamy na ściśnięcie)"
            status_color = '#00ff55'

        ax1.set_title(f"DETEKTOR WYBUCHU: {status}", fontsize=16, color=status_color, fontweight='bold')
        
        # --- WSPÓLNA LEGENDA ---
        # Musimy połączyć legendy z dwóch osi w jedną
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        ax1.grid(True, alpha=0.15, color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg']); ax2.set_facecolor(t['bg']) # Fix dla tła
        ax2.patch.set_alpha(0.0) # Druga oś musi być przezroczysta
        
        return fig

    # --- UPDATE: CRYPTO BUBBLES DATA (Wersja MEGA - 65+ Coinów) ---
    def get_crypto_bubbles_data(self):
        """
        Pobiera dane dla BARDZO SZEROKIEGO RYNKU (ok. 65 coinów).
        Celem jest zapchanie ekranu bąbelkami.
        """
        # Lista "Fat Pack": Majors, L1, L2, AI, Meme, DeFi, Legacy
        coins = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 
            'AVAX-USD', 'TRX-USD', 'DOT-USD', 'LINK-USD', 'TON-USD', 'MATIC-USD', 'SHIB-USD',
            'BCH-USD', 'LTC-USD', 'UNI-USD', 'NEAR-USD', 'APT-USD', 'ICP-USD', 'PEPE-USD',
            'ETC-USD', 'XLM-USD', 'XMR-USD', 'FIL-USD', 'HBAR-USD', 'LDO-USD', 'ARB-USD',
            'RNDR-USD', 'ATOM-USD', 'STX-USD', 'IMX-USD', 'INJ-USD', 'OP-USD', 'GRT-USD',
            # Nowości i Memy (Zapychacze przestrzeni)
            'VET-USD', 'TAO-USD', 'TIA-USD', 'SUI-USD', 'SEI-USD', 'KAS-USD', 'FET-USD',
            'AGIX-USD', 'BONK-USD', 'WIF-USD', 'FLOKI-USD', 'RUNE-USD', 'AAVE-USD', 'MKR-USD',
            'QNT-USD', 'ALGO-USD', 'FLOW-USD', 'SAND-USD', 'MANA-USD', 'EGLD-USD', 'AXS-USD',
            'THETA-USD', 'FTM-USD', 'SNX-USD', 'NEO-USD', 'EOS-USD', 'XTZ-USD', 'CHZ-USD'
        ]
        
        try:
            # Pobieramy dane (tylko 2 ostatnie dni dla szybkości, potrzebujemy zmiany 24h)
            df = yf.download(coins, period="2d", progress=False)['Close']
            
            bubble_data = []
            
            for ticker in coins:
                if ticker in df.columns:
                    series = df[ticker].dropna()
                    if len(series) > 1:
                        last_price = series.iloc[-1]
                        prev_price_24h = series.iloc[-2]
                        
                        change_24h = ((last_price - prev_price_24h) / prev_price_24h) * 100
                        name = ticker.replace('-USD', '')
                        
                        bubble_data.append({
                            'name': name,
                            'change': change_24h,
                            'price': last_price
                        })
            
            return pd.DataFrame(bubble_data)
        except Exception as e:
            print(f"Błąd bubbles: {e}")
            return None

    # --- UPDATE: CRYPTO BUBBLES PLOT (Wersja Gęsta) ---
    def plot_crypto_bubbles(self, df):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(14, 8)) # Jeszcze szerszy wykres
        ax = fig.add_subplot(111)
        
        # Generujemy losowe pozycje
        # Zmieniamy zakres na 25, żeby przy 60 bąblach miały trochę luzu, ale nie za dużo
        np.random.seed(42) 
        df['x'] = np.random.rand(len(df)) * 25
        df['y'] = np.random.rand(len(df)) * 25
        
        # --- ZMIANA: WIELKOŚĆ ---
        # Base size 900 (było 600) -> Żeby zajmowały więcej tła
        base_size = 900
        multiplier = 250
        sizes = base_size + (df['change'].abs() * multiplier)
        
        # Kolory
        colors = ['#00ff55' if x >= 0 else '#ff0055' for x in df['change']]
        
        # Rysujemy
        ax.scatter(df['x'], df['y'], s=sizes, c=colors, alpha=0.8, edgecolors=t['bg'], linewidth=1)
        
        # Etykiety
        for i, row in df.iterrows():
            # Skalowanie czcionki (żeby małe bąble miały czytelny tekst)
            # Im większa zmiana, tym większy tekst
            font_size = 8 + (abs(row['change']) * 0.4)
            font_size = min(max(font_size, 8), 14) # Limit 8-14
            
            # Nazwa
            ax.text(row['x'], row['y'] + 0.35, row['name'], 
                    ha='center', va='center', fontsize=font_size, fontweight='bold', color='black')
            
            # Procent
            sign = "+" if row['change'] > 0 else ""
            ax.text(row['x'], row['y'] - 0.35, f"{sign}{row['change']:.1f}%", 
                    ha='center', va='center', fontsize=font_size-1, color='black')

        ax.set_title(f"CRYPTO BUBBLES ({len(df)} Assets)", fontsize=20, color=t['text'], fontweight='bold')
        ax.axis('off')
        
        # Usuwamy marginesy, żeby bąble mogły dotykać krawędzi
        plt.tight_layout()
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        return fig

    # --- UPDATE: SNAJPER OKAZJI (Hunter Mode Pro - 100+ Coinów) ---
    def get_altcoin_squeeze_rank(self):
        """
        Skanuje PONAD 100 KRYPTOWALUT (Cały istotny rynek).
        Zwraca tylko TOP 15 z najbardziej naciągniętą sprężyną (najniższy Score).
        Dzięki temu nie musisz ręcznie wybierać coinów - program sam znajdzie okazje.
        """
        # MEGA LISTA (Hardcoded Market Scan)
        # Podzielona kategoriami, żeby nic nie umknęło.
        coins = [
            # 1. MAJORS & L1 (Fundamenty)
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 
            'DOT-USD', 'TRX-USD', 'LINK-USD', 'MATIC-USD', 'LTC-USD', 'BCH-USD', 'NEAR-USD', 
            'ATOM-USD', 'ICP-USD', 'APT-USD', 'SUI-USD', 'SEI-USD', 'KAS-USD', 'TON-USD',
            'ALGO-USD', 'HBAR-USD', 'EGLD-USD', 'XTZ-USD', 'EOS-USD', 'NEO-USD', 'FLOW-USD',
            'MINA-USD', 'QNT-USD', 'ASTR-USD', 'ROSE-USD', 'IOTA-USD', 'ZEC-USD', 'XMR-USD',
            
            # 2. LAYER 2 (Skalowanie)
            'ARB-USD', 'OP-USD', 'IMX-USD', 'MNT-USD', 'STX-USD', 'METIS-USD', 'LRC-USD',
            'SKL-USD', 'GNO-USD', 'STRK-USD', 'MANTA-USD',
            
            # 3. AI & DePIN (Sztuczna Inteligencja)
            'RNDR-USD', 'FET-USD', 'AGIX-USD', 'TAO-USD', 'TIA-USD', 'WLD-USD', 'GRT-USD',
            'OCEAN-USD', 'AKT-USD', 'JASMY-USD', 'AIOZ-USD', 'GLM-USD', 'ARKM-USD',
            
            # 4. RWA (Real World Assets - Hype)
            'ONDO-USD', 'PENDLE-USD', 'MKR-USD', 'SNX-USD', 'CFG-USD', 'MPL-USD', 'TRU-USD',
            'POLYX-USD', 'CPOOL-USD',
            
            # 5. MEME (Spekulacja)
            'DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'BONK-USD', 'WIF-USD', 'FLOKI-USD', 'MEME-USD',
            'BOME-USD', 'ORDI-USD', 'SATS-USD', 'DOG-USD',
            
            # 6. DeFi & DEX
            'UNI-USD', 'AAVE-USD', 'LDO-USD', 'CRV-USD', 'RUNE-USD', 'CAKE-USD', 'JUP-USD',
            'DYDX-USD', 'GMX-USD', 'COMP-USD', '1INCH-USD', 'SUSHI-USD', 'ENS-USD',
            
            # 7. GAMING & METAVERSE
            'ICP-USD', 'AXS-USD', 'SAND-USD', 'MANA-USD', 'GALA-USD', 'BEAM-USD', 'APE-USD',
            'ILV-USD', 'PRIME-USD', 'YGG-USD', 'PIXEL-USD'
        ]
        
        ranking = []
        
        try:
            # Pobieramy dane dla całej armii (To może chwilę potrwać, ok. 5-10 sekund)
            # period="6mo" jest optymalny
            df_all = yf.download(coins, period="6mo", progress=False)
            
            # Bezpieczne wyciąganie 'Close'
            if isinstance(df_all.columns, pd.MultiIndex):
                try: closes = df_all['Close']
                except KeyError: closes = df_all
            else:
                closes = df_all['Close'] if 'Close' in df_all.columns else df_all
            
            # Pętla po każdym coinie
            for ticker in coins:
                if ticker in closes.columns:
                    series = closes[ticker].dropna()
                    
                    # Filtrujemy:
                    if len(series) < 30: continue # Za młode
                    if (pd.Timestamp.now() - series.index[-1]).days > 3: continue # Nieaktualne dane

                    # 1. Bollinger Bands
                    sma = series.rolling(20).mean()
                    std = series.rolling(20).std()
                    upper = sma + (std * 2)
                    lower = sma - (std * 2)
                    
                    # 2. Szerokość
                    bb_width = (upper - lower) / sma
                    
                    if bb_width.isnull().iloc[-1]: continue

                    current_width = bb_width.iloc[-1]
                    min_w = bb_width.min()
                    max_w = bb_width.max()
                    
                    # 3. SCORE (0 = Max Squeeze / Cisza)
                    # Jeśli zmienność jest zerowa (błąd danych lub stablecoin), dajemy 0.5 (neutral)
                    if max_w == min_w: score = 0.5
                    else: score = (current_width - min_w) / (max_w - min_w)
                    
                    name = ticker.replace('-USD', '')
                    
                    ranking.append({
                        'coin': name,
                        'score': score, 
                        'width': current_width
                    })
            
            if not ranking: return None
            
            # --- SORTOWANIE I SELEKCJA ---
            rank_df = pd.DataFrame(ranking)
            
            # Sortujemy od najmniejszego Score (Najbardziej ściśnięte na górze)
            rank_df = rank_df.sort_values(by='score', ascending=True)
            
            # Zwracamy TOP 15 "Najgorętszych" okazji z całej setki
            return rank_df.head(15)
            
        except Exception as e:
            print(f"Błąd snajpera: {e}")
            return None

    # --- NOWOŚĆ: SKANER ALTCOINÓW (Wizualizacja) ---
    def plot_altcoin_squeeze_radar(self, df):
        """
        Rysuje ranking 'Squeeze'.
        Czerwone paski = Gotowe do wybuchu.
        """
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 8)) # Wysoki wykres
        ax = fig.add_subplot(111)
        
        # Odwracamy kolejność do rysowania (żeby najlepsze były na górze wykresu)
        df_plot = df.iloc[::-1]
        
        # Kolory pasków zależne od Score
        # < 0.15 (15%) = Czerwony (Squeeze)
        # Reszta = Szary/Niebieski
        colors = []
        for s in df_plot['score']:
            if s <= 0.15: colors.append('#ff0055') # ALARM
            elif s <= 0.30: colors.append('#ffeb3b') # Ostrzeżenie
            else: colors.append('#00e5ff') # Luz
            
        # Rysujemy paski poziome
        bars = ax.barh(df_plot['coin'], df_plot['score'], color=colors, alpha=0.8, height=0.6)
        
        # Linia progu (15%)
        ax.axvline(0.15, color='#ff0055', linestyle='--', linewidth=2)
        ax.text(0.16, len(df_plot)-1, 'STREFA WYBUCHU (<15%)', color='#ff0055', fontweight='bold', va='center')
        
        # Etykiety wartości na paskach
        for bar, score in zip(bars, df_plot['score']):
            width = bar.get_width()
            label_text = f"{score*100:.1f}%"
            # Jeśli pasek krótki (squeeze), tekst obok paska
            if score < 0.2:
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, label_text, 
                        va='center', color='white', fontweight='bold', fontsize=9)
            else:
                # Jeśli długi, w środku
                ax.text(width - 0.05, bar.get_y() + bar.get_height()/2, label_text, 
                        va='center', ha='right', color='black', fontweight='bold', fontsize=9)

        ax.set_title("SKANER ALTCOINÓW: Kto zaraz wybuchnie?", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Poziom Naciągnięcia Sprężyny (0% = Max Squeeze)', color=t['text'])
        
        # Formatowanie osi X na procenty
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
        ax.set_xlim(0, 1.0) # Skala 0-100%
        
        ax.grid(True, axis='x', alpha=0.15, color=t['grid'])
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text'])
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        
        return fig

    # --- NOWOŚĆ: ETH DETEKTOR WYBUCHU (Obliczenia) ---
    def get_eth_volatility_squeeze_data(self):
        """
        Oblicza Squeeze specyficznie dla Ethereum (ETH-USD).
        Nie możemy używać danych BTC, bo ETH ma swój własny cykl zmienności.
        """
        try:
            # Pobieramy dane dla ETH
            df = yf.download('ETH-USD', period="1y", progress=False)
            
            # Bezpieczne wyciągnięcie Close
            if isinstance(df.columns, pd.MultiIndex):
                try: data = df['Close']
                except: data = df
            else:
                data = df['Close'] if 'Close' in df.columns else df

            # Kopia do obliczeń
            df_eth = pd.DataFrame()
            df_eth['Close'] = data['ETH-USD'] if 'ETH-USD' in data.columns else data.iloc[:, 0]
            
            # 1. Wstęgi Bollingera (20, 2)
            df_eth['SMA_20'] = df_eth['Close'].rolling(window=20).mean()
            df_eth['StdDev'] = df_eth['Close'].rolling(window=20).std()
            df_eth['Upper'] = df_eth['SMA_20'] + (df_eth['StdDev'] * 2)
            df_eth['Lower'] = df_eth['SMA_20'] - (df_eth['StdDev'] * 2)
            
            # 2. Szerokość (Squeeze Indicator)
            df_eth['BB_Width'] = (df_eth['Upper'] - df_eth['Lower']) / df_eth['SMA_20']
            
            # 3. Próg Wybuchu (Dla ETH może być inny niż dla BTC, więc liczymy dynamicznie)
            # ETH jest bardziej zmienne, więc próg "ciszy" może być wyżej.
            dynamic_threshold = df_eth['BB_Width'].tail(180).quantile(0.20)
            threshold = max(dynamic_threshold, 0.12) # ETH rzadko schodzi poniżej 0.12
            
            return df_eth, threshold
            
        except Exception as e:
            print(f"Błąd ETH Squeeze: {e}")
            return None, 0

    # --- NOWOŚĆ: ETH DETEKTOR WYBUCHU (Wizualizacja) ---
    def plot_eth_volatility_squeeze(self, data, threshold):
        """
        Rysuje Squeeze dla ETH z ceną ETH.
        """
        if data is None or data.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        
        # --- OŚ 1 (LEWA): SQUEEZE (Fioletowa linia dla odróżnienia od BTC) ---
        current_width = data['BB_Width'].iloc[-1]
        
        # Linia zmienności
        line1, = ax1.plot(data.index, data['BB_Width'], color='#d500f9', linewidth=2, label='Naciąg Sprężyny ETH')
        
        # Próg (Czerwona przerywana)
        line2 = ax1.axhline(threshold, color='#ff0055', linestyle='--', linewidth=2, label=f'Próg Wybuchu ({threshold:.3f})')
        
        # Wypełnienie strefy
        ax1.fill_between(data.index, 0, threshold, color='#ff0055', alpha=0.15)
        
        ax1.set_ylabel('Szerokość Wstęgi (ETH)', color='#d500f9', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#d500f9', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        # --- OŚ 2 (PRAWA): CENA ETH (Srebrna/Biała) ---
        ax2 = ax1.twinx()
        line3, = ax2.plot(data.index, data['Close'], color='#e0e0e0', linewidth=1, alpha=0.6, linestyle='-', label='Cena ETH ($)')
        
        ax2.set_ylabel('Cena ETH ($)', color='#e0e0e0', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#e0e0e0', colors=t['text'])
        
        # Kosmetyka osi
        ax2.spines['top'].set_visible(False); ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax2.spines['right'].set_color(t['text']); ax2.spines['left'].set_visible(False)

        # Logika Statusu
        if current_width <= threshold:
            status = "⚠️ ETH SQUEEZE! (Gotowy do ruchu)"
            status_color = '#ff0055'
            ax1.scatter(data.index[-1], current_width, s=150, color='red', zorder=10, edgecolors='white')
        else:
            status = "Luz (Czekamy)"
            status_color = '#d500f9'

        ax1.set_title(f"ETH DETEKTOR: {status}", fontsize=16, color=status_color, fontweight='bold')
        
        # Legenda
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        ax1.grid(True, alpha=0.15, color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg']); ax2.set_facecolor(t['bg'])
        ax2.patch.set_alpha(0.0)
        
        return fig

    # --- NOWOŚĆ: SEZONOWOŚĆ (Metoda GET) ---
    def get_seasonality_data(self, symbol='BTC-USD'):
        """
        Pobiera historię i przetwarza ją na macierz: Rok x Miesiąc.
        Oblicza miesięczne zwroty procentowe.
        """
        try:
            # 1. Pobieramy całą historię
            df = yf.download(symbol, period="max", progress=False)
            
            # Obsługa formatów yfinance
            if isinstance(df.columns, pd.MultiIndex):
                close = df['Close'][symbol] if symbol in df['Close'].columns else df['Close'].iloc[:, 0]
            else:
                close = df['Close']
            
            # 2. Resampling do miesięcy (Konied miesiąca 'ME')
            # Obliczamy zmianę procentową
            monthly_returns = close.resample('ME').last().pct_change() * 100
            
            # 3. Tworzymy tabelę przestawną (Pivot)
            seasonality = pd.DataFrame()
            seasonality['Year'] = monthly_returns.index.year
            seasonality['Month'] = monthly_returns.index.month
            seasonality['Return'] = monthly_returns.values
            
            # Pivot: Wiersze = Lata, Kolumny = Miesiące
            heatmap_data = seasonality.pivot_table(index='Year', columns='Month', values='Return')
            
            # Sortujemy lata od najnowszych
            heatmap_data = heatmap_data.sort_index(ascending=False)
            
            # Zmieniamy numery miesięcy na nazwy
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                           7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            heatmap_data.columns = [month_names.get(c, c) for c in heatmap_data.columns]
            
            return heatmap_data, symbol
            
        except Exception as e:
            print(f"Błąd get_seasonality: {e}")
            return None, symbol

    # --- NOWOŚĆ: SEZONOWOŚĆ (Metoda PLOT) ---
    def plot_seasonality_heatmap(self, heatmap_data, symbol):
        """
        Rysuje mapę ciepła (Heatmap) używając biblioteki Seaborn.
        """
        if heatmap_data is None or heatmap_data.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        
        # Kolory: Czerwony (Spadek) -> Żółty (Zero) -> Zielony (Wzrost)
        try:
            sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", center=0, 
                        cbar=False, ax=ax, linewidths=1, linecolor=t['bg'],
                        annot_kws={"size": 9, "weight": "bold"})
            
            # Dodajemy "%" do liczb i poprawiamy kontrast
            for text in ax.texts:
                try:
                    val = float(text.get_text())
                    text.set_text(f"{val:+.1f}%")
                    # Jeśli wartość jest skrajna (>20% lub <-20%), zmień kolor tekstu na biały/czarny dla kontrastu
                    if abs(val) > 25: text.set_color('white')
                    else: text.set_color('black')
                except: pass
        except NameError:
            ax.text(0.5, 0.5, "Brak biblioteki 'seaborn'.\nZainstaluj: pip install seaborn", 
                    ha='center', va='center', color='white')

        ax.set_title(f"KALENDARZ ZYSKÓW: {symbol.replace('-USD','')}", fontsize=18, color=t['text'], fontweight='bold', pad=15)
        
        # Opisy osi
        ax.set_ylabel('Rok', color=t['text'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Miesiąc', color=t['text'], fontsize=12, fontweight='bold')
        
        # Kolory osi
        ax.tick_params(axis='x', colors=t['text'], rotation=0)
        ax.tick_params(axis='y', colors=t['text'], rotation=0)
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        return fig

    # --- NOWOŚĆ: DRAWDOWN SCANNER (Dane) ---
    def get_ath_drawdown_data(self):
        """
        Pobiera dane historyczne (MAX) i oblicza spadek od ATH (All-Time High).
        Zwraca posortowany ranking 'Bólu'.
        """
        # Lista coinów do sprawdzenia (Mix: Stare, Nowe, DeFi, L2)
        coins = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD',
            'AVAX-USD', 'DOT-USD', 'LINK-USD', 'MATIC-USD', 'SHIB-USD', 'LTC-USD', 'UNI-USD',
            'ATOM-USD', 'ICP-USD', 'FIL-USD', 'NEAR-USD', 'AAVE-USD', 'QNT-USD', 'ALGO-USD',
            'SAND-USD', 'MANA-USD', 'AXS-USD', 'EOS-USD', 'NEO-USD', 'IOTA-USD', 'ZEC-USD',
            'DASH-USD', 'BCH-USD', 'MKR-USD', 'SNX-USD', 'CRV-USD', 'GRT-USD', 'FTM-USD',
            'ARB-USD', 'OP-USD', 'APT-USD', 'SUI-USD', 'TIA-USD', 'RNDR-USD', 'PEPE-USD'
        ]
        
        ranking = []
        
        try:
            # Pobieramy MAX historię, żeby znaleźć prawdziwy szczyt
            # period="max" jest kluczowy
            df = yf.download(coins, period="max", progress=False)
            
            # Bezpieczne wyciąganie Close
            if isinstance(df.columns, pd.MultiIndex):
                try: closes = df['Close']
                except KeyError: closes = df
            else:
                closes = df['Close'] if 'Close' in df.columns else df
            
            for ticker in coins:
                if ticker in closes.columns:
                    series = closes[ticker].dropna()
                    if len(series) < 10: continue
                    
                    # 1. Znajdź ATH (Najwyższa cena w historii)
                    ath_price = series.max()
                    
                    # 2. Aktualna cena
                    current_price = series.iloc[-1]
                    
                    # 3. Oblicz Drawdown (%)
                    # Wynik będzie ujemny (np. -85.5)
                    drawdown = ((current_price - ath_price) / ath_price) * 100
                    
                    name = ticker.replace('-USD', '')
                    
                    ranking.append({
                        'coin': name,
                        'drawdown': drawdown,
                        'price': current_price,
                        'ath': ath_price
                    })
            
            # Tworzymy DataFrame
            rank_df = pd.DataFrame(ranking)
            # Sortujemy: Od największego bólu (-99%) do najmniejszego (-5%)
            rank_df = rank_df.sort_values(by='drawdown', ascending=True)
            
            return rank_df
            
        except Exception as e:
            print(f"Błąd drawdown: {e}")
            return None

    # --- NOWOŚĆ: DRAWDOWN SCANNER (Wykres) ---
    def plot_ath_drawdown(self, df):
        """
        Rysuje poziomy wykres słupkowy pokazujący odległość od ATH.
        """
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        # Wysoki wykres, żeby pomieścić wszystkie coiny
        fig = plt.figure(figsize=(10, 12)) 
        ax = fig.add_subplot(111)
        
        # Kolory w zależności od bólu
        # Ciemny Czerwony = Śmierć (-90% i gorzej)
        # Czerwony = Bessa (-50% do -90%)
        # Żółty = Korekta (-20% do -50%)
        # Zielony = Blisko szczytu (>-20%)
        colors = []
        for d in df['drawdown']:
            if d <= -90: colors.append('#8b0000') # Dark Red
            elif d <= -50: colors.append('#ff0055') # Red
            elif d <= -20: colors.append('#ffeb3b') # Yellow
            else: colors.append('#00ff55') # Green
            
        # Wykres poziomy (barh)
        bars = ax.barh(df['coin'], df['drawdown'], color=colors, alpha=0.8)
        
        # Etykiety wartości przy słupkach
        for bar, val in zip(bars, df['drawdown']):
            width = bar.get_width() # To jest liczba ujemna, np. -80
            
            # Tekst po lewej stronie słupka (dla czytelności)
            label_text = f"{val:.1f}%"
            ax.text(width - 2, bar.get_y() + bar.get_height()/2, label_text, 
                    va='center', ha='right', color='white', fontsize=9, fontweight='bold')

        ax.set_title("SKANER BÓLU: Odległość od ATH (All-Time High)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Procent spadku od szczytu', color=t['text'])
        
        # Oś X (od -100 do 0)
        ax.set_xlim(-105, 5) 
        
        ax.grid(True, axis='x', alpha=0.15, color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False) # Ukrywamy oś Y, bo mamy etykiety
        
        # Odwracamy oś Y, żeby te z największym spadkiem były na dole (lub górze, zależy od preferencji)
        # Tutaj: Największy ból na dole listy
        
        return fig

    # --- NOWOŚĆ: RSI HEATMAP (Obliczenia) ---
    def get_rsi_heatmap_data(self):
        """
        Pobiera dane dla 40 topowych coinów i oblicza RSI (14).
        Zwraca dane przygotowane do wyświetlenia w siatce (Heatmap).
        """
        coins = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD',
            'DOT-USD', 'LINK-USD', 'MATIC-USD', 'LTC-USD', 'UNI-USD', 'ATOM-USD', 'ETC-USD', 'XLM-USD',
            'FIL-USD', 'HBAR-USD', 'LDO-USD', 'ARB-USD', 'OP-USD', 'APT-USD', 'RNDR-USD', 'NEAR-USD',
            'INJ-USD', 'STX-USD', 'IMX-USD', 'GRT-USD', 'SNX-USD', 'AAVE-USD', 'ALGO-USD', 'QNT-USD',
            'EOS-USD', 'SAND-USD', 'MANA-USD', 'THETA-USD', 'FTM-USD', 'AXS-USD', 'NEO-USD', 'FLOW-USD'
        ]
        
        rsi_data = []
        
        try:
            # Pobieramy krótki okres, bo potrzebujemy tylko RSI z dzisiaj
            df_all = yf.download(coins, period="3mo", progress=False)
            
            # Bezpieczne wyciąganie Close
            if isinstance(df_all.columns, pd.MultiIndex):
                try: closes = df_all['Close']
                except KeyError: closes = df_all
            else:
                closes = df_all['Close'] if 'Close' in df_all.columns else df_all
            
            for ticker in coins:
                if ticker in closes.columns:
                    series = closes[ticker].dropna()
                    if len(series) < 15: continue
                    
                    # Obliczanie RSI (14)
                    delta = series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    current_rsi = rsi.iloc[-1]
                    name = ticker.replace('-USD', '')
                    
                    rsi_data.append({'coin': name, 'rsi': current_rsi})
            
            # Tworzymy DataFrame
            df_rsi = pd.DataFrame(rsi_data)
            df_rsi = df_rsi.sort_values(by='rsi', ascending=True) # Od najniższego RSI
            
            return df_rsi
            
        except Exception as e:
            print(f"Błąd RSI heatmap: {e}")
            return None

    # --- NOWOŚĆ: RSI HEATMAP (Wizualizacja Siatki) ---
    def plot_rsi_heatmap_grid(self, df):
        """
        Rysuje kafelki z RSI. 
        Zielone (<30) = Wyprzedane (Okazja?)
        Czerwone (>70) = Wykupione (Ryzyko?)
        """
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        
        # Musimy ułożyć dane w macierz (np. 5 wierszy x 8 kolumn)
        # Żeby heatmapa była prostokątem
        n_coins = len(df)
        cols = 8 # Liczba kolumn
        rows = (n_coins // cols) + 1
        
        # Dopełniamy pustymi danymi, żeby macierz była pełna
        matrix_rsi = np.full((rows, cols), np.nan)
        matrix_labels = np.full((rows, cols), "", dtype=object)
        
        # Sortujemy tak, żeby skrajne wartości były widoczne (np. najmniejsze RSI na początku)
        # Tutaj df jest już posortowane rosnąco
        
        for i in range(n_coins):
            r = i // cols
            c = i % cols
            
            rsi_val = df.iloc[i]['rsi']
            coin_name = df.iloc[i]['coin']
            
            matrix_rsi[r, c] = rsi_val
            matrix_labels[r, c] = f"{coin_name}\n{rsi_val:.1f}"

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Customowa mapa kolorów (Zielony -> Szary -> Czerwony)
        # RSI 0-30 (Zielony), 30-70 (Szary), 70-100 (Czerwony)
        # Używamy tricku z 'vlag' lub innej palety, ale najlepiej zdefiniować własną logicznie
        import seaborn as sns
        
        # Tworzymy Heatmapę
        # vmin=20, vmax=80 ustawia kontrast (poniżej 20 mega zielony, powyżej 80 mega czerwony)
        sns.heatmap(matrix_rsi, annot=matrix_labels, fmt="", cmap="RdYlGn_r", 
                    vmin=20, vmax=80, center=50,
                    ax=ax, linewidths=2, linecolor=t['bg'], cbar=False,
                    annot_kws={"size": 9, "weight": "bold"})
        
        # Usuwamy osie
        ax.axis('off')
        ax.set_title("RSI HEATMAP: Polowanie na dołki (Zielone < 30)", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda tekstowa
        plt.figtext(0.5, 0.02, "CIEMNA ZIELEŃ = Wyprzedanie (Szukaj Dywergencji!)  |  CZERWONY = Wykupienie (Uważaj)", 
                    ha="center", fontsize=10, color=t['text'], fontweight='bold')

        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        return fig

    # --- UPDATE: MARKET BREADTH + BTC PRICE (Wariograf z Ceną) ---
    def get_market_breadth_data(self):
        """
        Oblicza 'Szerokość Rynku' ORAZ pobiera cenę Bitcoina.
        Zwraca dwa zestawy danych: Wskaźnik Breadth i Cenę BTC.
        """
        # Lista Top 30 Coinów
        coins = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'DOGE-USD',
            'DOT-USD', 'LINK-USD', 'TRX-USD', 'MATIC-USD', 'LTC-USD', 'SHIB-USD', 'UNI-USD',
            'ATOM-USD', 'XLM-USD', 'ETC-USD', 'FIL-USD', 'HBAR-USD', 'APT-USD', 'ICP-USD',
            'NEAR-USD', 'LDO-USD', 'ARB-USD', 'OP-USD', 'RNDR-USD', 'INJ-USD', 'STX-USD', 'QNT-USD'
        ]
        
        try:
            # Pobieramy historię (1 rok)
            df_all = yf.download(coins, period="1y", progress=False)
            
            # Bezpieczne wyciąganie Close
            if isinstance(df_all.columns, pd.MultiIndex):
                try: closes = df_all['Close']
                except KeyError: closes = df_all
            else:
                closes = df_all['Close'] if 'Close' in df_all.columns else df_all
            
            # --- 1. OBLICZANIE BREADTH ---
            above_sma = pd.DataFrame(index=closes.index)
            valid_coins_count = 0
            
            for ticker in coins:
                if ticker in closes.columns:
                    series = closes[ticker].dropna()
                    if len(series) < 50: continue
                    
                    sma_50 = series.rolling(window=50).mean()
                    above_sma[ticker] = (series > sma_50).astype(int)
                    valid_coins_count += 1
            
            if valid_coins_count == 0: return None, None
            
            breadth = above_sma.mean(axis=1) * 100
            breadth_smoothed = breadth.rolling(3).mean().dropna()
            
            # --- 2. WYCIĄGANIE CENY BTC ---
            # Musimy dopasować indeksy (daty)
            if 'BTC-USD' in closes.columns:
                btc_price = closes['BTC-USD'].reindex(breadth_smoothed.index)
            else:
                btc_price = None

            return breadth_smoothed, btc_price
            
        except Exception as e:
            print(f"Błąd Market Breadth: {e}")
            return None, None

    # --- UPDATE: MARKET BREADTH PLOT (Dual Axis) ---
    def plot_market_breadth(self, breadth_data, price_data):
        """
        Rysuje wskaźnik szerokości rynku (Lewa Oś) ORAZ cenę Bitcoina (Prawa Oś).
        Pozwala wykrywać dywergencje między ceną a siłą rynku.
        """
        if breadth_data is None or breadth_data.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        
        # --- OŚ 1 (LEWA): MARKET BREADTH ---
        # Rysujemy linię wariografu
        line1, = ax1.plot(breadth_data.index, breadth_data, color='#00e5ff', linewidth=2, label='% Rynku > SMA50 (Lewa)')
        
        # Strefy
        ax1.axhline(80, color='#ff0055', linestyle='--', alpha=0.3)
        ax1.fill_between(breadth_data.index, 80, 100, color='#ff0055', alpha=0.10)
        
        ax1.axhline(20, color='#00ff55', linestyle='--', alpha=0.3)
        ax1.fill_between(breadth_data.index, 0, 20, color='#00ff55', alpha=0.10)
        
        ax1.set_ylabel('% Monet w Trendzie', color='#00e5ff', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#00e5ff', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        ax1.set_ylim(0, 100)
        
        # --- OŚ 2 (PRAWA): CENA BTC ---
        if price_data is not None:
            ax2 = ax1.twinx()
            line2, = ax2.plot(price_data.index, price_data, color='#ffd700', linewidth=1.5, linestyle='-', alpha=0.8, label='Cena BTC (Prawa)')
            ax2.set_ylabel('Cena BTC ($)', color='#ffd700', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#ffd700', colors=t['text'])
            
            # Ukrywamy ramki drugiej osi
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_color(t['text'])
            ax2.spines['bottom'].set_visible(False)
            ax2.patch.set_alpha(0.0)
        
        # Tytuł i Status
        last_val = breadth_data.iloc[-1]
        if last_val > 80: status = "PRZEGRZANIE (Ryzyko)"
        elif last_val < 20: status = "WYPRZEDANIE (Okazja)"
        elif last_val > 50: status = "TREND WZROSTOWY"
        else: status = "SŁABOŚĆ RYNKU"
        
        ax1.set_title(f"WARIOGRAF RYNKU + BTC: {status} ({last_val:.1f}%)", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda (Scalona)
        lines = [line1]
        if price_data is not None: lines.append(line2)
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        ax1.grid(True, alpha=0.15, color=t['grid'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- NOWOŚĆ: RRG (Relative Rotation Graph) - Obliczenia ---
    def get_rrg_data(self):
        """
        Oblicza dane do wykresu rotacji kapitału (RRG).
        Porównuje siłę Altcoinów względem BTC (Benchmark).
        Używa normalizacji (Z-Score), aby umieścić duże i małe coiny na jednym wykresie.
        """
        # Koszyk reprezentatywny (L1, L2, DeFi, Meme)
        # Porównujemy je do BTC, więc BTC nie ma na liście (jest punktem odniesienia 0,0)
        coins = [
            'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'DOGE-USD',
            'DOT-USD', 'LINK-USD', 'MATIC-USD', 'LTC-USD', 'UNI-USD', 'ATOM-USD', 'ARB-USD',
            'OP-USD', 'NEAR-USD', 'APT-USD', 'RNDR-USD', 'INJ-USD', 'SHIB-USD', 'PEPE-USD',
            'FET-USD', 'TIA-USD', 'SUI-USD', 'ONDO-USD'
        ]
        
        rrg_points = []
        
        try:
            # 1. Pobieramy dane (Alty + BTC)
            tickers_to_download = coins + ['BTC-USD']
            df_all = yf.download(tickers_to_download, period="6mo", progress=False)
            
            # Bezpieczne wyciąganie Close
            if isinstance(df_all.columns, pd.MultiIndex):
                try: closes = df_all['Close']
                except KeyError: closes = df_all
            else:
                closes = df_all['Close'] if 'Close' in df_all.columns else df_all
                
            if 'BTC-USD' not in closes.columns: return None

            btc_series = closes['BTC-USD']
            
            # 2. Pętla po Altach
            for ticker in coins:
                if ticker in closes.columns:
                    alt_series = closes[ticker].dropna()
                    if len(alt_series) < 60: continue
                    
                    # Wyrównanie dat
                    common_index = alt_series.index.intersection(btc_series.index)
                    alt_aligned = alt_series.loc[common_index]
                    btc_aligned = btc_series.loc[common_index]
                    
                    # --- MATEMATYKA RRG ---
                    # A. Siła Relatywna (RS) = Cena Alta / Cena BTC
                    rs = alt_aligned / btc_aligned
                    
                    # B. Normalizacja (JdK RS-Ratio - uproszczone)
                    # Używamy średniej i odchylenia z 20 dni, żeby sprowadzić wszystko do wspólnej skali
                    mean = rs.rolling(window=20).mean()
                    std = rs.rolling(window=20).std()
                    z_score_ratio = (rs - mean) / std # To jest Oś X
                    
                    # C. Momentum Siły Relatywnej (JdK RS-Momentum)
                    # Jak szybko zmienia się ten stosunek? (ROC z 10 dni na Z-Score)
                    momentum = z_score_ratio.diff(5) * 10 # To jest Oś Y (skalowane)
                    
                    # Bierzemy ostatnie wartości
                    last_x = z_score_ratio.iloc[-1]
                    last_y = momentum.iloc[-1]
                    
                    # Filtrujemy błędy (NaN)
                    if pd.isna(last_x) or pd.isna(last_y): continue
                    
                    name = ticker.replace('-USD', '')
                    rrg_points.append({'coin': name, 'x': last_x, 'y': last_y})
            
            return pd.DataFrame(rrg_points)

        except Exception as e:
            print(f"Błąd RRG: {e}")
            return None

    # --- NOWOŚĆ: RRG (Wykres Ćwiartek) ---
    def plot_rrg_chart(self, df):
        """
        Rysuje wykres RRG z 4 ćwiartkami:
        1. Leading (Zielona) - Silne i rosną
        2. Weakening (Żółta) - Silne ale słabną
        3. Lagging (Czerwona) - Słabe i spadają
        4. Improving (Niebieska) - Słabe ale rosną
        """
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 10)) # Kwadratowy wykres
        ax = fig.add_subplot(111)
        
        # --- RYSOWANIE TŁA (ĆWIARTKI) ---
        # Ustawiamy limity osi dynamicznie lub na sztywno (np. -3 do 3)
        limit = max(df['x'].abs().max(), df['y'].abs().max()) + 0.5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # 1. Prawa Górna (Leading) - Zielona
        ax.fill_between([0, limit], 0, limit, color='#00ff55', alpha=0.1)
        ax.text(limit/2, limit*0.9, "LEADING (Liderzy)\nSilne vs BTC + Rosną", 
                ha='center', color='#00ff55', fontweight='bold', alpha=0.7)
        
        # 2. Prawa Dolna (Weakening) - Żółta
        ax.fill_between([0, limit], -limit, 0, color='#ffeb3b', alpha=0.1)
        ax.text(limit/2, -limit*0.9, "WEAKENING (Słabnące)\nRealizacja Zysków", 
                ha='center', color='#ffeb3b', fontweight='bold', alpha=0.7)
        
        # 3. Lewa Dolna (Lagging) - Czerwona
        ax.fill_between([-limit, 0], -limit, 0, color='#ff0055', alpha=0.1)
        ax.text(-limit/2, -limit*0.9, "LAGGING (Maruderzy)\nSłabe vs BTC", 
                ha='center', color='#ff0055', fontweight='bold', alpha=0.7)
        
        # 4. Lewa Górna (Improving) - Niebieska
        ax.fill_between([-limit, 0], 0, limit, color='#00e5ff', alpha=0.1)
        ax.text(-limit/2, limit*0.9, "IMPROVING (Odbijające)\nŁapanie Dołka", 
                ha='center', color='#00e5ff', fontweight='bold', alpha=0.7)
        
        # Osie środkowe
        ax.axhline(0, color=t['text'], linestyle='-', linewidth=1)
        ax.axvline(0, color=t['text'], linestyle='-', linewidth=1)
        
        # --- PUNKTY (SCATTER) ---
        # Kolorujemy punkty zależnie od ćwiartki
        colors = []
        for index, row in df.iterrows():
            if row['x'] > 0 and row['y'] > 0: colors.append('#00ff55') # Leading
            elif row['x'] > 0 and row['y'] < 0: colors.append('#ffeb3b') # Weakening
            elif row['x'] < 0 and row['y'] < 0: colors.append('#ff0055') # Lagging
            else: colors.append('#00e5ff') # Improving

        ax.scatter(df['x'], df['y'], c=colors, s=100, edgecolors='white', zorder=10)
        
        # Podpisy coinów
        import adjustText # Opcjonalnie, ale matplotlib ma problem z nakładaniem
        texts = []
        for i, txt in enumerate(df['coin']):
            # Prosty tekst z lekkim przesunięciem
            ax.text(df['x'].iloc[i]+0.1, df['y'].iloc[i]+0.1, txt, color='white', fontsize=9, fontweight='bold')

        ax.set_title("RRG: Mapa Przepływu Kapitału (vs BTC)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Siła Relatywna vs BTC (Prawo = Silniejszy)', color=t['text'])
        ax.set_ylabel('Momentum (Góra = Przyspiesza)', color=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        ax.tick_params(colors=t['text'])
        
        return fig

    # --- NOWOŚĆ: HURST EXPONENT (Quant Tool) - Obliczenia ---
    def get_hurst_ranking(self):
        """
        Oblicza Wykładnik Hursta (H) dla topowych aktywów.
        Metoda: Analiza przeskalowanego zakresu (R/S) lub wariancji.
        Mówi nam, czy rynek trenduje (H > 0.5) czy wraca do średniej (H < 0.5).
        """
        coins = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD',
            'DOGE-USD', 'LINK-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD', 'UNI-USD', 'ATOM-USD',
            'ARB-USD', 'OP-USD', 'TIA-USD', 'RNDR-USD', 'INJ-USD', 'NEAR-USD', 'APT-USD',
            'SUI-USD', 'SEI-USD', 'PEPE-USD', 'WIF-USD', 'ONDO-USD', 'FET-USD'
        ]
        
        ranking = []
        
        try:
            # Potrzebujemy dłuższego okresu dla statystycznej istotności (np. 1 rok)
            df_all = yf.download(coins, period="1y", progress=False)
            
            # Bezpieczne wyciąganie Close
            if isinstance(df_all.columns, pd.MultiIndex):
                try: closes = df_all['Close']
                except KeyError: closes = df_all
            else:
                closes = df_all['Close'] if 'Close' in df_all.columns else df_all
            
            for ticker in coins:
                if ticker in closes.columns:
                    series = closes[ticker].dropna()
                    # Minimum 100 dni danych
                    if len(series) < 100: continue
                    
                    # --- MATEMATYKA HURSTA (Uproszczona metoda wariancji) ---
                    # H = nachylenie wykresu log-log zmienności w funkcji opóźnienia (lag)
                    lags = range(2, 20)
                    tau = []
                    
                    # Obliczamy odchylenie standardowe różnic dla różnych opóźnień (lags)
                    # Variogram approach
                    for lag in lags:
                        # Różnica ceny po 'lag' dniach
                        diff = series.diff(lag).dropna()
                        if len(diff) == 0: continue
                        tau.append(np.std(diff))
                    
                    if len(tau) != len(lags): continue

                    # Regresja liniowa na logarytmach
                    # log(std) ~ H * log(lag)
                    x = np.log(lags)
                    y = np.log(tau)
                    
                    # Polyfit zwraca [slope, intercept]
                    # Slope (nachylenie) to nasz Wykładnik Hursta
                    poly = np.polyfit(x, y, 1)
                    H = poly[0]
                    
                    name = ticker.replace('-USD', '')
                    
                    # Status słowny
                    if H > 0.60: state = "SILNY TREND"
                    elif H > 0.50: state = "Lekki Trend"
                    elif H > 0.40: state = "Losowy / Szum"
                    else: state = "PING-PONG (Mean Revert)"

                    ranking.append({'coin': name, 'hurst': H, 'status': state})
            
            # DataFrame i sortowanie (Najsilniejszy trend na górze)
            rank_df = pd.DataFrame(ranking)
            rank_df = rank_df.sort_values(by='hurst', ascending=False)
            
            return rank_df

        except Exception as e:
            print(f"Błąd Hurst: {e}")
            return None

    # --- NOWOŚĆ: HURST EXPONENT (Wykres) ---
    def plot_hurst_ranking(self, df):
        """
        Rysuje ranking Wykładnika Hursta.
        Strefy kolorystyczne pokazują naturę rynku.
        """
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Kolory słupków zależne od wartości H
        colors = []
        for h in df['hurst']:
            if h >= 0.60: colors.append('#00ff55') # Silny Trend (Zielony)
            elif h >= 0.50: colors.append('#ccff90') # Słaby Trend (Jasny zielony)
            elif h >= 0.45: colors.append('#bdbdbd') # Szum / Losowość (Szary)
            else: colors.append('#ff0055') # Mean Reversion (Czerwony - uwaga na wybicia)
            
        # Wykres poziomy
        # Odwracamy df do rysowania, żeby najlepsze były na górze
        df_plot = df.iloc[::-1]
        c_plot = colors[::-1]
        
        bars = ax.barh(df_plot['coin'], df_plot['hurst'], color=c_plot, alpha=0.8)
        
        # Linie graniczne
        ax.axvline(0.5, color='white', linestyle='--', alpha=0.5, label='0.5 (Losowość / Rzut monetą)')
        ax.axvline(0.6, color='#00ff55', linestyle=':', alpha=0.5, label='> 0.6 (Silny Trend)')
        
        # Opisy na słupkach
        for bar, h, status in zip(bars, df_plot['hurst'], df_plot['status']):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{h:.2f} ({status})", 
                    va='center', color='white', fontsize=8, fontweight='bold')

        ax.set_title("HURST EXPONENT: Jakość Trendu", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Wartość H (0.5 = Chaos, >0.5 = Trend, <0.5 = Konsolidacja)', color=t['text'])
        ax.set_xlim(0.3, 0.8) # Skupiamy się na istotnym zakresie
        
        # Legenda stref (tekstowa na dole)
        plt.figtext(0.5, 0.02, "ZIELONE = Trenduje (Follow the Trend)  |  CZERWONE = Wraca do średniej (Kupuj dołki w konsolidacji)", 
                    ha="center", fontsize=9, color=t['text'], fontweight='bold')

        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.tick_params(colors=t['text'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        
        return fig

    # --- 1. MONTE CARLO SIMULATION (Przyszłość) ---
    def get_monte_carlo_simulation(self):
        """
        Symuluje 50 możliwych ścieżek ceny BTC na najbliższe 30 dni
        używając geometrycznych ruchów Browna (GBM).
        """
        try:
            # Pobieramy dane
            df = yf.download('BTC-USD', period="1y", progress=False)
            close = df['Close']['BTC-USD'] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            
            # Parametry do symulacji
            last_price = close.iloc[-1]
            returns = close.pct_change().dropna()
            daily_vol = returns.std()
            daily_drift = returns.mean() - (0.5 * daily_vol**2) # Wzór Itō
            
            # Konfiguracja
            days_forward = 30
            simulations = 50
            
            simulation_df = pd.DataFrame()
            
            for i in range(simulations):
                # Generujemy losowe szoki
                daily_shocks = np.random.normal(0, 1, days_forward)
                price_path = [last_price]
                
                for x in daily_shocks:
                    # Wzór na GBM
                    next_price = price_path[-1] * np.exp(daily_drift + daily_vol * x)
                    price_path.append(next_price)
                
                simulation_df[f'Sim_{i}'] = price_path
                
            return simulation_df, last_price

        except Exception as e:
            print(f"Błąd Monte Carlo: {e}")
            return None, None

    def plot_monte_carlo(self, sim_df, last_price):
        if sim_df is None: return None
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Rysujemy wszystkie ścieżki
        for col in sim_df.columns:
            ax.plot(sim_df.index, sim_df[col], color='#00e5ff', alpha=0.15, linewidth=1)
            
        # Rysujemy średnią ścieżkę (Oczekiwaną)
        mean_path = sim_df.mean(axis=1)
        ax.plot(mean_path.index, mean_path, color='#ffd700', linewidth=3, linestyle='--', label='Średnia Prognoza')
        
        # Statystyka końcowa
        final_prices = sim_df.iloc[-1]
        prob_up = (final_prices > last_price).mean() * 100
        
        ax.set_title(f"MONTE CARLO: 30-dniowa Prognoza (Szansa Wzrostu: {prob_up:.0f}%)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_ylabel('Symulowana Cena ($)', color=t['text'])
        ax.set_xlabel('Dni w przyszłość', color=t['text'])
        
        # Start Price Line
        ax.axhline(last_price, color='#ff0055', linestyle=':', label='Cena Startowa')
        
        ax.legend(facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.grid(True, alpha=0.1, color=t['grid'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        
        return fig

    # --- 2. SHARPE RATIO SCANNER (Jakość Inwestycji) ---
    def get_sharpe_ranking(self):
        """
        Oblicza wskaźnik Sharpe'a dla Top Coinów.
        Sharpe = (Zwrot - RiskFree) / Zmienność.
        Mówi: Który coin daje najwięcej zysku na jednostkę strachu.
        """
        coins = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 
            'AVAX-USD', 'DOGE-USD', 'LINK-USD', 'MATIC-USD', 'LTC-USD', 'UNI-USD',
            'ARB-USD', 'OP-USD', 'RNDR-USD', 'INJ-USD', 'PEPE-USD', 'ONDO-USD'
        ]
        
        ranking = []
        try:
            df = yf.download(coins, period="6mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                closes = df['Close'] if 'Close' in df.columns else df
            else:
                closes = df['Close']
            
            for ticker in coins:
                if ticker in closes.columns:
                    series = closes[ticker].dropna()
                    if len(series) < 30: continue
                    
                    # Obliczenia roczne
                    returns = series.pct_change()
                    mean_return = returns.mean() * 252 # Roczny zwrot
                    std_dev = returns.std() * np.sqrt(252) # Roczna zmienność
                    
                    if std_dev == 0: continue
                    
                    # Sharpe Ratio (zakładamy Risk Free Rate = 4% = 0.04)
                    sharpe = (mean_return - 0.04) / std_dev
                    
                    name = ticker.replace('-USD', '')
                    ranking.append({'coin': name, 'sharpe': sharpe, 'return': mean_return})
            
            rank_df = pd.DataFrame(ranking)
            rank_df = rank_df.sort_values(by='sharpe', ascending=False)
            return rank_df

        except Exception as e:
            print(f"Błąd Sharpe: {e}")
            return None

    def plot_sharpe_ranking(self, df):
        if df is None: return None
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Kolory: Zielone (>2 - Wybitne), Żółte (1-2 - Dobre), Czerwone (<1 - Słabe)
        colors = []
        for s in df['sharpe']:
            if s >= 2.0: colors.append('#00ff55')
            elif s >= 1.0: colors.append('#ffeb3b')
            else: colors.append('#ff0055')
            
        df_plot = df.iloc[::-1] # Najlepsze na górze
        c_plot = colors[::-1]
        
        bars = ax.barh(df_plot['coin'], df_plot['sharpe'], color=c_plot, alpha=0.8)
        
        for bar, s in zip(bars, df_plot['sharpe']):
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{s:.2f}", 
                    va='center', color='white', fontsize=9, fontweight='bold')
            
        ax.set_title("SHARPE RATIO: Jakość Zysków", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Sharpe Ratio (Im wyżej tym bezpieczniejszy zysk)', color=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        ax.tick_params(colors=t['text'])
        return fig

    # --- 3. SPREAD Z-SCORE (Arbitraż BTC vs ETH) ---
    def get_btc_eth_spread(self):
        """
        Oblicza Z-Score spreadu między BTC a ETH.
        Wykrywa anomalie cenowe.
        """
        try:
            df = yf.download(['BTC-USD', 'ETH-USD'], period="1y", progress=False)['Close']
            
            # Obliczamy stosunek cen (Ile ETH kupisz za 1 BTC)
            ratio = df['BTC-USD'] / df['ETH-USD']
            
            # Obliczamy Z-Score (Odchylenie od średniej 20-dniowej)
            mean = ratio.rolling(window=30).mean()
            std = ratio.rolling(window=30).std()
            z_score = (ratio - mean) / std
            
            return z_score.dropna(), ratio
        except Exception as e:
            print(f"Błąd Spread: {e}")
            return None, None

    def plot_stat_arb_spread(self, z_score):
        if z_score is None: return None
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Wykres Z-Score
        ax.plot(z_score.index, z_score, color='#d500f9', linewidth=1.5, label='Spread BTC/ETH (Z-Score)')
        
        # Linie Sygnałowe (+2 / -2 odchylenia)
        ax.axhline(2.0, color='#ff0055', linestyle='--', label='Sprzedaj BTC / Kup ETH')
        ax.axhline(-2.0, color='#00ff55', linestyle='--', label='Kup BTC / Sprzedaj ETH')
        ax.axhline(0, color=t['text'], linestyle=':', alpha=0.3)
        
        # Wypełnienia
        ax.fill_between(z_score.index, 2.0, 4.0, color='#ff0055', alpha=0.1)
        ax.fill_between(z_score.index, -4.0, -2.0, color='#00ff55', alpha=0.1)
        
        last_val = z_score.iloc[-1]
        
        if last_val > 2.0: status = "BTC ZA DROGI (Short BTC/Long ETH)"
        elif last_val < -2.0: status = "BTC ZA TANI (Long BTC/Short ETH)"
        else: status = "W NORMIE (Brak Arbitrażu)"
        
        ax.set_title(f"ARBITRAŻ STATYSTYCZNY: {status}", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_ylabel('Odchylenie Standardowe (Sigma)', color=t['text'])
        ax.set_ylim(-3.5, 3.5)
        
        ax.legend(facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.grid(True, alpha=0.15, color=t['grid'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        return fig

    # --- NOWOŚĆ: PORTFOLIO OPTIMIZER (Markowitz Efficient Frontier) ---
    def get_portfolio_optimization(self):
        """
        Symuluje 5000 portfeli składających się z wybranych coinów.
        Znajduje 'Max Sharpe Portfolio' (Najlepszy stosunek zysku do ryzyka).
        To jest matematyka noblisty Harry'ego Markowitza.
        """
        # Lista składników portfela (Możesz tu dodać co chcesz)
        coins = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 
            'ADA-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD', 'DOGE-USD',
            'RNDR-USD', 'INJ-USD', 'FET-USD', 'ARB-USD', 'OP-USD', 'XLM-USD', 'ONDO-USD', 'ALGO-USD',
            'JASMY-USD', 'LUNC-USD'
        ]
        
        try:
            # 1. Pobieramy dane (1 rok)
            df = yf.download(coins, period="1y", progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                closes = df['Close'] if 'Close' in df.columns else df
            else:
                closes = df['Close']
            
            # Czyścimy puste
            closes = closes.dropna(axis=1, how='all').dropna()
            if closes.empty: return None, None, None
            
            # 2. Obliczamy dzienne zwroty logarytmiczne
            log_returns = np.log(closes / closes.shift(1))
            
            num_assets = len(closes.columns)
            num_portfolios = 5000 # Ilość symulacji
            
            # Tablice na wyniki
            p_ret = [] # Zwrot portfela
            p_vol = [] # Ryzyko (zmienność) portfela
            p_weights = [] # Wagi (skład)
            
            # Macierz kowariancji (jak aktywa korelują ze sobą)
            cov_matrix = log_returns.cov() * 252 
            
            # --- SYMULACJA MONTE CARLO ---
            for _ in range(num_portfolios):
                # Losowe wagi
                weights = np.random.random(num_assets)
                weights /= np.sum(weights) # Suma musi być 100% (1.0)
                p_weights.append(weights)
                
                # Oczekiwany zwrot roczny
                returns = np.sum(weights * log_returns.mean()) * 252
                p_ret.append(returns)
                
                # Oczekiwana zmienność (Ryzyko)
                var = np.dot(weights.T, np.dot(cov_matrix, weights))
                sd = np.sqrt(var)
                p_vol.append(sd)
            
            # Konwersja na array
            p_ret = np.array(p_ret)
            p_vol = np.array(p_vol)
            p_weights = np.array(p_weights)
            
            # 3. Szukamy Najlepszego Portfela (Max Sharpe Ratio)
            # Sharpe = (Zwrot - 0%) / Ryzyko
            sr = p_ret / p_vol
            max_sr_idx = sr.argmax()
            
            # Dane najlepszego portfela
            best_return = p_ret[max_sr_idx]
            best_vol = p_vol[max_sr_idx]
            best_weights = p_weights[max_sr_idx]
            
            # Tworzymy słownik z wynikami dla najlepszego składu
            allocation = {}
            for i, ticker in enumerate(closes.columns):
                w = best_weights[i]
                if w > 0.01: # Pokaż tylko te powyżej 1%
                    allocation[ticker.replace('-USD','')] = w
            
            # Sortujemy alokację
            allocation = dict(sorted(allocation.items(), key=lambda item: item[1], reverse=True))
            
            # Zwracamy wszystko co potrzebne do wykresu
            sim_data = {
                'volatility': p_vol,
                'returns': p_ret,
                'sharpe': sr
            }
            
            best_portfolio = {
                'return': best_return,
                'volatility': best_vol,
                'allocation': allocation
            }
            
            return sim_data, best_portfolio, closes.columns

        except Exception as e:
            print(f"Błąd Optimizer: {e}")
            return None, None, None

    # --- NOWOŚĆ: PORTFOLIO OPTIMIZER (Wykres + Pie Chart) ---
    def plot_efficient_frontier(self, sim_data, best_portfolio):
        """
        Rysuje Efektywną Granicę (Wykres punktowy) oraz
        Wykres Kołowy (Pie Chart) z idealnym podziałem portfela.
        """
        if sim_data is None: return None
        
        t = self.get_theme_colors()
        
        # Ustawiamy układ: Lewo (Scatter), Prawo (Pie Chart)
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        
        ax1 = fig.add_subplot(gs[0]) # Scatter
        ax2 = fig.add_subplot(gs[1]) # Pie
        
        # --- 1. EFFICIENT FRONTIER (Scatter) ---
        # Kolorujemy punkty wg Sharpe Ratio (im jaśniej tym lepiej)
        sc = ax1.scatter(sim_data['volatility'], sim_data['returns'], 
                         c=sim_data['sharpe'], cmap='viridis', s=10, alpha=0.5)
        
        # Zaznaczamy GWIAZDĄ najlepszy portfel
        ax1.scatter(best_portfolio['volatility'], best_portfolio['return'], 
                    c='#ff0055', s=300, marker='*', edgecolors='white', label='Max Sharpe (Idealny Portfel)')
        
        ax1.set_title("EFEKTYWNA GRANICA (5000 Symulacji)", fontsize=14, color=t['text'], fontweight='bold')
        ax1.set_xlabel('Ryzyko (Zmienność Roczna)', color=t['text'])
        ax1.set_ylabel('Oczekiwany Zwrot Roczny', color=t['text'])
        ax1.legend(facecolor=t['bg'], labelcolor=t['text'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        
        # Pasek kolorów
        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label('Sharpe Ratio (Jakość)', color=t['text'])
        cbar.ax.yaxis.set_tick_params(color=t['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=t['text'])

        # --- 2. OPTIMAL ALLOCATION (Pie Chart) ---
        labels = list(best_portfolio['allocation'].keys())
        sizes = list(best_portfolio['allocation'].values())
        
        # Kolory dla pie charta (Neonowe)
        pie_colors = ['#00e5ff', '#d500f9', '#ffd700', '#00ff55', '#ff0055', '#2979ff', '#ff9100']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                           startangle=140, colors=pie_colors[:len(labels)],
                                           textprops={'color': t['text']})
        
        # Stylizacja tekstów na wykresie kołowym
        for text in texts: text.set_color(t['text'])
        for autotext in autotexts: 
            autotext.set_color('black')
            autotext.set_weight('bold')
            
        ax2.set_title("TWOJE IDEALNE PROPORCJE", fontsize=14, color=t['text'], fontweight='bold')
        
        # Tło
        fig.patch.set_facecolor(t['bg'])
        ax1.set_facecolor(t['bg'])
        # Ax2 jest kołowy, nie ma tła
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- NOWOŚĆ: VALUE AT RISK (VaR) - Zarządzanie Ryzykiem ---
    def get_var_data(self):
        """
        Oblicza Value at Risk (VaR) i Expected Shortfall (ES).
        Metoda Historyczna (Historical Simulation).
        Mówi: Gdzie jest granica bólu, której statystycznie nie przekroczymy?
        """
        try:
            # Pobieramy dane BTC
            df = yf.download('BTC-USD', period="2y", progress=False)
            close = df['Close']['BTC-USD'] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            
            # Zwroty procentowe dzienne
            returns = close.pct_change().dropna()
            
            # --- MATEMATYKA RYZYKA ---
            # VaR 95% - Odetnij 5% najgorszych dni w historii
            var_95 = np.percentile(returns, 5)
            
            # VaR 99% - Odetnij 1% najgorszych dni (Krachy)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall (CVaR) - Średnia strata, jeśli już przebijemy VaR (Czarny Łabędź)
            es_95 = returns[returns <= var_95].mean()
            
            return returns, var_95, var_99, es_95

        except Exception as e:
            print(f"Błąd VaR: {e}")
            return None, None, None, None

    def plot_var_distribution(self, returns, var_95, var_99, es_95):
        """
        Rysuje rozkład zwrotów (Histogram) z zaznaczonymi strefami ryzyka.
        """
        if returns is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Histogram (Rozkład statystyczny)
        # Bins = 50 słupków
        n, bins, patches = ax.hist(returns, bins=70, color='#00e5ff', alpha=0.3, edgecolor=t['bg'], density=True)
        
        # Krzywa gęstości (KDE - opcjonalne, tutaj prosta linia obwiedni)
        # Dla uproszczenia zostajemy przy histogramie, bo jest czytelniejszy dla VaR
        
        # --- LINIE ŚMIERCI (VaR) ---
        # VaR 95% (Żółta)
        ax.axvline(var_95, color='#ffd700', linestyle='--', linewidth=2, label=f'VaR 95% ({var_95:.1%})')
        
        # VaR 99% (Czerwona)
        ax.axvline(var_99, color='#ff0055', linestyle='-', linewidth=2, label=f'VaR 99% ({var_99:.1%})')
        
        # Strefa Expected Shortfall (To co za linią VaR 95)
        # Kolorujemy ogon strat
        for i in range(len(patches)):
            if bins[i] < var_95:
                patches[i].set_facecolor('#ff0055') # Krwawy ogon
                patches[i].set_alpha(0.6)
        
        ax.set_title("VALUE AT RISK: Mapa Ryzyka (Gdzie jest ból?)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Dzienny Zwrot (%)', color=t['text'])
        ax.set_ylabel('Częstotliwość występowania', color=t['text'])
        
        # Formatowanie osi X na procenty
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Tekst informacyjny
        info_text = f"Średnia Strata w Krachu (ES): {es_95:.1%}\n(Jeśli przebijemy żółtą linię, \ntyle średnio tracimy)"
        ax.text(var_95 - 0.02, ax.get_ylim()[1]*0.5, info_text, color='white', ha='right', fontsize=9)
        
        ax.legend(facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.grid(True, alpha=0.1, color=t['grid'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        ax.tick_params(colors=t['text'])
        
        return fig

    # --- NOWOŚĆ: ALGORYTMICZNY SĘDZIA (Technical Consensus) ---
    def get_technical_verdict(self):
        """
        Zbiera 10 wskaźników dla BTC i przeprowadza głosowanie.
        Zwraca tabelę wyników i ostateczny werdykt (MOCNE KUPUJ / SPRZEDAJ).
        """
        try:
            # Pobieramy dane (200 dni potrzebne do SMA200)
            df = yf.download('BTC-USD', period="1y", progress=False)
            close = df['Close']['BTC-USD'] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            
            # Wskaźniki
            current_price = close.iloc[-1]
            
            # 1. Średnie Kroczące (Trend)
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]
            
            # 2. RSI (Oscylator)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]
            
            # 3. MACD (Momentum)
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_val = macd_line.iloc[-1] - signal_line.iloc[-1] # Histogram
            
            # 4. Bollinger Bands (Zmienność)
            std = close.rolling(20).std().iloc[-1]
            lower_band = sma_20 - (2 * std)
            upper_band = sma_20 + (2 * std)
            
            # --- GŁOSOWANIE ---
            score = 0 # Startujemy od 0
            signals = []
            
            # Logika Średnich
            if current_price > sma_20: score += 1; signals.append("SMA 20: BULL 🟢")
            else: score -= 1; signals.append("SMA 20: BEAR 🔴")
            
            if current_price > sma_50: score += 1; signals.append("SMA 50: BULL 🟢")
            else: score -= 1; signals.append("SMA 50: BEAR 🔴")
            
            if current_price > sma_200: score += 2; signals.append("SMA 200: BULL 🟢") # Ważna!
            else: score -= 2; signals.append("SMA 200: BEAR 🔴")
            
            # Logika RSI
            if rsi_val < 30: score += 2; signals.append("RSI: OVERSOLD (Kupuj) 🟢")
            elif rsi_val > 70: score -= 2; signals.append("RSI: OVERBOUGHT (Sprzedaj) 🔴")
            else: signals.append("RSI: Neutral ⚪")
            
            # Logika MACD
            if macd_val > 0: score += 1; signals.append("MACD: BULL 🟢")
            else: score -= 1; signals.append("MACD: BEAR 🔴")
            
            # Logika Bollinger
            if current_price < lower_band: score += 2; signals.append("BB: Cena niska (Okazja) 🟢")
            elif current_price > upper_band: score -= 2; signals.append("BB: Cena wysoka (Ryzyko) 🔴")
            else: signals.append("BB: Neutral ⚪")
            
            # Wynik
            total_max = 9 # Max punktów (zależy od wag)
            sentiment = score / total_max # -1 do 1
            
            return sentiment, signals, current_price

        except Exception as e:
            print(f"Błąd Werdykt: {e}")
            return None, None, None

    # --- FIX: ALGORYTMICZNY SĘDZIA (Poprawiony kolor tła) ---
    def plot_verdict_gauge(self, sentiment):
        """
        Rysuje prosty pasek "Werdyktu".
        Poprawiono błąd z kolorem '#bg'.
        """
        if sentiment is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(111)
        
        # Pasek tła (Gradient) - Niewidoczny pasek ustalający skalę
        # ZMIANA: color='none' zamiast błędnego '#bg'
        ax.barh(0, 2, left=-1, height=0.5, color='none', alpha=0) 
        
        # Strefa Sprzedaży (Czerwona)
        ax.fill_between([-1, -0.3], -0.2, 0.2, color='#ff0055', alpha=0.5)
        ax.text(-0.65, 0, "SPRZEDAJ", ha='center', va='center', color='white', fontweight='bold')
        
        # Strefa Neutralna (Szara)
        ax.fill_between([-0.3, 0.3], -0.2, 0.2, color='grey', alpha=0.3)
        ax.text(0, 0, "NEUTRAL", ha='center', va='center', color='white', fontweight='bold')
        
        # Strefa Kupna (Zielona)
        ax.fill_between([0.3, 1], -0.2, 0.2, color='#00ff55', alpha=0.5)
        ax.text(0.65, 0, "KUPUJ", ha='center', va='center', color='white', fontweight='bold')
        
        # Wskaźnik (Marker)
        # Ograniczamy do zakresu -1 do 1, żeby nie wyleciał poza wykres
        val = max(min(sentiment, 1), -1)
        ax.scatter(val, 0.25, s=400, marker='v', color='white', edgecolors='black', zorder=10)
        
        # Opis wyniku
        if val > 0.5: desc = "MOCNE KUPUJ 🚀"
        elif val > 0.1: desc = "KUPUJ ↗️"
        elif val < -0.5: desc = "MOCNE SPRZEDAJ 🩸"
        elif val < -0.1: desc = "SPRZEDAJ ↘️"
        else: desc = "CZEKAJ 💤"
        
        ax.set_title(f"WERDYKT ALGORYTMU: {desc}", fontsize=18, color=t['text'], fontweight='bold')
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.3, 0.5)
        ax.axis('off')
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        return fig

    # --- 1. KRYTERIUM KELLY'EGO (Bet Sizing) ---
    def get_kelly_criterion(self):
        """
        Analizuje historyczne zwroty BTC, aby obliczyć optymalną wielkość pozycji (Kelly Bet).
        Wzór: f* = (p*b - q) / b
        Gdzie: p = prawodpodobieństwo wygranej, b = stosunek zysku do ryzyka (odds).
        """
        try:
            # Pobieramy dane
            df = yf.download('BTC-USD', period="2y", progress=False)
            close = df['Close']['BTC-USD'] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            
            # Obliczamy dzienne zmiany
            returns = close.pct_change().dropna()
            
            # Statystyka dni wygranych vs przegranych
            winning_days = returns[returns > 0]
            losing_days = returns[returns < 0]
            
            p = len(winning_days) / len(returns) # Prawdopodobieństwo wygranej sesji
            q = 1 - p # Prawdopodobieństwo przegranej
            
            avg_win = winning_days.mean()
            avg_loss = abs(losing_days.mean())
            
            # Payoff ratio (b)
            b = avg_win / avg_loss
            
            # Wzór Kelly'ego (Pełny)
            kelly_fraction = (p * b - q) / b
            
            # W praktyce używa się "Half Kelly" (Połowa stawki), żeby zmniejszyć zmienność
            safe_kelly = kelly_fraction / 2
            
            return kelly_fraction, safe_kelly, p, b

        except Exception as e:
            print(f"Błąd Kelly: {e}")
            return None, None, None, None

    def plot_kelly_gauge(self, full_kelly, safe_kelly):
        """
        Rysuje licznik sugerowanej wielkości pozycji.
        """
        if full_kelly is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        
        # Tło paska
        ax.barh(0, 100, color=t['text'], alpha=0.1)
        
        # Wartość Kelly (limitujemy do 50% dla bezpieczeństwa wizualizacji)
        k_val = min(max(full_kelly * 100, 0), 100)
        safe_val = min(max(safe_kelly * 100, 0), 100)
        
        # Pasek Full Kelly (Ryzykowany)
        ax.barh(0, k_val, color='#ff0055', alpha=0.5, label='Full Kelly (Agresywnie)')
        
        # Pasek Half Kelly (Bezpieczny)
        ax.barh(0, safe_val, color='#00ff55', alpha=1.0, label='Half Kelly (Optymalnie)')
        
        # Marker tekstowy
        ax.text(safe_val + 1, 0, f"{safe_val:.1f}% Portfela", va='center', fontweight='bold', color='#00ff55', fontsize=14)
        
        ax.set_title("KRYTERIUM KELLY'EGO: Ile postawić?", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('% Całego Kapitału', color=t['text'])
        ax.set_xlim(0, 50) # Skala do 50% max
        ax.set_yticks([])
        
        # Opis
        desc = f"Matematyka mówi: Inwestuj **{safe_val:.1f}%** kapitału w jedną transakcję.\n" \
               f"Full Kelly ({k_val:.1f}%) daje max zwrot, ale ryzykujesz bankructwo (drawdown)."
        
        plt.figtext(0.5, -0.1, desc, ha="center", fontsize=10, color=t['text'])
        
        ax.legend(loc='upper right', facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        
        return fig

    # --- 2. SMART MONEY TRACKER (Accumulation/Distribution) ---
    def get_smart_money_data(self):
        """
        Oblicza wskaźnik Accumulation/Distribution (A/D) i sprawdza dywergencje z ceną.
        Wykrywa, czy ruch ceny jest poparty wolumenem.
        """
        try:
            df = yf.download('BTC-USD', period="6mo", progress=False)
            
            # Obsługa MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                close = df['Close']['BTC-USD']
                high = df['High']['BTC-USD']
                low = df['Low']['BTC-USD']
                vol = df['Volume']['BTC-USD']
            else:
                close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
            
            # Wskaźnik A/D (Accumulation/Distribution Line)
            # MFV = ((Close - Low) - (High - Close)) / (High - Low)
            # AD = MFV * Volume + Prev AD
            mfv = ((close - low) - (high - close)) / (high - low)
            # Zabezpieczenie przed dzieleniem przez zero
            mfv = mfv.fillna(0)
            
            ad_line = (mfv * vol).cumsum()
            
            return close, ad_line

        except Exception as e:
            print(f"Błąd Smart Money: {e}")
            return None, None

    def plot_smart_money(self, price, ad_line):
        if price is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        
        # Normalizacja danych do wykresu (0-100%) żeby nałożyć linię na cenę
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())
        
        norm_price = normalize(price)
        norm_ad = normalize(ad_line)
        
        # Wykres Ceny (Żółty)
        ax1.plot(norm_price.index, norm_price, color='#ffd700', label='Cena BTC', alpha=0.6)
        
        # Wykres Smart Money (Niebieski)
        ax1.plot(norm_ad.index, norm_ad, color='#00e5ff', label='Smart Money (Wolumen)', linewidth=2)
        
        # Wypełnienie różnicy (Dywergencja)
        ax1.fill_between(norm_price.index, norm_price, norm_ad, where=(norm_price > norm_ad), 
                         color='#ff0055', alpha=0.3, label='FEJK (Cena rośnie bez wolumenu)')
        
        ax1.fill_between(norm_price.index, norm_price, norm_ad, where=(norm_price < norm_ad), 
                         color='#00ff55', alpha=0.3, label='AKUMULACJA (Wolumen wyprzedza cenę)')
        
        ax1.set_title("SMART MONEY: Kto kłamie? Cena czy Wolumen?", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_yticks([]) # Ukrywamy osie wartości, bo są znormalizowane
        
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_visible(False)
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- NOWOŚĆ: ECHOLOKACJA RYNKU (Fourier Transform) ---
    def get_fourier_projection(self):
        """
        Rozkłada cenę Bitcoina na fale sinusoidalne (FFT).
        Identyfikuje dominujące cykle i projektuje je w przyszłość.
        To pozwala przewidzieć punkty zwrotne (Górki/Dołki) wynikające z cykliczności.
        """
        try:
            # Pobieramy dane (dłuższy okres dla lepszej detekcji fal)
            df = yf.download('BTC-USD', period="2y", progress=False)
            close = df['Close']['BTC-USD'] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            
            data = close.values
            N = len(data)
            
            # 1. Usuwamy trend liniowy (żeby badać same cykle)
            x = np.arange(N)
            poly = np.polyfit(x, data, 1)
            trend = poly[0] * x + poly[1]
            detrended = data - trend
            
            # 2. Fast Fourier Transform (FFT)
            fft_coeffs = np.fft.fft(detrended)
            frequencies = np.fft.fftfreq(N)
            
            # 3. Filtrowanie (Bierzemy tylko TOP 10 najsilniejszych fal/cykli)
            # Reszta to szum, który chcemy usunąć
            magnitudes = np.abs(fft_coeffs)
            indices = np.argsort(magnitudes)[::-1] # Sortujemy od najsilniejszych
            
            top_n = 20 # Ilość fal składowych
            clean_fft = np.zeros_like(fft_coeffs)
            clean_fft[indices[:top_n]] = fft_coeffs[indices[:top_n]]
            
            # 4. Rekonstrukcja sygnału (Sygnał bez szumu)
            reconstructed_cycle = np.fft.ifft(clean_fft).real
            
            # 5. Projekcja w przyszłość (Extrapolacja fal)
            future_days = 30
            future_x = np.arange(N + future_days)
            future_trend = poly[0] * future_x + poly[1]
            
            # Matematyczna ekstrapolacja sumy sinusuid
            # To jest trudne w czystym numpy, więc robimy trick:
            # Generujemy falę na nowym, dłuższym zakresie czasu bazując na częstotliwościach
            
            restored_wave = np.zeros(N + future_days)
            t = np.arange(N + future_days)
            
            for i in indices[:top_n]:
                freq = frequencies[i]
                ampl = np.abs(fft_coeffs[i]) / N
                phase = np.angle(fft_coeffs[i])
                
                # Dodajemy każdą falę składową: A * cos(2*pi*f*t + phase)
                restored_wave += ampl * np.cos(2 * np.pi * freq * t + phase)
            
            # Dodajemy z powrotem trend, żeby uzyskać cenę
            final_projection = restored_wave + future_trend
            
            return close, final_projection, future_days

        except Exception as e:
            print(f"Błąd Fourier: {e}")
            return None, None, None

    def plot_fourier_cycles(self, close, projection, future_days):
        if close is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        
        # Oś czasu
        dates = close.index
        # Generujemy daty przyszłe
        last_date = dates[-1]
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
        all_dates = list(dates) + future_dates
        
        # Przycinamy projekcję do długości dat (czasem może się różnić o 1)
        projection = projection[:len(all_dates)]
        
        # 1. Prawdziwa Cena (Szara/Żółta)
        ax.plot(dates, close.values, color='#ffd700', label='Cena Rynkowa (Szum + Sygnał)', alpha=0.5, linewidth=1)
        
        # 2. CZYSTY SYGNAŁ (Fioletowy) - To jest "Serce Rynku"
        # Rysujemy część historyczną
        ax.plot(dates, projection[:len(dates)], color='#d500f9', linewidth=2.5, label='Dominujący Cykl (Fourier)')
        
        # 3. PRZYSZŁOŚĆ (Neonowa zieleń) - To jest predykcja
        ax.plot(future_dates, projection[len(dates):], color='#00ff55', linewidth=3, linestyle='--', label='PROJEKCJA PRZYSZŁOŚCI')
        
        # Kropka "Tu jesteśmy"
        ax.scatter(dates[-1], projection[len(dates)-1], s=200, color='white', edgecolors='black', zorder=10)
        ax.text(dates[-1], projection[len(dates)-1], " TERAZ", color='white', fontweight='bold', ha='left')

        # Znajdźmy lokalny szczyt/dołek w przyszłości
        future_vals = projection[len(dates):]
        if len(future_vals) > 0:
            local_max = np.argmax(future_vals)
            local_min = np.argmin(future_vals)
            
            # Oznaczamy szczyt
            ax.scatter(future_dates[local_max], future_vals[local_max], color='#ff0055', s=100, zorder=10)
            ax.text(future_dates[local_max], future_vals[local_max], " Przewidywany SZCZYT", color='#ff0055', fontsize=9, fontweight='bold')
            
            # Oznaczamy dołek
            ax.scatter(future_dates[local_min], future_vals[local_min], color='#00e5ff', s=100, zorder=10)
            ax.text(future_dates[local_min], future_vals[local_min], " Przewidywany DOŁEK", color='#00e5ff', fontsize=9, fontweight='bold')

        ax.set_title("ECHOLOKACJA RYNKU: Dekodowanie Cykli (Fourier)", fontsize=16, color=t['text'], fontweight='bold')
        
        ax.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.grid(True, alpha=0.15, color=t['grid'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        
        return fig

    # --- NOWOŚĆ: GENOM RYNKU (3D Phase Space / Strange Attractor) ---
    def get_phase_space_data(self):
        """
        Tworzy trójwymiarową mapę chaosu (Przestrzeń Fazowa).
        Używa opóźnień czasowych (Time Delay Embedding), aby zrekonstruować 'atraktor' rynku.
        X = Cena(t)
        Y = Cena(t - lag)
        Z = Cena(t - 2*lag)
        """
        try:
            # Pobieramy dane (1 rok to minimum dla ładnego atraktora)
            df = yf.download('BTC-USD', period="1y", progress=False)
            close = df['Close']['BTC-USD'] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            
            # Parametr opóźnienia (Lag) - kluczowy w teorii chaosu
            # Dla krypto 5-7 dni działa najlepiej
            lag = 5 
            
            # Tworzymy współrzędne 3D
            x = close.iloc[2*lag : ]           # Cena teraz
            y = close.shift(lag).iloc[2*lag :] # Cena tydzień temu
            z = close.shift(2*lag).iloc[2*lag :] # Cena 2 tygodnie temu
            
            # Musimy wyrównać indeksy
            data_3d = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=x.index)
            
            return data_3d, close.iloc[-1]

        except Exception as e:
            print(f"Błąd Phase Space: {e}")
            return None, None

    def plot_phase_space_3d(self, data_3d, current_price):
        """
        Rysuje GENOM RYNKU w 3D.
        To nie jest zwykły wykres. To struktura zachowania rynku.
        """
        if data_3d is None: return None
        
        t = self.get_theme_colors()
        
        # Tworzymy figurę 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Kolorowanie czasem (Stare punkty = Ciemne, Nowe = Jasne/Neonowe)
        # To pozwala widzieć "ścieżkę", którą podąża rynek
        num_points = len(data_3d)
        colors = plt.cm.plasma(np.linspace(0, 1, num_points)) # Mapa kolorów Plasma (fiolet -> żółty)
        
        # Rysujemy punkty (Scatter)
        # s=20 (rozmiar), alpha=0.6 (przezroczystość)
        ax.scatter(data_3d['x'], data_3d['y'], data_3d['z'], c=colors, s=20, alpha=0.6, edgecolors='none')
        
        # Rysujemy linię łączącą ostatnie 30 dni (Ogon Komety) - żeby widzieć gdzie teraz lecimy
        last_30 = data_3d.tail(30)
        ax.plot(last_30['x'], last_30['y'], last_30['z'], color='#00ff55', linewidth=2, label='Ostatnie 30 dni')
        
        # Ostatni punkt (GŁOWA WĘŻA) - Tu jesteśmy TERAZ
        last_pt = data_3d.iloc[-1]
        ax.scatter(last_pt['x'], last_pt['y'], last_pt['z'], color='#fff', s=300, marker='*', edgecolors='#ff0055', zorder=20, label='TERAZ')
        
        # Kosmetyka 3D
        ax.set_title("GENOM RYNKU: Atraktor Chaosu (3D)", fontsize=16, color=t['text'], fontweight='bold')
        
        # Osie
        ax.set_xlabel('Cena (t)', color=t['text'])
        ax.set_ylabel(f'Cena (t-5)', color=t['text'])
        ax.set_zlabel(f'Cena (t-10)', color=t['text'])
        
        # Kolory osi i tła
        ax.xaxis.set_tick_params(colors=t['text'])
        ax.yaxis.set_tick_params(colors=t['text'])
        ax.zaxis.set_tick_params(colors=t['text'])
        
        # Tło panelu 3D (Przezroczyste lub ciemne)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False) # Wyłączamy siatkę dla czystości "kosmosu"
        
        # Czarne tło dla efektu WOW
        fig.patch.set_facecolor(t['bg'])
        ax.set_facecolor(t['bg'])
        
        # Ustawiamy widok (kąt kamery)
        ax.view_init(elev=30, azim=45)
        
        plt.legend(facecolor=t['bg'], labelcolor=t['text'])
        return fig

    # --- UPDATE: FRACTAL PATTERN MATCHER (Wersja 'Znajdź cokolwiek') ---
    def get_fractal_matches(self):
        """
        Znajduje w historii BTC momenty (bliźniaki).
        WERSJA POPRAWIONA: Zmniejszono rygor, żeby zawsze znajdował wzorce,
        nawet gdy rynek jest 'unikalny'.
        """
        try:
            # Pobieramy MAX historii
            df = yf.download('BTC-USD', period="max", progress=False)
            close = df['Close']['BTC-USD'] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            
            # --- ZMIANA 1: Krótsze okno (łatwiej dopasować) ---
            window_size = 45  # Było 60
            projection_days = 30 
            
            current_sequence = close.iloc[-window_size:].values
            
            # Normalizacja "Teraz"
            def normalize(arr):
                # Zabezpieczenie przed dzieleniem przez zero (płaski rynek)
                diff = np.max(arr) - np.min(arr)
                if diff == 0: return np.zeros_like(arr)
                return (arr - np.min(arr)) / diff
            
            norm_current = normalize(current_sequence)
            
            # Skanowanie historii
            matches = []
            history_len = len(close) - window_size - projection_days
            
            # Przesuwamy okno po historii (krok co 3 dni dla szybkości)
            for i in range(0, history_len, 3):
                # Wycinek historyczny
                hist_window = close.iloc[i : i + window_size].values
                
                # Normalizacja wycinka
                norm_hist = normalize(hist_window)
                
                # Obliczamy korelację
                # Jeśli wycinek jest płaski (std=0), korelacja zwróci NaN -> zamieniamy na 0
                try:
                    corr = np.corrcoef(norm_current, norm_hist)[0, 1]
                except:
                    corr = 0
                
                if np.isnan(corr): corr = 0
                
                # --- ZMIANA 2: Niższy próg (0.60 zamiast 0.80) ---
                # Szukamy "Kuzynów", skoro nie ma "Bliźniaków"
                if corr > 0.80:
                    future_window = close.iloc[i + window_size : i + window_size + projection_days].values
                    
                    # Skalowanie przyszłości
                    scale_factor = current_sequence[-1] / hist_window[-1]
                    scaled_future = future_window * scale_factor
                    
                    matches.append({
                        'date': close.index[i],
                        'corr': corr,
                        'future': scaled_future
                    })
            
            # Sortujemy i bierzemy TOP 5 (nawet jeśli są słabe, bierzemy najlepsze z najgorszych)
            matches = sorted(matches, key=lambda x: x['corr'], reverse=True)[:5]
            
            # Jeśli nadal pusto (bardzo dziwny rynek), to zwracamy nic
            if not matches:
                return current_sequence, [], None, projection_days
            
            # Obliczamy Średnią Ścieżkę
            avg_future = np.mean([m['future'] for m in matches], axis=0)
                
            return current_sequence, matches, avg_future, projection_days

        except Exception as e:
            print(f"Błąd Fraktal: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

    def plot_fractal_matches(self, current_seq, matches, avg_future, proj_days):
        if current_seq is None or not matches: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        
        # Oś X (Dni)
        x_past = np.arange(-len(current_seq) + 1, 1) # Od -60 do 0
        x_future = np.arange(1, proj_days + 1)      # Od 1 do 30
        
        # 1. Rysujemy "Bliźniaki" (Szare linie)
        for m in matches:
            # Doklejamy ostatni punkt teraźniejszości, żeby linie były ciągłe
            fut = np.concatenate(([current_seq[-1]], m['future']))
            x_fut_full = np.concatenate(([0], x_future))
            
            ax.plot(x_fut_full, fut, color='grey', alpha=0.4, linestyle='--', linewidth=1)
            
            # Podpisujemy daty bliźniaków (Gdzie to znaleźliśmy?)
            # Np. "2017"
            ax.text(x_future[-1], fut[-1], f"{m['date'].strftime('%Y')}\n({m['corr']:.2f})", 
                    color='grey', fontsize=8, alpha=0.7)
            
        # 2. Rysujemy OBECNĄ CENĘ (Gruba Żółta)
        ax.plot(x_past, current_seq, color='#ffd700', linewidth=3, label='TWOJA SYTUACJA (Teraz)')
        
        # 3. Rysujemy ŚREDNIĄ PROJEKCJĘ (Neonowa)
        if avg_future is not None:
            avg_fut_full = np.concatenate(([current_seq[-1]], avg_future))
            x_fut_full = np.concatenate(([0], x_future))
            
            # Kolor zależny od kierunku (Rośnie czy Spada?)
            final_change = (avg_future[-1] - current_seq[-1]) / current_seq[-1]
            c_proj = '#00ff55' if final_change > 0 else '#ff0055'
            
            ax.plot(x_fut_full, avg_fut_full, color=c_proj, linewidth=4, label=f'Projekcja Fraktalna ({final_change:+.1%})')
            
            # Kropka na końcu
            ax.scatter(x_future[-1], avg_future[-1], s=200, color=c_proj, edgecolors='white', zorder=10)

        # Linia "TERAZ"
        ax.axvline(0, color='white', linestyle=':', alpha=0.5)
        ax.text(0, current_seq[-1], " TERAZ", color='white', fontweight='bold', ha='right')

        ax.set_title("FRACTAL PATTERN MATCHER: Klonowanie Historii", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Dni (0 = Dzisiaj)', color=t['text'])
        ax.set_ylabel('Cena ($)', color=t['text'])
        
        ax.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.grid(True, alpha=0.1, color=t['grid'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        
        return fig

    # --- FINAL BOSS: LIQUIDATION HEATMAP (Symulacja Stref Śmierci) ---
    def get_liquidation_levels(self):
        """
        Symuluje poziomy likwidacji traderów grających na dźwigni (x10, x25, x50, x100).
        Banki celują w te poziomy, aby pozyskać płynność (Stop Hunt).
        """
        try:
            df = yf.download('BTC-USD', period="3mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                high = df['High']['BTC-USD']
                low = df['Low']['BTC-USD']
                close = df['Close']['BTC-USD']
            else:
                high = df['High']; low = df['Low']; close = df['Close']
            
            # Znajdujemy lokalne szczyty i dołki (Swingi)
            # To tam ulica stawia Stop Lossy lub wchodzi w pozycje
            window = 5
            df['Swing_High'] = high[(high.shift(1) < high) & (high.shift(-1) < high)]
            df['Swing_Low'] = low[(low.shift(1) > low) & (low.shift(-1) > low)]
            
            levels = []
            
            # 1. Gdzie są likwidacje SHORTÓW? (Powyżej szczytów)
            # Jeśli ktoś szortował szczyt na dźwigni x50, zginie +2% wyżej.
            # Jeśli x100, zginie +1% wyżej.
            for date, price in df['Swing_High'].dropna().items():
                if price < close.iloc[-1] * 0.8: continue # Ignorujemy stare/dalekie
                
                levels.append({'price': price * 1.01, 'lev': '100x', 'type': 'Short Liq', 'alpha': 0.1})
                levels.append({'price': price * 1.02, 'lev': '50x', 'type': 'Short Liq', 'alpha': 0.2})
                levels.append({'price': price * 1.05, 'lev': '20x', 'type': 'Short Liq', 'alpha': 0.3})

            # 2. Gdzie są likwidacje LONGÓW? (Poniżej dołków)
            for date, price in df['Swing_Low'].dropna().items():
                if price > close.iloc[-1] * 1.2: continue
                
                levels.append({'price': price * 0.99, 'lev': '100x', 'type': 'Long Liq', 'alpha': 0.1})
                levels.append({'price': price * 0.98, 'lev': '50x', 'type': 'Long Liq', 'alpha': 0.2})
                levels.append({'price': price * 0.95, 'lev': '20x', 'type': 'Long Liq', 'alpha': 0.3})
                
            return close, levels

        except Exception as e:
            print(f"Błąd Liq Map: {e}")
            return None, None

    def plot_liquidation_heatmap(self, close, levels):
        if close is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        
        # Rysujemy cenę
        ax.plot(close.index, close, color='#e0e0e0', linewidth=1, label='Cena BTC')
        
        # Rysujemy poziomy likwidacji jako "Chmury Płynności"
        # Im więcej poziomów się nakłada, tym ciemniejszy kolor (Cluster)
        
        for lvl in levels:
            color = '#ff0055' if lvl['type'] == 'Short Liq' else '#00ff55'
            # Rysujemy poziomą linię przez cały wykres
            ax.axhline(lvl['price'], color=color, alpha=0.03, linewidth=4)
            
        # Oznaczenie aktualnej ceny
        last_price = close.iloc[-1]
        ax.axhline(last_price, color='#ffd700', linestyle=':', label='TERAZ')
        
        # Znajdźmy "Najgorętsze strefy" (Gdzie jest najwięcej kresek?)
        # To prosta wizualizacja zagęszczenia
        
        ax.set_title("LIQUIDATION HEATMAP: Gdzie banki szukają ofiar?", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_ylabel('Cena ($)', color=t['text'])
        
        # Legenda "Hackowana"
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='#ff0055', lw=4, alpha=0.5),
                        Line2D([0], [0], color='#00ff55', lw=4, alpha=0.5),
                        Line2D([0], [0], color='#ffd700', lw=1, linestyle=':')]
        
        ax.legend(custom_lines, ['Likwidacje Shortów (Magnes w górę)', 'Likwidacje Longów (Magnes w dół)', 'Cena Aktualna'], 
                  loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.grid(False) # Bez siatki, żeby widzieć chmury
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        
        return fig

    # --- RESET: FED NET LIQUIDITY (Wersja Podstawowa / Czysta) ---
    def get_true_fed_liquidity(self):
        """
        Oblicza Fed Net Liquidity w najprostszy, niezawodny sposób.
        Bez skomplikowanych filtrów, które psuły wykres.
        Wzór: Assets - TGA - RRP.
        """
        try:
            start_date = '2018-01-01'
            end_date = datetime.now()
            
            # 1. Pobieramy dane z FRED (Tylko 3 podstawowe składniki)
            tickers = ['WALCL', 'WTREGEN', 'RRPONTSYD']
            
            # Pobieranie z obsługą błędów
            try:
                df = web.DataReader(tickers, 'fred', start_date, end_date)
            except Exception as e:
                print(f"Błąd pobierania FRED: {e}")
                return None, None

            # Uzupełniamy braki (najpierw w przód, potem w tył dla bezpieczeństwa)
            df = df.ffill().bfill()

            # 2. KONWERSJA NA MILIARDY (Żeby jednostki się zgadzały)
            # WALCL (Assets) jest w Milionach -> Dzielimy przez 1000
            assets = df['WALCL'] / 1000
            
            # WTREGEN (TGA) i RRPONTSYD (RRP) są już w Miliardach
            tga = df['WTREGEN']
            rrp = df['RRPONTSYD']
            
            # 3. WZÓR NA PŁYNNOŚĆ NETTO
            net_liquidity = assets - tga - rrp
            
            # 4. RESAMPLING (Tygodniowy)
            # Bierzemy stan na koniec tygodnia (piątek)
            net_liq_wk = net_liquidity.resample('W').last().ffill()
            
            # 5. OBLICZENIE YoY % (Zmiana roczna)
            # Używamy prostego wygładzania (4 tygodnie), żeby wykres był czytelny,
            # ale nie zniekształcony.
            liq_smooth = net_liq_wk.rolling(window=4).mean()
            liq_yoy = liq_smooth.pct_change(52) * 100
            
            # --- BTC ---
            btc = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
            
            btc_wk = btc.resample('W').last().ffill()
            # BTC też lekko wygładzamy (miesiąc), żeby pasowało do makro
            btc_smooth = btc_wk.rolling(window=4).mean()
            btc_yoy = btc_smooth.pct_change(52) * 100
            
            # Wyrównanie dat (Od 2019)
            common_idx = liq_yoy.index.intersection(btc_yoy.index)
            common_idx = common_idx[common_idx >= '2019-01-01']
            
            return btc_yoy.loc[common_idx], liq_yoy.loc[common_idx]

        except Exception as e:
            print(f"Błąd Basic Fed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_true_fed_liquidity(self, btc_yoy, liq_yoy):
        """
        Rysuje wykres w standardowej skali.
        Automatyczna skala osi (bez ręcznego ucinania).
        """
        if btc_yoy is None or liq_yoy is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- OŚ LEWA: FED LIQUIDITY (Biała/Czarna) ---
        liq_color = t['text']
        
        # Rysujemy bez przesunięć i kombinacji
        ax1.plot(liq_yoy.index, liq_yoy, color=liq_color, linewidth=2, label='Fed Net Liquidity YoY%')
        
        # Pozwalamy matplotlibowi samemu dobrać skalę, żeby nic nie ucięło
        # Dodajemy tylko siatkę na zerze
        ax1.axhline(0, color=t['text'], linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax1.set_ylabel('Fed Net Liquidity YoY (%)', color=liq_color, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=liq_color, colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        # --- OŚ PRAWA: BITCOIN (Różowa/Neonowa) ---
        ax2 = ax1.twinx()
        btc_color = '#ff007f'
        
        ax2.plot(btc_yoy.index, btc_yoy, color=btc_color, linewidth=1.5, label='Bitcoin YoY% (RHS)')
        
        ax2.set_ylabel('Bitcoin YoY (%)', color=btc_color, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=btc_color, colors=t['text'])
        
        # Kosmetyka
        ax2.spines['top'].set_visible(False); ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False); ax2.spines['bottom'].set_visible(False)

        ax1.set_title("FED NET LIQUIDITY vs BTC (Standard)", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', ncol=2, frameon=False, labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- FIX: M2 MONEY SUPPLY (Naprawa błędu Alignment) ---
    def get_m2_supply_data(self):
        """
        Pobiera podaż pieniądza M2 z FRED.
        POPRAWKA: Zwraca czyste Series, aby uniknąć błędu 'Operands are not aligned'.
        """
        try:
            start_date = '2019-01-01'
            end_date = datetime.now()
            
            # Pobieramy M2 (FRED)
            m2_data = web.DataReader('M2SL', 'fred', start_date, end_date)
            
            # Pobieramy BTC (Yahoo)
            btc = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            if isinstance(btc, pd.DataFrame): 
                # Jeśli pobrało jako DataFrame, spłaszczamy do Series
                btc = btc.iloc[:, 0]
            
            # Wyrównanie danych (Resampling do tygodniówek)
            # 'M2SL' to nazwa kolumny w danych z FRED
            m2_weekly = m2_data['M2SL'].resample('W').ffill()
            btc_weekly = btc.resample('W').last().ffill()
            
            # Wyrównujemy indeksy (część wspólna)
            common_idx = m2_weekly.index.intersection(btc_weekly.index)
            
            m2 = m2_weekly.loc[common_idx]
            btc = btc_weekly.loc[common_idx]
            
            # Obliczamy M2 YoY% (Roczna zmiana dodruku)
            m2_yoy = m2.pct_change(52) * 100
            
            return btc, m2, m2_yoy

        except Exception as e:
            print(f"Błąd M2: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def plot_m2_vs_btc(self, btc, m2, m2_yoy):
        """
        Rysuje BTC vs M2.
        POPRAWKA: Dostosowano do obsługi Series (naprawa błędu fill_between).
        """
        if btc is None or m2 is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- Wykres 1: BTC vs M2 (Logarytmiczny) ---
        
        # Normalizacja do startu (Start = 100%)
        # Teraz m2 jest Series, więc używamy iloc[0] bezpośrednio (bez nazwy kolumny)
        btc_norm = (btc / btc.iloc[0]) * 100
        m2_norm = (m2 / m2.iloc[0]) * 100
        
        ax1.plot(btc_norm.index, btc_norm, color='#ff9900', linewidth=2, label='Bitcoin (Wzrost ceny)')
        ax1.plot(m2_norm.index, m2_norm, color='#00e5ff', linewidth=2, linestyle='--', label='Podaż Dolara M2 (Inflacja)')
        
        # Wypełnienie: Kiedy BTC jest NAD M2 = Prawdziwy Zysk
        # Teraz btc_norm i m2_norm mają ten sam kształt, więc to zadziała
        ax1.fill_between(btc_norm.index, btc_norm, m2_norm, where=(btc_norm > m2_norm), 
                         color='#00ff55', alpha=0.1, label='REALNY ZYSK (Above Inflation)')
        
        ax1.set_yscale('log') # Skala logarytmiczna
        ax1.set_ylabel('Wzrost wartości (Start=100)', color=t['text'])
        ax1.set_title("BTC vs DRUKARKA (M2): Czy wyprzedzasz inflację?", fontsize=16, color=t['text'], fontweight='bold')
        
        # Mały wykres na dole (M2 YoY) na prawej osi
        ax2 = ax1.twinx()
        # m2_yoy też jest już Series
        ax2.fill_between(m2_yoy.index, m2_yoy, 0, color='#ffd700', alpha=0.15)
        ax2.set_ylabel('Tempo dodruku M2 (YoY %)', color='#ffd700')
        ax2.tick_params(axis='y', labelcolor='#ffd700')
        ax2.spines['top'].set_visible(False); ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        
        # Legenda
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- NOWOŚĆ: FRED MACRO PACK (Recesja, Grawitacja, Strach) ---
    def get_fred_macro_pack(self):
        """
        Pobiera 3 kluczowe wskaźniki makro z FRED:
        1. T10Y2Y (Krzywa dochodowości) - Wykrywacz recesji.
        2. DFII10 (Realne Stopy) - Grawitacja dla aktywów Risk-On.
        3. BAMLC0A0CM (Credit Spread) - Strach na rynku długu.
        """
        try:
            start_date = '2019-01-01'
            end_date = datetime.now()
            
            # Pobieramy wszystko w jednym strzale
            tickers = ['T10Y2Y', 'DFII10', 'BAMLC0A0CM']
            df = web.DataReader(tickers, 'fred', start_date, end_date)
            
            # Wypełniamy braki (dane makro nie są codzienne)
            df = df.ffill().dropna()
            
            return df
        except Exception as e:
            print(f"Błąd FRED Macro: {e}")
            return None

    def plot_macro_detonators(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        
        # Tworzymy 3 mniejsze wykresy jeden pod drugim
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # --- 1. KRZYWA DOCHODOWOŚCI (T10Y2Y) ---
        yield_curve = df['T10Y2Y']
        # Kolorujemy: Czerwone (Inwersja < 0), Zielone (Normalnie > 0)
        ax1.plot(yield_curve.index, yield_curve, color='white', linewidth=1)
        ax1.fill_between(yield_curve.index, yield_curve, 0, where=(yield_curve < 0), color='#ff0055', alpha=0.5, label='Inwersja (Recesja nadchodzi)')
        ax1.fill_between(yield_curve.index, yield_curve, 0, where=(yield_curve >= 0), color='#00ff55', alpha=0.2, label='Normalność')
        ax1.axhline(0, color='white', linestyle=':')
        
        last_y = yield_curve.iloc[-1]
        status_y = "⚠️ DETONACJA (Powrót nad 0)" if last_y > -0.1 and last_y < 0.3 else "CZEKANIE"
        if last_y < -0.1: status_y = "ŁADOWANIE BOMBY (Inwersja)"
        
        ax1.set_title(f"1. KRZYWA DOCHODOWOŚCI (10Y-2Y): {status_y}", fontsize=12, color=t['text'], fontweight='bold')
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'], fontsize=8)
        
        # --- 2. REALNE STOPY PROCENTOWE (DFII10) ---
        real_rates = df['DFII10']
        # Odwracamy kolory logicznie: Wysokie stopy = Źle dla BTC (Czerwone), Niskie = Dobrze (Zielone)
        ax2.plot(real_rates.index, real_rates, color='#00e5ff', linewidth=1.5)
        ax2.axhline(2.0, color='#ff0055', linestyle='--', label='Strefa Bólu (>2%)')
        ax2.axhline(0.0, color='#00ff55', linestyle='--', label='Strefa Euforii (<0%)')
        
        last_r = real_rates.iloc[-1]
        ax2.set_title(f"2. REALNE STOPY (Grawitacja BTC): {last_r:.2f}%", fontsize=12, color=t['text'], fontweight='bold')
        ax2.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'], fontsize=8)

        # --- 3. STRES KREDYTOWY (Credit Spreads) ---
        spreads = df['BAMLC0A0CM']
        ax3.plot(spreads.index, spreads, color='#ffd700', linewidth=1.5)
        
        # Jeśli spread rośnie gwałtownie -> Krach
        last_s = spreads.iloc[-1]
        status_s = "PANIKA!" if last_s > 5.0 else "Spokój"
        
        ax3.set_title(f"3. STRES KREDYTOWY (High Yield Spread): {status_s}", fontsize=12, color=t['text'], fontweight='bold')
        ax3.fill_between(spreads.index, spreads, 3.5, where=(spreads > 3.5), color='#ff0055', alpha=0.5, label='Ryzyko Bankructw')
        
        # Kosmetyka całości
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.15, color=t['grid'])
            ax.set_facecolor(t['bg'])
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
            ax.tick_params(colors=t['text'])
            
        fig.patch.set_facecolor(t['bg'])
        return fig

    # --- FINAL BOSS: GLOBAL CENTRAL BANK LIQUIDITY (FED + ECB + BOJ) ---
    def get_global_liquidity_index(self):
        """
        Oblicza GLOBALNĄ PŁYNNOŚĆ (Global Liquidity Index).
        Sumuje bilanse:
        1. FED (USA) - skorygowany o TGA i RRP (Net Liquidity)
        2. ECB (Europa) - przeliczony na USD
        3. BOJ (Japonia) - przeliczony na USD
        
        Zwraca: BTC oraz Sumaryczny Indeks Płynności w Bilionach USD.
        """
        try:
            start_date = '2019-01-01'
            end_date = datetime.now()
            
            # 1. Pobieramy BTC
            btc = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
            
            # 2. Pobieramy dane Makro z FRED
            # WALCL = Fed Assets (Miliony USD)
            # WTREGEN = TGA (Miliardy USD) -> trzeba * 1000
            # RRPONTSYD = RRP (Miliardy USD) -> trzeba * 1000
            # ECBASSETSW = ECB Assets (Miliony EUR)
            # JPNASSETS = BOJ Assets (100 Milionów JPY)
            # DEXUSEU = Kurs USD/EUR (Do przeliczenia ECB)
            # DEXJPUS = Kurs JPY/USD (Do przeliczenia BOJ)
            
            tickers = [
                'WALCL', 'WTREGEN', 'RRPONTSYD',  # USA
                'ECBASSETSW', 'DEXUSEU',          # EUROPA
                'JPNASSETS', 'DEXJPUS'            # JAPONIA
            ]
            
            macro_data = web.DataReader(tickers, 'fred', start_date, end_date)
            macro_data = macro_data.ffill().dropna()
            
            # 3. Obliczenia (Konwersja wszystkiego na MILIONY USD)
            
            # --- A. USA (FED NET LIQUIDITY) ---
            # Assets - TGA - RRP
            fed_net = macro_data['WALCL'] - (macro_data['WTREGEN'] * 1000) - (macro_data['RRPONTSYD'] * 1000)
            
            # --- B. EUROPA (ECB w USD) ---
            # Assets (EUR) * Kurs (USD/EUR)
            ecb_usd = macro_data['ECBASSETSW'] * macro_data['DEXUSEU']
            
            # --- C. JAPONIA (BOJ w USD) ---
            # JPNASSETS jest w jednostkach "100 Milionów Jenów".
            # Żeby mieć Miliony Jenów -> * 100.
            # Żeby mieć USD -> Dzielimy przez kurs JPY/USD
            boj_usd = (macro_data['JPNASSETS'] * 100) / macro_data['DEXJPUS']
            
            # BOJ jest miesięczny, reszta tygodniowa/dzienna. Musimy wyrównać BOJ.
            boj_usd = boj_usd.resample('D').interpolate(method='linear')
            
            # 4. SUMA GLOBALNA (Wszystko sprowadzone do wspólnego indeksu)
            # Musimy wyrównać indeksy, bo różne banki publikują w różne dni
            df_liq = pd.DataFrame({
                'FED': fed_net,
                'ECB': ecb_usd,
                'BOJ': boj_usd
            }).ffill().dropna()
            
            # Suma w Bilionach (Trillions) dla czytelności (dzielimy przez 1,000,000)
            df_liq['GLOBAL_INDEX'] = (df_liq['FED'] + df_liq['ECB'] + df_liq['BOJ']) / 1000000
            
            # Wyrównanie z BTC (Tygodniówki są najlepsze do szumów)
            btc_wk = btc.resample('W').last().ffill()
            liq_wk = df_liq['GLOBAL_INDEX'].resample('W').last().ffill()
            
            common_idx = btc_wk.index.intersection(liq_wk.index)
            
            return btc_wk.loc[common_idx], liq_wk.loc[common_idx]

        except Exception as e:
            print(f"Błąd Global Liquidity: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_global_liquidity_index(self, btc, liq_index):
        if btc is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- BTC (Logarytmicznie) ---
        ax1.plot(btc.index, btc, color='#ff9900', linewidth=1.5, label='Bitcoin Price')
        ax1.set_yscale('log') # Skala log dla BTC, żeby pasowała do makro
        ax1.set_ylabel('Cena BTC (Log)', color='#ff9900', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#ff9900', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])

        # --- GLOBAL LIQUIDITY (Oś Prawa) ---
        ax2 = ax1.twinx()
        
        # Kolor Cyjan/Biały
        liq_color = '#00e5ff'
        
        ax2.plot(liq_index.index, liq_index, color=liq_color, linewidth=2.5, label='Global Liquidity Index (FED+ECB+BOJ)')
        
        # Wypełnienie pod wykresem płynności (Efekt "Fali")
        ax2.fill_between(liq_index.index, liq_index, liq_index.min(), color=liq_color, alpha=0.1)
        
        ax2.set_ylabel('Płynność Globalna (Bilony USD)', color=liq_color, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=liq_color, colors=t['text'])
        
        # Ukrywamy ramki
        ax2.spines['top'].set_visible(False); ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False); ax2.spines['bottom'].set_visible(False)

        ax1.set_title("🌏 GLOBAL LIQUIDITY INDEX: Suma Drukarek (USA + EU + Japonia)", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- FIX: BANK CREDIT APPETITE (Safe Mode) ---
    def get_credit_conditions(self):
        """
        Pobiera dane o chęci banków do udzielania kredytów (SLOOS) vs Reverse Repo.
        NAPRAWA: Rozdzielono pobieranie na niezależne bloki.
        Zmieniono ticker hipoteczny na DRTSP (bardziej stabilny).
        """
        try:
            start_date = '2018-01-01'
            end_date = datetime.now()
            
            # 1. Pobieramy REPO (To powinno zawsze działać)
            try:
                rrp = web.DataReader('RRPONTSYD', 'fred', start_date, end_date)
                rrp.columns = ['RRP']
            except Exception as e:
                print(f"Błąd RRP: {e}")
                return None # Bez Repo wykres nie ma sensu
                
            # 2. Pobieramy SLOOS (Ankiety Bankowe)
            # DRTSCLCC = Karty Kredytowe
            # DRTSP = Hipoteki (Standard Prime) - ZMIANA Z DRTSPM
            sloos_data = pd.DataFrame()
            
            try:
                # Pobieramy osobno, żeby jeden błąd nie zabił drugiego
                cc = web.DataReader('DRTSCLCC', 'fred', start_date, end_date)
                cc.columns = ['CreditCards']
                sloos_data = cc
                
                mtg = web.DataReader('DRTSP', 'fred', start_date, end_date)
                mtg.columns = ['Mortgages']
                
                # Łączymy
                if not sloos_data.empty:
                    sloos_data = sloos_data.join(mtg, how='outer')
                else:
                    sloos_data = mtg
                    
            except Exception as e:
                print(f"Ostrzeżenie - brak danych kredytowych: {e}")
                # Nie zwracamy None, jedziemy dalej z samym Repo
            
            # 3. ŁĄCZENIE
            # Rozciągamy SLOOS (Kwartalne) na dzienne (ffill)
            if not sloos_data.empty:
                sloos_daily = sloos_data.resample('D').ffill()
                # Łączymy z Repo
                df = rrp.join(sloos_daily, how='left').ffill()
            else:
                df = rrp
                df['CreditCards'] = 0 # Zaślepka
                df['Mortgages'] = 0   # Zaślepka

            return df.dropna()

        except Exception as e:
            print(f"Krytyczny błąd Credit Conditions: {e}")
            return None

    def plot_credit_conditions(self, df):
        # Zabezpieczenie przed pustym wykresem
        if df is None or len(df) == 0: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- 1. REVERSE REPO (Obszar w tle) ---
        ax1.fill_between(df.index, df['RRP'], 0, color='#00e5ff', alpha=0.15, label='Reverse Repo (Nadmiar gotówki)')
        ax1.set_ylabel('Reverse Repo (Miliardy $)', color='#00e5ff', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#00e5ff', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        # --- 2. CHĘĆ DAWANIA KREDYTÓW (Oś Prawa) ---
        ax2 = ax1.twinx()
        
        # Rysujemy tylko jeśli mamy dane (nie są zerami)
        has_credit_data = False
        
        if 'CreditCards' in df.columns and df['CreditCards'].sum() != 0:
            ax2.plot(df.index, df['CreditCards'], color='#ff0055', linewidth=2.5, label='Trudność Kredytu: Karty (Ludzie)')
            has_credit_data = True
            
        if 'Mortgages' in df.columns and df['Mortgages'].sum() != 0:
            ax2.plot(df.index, df['Mortgages'], color='#ffd700', linewidth=1.5, linestyle='--', label='Trudność Kredytu: Hipoteki (Domy)')
            has_credit_data = True
        
        # Linia ZERO
        ax2.axhline(0, color=t['text'], linestyle=':', linewidth=1)
        
        if has_credit_data:
            ax2.set_ylabel('% Banków zaostrzających kryteria', color='#ff0055', fontweight='bold')
            # Teksty pomocnicze
            try:
                ax2.text(df.index[0], 5, "TRUDNO (Tightening)", color=t['text'], fontsize=8, va='bottom')
                ax2.text(df.index[0], -5, "ŁATWO (Loosening)", color=t['text'], fontsize=8, va='top')
            except:
                pass
        else:
             ax2.set_ylabel('Brak danych kredytowych z FRED', color=t['text'])

        ax2.tick_params(axis='y', labelcolor='#ff0055', colors=t['text'])
        
        # Ukrywamy ramki
        ax2.spines['top'].set_visible(False); ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False); ax2.spines['bottom'].set_visible(False)

        ax1.set_title("TEORIA BANKOWA: Repo vs Kredyty (SLOOS)", fontsize=14, color=t['text'], fontweight='bold')
        
        # Legenda
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        
        return fig

    # --- NOWOŚĆ: FINANCIAL CONDITIONS INDEX (NFCI) - Autostrada Pieniądza ---
    def get_financial_conditions_index(self):
        """
        Pobiera Chicago Fed National Financial Conditions Index (NFCI).
        Ticker: NFCI
        Wartości ujemne = Luźne warunki (Dobrze dla krypto).
        Wartości dodatnie = Ciasne warunki (Źle dla krypto).
        """
        try:
            start_date = '2018-01-01'
            end_date = datetime.now()
            
            # 1. Pobieramy NFCI (Tygodniowy)
            try:
                nfci = web.DataReader('NFCI', 'fred', start_date, end_date)
            except Exception as e:
                print(f"Błąd NFCI: {e}")
                return None, None
                
            # 2. Pobieramy BTC
            btc = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
            
            # Resampling BTC do tygodniówek (piątek), żeby pasowało do NFCI
            btc_wk = btc.resample('W-FRI').last().ffill()
            nfci_wk = nfci.resample('W-FRI').last().ffill()
            
            # Wyrównanie
            common_idx = nfci_wk.index.intersection(btc_wk.index)
            
            return btc_wk.loc[common_idx], nfci_wk.loc[common_idx]

        except Exception as e:
            print(f"Krytyczny błąd NFCI: {e}")
            return None, None

    def plot_financial_conditions(self, btc, nfci):
        if btc is None or nfci is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- OŚ LEWA: NFCI (Odwrócony) ---
        # Raoul Pal zawsze odwraca ten wykres.
        # Im NIŻEJ na wykresie NFCI (czyli wyżej matematycznie), tym GORZEJ.
        # My zrobimy tak:
        # Linia idzie w GÓRĘ = Warunki się luzują (DOBRZE).
        # Linia idzie w DÓŁ = Warunki się zacieśniają (ŹLE).
        
        # Mnożymy przez -1, żeby wykres szedł w górę, gdy jest dobrze
        inverted_nfci = nfci['NFCI'] * -1
        
        # Kolorujemy tło w zależności od warunków
        # NFCI > 0 (czyli tutaj < 0 po odwróceniu) = STRES (Czerwone)
        # NFCI < 0 (czyli tutaj > 0 po odwróceniu) = LUZ (Zielone)
        
        ax1.fill_between(inverted_nfci.index, inverted_nfci, 0, where=(inverted_nfci >= 0), 
                         color='#00ff55', alpha=0.15, label='Warunki LUŹNE (Risk-On)')
        ax1.fill_between(inverted_nfci.index, inverted_nfci, 0, where=(inverted_nfci < 0), 
                         color='#ff0055', alpha=0.15, label='Warunki CIASNE (Risk-Off)')
        
        ax1.plot(inverted_nfci.index, inverted_nfci, color='#00e5ff', linewidth=2, label='Financial Conditions (Inverted)')
        
        ax1.set_ylabel('Jakość Warunków Finansowych (Im wyżej tym lepiej)', color='#00e5ff', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#00e5ff', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        ax1.axhline(0, color=t['text'], linestyle='-', linewidth=0.5)

        # --- OŚ PRAWA: BITCOIN (Logarytmiczna) ---
        ax2 = ax1.twinx()
        
        # Skala Log dla BTC
        ax2.set_yscale('log')
        ax2.plot(btc.index, btc, color='#ff9900', linewidth=1.5, linestyle='--', label='Cena Bitcoina (Log)')
        
        ax2.set_ylabel('Cena BTC ($)', color='#ff9900', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#ff9900', colors=t['text'])
        
        # Ukrywamy ramki
        ax2.spines['top'].set_visible(False); ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False); ax2.spines['bottom'].set_visible(False)

        ax1.set_title("RISK ON / RISK OFF: Warunki Finansowe (NFCI)", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        
        return fig

    # --- FIX: BUSINESS CYCLE (Switch to INDPRO) ---
    def get_business_cycle_data(self):
        """
        Pobiera Cykl Koniunkturalny.
        ZMIANA: Zamiast 'USSLIND' (który ma braki danych), używamy 'INDPRO'
        (Industrial Production Index). To jest twardy wskaźnik produkcji w USA.
        
        Wersja Raoul Pal: Patrzymy na zmianę roczną (YoY %).
        """
        try:
            start_date = '2018-01-01'
            end_date = datetime.now()
            
            # 1. Pobieramy INDPRO (Industrial Production: Total Index)
            # To jest najpewniejszy wskaźnik cyklu dostępny na FRED.
            try:
                # INDPRO = Industrial Production
                cycle_data = web.DataReader('INDPRO', 'fred', start_date, end_date)
            except Exception as e:
                print(f"Błąd INDPRO: {e}")
                return None, None
            
            # Pobieramy BTC
            btc = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
            
            # --- OBLICZENIA (YoY %) ---
            
            # Resampling do końca miesiąca (ME)
            cycle_monthly = cycle_data.resample('ME').last().ffill()
            
            # Zmiana Roczna (YoY)
            # To pokazuje czy produkcja przyspiesza (nad zerem) czy zwalnia (pod zerem)
            cycle_yoy = cycle_monthly.pct_change(12) * 100
            
            # Wygładzanie (3 miesiące), żeby wykres był czytelny
            cycle_yoy = cycle_yoy.rolling(window=3).mean()
            
            # Interpolacja do tygodniówek (Liniowa - bezpieczna)
            cycle_weekly = cycle_yoy.resample('W').interpolate(method='linear')
            
            # BTC
            btc_monthly = btc.resample('ME').last().ffill()
            btc_yoy = btc_monthly.pct_change(12) * 100
            btc_weekly = btc_yoy.resample('W').interpolate(method='linear')
            
            # Wyrównanie
            common_idx = cycle_weekly.index.intersection(btc_weekly.index)
            common_idx = common_idx[common_idx >= '2019-01-01']
            
            return btc_weekly.loc[common_idx], cycle_weekly.loc[common_idx]

        except Exception as e:
            print(f"Błąd Business Cycle: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_business_cycle(self, btc_yoy, cycle_yoy):
        if btc_yoy is None or cycle_yoy is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- OŚ LEWA: CYKL PRZEMYSŁOWY (Biała) ---
        cycle_color = t['text']
        
        ax1.plot(cycle_yoy.index, cycle_yoy, color=cycle_color, linewidth=3, label='Industrial Production YoY% (The Cycle)')
        
        # Kolorujemy tło
        # INDPRO < 0 = Recesja przemysłowa (Źle)
        # INDPRO > 0 = Wzrost (Dobrze)
        # Ponieważ 'cycle_yoy' to DataFrame, musimy odwołać się do kolumny 'INDPRO'
        ax1.fill_between(cycle_yoy.index, cycle_yoy['INDPRO'], 0, where=(cycle_yoy['INDPRO'] < 0), 
                         color='#ff0055', alpha=0.15, label='RECESJA PRZEMYSŁOWA')
        ax1.fill_between(cycle_yoy.index, cycle_yoy['INDPRO'], 0, where=(cycle_yoy['INDPRO'] >= 0), 
                         color='#00ff55', alpha=0.1, label='EKSPANSJA')

        ax1.set_ylabel('Produkcja Przemysłowa YoY (%)', color=cycle_color, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=cycle_color, colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        ax1.axhline(0, color=t['text'], linestyle=':', linewidth=1)

        # --- OŚ PRAWA: BITCOIN (Pomarańczowa) ---
        ax2 = ax1.twinx()
        btc_color = '#ff9900'
        
        ax2.plot(btc_yoy.index, btc_yoy, color=btc_color, linewidth=1.5, linestyle='--', label='Bitcoin YoY% (RHS)')
        
        ax2.set_ylabel('Bitcoin YoY (%)', color=btc_color, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=btc_color, colors=t['text'])
        
        # Kosmetyka
        ax2.spines['top'].set_visible(False); ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False); ax2.spines['bottom'].set_visible(False)

        ax1.set_title("BUSINESS CYCLE: Produkcja Przemysłowa vs BTC", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', ncol=2, frameon=False, labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        
        return fig

    # --- NOWOŚĆ: MACRO CONTEXT (BTC vs NASDAQ vs DXY) ---
    def get_macro_context_data(self):
        """
        Pobiera kontekst makroekonomiczny wg Raoula Pala.
        Zestawia BTC z:
        1. NASDAQ 100 (^NDX) - "Exponential Age". BTC powinno iść z tym w parze.
        2. DXY (DX-Y.NYB) - "Dollar Wrecking Ball". Siła dolara.
        """
        try:
            start_date = '2019-01-01'
            end_date = datetime.now()
            
            tickers = ['BTC-USD', '^NDX', 'DX-Y.NYB']
            
            data = yf.download(tickers, start=start_date, progress=False)['Close']
            
            # Obsługa MultiIndex (jeśli yfinance zwróci tabelę z poziomami)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1) # Usuwamy 'Ticker' level jeśli jest
            
            # Uzupełnianie braków (Weekendy - krypto działa, giełda nie)
            # Forward fill sprawia, że w weekend cena akcji/dolara jest z piątku
            data = data.ffill()
            
            # Normalizacja do Procentów (Start = 0%)
            # Żeby zobaczyć "Wyścig Aktywów", musimy sprowadzić je do wspólnego mianownika.
            # Ale tutaj zrobimy inaczej - zwrócimy ceny, a normalizację zrobimy na wykresie
            # lub policzymy korelację.
            
            return data

        except Exception as e:
            print(f"Błąd Macro Context: {e}")
            return None

    def plot_macro_context(self, df):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- OŚ LEWA: BTC i NASDAQ (Risk On Assets) ---
        
        # 1. BITCOIN (Pomarańczowy)
        # Normalizujemy do zakresu 0-1 (MinMax) lub % od początku, żeby pasowały do siebie wizualnie?
        # Raoul lubi nakładać wykresy na siebie używając dwóch osi.
        # Dajmy BTC na lewej osi (logarytmicznie).
        
        ax1.plot(df.index, df['BTC-USD'], color='#ff9900', linewidth=2, label='Bitcoin (Price)')
        ax1.set_yscale('log') # Log dla BTC, bo to "Exponential Asset"
        ax1.set_ylabel('Cena BTC (Log)', color='#ff9900', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#ff9900', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        # --- OŚ PRAWA 1: NASDAQ (Technologia) ---
        ax2 = ax1.twinx()
        
        # NASDAQ (Niebieski Cyjan)
        ax2.plot(df.index, df['^NDX'], color='#00e5ff', linewidth=1.5, alpha=0.8, label='NASDAQ 100 (Tech Stocks)')
        
        # --- OŚ PRAWA 2: DXY (Dolar) ---
        # Tu jest trik - DXY rysujemy ODWRÓCONY (Inverted), bo Raoul tak robi.
        # Jak Odwrócony Dolar spada -> to znaczy że Dolar ROŚNIE -> to źle dla krypto.
        # Jak Odwrócony Dolar rośnie -> Dolar SŁABNIE -> dobrze dla krypto.
        # Ale żeby nie robić bałaganu z 3 osiami, po prostu narysujemy DXY na tej samej osi co Nasdaq,
        # ale przeskalowane, albo zrobimy "Third Axis" (trudne w matplotlib).
        
        # ZROBIMY INACZEJ: 
        # Na prawej osi damy NASDAQ.
        # A DXY damy jako tło (Fill Between) albo przerywaną linię, ODWRÓCONĄ.
        
        # Tworzymy wirtualną oś dla DXY, żeby nie zaburzała Nasdaqa
        ax3 = ax1.twinx()
        # Przesuwamy oś w prawo, żeby nie nachodziła na Nasdaq
        ax3.spines["right"].set_position(("axes", 1.1)) 
        
        dxy_inverted = df['DX-Y.NYB'] * -1 # Odwracamy
        ax3.plot(df.index, dxy_inverted, color='#ff0055', linewidth=1, linestyle='--', label='Dolar DXY (Odwrócony)')
        
        ax2.set_ylabel('NASDAQ 100 Points', color='#00e5ff', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#00e5ff', colors=t['text'])
        
        ax3.set_ylabel('DXY Strength (Inverted)', color='#ff0055', fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='#ff0055', colors=t['text'])
        
        # Ukrywamy zbędne ramki
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False); ax2.spines['bottom'].set_visible(False)
        ax3.spines['top'].set_visible(False); ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)

        ax1.set_title("MACRO CONTEXT: BTC vs Tech (NDX) vs Dolar (DXY)", fontsize=14, color=t['text'], fontweight='bold')
        
        # Legenda (Zbiorcza)
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        
        ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        
        return fig

    # --- UPDATE: TGA MONITOR (Historyczny od 2019) ---
    def get_tga_monitor_data(self):
        """
        Pobiera dane konta TGA (Treasury General Account).
        Ticker: WTREGEN (Weekly)
        ZMIANA: Zakres od 2019 roku, aby widzieć cykliczność podatkową.
        """
        try:
            start_date = '2019-01-01' # Zmiana na 2019
            end_date = datetime.now()
            
            # Pobieramy TGA
            try:
                tga = web.DataReader('WTREGEN', 'fred', start_date, end_date)
            except Exception as e:
                print(f"Błąd TGA: {e}")
                return None

            # Dane są w Miliardach (Billions)
            # Sprawdzenie jednostek dla pewności
            last_val = tga['WTREGEN'].iloc[-1]
            if last_val > 10000: # Jeśli to miliony, zamień na miliardy
                 tga = tga / 1000
            
            return tga

        except Exception as e:
            print(f"Krytyczny błąd TGA: {e}")
            return None

    def plot_tga_monitor(self, tga):
        if tga is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- Rysujemy TGA ---
        # Kolor: Turkusowy/Morski
        line_color = '#00e5ff'
        
        ax1.plot(tga.index, tga['WTREGEN'], color=line_color, linewidth=3, label='Stan Konta TGA (Mld $)')
        
        # --- TARGET RZĄDOWY (850-900 mld) ---
        # Rysujemy strefę docelową, o której pisałeś
        ax1.axhspan(850, 900, color='#ff0055', alpha=0.15, label='Cel Rządu (Drenaż Podatkowy)')
        ax1.text(tga.index[0], 875, "CEL PO PODATKACH (900 mld)", color='#ff0055', fontsize=10, va='center')

        # --- OPISY STRZAŁKAMI (Edukacyjne) ---
        # Żebyś od razu wiedział co się dzieje
        last_date = tga.index[-1]
        last_val = tga['WTREGEN'].iloc[-1]
        
        ax1.annotate('📉 SPADEK = WYPŁATY = POMPA (Teraz)', 
                     xy=(last_date, last_val), 
                     xytext=(last_date - pd.Timedelta(weeks=12), last_val - 100),
                     arrowprops=dict(facecolor='#00ff55', shrink=0.05),
                     color='#00ff55', fontweight='bold')
                     
        ax1.text(tga.index[int(len(tga)*0.1)], 200, "⬆️ WZROST = PODATKI = RUG PULL (Kwiecień)", color='#ff0055', fontsize=10)

        # Ustawienia osi
        ax1.set_ylabel('Stan Konta TGA (Miliardy USD)', color=line_color, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=line_color, colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        # Tytuł
        ax1.set_title(f"TGA MONITOR: Obserwuj Linię (Aktualnie: {last_val:.0f} mld $)", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- NOWOŚĆ: ETH/BTC RATIO (Altcoin Season Signal) ---
    def get_crypto_rotation_data(self):
        """
        Pobiera dane dla pary ETH/BTC.
        To jest wskaźnik rotacji kapitału wg Raoula Pala.
        Jeśli ETH/BTC rośnie -> Kapitał płynie do Altcoinów (Risk On).
        Jeśli ETH/BTC spada -> Kapitał ucieka do Bitcoina (Risk Off).
        """
        try:
            start_date = '2019-01-01'
            end_date = datetime.now()
            
            tickers = ['BTC-USD', 'ETH-USD']
            data = yf.download(tickers, start=start_date, progress=False)['Close']
            
            # Obsługa MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Uzupełnianie braków
            data = data.ffill()
            
            # Obliczamy Ratio
            data['ETH_BTC_RATIO'] = data['ETH-USD'] / data['BTC-USD']
            
            # Dodajemy średnią (np. 20 tygodniową), żeby widzieć trend
            data['SMA_20'] = data['ETH_BTC_RATIO'].rolling(window=140).mean() # 140 dni ~ 20 tygodni
            
            return data

        except Exception as e:
            print(f"Błąd Crypto Rotation: {e}")
            return None

    def plot_crypto_rotation(self, df):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- ETH/BTC RATIO ---
        # Kolor: Fioletowy (Ethereum)
        eth_color = '#a020f0' 
        
        ax1.plot(df.index, df['ETH_BTC_RATIO'], color=eth_color, linewidth=2, label='ETH / BTC Ratio')
        
        # Średnia (Trend)
        ax1.plot(df.index, df['SMA_20'], color='white', linewidth=1, linestyle='--', label='Trend (20-week SMA)', alpha=0.7)
        
        # Wypełnienie tła - Sygnał
        # Jeśli Ratio > SMA -> ALT SEASON (Zielone tło)
        # Jeśli Ratio < SMA -> BTC SEASON (Czerwone tło)
        
        ax1.fill_between(df.index, df['ETH_BTC_RATIO'], df['SMA_20'], 
                         where=(df['ETH_BTC_RATIO'] > df['SMA_20']), 
                         color='#00ff55', alpha=0.1, label='ALTCOIN SEASON (ETH Strong)')
                         
        ax1.fill_between(df.index, df['ETH_BTC_RATIO'], df['SMA_20'], 
                         where=(df['ETH_BTC_RATIO'] <= df['SMA_20']), 
                         color='#ff0055', alpha=0.1, label='BITCOIN SEASON (ETH Weak)')

        ax1.set_ylabel('Cena ETH wyrażona w BTC', color=eth_color, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=eth_color, colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        # Tytuł
        last_val = df['ETH_BTC_RATIO'].iloc[-1]
        trend_val = df['SMA_20'].iloc[-1]
        status = "ALTCOIN SEASON 🚀" if last_val > trend_val else "BITCOIN DOMINANCE 🛡️"
        
        ax1.set_title(f"ETH/BTC RATIO: {status}", fontsize=16, color=t['text'], fontweight='bold')
        
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- PRO TOOLS: BITCOIN POWER LAW (Forecast 2032) ---
    def get_bitcoin_power_law(self):
        """
        Oblicza Model Potęgowy Bitcoina (Power Law Corridor).
        ZMIANA: Generuje prognozę (linie tunelu) aż do 2032 roku.
        """
        try:
            start_date = '2010-01-01'
            end_date = datetime.now()
            
            # 1. Pobieramy Historię
            data = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            
            if isinstance(data, pd.DataFrame): 
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs('Close', axis=1, level=0)
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
            
            data = data[data > 0].dropna()
            
            # 2. PRZYGOTOWANIE DANYCH HISTORYCZNYCH
            genesis_date = pd.Timestamp('2009-01-03')
            df = pd.DataFrame({'price': data})
            df['date'] = df.index
            df['days_since_genesis'] = (df['date'] - genesis_date).dt.days
            df = df[df['days_since_genesis'] > 0]
            
            # Logarytmy do regresji
            df['log_days'] = np.log10(df['days_since_genesis'])
            df['log_price'] = np.log10(df['price'])
            
            # 3. OBLICZANIE MODELU (REGRESJA)
            # Uczymy model TYLKO na danych historycznych
            slope, intercept = np.polyfit(df['log_days'], df['log_price'], 1)
            
            # Obliczamy odchylenie standardowe (szerokość kanału)
            # Najpierw liczymy linię środkową dla historii
            model_log_history = (slope * df['log_days']) + intercept
            residuals = df['log_price'] - model_log_history
            std_dev = residuals.std()
            
            # 4. GENEROWANIE PRZYSZŁOŚCI (FORECAST)
            # Tworzymy daty od dzisiaj do 2032-12-31
            last_date = df['date'].iloc[-1]
            future_date_end = pd.Timestamp('2032-12-31')
            
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=future_date_end, freq='D')
            
            # Tworzymy pusty DataFrame dla przyszłości
            df_future = pd.DataFrame({'date': future_dates})
            df_future['price'] = np.nan # Przyszła cena jest nieznana (na razie)
            df_future['days_since_genesis'] = (df_future['date'] - genesis_date).dt.days
            df_future['log_days'] = np.log10(df_future['days_since_genesis'])
            
            # 5. ŁĄCZENIE (Historia + Przyszłość)
            # Używamy pd.concat zamiast append (nowy standard pandas)
            df_total = pd.concat([df, df_future], ignore_index=True)
            
            # 6. APLIKOWANIE MODELU NA CAŁOŚĆ (Historia + Przyszłość)
            # Teraz wyliczamy linie tunelu dla WSZYSTKICH dat (starych i nowych)
            df_total['model_log_price'] = (slope * df_total['log_days']) + intercept
            df_total['model_price'] = 10 ** df_total['model_log_price']
            
            # Bandy (Strefy)
            df_total['bottom_band'] = 10 ** (df_total['model_log_price'] - (1.5 * std_dev))
            df_total['top_band'] = 10 ** (df_total['model_log_price'] + (2.0 * std_dev))
            
            return df_total

        except Exception as e:
            print(f"Błąd Power Law: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_power_law(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- Rysujemy Wykres (Skala Logarytmiczna jest kluczowa) ---
        ax1.set_yscale('log')
        
        # 1. Rzeczywista cena BTC
        ax1.plot(df['date'], df['price'], color='white', linewidth=1, label='Cena Bitcoina')
        
        # 2. Tunel Potęgowy
        # Górna Banda (Czerwona - Sprzedawaj)
        ax1.plot(df['date'], df['top_band'], color='#ff0055', linewidth=2, linestyle='--', label='SZCZYT BAŃKI (Bubble Territory)')
        
        # Dolna Banda (Zielona - Kupuj)
        ax1.plot(df['date'], df['bottom_band'], color='#00ff55', linewidth=2, linestyle='--', label='DNO GENERACYJNE (Buy Zone)')
        
        # Środek (Fair Value)
        ax1.plot(df['date'], df['model_price'], color='#00e5ff', linewidth=1, linestyle=':', alpha=0.5, label='Fair Value (Środek)')
        
        # Kolorowanie stref
        ax1.fill_between(df['date'], df['top_band'], df['model_price'], color='#ff0055', alpha=0.05)
        ax1.fill_between(df['date'], df['model_price'], df['bottom_band'], color='#00ff55', alpha=0.05)
        
        # Opisy
        last_price = df['price'].iloc[-1]
        fair_val = df['model_price'].iloc[-1]
        
        # Sprawdzamy, czy jest tanio czy drogo
        status = "TANIO (Okazja)" if last_price < fair_val else "DROGO (Ryzyko)"
        color_status = '#00ff55' if last_price < fair_val else '#ff0055'
        
        ax1.set_title(f"BITCOIN POWER LAW: {status}", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena BTC (Log)', color=t['text'])
        
        # Legenda
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        # Formatowanie
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'], which='both') # Siatka logarytmiczna
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- PRO TOOLS: PI CYCLE TOP INDICATOR (Max History) ---
    def get_pi_cycle_data(self):
        """
        Oblicza Pi Cycle Top Indicator.
        ZMIANA: Start od 2010-01-01.
        Próbujemy pobrać absolutnie całą historię dostępną na Yahoo Finance.
        Nawet jeśli dane zaczną się w 2014 (ograniczenie Yahoo), 
        to da nam to najwcześniejszy możliwy start wskaźnika (ok. połowy 2015).
        """
        try:
            # Ustawiamy 2010, żeby wymusić pobranie wszystkiego co istnieje
            start_date = '2010-01-01' 
            end_date = datetime.now()
            
            data = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            
            if isinstance(data, pd.DataFrame): 
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs('Close', axis=1, level=0)
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
            
            df = pd.DataFrame({'price': data})
            
            # Obliczamy średnie
            # 111 DMA (Szybka)
            df['MA_111'] = df['price'].rolling(window=111).mean()
            
            # 350 DMA x 2 (Wolna * 2)
            # To ta linia "zjada" pierwsze 350 dni danych
            df['MA_350_x2'] = df['price'].rolling(window=350).mean() * 2
            
            # Usuwamy puste okresy rozbiegowe
            df = df.dropna()
            
            return df

        except Exception as e:
            print(f"Błąd Pi Cycle: {e}")
            return None

    def plot_pi_cycle(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # Wykres ceny
        ax1.plot(df.index, df['price'], color='white', linewidth=1, alpha=0.5, label='Cena BTC')
        
        # Linie Pi Cycle
        # 111 DMA (Pomarańczowa)
        ax1.plot(df.index, df['MA_111'], color='#ff9900', linewidth=1.5, label='111 DMA')
        
        # 350 DMA x2 (Zielona)
        ax1.plot(df.index, df['MA_350_x2'], color='#00ff55', linewidth=1.5, label='350 DMA x2')
        
        # --- FIX: WYKRYWANIE PRZECIĘCIA (CROSSOVER) ---
        # Zamiast zaznaczać każdy dzień, kiedy warunek jest spełniony,
        # zaznaczamy tylko MOMENT zmiany (gdy wczoraj było pod, a dziś jest nad).
        
        # Warunek 1: Dziś 111 > 350x2
        condition_now = df['MA_111'] >= df['MA_350_x2']
        
        # Warunek 2: Wczoraj 111 < 350x2 (używamy shift(1) żeby sprawdzić poprzedni dzień)
        condition_yesterday = ~condition_now.shift(1).fillna(False)
        
        # Koniunkcja: Dziś TAK i Wczoraj NIE = Przecięcie
        crossover_mask = condition_now & condition_yesterday
        
        # Filtrujemy dane
        crosses = df[crossover_mask]
        
        if not crosses.empty:
            # Rysujemy duże, wyraźne kropki tylko w momentach przecięcia
            ax1.scatter(crosses.index, crosses['price'], color='#ff0000', s=100, zorder=5, edgecolors='white', linewidth=1.5, label='SYGNAŁ SPRZEDAŻY (Cross)')
            
            # Opcjonalnie: Dodajemy datę przy sygnale, żebyś wiedział dokładnie kiedy to było
            for date, price in zip(crosses.index, crosses['price']):
                ax1.text(date, price * 1.3, f"{date.strftime('%Y-%m-%d')}", color='#ff0000', fontsize=8, ha='center', rotation=45)
        
        ax1.set_title("PI CYCLE TOP: Czy to już szczyt?", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena ($)', color=t['text'])
        ax1.set_yscale('log')
        
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- PRO TOOLS: PUELL MULTIPLE (Miner Stress / On-Chain) ---
    def get_puell_multiple_data(self):
        """
        Oblicza PUELL MULTIPLE.
        Wskaźnik on-chain badający przychody górników.
        Wzór: Dzienna Wartość Emisji / 365-dniowa Średnia Wartości Emisji.
        """
        try:
            start_date = '2014-01-01'
            end_date = datetime.now()
            
            # 1. Pobieramy Cenę BTC
            data = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            
            if isinstance(data, pd.DataFrame): 
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs('Close', axis=1, level=0)
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
            
            df = pd.DataFrame({'price': data})
            df = df.dropna()
            
            # 2. SYMULACJA EMISJI (Nagroda za blok * 144 bloki dziennie)
            # Musimy wiedzieć, ile BTC trafiało na rynek każdego dnia.
            # Halvingi:
            # - Pre-2012: 50 BTC
            # - 2012-11-28: 25 BTC
            # - 2016-07-09: 12.5 BTC
            # - 2020-05-11: 6.25 BTC
            # - 2024-04-20: 3.125 BTC
            
            # Tworzymy kolumnę z nagrodą (zakładamy 144 bloki dziennie)
            df['coins_issued'] = 0.0
            
            dates = df.index
            
            # Przypisujemy nagrody wg dat (progi halvingowe)
            # Uwaga: Daty są przybliżone (halving zależy od bloku, nie daty), ale błąd jest pomijalny dla makro.
            
            # Definiujemy okresy
            date_2016 = pd.Timestamp('2016-07-09')
            date_2020 = pd.Timestamp('2020-05-11')
            date_2024 = pd.Timestamp('2024-04-20')
            
            # Logika wektorowa (szybka)
            df.loc[df.index < date_2016, 'coins_issued'] = 25.0 * 144
            df.loc[(df.index >= date_2016) & (df.index < date_2020), 'coins_issued'] = 12.5 * 144
            df.loc[(df.index >= date_2020) & (df.index < date_2024), 'coins_issued'] = 6.25 * 144
            df.loc[df.index >= date_2024, 'coins_issued'] = 3.125 * 144
            
            # 3. OBLICZANIE PUELL MULTIPLE
            # Miner Revenue ($) = Coins Issued * Price
            df['miner_revenue'] = df['coins_issued'] * df['price']
            
            # Mianownik: 365-dniowa średnia przychodów
            df['revenue_ma_365'] = df['miner_revenue'].rolling(window=365).mean()
            
            # Puell Multiple
            df['puell_multiple'] = df['miner_revenue'] / df['revenue_ma_365']
            
            return df.dropna()

        except Exception as e:
            print(f"Błąd Puell Multiple: {e}")
            return None

    def plot_puell_multiple(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- OŚ LEWA: PUELL MULTIPLE (Wskaźnik) ---
        # Rysujemy to jako linię lub obszar
        puell_color = '#ff9900' # Pomarańczowy Bitcoinowy
        
        ax1.plot(df.index, df['puell_multiple'], color=puell_color, linewidth=1.5, label='Puell Multiple')
        
        # STREFY (Klucz do zarabiania)
        # BUY ZONE (Miner Capitulation): < 0.5 (Historycznie idealne dno)
        ax1.axhspan(0, 0.5, color='#00ff55', alpha=0.15, label='BUY ZONE (Miner Capitulation)')
        ax1.axhline(0.5, color='#00ff55', linestyle='--', linewidth=1)
        
        # SELL ZONE (Miner Euphoria): > 4.0 (Szczyty baniek)
        # W ostatnich cyklach szczyty są niższe, więc 3.5-4.0 to strefa ostrzegawcza.
        ax1.axhspan(4.0, 10, color='#ff0055', alpha=0.15, label='SELL ZONE (Euphoria)')
        ax1.axhline(4.0, color='#ff0055', linestyle='--', linewidth=1)
        
        ax1.set_ylabel('Puell Multiple Value', color=puell_color, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=puell_color, colors=t['text'])
        ax1.set_ylim(0, 6) # Ograniczamy widok, żeby szpilki nie spłaszczyły wykresu
        
        # --- OŚ PRAWA: CENA BTC (Logarytmiczna) ---
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        
        ax2.plot(df.index, df['price'], color='white', linewidth=1, alpha=0.5, label='Cena BTC (Log)')
        ax2.set_ylabel('Cena BTC ($)', color='white')
        ax2.tick_params(axis='y', labelcolor='white', colors=t['text'])

        # Tytuł i Legenda
        last_val = df['puell_multiple'].iloc[-1]
        
        # Interpretacja
        if last_val < 0.5: status = "KUPUJ (Kapitulacja Górników) 🟢"
        elif last_val > 4.0: status = "SPRZEDAWAJ (Euforia) 🔴"
        else: status = "HODL (Neutral) ⚪"
        
        ax1.set_title(f"PUELL MULTIPLE: {status} (Val: {last_val:.2f})", fontsize=16, color=t['text'], fontweight='bold')
        
        # Łączona legenda
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', ncol=3, frameon=False, labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- PRO TOOLS: 2-YEAR MA MULTIPLIER (Investor Tool) ---
    def get_two_year_multiplier_data(self):
        """
        Oblicza 2-Year MA Multiplier.
        Narzędzie dla inwestorów długoterminowych.
        Zielona Linia: Średnia 2-letnia (730 dni). Strefa zakupów.
        Czerwona Linia: Średnia 2-letnia x 5. Strefa sprzedaży (FOMO).
        """
        try:
            # Potrzebujemy dużo danych wstecz (2 lata na rozbieg)
            start_date = '2012-01-01'
            end_date = datetime.now()
            
            data = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            
            if isinstance(data, pd.DataFrame): 
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs('Close', axis=1, level=0)
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
            
            df = pd.DataFrame({'price': data})
            df = df.dropna()
            
            # 1. Obliczamy 2-Year MA (730 dni)
            # To jest nasza "Podłoga"
            df['MA_2YR'] = df['price'].rolling(window=730).mean()
            
            # 2. Obliczamy Sufit (x5)
            # Historycznie, gdy cena przekracza 5x średnią 2-letnią, bańka pęka.
            df['MA_2YR_x5'] = df['MA_2YR'] * 5
            
            # Usuwamy okres rozbiegowy
            df = df.dropna()
            
            return df

        except Exception as e:
            print(f"Błąd 2-Year Multiplier: {e}")
            return None

    def plot_two_year_multiplier(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # Obowiązkowa skala logarytmiczna
        ax1.set_yscale('log')
        
        # 1. LINIE WSKAŹNIKA (Warstwa 2)
        ax1.plot(df.index, df['MA_2YR'], color='#00ff55', linewidth=2, label='STREFA ZAKUPU (2yr MA)', zorder=2)
        ax1.plot(df.index, df['MA_2YR_x5'], color='#ff0055', linewidth=2, label='STREFA SPRZEDAŻY (2yr MA x5)', zorder=2)
        
        # 2. RZECZYWISTA CENA (VISUAL LIFT - Warstwa 3)
        # Przesuwamy o 15% w górę
        visual_price = df['price'] * 1.15 
        ax1.plot(df.index, visual_price, color='white', linewidth=1, alpha=0.9, label='Cena BTC (Przesunięta)', zorder=3)
        
        # 3. KOLOROWANIE OBSZARÓW (NAPRAWA LOGIKI)
        
        # Sygnał KUPNA (Zielony)
        # Tu zostawiamy oryginalną cenę (df['price']), bo chcemy kupować tylko gdy jest NAPRAWDĘ tanio.
        # Gdybyśmy użyli visual_price, trudniej byłoby dotknąć dna.
        ax1.fill_between(df.index, 0, df['MA_2YR'], 
                         where=(df['price'] < df['MA_2YR']), 
                         color='#00ff55', alpha=0.4, label='SUPER OKAZJA (Under Valued)', zorder=1)
                         
        # Sygnał SPRZEDAŻY (Czerwony - FIX)
        # ZMIANA: Używamy 'visual_price' do warunku.
        # Skoro biała linia dotyka czerwonej, to uznajemy to za strefę bańki.
        # To naprawi "dziury" w podświetleniu na szczytach.
        ax1.fill_between(df.index, df['MA_2YR_x5'], visual_price * 1.5, 
                         where=(visual_price > df['MA_2YR_x5']), 
                         color='#ff0055', alpha=0.4, label='BAŃKA (Visual Touch)', zorder=1)

        # Opisy
        last_val = visual_price.iloc[-1] # Patrzymy na linię wizualną
        floor = df['MA_2YR'].iloc[-1]
        ceiling = df['MA_2YR_x5'].iloc[-1]
        
        if df['price'].iloc[-1] < floor: # Do oceny dna używamy realnej ceny
            status = "SUPER TANIO (Kupuj!)"
        elif last_val > ceiling: # Do oceny szczytu używamy wizualnej (wcześniejsze ostrzeżenie)
            status = "EKSTREMALNIE DROGO (Uciekaj!)"
        else:
            status = "HODL (Czekaj)"
            
        ax1.set_title(f"2-YEAR MA MULTIPLIER: {status}", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena BTC (Log) - Offset +15%', color=t['text'])
        
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'], which='both')
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- PRO TOOLS: GOLDEN RATIO MULTIPLIER (Math of God) ---
    def get_golden_ratio_data(self):
        """
        Oblicza Golden Ratio Multiplier.
        Oparte na średniej 350 DMA (tak jak Pi Cycle), ale mnożonej przez ciąg Fibonacciego.
        To wyznacza 'naturalne' poziomy oporu dla Bitcoina.
        """
        try:
            start_date = '2010-01-01' # Max historia
            end_date = datetime.now()
            
            data = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            
            if isinstance(data, pd.DataFrame): 
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs('Close', axis=1, level=0)
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
            
            df = pd.DataFrame({'price': data})
            df = df.dropna()
            
            # BAZA: 350 DMA (To jest "Kręgosłup" rynku)
            df['MA_350'] = df['price'].rolling(window=350).mean()
            
            # POZIOMY FIBONACCIEGO
            # 1.618 (Golden Ratio) - Często lokalny szczyt lub silny opór
            df['GR_1.6'] = df['MA_350'] * 1.618
            
            # 2.0 - Poziom realizowania zysków
            df['GR_2.0'] = df['MA_350'] * 2.0
            
            # 3.0 - Szczyt Cyklu (Historycznie)
            df['GR_3.0'] = df['MA_350'] * 3.0
            
            # Usuwamy okres rozbiegowy
            df = df.dropna()
            
            return df

        except Exception as e:
            print(f"Błąd Golden Ratio: {e}")
            return None

    def plot_golden_ratio(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # Logarytm obowiązkowy
        ax1.set_yscale('log')
        
        # 1. CENA (Biała)
        ax1.plot(df.index, df['price'], color='white', linewidth=1, alpha=0.8, label='Cena BTC', zorder=3)
        
        # 2. KRĘGOSŁUP (350 DMA - Szara/Pomarańczowa)
        ax1.plot(df.index, df['MA_350'], color='#ff9900', linewidth=1.5, linestyle='--', label='Baza (350 DMA)', zorder=2)
        
        # 3. POZIOMY FIBONACCIEGO (Schody)
        
        # x1.6 (Golden Ratio - Żółta)
        ax1.plot(df.index, df['GR_1.6'], color='#ffd700', linewidth=1.5, label='OPÓR (x1.6 Golden Ratio)', zorder=2)
        ax1.fill_between(df.index, df['MA_350'], df['GR_1.6'], color='#ffd700', alpha=0.05)
        
        # x2.0 (Czerwona - Take Profit)
        ax1.plot(df.index, df['GR_2.0'], color='#ff0055', linewidth=1.5, label='TAKE PROFIT (x2.0)', zorder=2)
        
        # x3.0 (Fioletowa - MAX PAIN / TOP)
        ax1.plot(df.index, df['GR_3.0'], color='#a020f0', linewidth=2, label='SZCZYT CYKLU (x3.0)', zorder=2)
        
        # Wypełnienie strefy FOMO (między x2 a x3)
        ax1.fill_between(df.index, df['GR_2.0'], df['GR_3.0'], color='#ff0055', alpha=0.1, label='STREFA FOMO')

        # Marker na dzisiaj
        last_price = df['price'].iloc[-1]
        next_target = 0
        target_name = ""
        
        # Sprawdzamy gdzie jesteśmy
        if last_price < df['GR_1.6'].iloc[-1]:
            next_target = df['GR_1.6'].iloc[-1]
            target_name = "x1.6"
        elif last_price < df['GR_2.0'].iloc[-1]:
            next_target = df['GR_2.0'].iloc[-1]
            target_name = "x2.0"
        else:
            next_target = df['GR_3.0'].iloc[-1]
            target_name = "SZCZYT (x3.0)"
            
        ax1.set_title(f"GOLDEN RATIO MULTIPLIER: Cel -> {target_name} (${next_target:,.0f})", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena BTC (Log)', color=t['text'])
        
        # Legenda
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'], which='both')
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- PRO TOOLS: MVRV Z-SCORE (Start 2015) ---
    def get_mvrv_z_score_data(self):
        """
        Oblicza MVRV Z-Score.
        ZMIANA: Wyświetlanie od 2015 roku.
        Technicznie pobieramy dane od 2011, żeby 'nakarmić' średnią 4-letnią (1400 dni),
        ale wynik przycinamy, żebyś widział wykres od 2015.
        """
        try:
            # 1. Pobieramy dane z zapasem (musimy mieć historię dla średniej)
            start_download = '2011-01-01' 
            end_date = datetime.now()
            
            data = yf.download('BTC-USD', start=start_download, progress=False)['Close']
            
            if isinstance(data, pd.DataFrame): 
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs('Close', axis=1, level=0)
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
            
            df = pd.DataFrame({'price': data})
            df = df.dropna()
            
            # 2. Obliczenia (Wymagają 1400 dni historii)
            # Symulacja Realized Price
            df['realized_price_proxy'] = df['price'].rolling(window=1400).mean()
            
            # Odchylenie Standardowe
            df['std_dev'] = df['price'].rolling(window=1400).std()
            
            # Z-Score
            df['z_score'] = (df['price'] - df['realized_price_proxy']) / df['std_dev']
            
            # 3. FILTROWANIE (Wyświetlamy tylko od 2015)
            # Dopiero teraz, gdy mamy obliczone wartości, ucinamy stare lata.
            view_start_date = '2015-01-01'
            df_view = df[df.index >= view_start_date]
            
            return df_view.dropna()

        except Exception as e:
            print(f"Błąd MVRV Z-Score: {e}")
            return None

    def plot_mvrv_z_score(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
        plt.subplots_adjust(hspace=0.05) # Zmniejszamy odstęp między wykresami
        
        # --- GÓRNY WYKRES: CENA vs FAIR VALUE ---
        ax1.set_yscale('log')
        ax1.plot(df.index, df['price'], color='white', linewidth=1, label='Cena Rynkowa (Market Price)')
        ax1.plot(df.index, df['realized_price_proxy'], color='#00e5ff', linewidth=1.5, linestyle='--', label='Cena Zrealizowana (Fair Value)')
        
        ax1.set_title("MVRV CONTEXT: Cena vs Wartość Godziwa", fontsize=14, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena BTC (Log)', color=t['text'])
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        
        # --- DOLNY WYKRES: Z-SCORE (OSCYLATOR) ---
        # To jest "Serce" tego wskaźnika
        z_val = df['z_score']
        
        # Kolorowanie linii w zależności od wartości
        # Tworzymy mapę kolorów, ale prościej narysować po prostu linię i wypełnić strefy
        ax2.plot(df.index, z_val, color='#ff9900', linewidth=1.5, label='MVRV Z-Score')
        
        # STREFA SPRZEDAŻY (CZERWONA)
        # Historycznie szczyty są powyżej wartości 6-7
        ax2.axhspan(7, 12, color='#ff0055', alpha=0.2, label='STREFA BAŃKI (Top Zone)')
        ax2.axhline(7, color='#ff0055', linestyle='--', linewidth=1)
        
        # STREFA ZAKUPU (ZIELONA)
        # Historycznie dna są poniżej 0 (lub blisko 0)
        ax2.axhspan(-2, 0.1, color='#00ff55', alpha=0.2, label='STREFA OKAZJI (Bottom Zone)')
        ax2.axhline(0, color='#00ff55', linestyle='--', linewidth=1)
        
        # Linia środkowa
        ax2.axhline(3.5, color='grey', linestyle=':', alpha=0.3)
        
        ax2.set_ylabel('Z-Score Value', color='#ff9900', fontweight='bold')
        ax2.set_ylim(-1.5, 11) # Zakres typowy dla Z-Score
        
        # Opis aktualnego stanu
        last_z = z_val.iloc[-1]
        if last_z < 0.1: 
            status = "DNO (Kupuj agresywnie)"
            stat_color = '#00ff55'
        elif last_z > 7: 
            status = "SZCZYT (Sprzedawaj wszystko)"
            stat_color = '#ff0055'
        else: 
            status = "W RYTMIE (HODL)"
            stat_color = t['text']
            
        ax2.text(df.index[-1], last_z + 0.5, f"Z: {last_z:.2f}\n{status}", color=stat_color, fontweight='bold', ha='right')
        
        # Kosmetyka
        for ax in [ax1, ax2]:
            ax.patch.set_facecolor(t['bg'])
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
            ax.tick_params(colors=t['text'])
        
        fig.patch.set_facecolor(t['bg'])
        
        return fig

    # --- PRO TOOLS: YIELD CURVE + NBER RECESSIONS (Start 2000) ---
    def get_yield_curve_data(self):
        """
        Pobiera Krzywą Dochodowości (T10Y2Y) oraz Oficjalne Recesje (USREC).
        Dane od roku 2000.
        """
        try:
            start_date = '2000-01-01'
            end_date = datetime.now()
            
            # Pobieramy:
            # T10Y2Y = Spread (Yield Curve)
            # USREC = NBER Recession Indicator (1 = Recesja, 0 = Norma)
            tickers = ['T10Y2Y', 'USREC']
            
            # Pobieramy dane makro
            macro_data = web.DataReader(tickers, 'fred', start_date, end_date)
            
            # USREC jest miesięczny, T10Y2Y dzienny. Musimy to wyrównać.
            # Robimy forward fill dla recesji (żeby każdy dzień miesiąca miał status '1')
            macro_data = macro_data.resample('D').ffill()
            
            # Pobieramy BTC (Pojawi się dopiero ok 2014 roku, wcześniej będzie NaN - to OK)
            btc = yf.download('BTC-USD', start=start_date, progress=False)['Close']
            
            if isinstance(btc, pd.DataFrame): 
                if isinstance(btc.columns, pd.MultiIndex):
                    btc = btc.xs('Close', axis=1, level=0)
                if isinstance(btc, pd.DataFrame):
                    btc = btc.iloc[:, 0]
            
            # Łączymy wszystko
            df = macro_data.join(btc.rename('btc'), how='outer')
            
            # Usuwamy tylko te wiersze, gdzie nie ma Yield Curve (BTC może być puste)
            df = df.dropna(subset=['T10Y2Y'])
            
            return df

        except Exception as e:
            print(f"Błąd Yield Curve: {e}")
            return None

    def plot_yield_curve(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- TŁO: PASY RECESJI (NBER) ---
        # Rysujemy szare pasy tam, gdzie USREC == 1
        # Robimy to "pod spodem" (zorder=0)
        
        # Trik: fill_between działa na osi X, a my chcemy całą wysokość.
        # Ale możemy użyć transformacji lub po prostu ustawić duże wartości Y.
        # Yield curve rzadko wychodzi poza -2 i +5.
        
        ax1.fill_between(df.index, -5, 10, where=(df['USREC'] == 1), 
                         color='grey', alpha=0.25, label='OFICJALNA RECESJA (NBER)', zorder=0)
        
        # --- OŚ LEWA: SPREAD 10Y-2Y ---
        
        # 1. Strefa Inwersji (Czerwona) - SYGNAŁ OSTRZEGAWCZY
        ax1.fill_between(df.index, df['T10Y2Y'], 0, where=(df['T10Y2Y'] < 0), 
                         color='#ff0055', alpha=0.4, label='INWERSJA (Sygnał)', zorder=1)
        
        # 2. Strefa Normalna (Zielona)
        ax1.fill_between(df.index, df['T10Y2Y'], 0, where=(df['T10Y2Y'] >= 0), 
                         color='#00ff55', alpha=0.1, label='Normalność', zorder=1)
        
        # 3. Linia Spreadu (Biała)
        ax1.plot(df.index, df['T10Y2Y'], color='white', linewidth=1.5, label='Spread 10Y-2Y', zorder=2)
        
        # 4. Linia Zerowa
        ax1.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        
        ax1.set_ylabel('Spread 10Y - 2Y (%)', color='white', fontweight='bold')
        ax1.set_ylim(-2, 4) # Ograniczamy widok Y, żeby szare pasy były ładne
        ax1.tick_params(axis='y', labelcolor='white', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        # --- OŚ PRAWA: BTC (Od 2014) ---
        # Rysujemy tylko tam, gdzie są dane (automatycznie)
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.plot(df.index, df['btc'], color='#ff9900', linewidth=1.5, linestyle=':', alpha=0.9, label='Cena BTC (Log)', zorder=3)
        ax2.set_ylabel('Cena BTC ($)', color='#ff9900')
        ax2.tick_params(axis='y', labelcolor='#ff9900', colors=t['text'])
        
        # Tytuł i Status
        last_val = df['T10Y2Y'].iloc[-1]
        
        if df['USREC'].iloc[-1] == 1:
            status = "TRWA RECESJA 🚨"
        elif last_val < 0: 
            status = "INWERSJA (Czekamy na wybuch) ⚠️"
        elif last_val < 0.5: # Tuż po wyjściu z inwersji
            status = "DE-INWERSJA (Strefa Śmierci) 💀"
        else: 
            status = "NORMALNIE 🟢"
        
        ax1.set_title(f"YIELD CURVE & RECESJE (2000-TERAZ): {status} [{last_val:.2f}%]", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda (Scalona)
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- PRO TOOLS: WHALE DIVERGENCE (Fix Danych) ---
    def get_whale_divergence_data(self):
        """
        Oblicza dywergencję między RSI (Cena/Emocje) a MFI (Wolumen/Pieniądze).
        POPRAWKA: Naprawiono błąd 'MultiIndex' z biblioteki yfinance.
        """
        try:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d') # 2 lata wstecz
            
            # Pobieramy OHLCV (Wolumen jest kluczowy!)
            data = yf.download('BTC-USD', start=start_date, progress=False)
            
            # --- NAPRAWA STRUKTURY DANYCH (CRITICAL FIX) ---
            if isinstance(data.columns, pd.MultiIndex):
                # Sprawdzamy, na którym poziomie są nazwy cen ('Close', 'Volume' itd.)
                # Poziom 0?
                if 'Close' in data.columns.get_level_values(0):
                    data.columns = data.columns.get_level_values(0)
                # Poziom 1?
                elif 'Close' in data.columns.get_level_values(1):
                    data.columns = data.columns.get_level_values(1)
            
            # Upewniamy się, że mamy potrzebne kolumny (w tym Volume)
            # Jeśli są z małej litery (np. 'volume'), zmieniamy na dużą ('Volume')
            data.columns = [c.capitalize() for c in data.columns]
            
            required_cols = ['Close', 'High', 'Low', 'Volume']
            
            # Sprawdzenie ostateczne
            if not all(col in data.columns for col in required_cols):
                print(f"Błąd struktur danych Whale Div: Dostępne kolumny to {data.columns}")
                return None

            df = data[required_cols].copy()
            
            # 1. OBLICZANIE RSI (14) - Wskaźnik "Ulicy"
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 2. OBLICZANIE MFI (14) - Wskaźnik "Wielorybów"
            # Typical Price
            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['RawMoneyFlow'] = df['TP'] * df['Volume']
            
            # Kierunek przepływu
            # Jeśli TP dzisiaj > TP wczoraj -> Positive Flow
            df['PositiveFlow'] = np.where(df['TP'] > df['TP'].shift(1), df['RawMoneyFlow'], 0)
            df['NegativeFlow'] = np.where(df['TP'] < df['TP'].shift(1), df['RawMoneyFlow'], 0)
            
            # Sumy kroczące 14-dniowe
            df['PosMF_Sum'] = df['PositiveFlow'].rolling(window=14).sum()
            df['NegMF_Sum'] = df['NegativeFlow'].rolling(window=14).sum()
            
            # MFI Formula
            # Zabezpieczenie przed dzieleniem przez zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mfi_ratio = df['PosMF_Sum'] / df['NegMF_Sum']
                df['MFI'] = 100 - (100 / (1 + mfi_ratio))
            
            # Wypełniamy ewentualne braki zerami lub poprzednią wartością
            df['MFI'] = df['MFI'].fillna(50) 
            
            # Wygładzamy lekko
            df['RSI'] = df['RSI'].rolling(window=3).mean()
            df['MFI'] = df['MFI'].rolling(window=3).mean()
            
            return df.dropna()

        except Exception as e:
            print(f"Błąd Whale Divergence: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_whale_divergence(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        plt.subplots_adjust(hspace=0.05)
        
        # --- GÓRA: CENA BTC ---
        ax1.set_yscale('log')
        ax1.plot(df.index, df['Close'], color='white', linewidth=1.5, label='Cena BTC')
        ax1.set_ylabel('Cena BTC ($)', color='white', fontweight='bold')
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        
        # --- DÓŁ: RSI vs MFI (WALKA) ---
        
        # 1. RSI (Ulica) - Fioletowa linia
        ax2.plot(df.index, df['RSI'], color='#a020f0', linewidth=1.5, label='RSI (Cena/Ulica)', alpha=0.7)
        
        # 2. MFI (Wieloryby) - Turkusowa linia (Grubsza)
        ax2.plot(df.index, df['MFI'], color='#00e5ff', linewidth=2, label='MFI (Wolumen/Wieloryby)')
        
        # 3. WYPEŁNIENIA (Klucz do czytania)
        
        # MFI > RSI (Zielone) -> Wolumen wspiera cenę. Zdrowy wzrost.
        ax2.fill_between(df.index, df['MFI'], df['RSI'], where=(df['MFI'] > df['RSI']), 
                         color='#00ff55', alpha=0.3, label='SIŁA WIELORYBÓW (Zdrowy Trend)')
        
        # RSI > MFI (Czerwone) -> Cena rośnie, ale pieniądze uciekają. PUŁAPKA.
        ax2.fill_between(df.index, df['MFI'], df['RSI'], where=(df['RSI'] > df['MFI']), 
                         color='#ff0055', alpha=0.3, label='DYWERSJA (Ucieczka Kapitału)')
        
        # Poziomy wykupienia/wyprzedania
        ax2.axhline(80, color='grey', linestyle=':', alpha=0.5)
        ax2.axhline(20, color='grey', linestyle=':', alpha=0.5)
        
        ax2.set_ylabel('Oscylator (0-100)', color=t['text'])
        ax2.set_ylim(0, 100)
        
        # Tytuł i Status
        last_rsi = df['RSI'].iloc[-1]
        last_mfi = df['MFI'].iloc[-1]
        diff = last_mfi - last_rsi
        
        if diff > 5: status = "MOCNE FUNDAMENTY (Wieloryby kupują) 🟢"
        elif diff < -5: status = "PUŁAPKA / DYWERSJA (Wieloryby sprzedają) 🔴"
        else: status = "NEUTRALNIE ⚪"
        
        ax1.set_title(f"WHALE DIVERGENCE: {status}", fontsize=16, color=t['text'], fontweight='bold')
        
        ax2.legend(loc='upper center', ncol=2, facecolor=t['bg'], labelcolor=t['text'], frameon=False)
        
        # Kosmetyka
        for ax in [ax1, ax2]:
            ax.patch.set_facecolor(t['bg'])
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
            ax.tick_params(colors=t['text'])
        
        fig.patch.set_facecolor(t['bg'])
        
        return fig

    # --- PRO TOOLS: LIQUIDATION HEATMAP (Lasery - Wersja Debug) ---
    def get_liquidation_heatmap_data(self):
        """
        Pobiera dane i wyświetla błędy na ekranie, jeśli coś pójdzie nie tak.
        """
        try:
            # 1. Pobieranie
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            data = yf.download('BTC-USD', start=start_date, progress=False)
            
            if data.empty:
                st.error("Błąd: Pobrane dane są puste (yfinance zwrócił 0 wierszy).")
                return None

            # 2. Naprawa MultiIndex (Krytyczne dla yfinance)
            if isinstance(data.columns, pd.MultiIndex):
                # Szukamy poziomu z nazwami kolumn
                if 'Close' in data.columns.get_level_values(0):
                    data.columns = data.columns.get_level_values(0)
                elif 'Close' in data.columns.get_level_values(1):
                    data.columns = data.columns.get_level_values(1)
            
            # 3. Formatowanie kolumn
            data.columns = [c.capitalize() for c in data.columns]
            
            # 4. Sprawdzenie kolumn
            required = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required):
                st.error(f"Błąd struktury danych. Dostępne kolumny: {data.columns.tolist()}")
                return None

            df = data[required].copy()
            
            # 5. Obliczenia Swingów
            df['is_swing_high'] = df['High'].rolling(window=3, center=True).max() == df['High']
            df['is_swing_low'] = df['Low'].rolling(window=3, center=True).min() == df['Low']
            
            return df

        except Exception as e:
            # TO JEST KLUCZ: Wyświetlamy błąd na stronie
            st.error(f"Krytyczny błąd w get_liquidation_heatmap_data: {e}")
            import traceback
            st.text(traceback.format_exc()) # Pokaż szczegóły błędu
            return None

    def plot_liquidation_heatmap(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        end_date = df.index[-1]
        
        # 1. Rysujemy LASERY (Poziome belki)
        swing_highs = df[df['is_swing_high']]
        swing_lows = df[df['is_swing_low']]
        
        # Shorty (Nad ceną) - Ostatnie 30
        for date, row in swing_highs.tail(30).iterrows():
            p = row['High']
            # Rysujemy od daty powstania (date) do końca wykresu (end_date)
            ax1.fill_between([date, end_date], p*1.01, p*1.012, color='#ffff00', alpha=0.6, linewidth=0) # 100x
            ax1.fill_between([date, end_date], p*1.02, p*1.023, color='#ff9900', alpha=0.4, linewidth=0) # 50x
            ax1.fill_between([date, end_date], p*1.04, p*1.045, color='#ff0055', alpha=0.3, linewidth=0) # 25x

        # Longi (Pod ceną) - Ostatnie 30
        for date, row in swing_lows.tail(30).iterrows():
            p = row['Low']
            ax1.fill_between([date, end_date], p*0.99, p*0.988, color='#ffff00', alpha=0.6, linewidth=0)
            ax1.fill_between([date, end_date], p*0.98, p*0.977, color='#ff9900', alpha=0.4, linewidth=0)
            ax1.fill_between([date, end_date], p*0.96, p*0.955, color='#ff0055', alpha=0.3, linewidth=0)

        # 2. Cena BTC
        ax1.plot(df.index, df['Close'], color='white', linewidth=1.5, label='Cena BTC', zorder=10)
        
        # 3. Stylizacja i Skala
        ax1.set_title("LIQUIDATION LASERS: Strefy Bólu", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena BTC ($)', color='white')
        
        # Auto-Zoom (Ważne!)
        y_min = df['Low'].min() * 0.95
        y_max = df['High'].max() * 1.05
        ax1.set_ylim(y_min, y_max)
        
        # Legenda (Hackowana dla kolorów)
        ax1.plot([], [], color='#ffff00', label='100x Zone', linewidth=4, alpha=0.8)
        ax1.plot([], [], color='#ff9900', label='50x Zone', linewidth=4, alpha=0.8)
        ax1.plot([], [], color='#ff0055', label='25x Zone', linewidth=4, alpha=0.8)
        
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- PRO TOOLS: SUPERTREND SNIPER (Short Term) ---
    def get_supertrend_data(self):
        """
        Oblicza SuperTrend (ATR Trailing Stop) oraz 21 EMA.
        To najlepsze narzędzie do łapania średnioterminowych trendów (tygodnie/miesiące).
        """
        try:
            # Pobieramy 2 lata wstecz - wystarczy dla krótkiego terminu
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            data = yf.download('BTC-USD', start=start_date, progress=False)
            
            # --- FIX MULTIINDEX ---
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0): data.columns = data.columns.get_level_values(0)
                elif 'Close' in data.columns.get_level_values(1): data.columns = data.columns.get_level_values(1)
            
            data.columns = [c.capitalize() for c in data.columns]
            df = data[['High', 'Low', 'Close']].copy()

            # 1. 21 EMA (Wykładnicza średnia - krótki termin)
            # Używamy 21 okresów na interwale dziennym (bardzo popularne) 
            # lub przeliczamy na tygodniowe (21 tygodni = 147 dni). 
            # Zróbmy 21 EMA DZIENNĄ dla szybszych sygnałów, bo to short-term.
            df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

            # 2. SUPERTREND CALCULATION
            # Parametry: Period 10, Multiplier 3 (Standard)
            period = 10
            multiplier = 3
            
            # ATR (Average True Range)
            df['tr0'] = abs(df['High'] - df['Low'])
            df['tr1'] = abs(df['High'] - df['Close'].shift(1))
            df['tr2'] = abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            df['ATR'] = df['TR'].rolling(period).mean()
            
            # Basic Bands
            df['hl2'] = (df['High'] + df['Low']) / 2
            df['basic_upper'] = df['hl2'] + (multiplier * df['ATR'])
            df['basic_lower'] = df['hl2'] - (multiplier * df['ATR'])
            
            # Final Bands (Rekurencja - pętla)
            df['final_upper'] = 0.0
            df['final_lower'] = 0.0
            df['supertrend'] = 0.0 # Wartość linii
            df['trend'] = True # True = Green, False = Red
            
            # Musimy iterować, bo SuperTrend zależy od poprzedniej wartości
            # Konwertujemy do numpy dla szybkości, ale pętla po indexach też zadziała
            
            for i in range(period, len(df)):
                # Upper Band Logic
                if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1]) or (df['Close'].iloc[i-1] > df['final_upper'].iloc[i-1]):
                    df.at[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                else:
                    df.at[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]
                
                # Lower Band Logic
                if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1]) or (df['Close'].iloc[i-1] < df['final_lower'].iloc[i-1]):
                    df.at[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                else:
                    df.at[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]
                
                # Trend Direction Logic
                prev_trend = df['trend'].iloc[i-1]
                
                if prev_trend: # Był wzrostowy
                    if df['Close'].iloc[i] < df['final_lower'].iloc[i]:
                        df.at[df.index[i], 'trend'] = False # Zmiana na spadkowy
                    else:
                        df.at[df.index[i], 'trend'] = True
                else: # Był spadkowy
                    if df['Close'].iloc[i] > df['final_upper'].iloc[i]:
                        df.at[df.index[i], 'trend'] = True # Zmiana na wzrostowy
                    else:
                        df.at[df.index[i], 'trend'] = False
                        
                # Przypisanie właściwej linii do SuperTrend
                if df['trend'].iloc[i]:
                    df.at[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                else:
                    df.at[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]

            return df.iloc[period:]

        except Exception as e:
            print(f"Błąd SuperTrend: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_supertrend(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # 1. TŁO TRENDU (SuperTrend Cloud)
        # Jeśli trend Green -> Tło zielone między Ceną a SuperTrendem
        # Jeśli trend Red -> Tło czerwone
        
        ax1.fill_between(df.index, df['supertrend'], df['Close'], 
                         where=df['trend'], color='#00ff55', alpha=0.15, label='Trend Wzrostowy (Long)')
        
        ax1.fill_between(df.index, df['supertrend'], df['Close'], 
                         where=~df['trend'], color='#ff0055', alpha=0.15, label='Trend Spadkowy (Short)')
        
        # 2. LINIE TRENDU
        # Rysujemy linię SuperTrend zmieniającą kolory
        # Musimy to podzielić na segmenty lub narysować kropkami
        
        ax1.plot(df.index, df['supertrend'], color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Zielona linia tam gdzie trend=True
        green_line = df['supertrend'].copy()
        green_line[~df['trend']] = np.nan
        ax1.plot(df.index, green_line, color='#00ff55', linewidth=2)
        
        # Czerwona linia tam gdzie trend=False
        red_line = df['supertrend'].copy()
        red_line[df['trend']] = np.nan
        ax1.plot(df.index, red_line, color='#ff0055', linewidth=2)

        # 3. 21 EMA (Żółta) - O to prosiłeś
        ax1.plot(df.index, df['EMA_21'], color='#ffff00', linewidth=1.5, label='21 EMA (Momentum)')

        # 4. CENA BTC
        ax1.plot(df.index, df['Close'], color='white', linewidth=1, alpha=0.8, label='Cena BTC')
        
        # SYGNAŁY KUPNA / SPRZEDAŻY (Strzałki)
        # Wykrywamy moment zmiany trendu
        df['trend_shift'] = df['trend'].astype(int).diff()
        
        # +1 = Zmiana na Green (Kupuj), -1 = Zmiana na Red (Sprzedaj)
        buy_signals = df[df['trend_shift'] == 1]
        sell_signals = df[df['trend_shift'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['supertrend'] * 0.98, marker='^', color='#00ff55', s=100, zorder=5, edgecolors='black')
        ax1.scatter(sell_signals.index, sell_signals['supertrend'] * 1.02, marker='v', color='#ff0055', s=100, zorder=5, edgecolors='black')

        # Tytuł i Status
        last_price = df['Close'].iloc[-1]
        is_bullish = df['trend'].iloc[-1]
        st_val = df['supertrend'].iloc[-1]
        ema_val = df['EMA_21'].iloc[-1]
        
        if is_bullish:
            status = "TREND WZROSTOWY 🟢"
            dist = ((last_price - st_val) / last_price) * 100
            subtitle = f"Stop Loss (Zmiana trendu): {st_val:.0f}$ (-{dist:.1f}%)"
        else:
            status = "TREND SPADKOWY 🔴"
            dist = ((st_val - last_price) / last_price) * 100
            subtitle = f"Opór (Zmiana trendu): {st_val:.0f}$ (+{dist:.1f}%)"
            
        ax1.set_title(f"SUPERTREND SNIPER: {status}\n{subtitle}", fontsize=14, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena BTC ($)', color='white')
        
        # Zoom na ostatnie 180 dni domyślnie, żeby było widać "krótki termin"
        cutoff_date = df.index[-1] - timedelta(days=180)
        ax1.set_xlim(left=cutoff_date)
        
        # Y Lim dynamiczne do widoku
        view_df = df[df.index >= cutoff_date]
        if not view_df.empty:
            ax1.set_ylim(view_df['Low'].min()*0.95, view_df['High'].max()*1.05)

        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- PRO TOOLS: TECH WAR (INTC vs AMD vs NVDA vs BTC) ---
    def get_tech_war_data(self):
        """
        Pobiera dane dla Intel, AMD, Nvidia i Bitcoin.
        Oblicza skumulowany zwrot procentowy (Relative Performance).
        Pozwala porównać, kto wygrywa wyścig technologiczny.
        """
        try:
            # 2 lata to dobry okres, żeby zobaczyć boom na AI
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            tickers = ['INTC', 'AMD', 'NVDA', 'BTC-USD']
            
            # Pobieramy wszystko naraz
            data = yf.download(tickers, start=start_date, progress=False)
            
            # --- FIX MULTIINDEX (Dla wielu tickerów yfinance zwraca (Price, Ticker)) ---
            # Interesuje nas tylko 'Close'
            if isinstance(data.columns, pd.MultiIndex):
                # Próbujemy znaleźć poziom z cenami zamknięcia
                try:
                    df = data.xs('Close', axis=1, level=0, drop_level=True)
                except KeyError:
                    try:
                        df = data.xs('Close', axis=1, level=1, drop_level=True)
                    except KeyError:
                        # Fallback: ręczne szukanie
                        print("Struktura nietypowa, próbuję spłaszczyć...")
                        return None
            else:
                # Jeśli to nie MultiIndex (mało prawdopodobne przy wielu tickerach)
                df = data['Close']

            # Upewniamy się, że mamy wszystkie kolumny (czasem coś nie pobierze)
            # Sortujemy, żeby kolejność była stała
            df = df[tickers].copy()
            df = df.ffill().dropna()

            # --- NORMALIZACJA (Wszystko startuje od 0%) ---
            # Wzór: (Cena / Cena_Początkowa - 1) * 100
            df_normalized = (df / df.iloc[0] - 1) * 100
            
            return df_normalized

        except Exception as e:
            print(f"Błąd Tech War: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_tech_war(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # Kolory marek
        colors = {
            'INTC': '#0071c5',   # Intel Blue
            'AMD': '#ed1c24',    # AMD Red
            'NVDA': '#76b900',   # Nvidia Green
            'BTC-USD': '#ff9900' # Bitcoin Orange
        }
        
        labels = {
            'INTC': 'Intel (INTC)',
            'AMD': 'AMD',
            'NVDA': 'Nvidia (NVDA)',
            'BTC-USD': 'Bitcoin (BTC)'
        }

        # Rysujemy linie
        for ticker in df.columns:
            # Nvidia i BTC grubsze, bo to liderzy
            lw = 2.5 if ticker in ['NVDA', 'BTC-USD'] else 1.5
            alpha = 1.0 if ticker in ['NVDA', 'BTC-USD'] else 0.7
            
            # Ostatnia wartość do etykiety
            last_val = df[ticker].iloc[-1]
            sign = "+" if last_val > 0 else ""
            
            ax1.plot(df.index, df[ticker], color=colors.get(ticker, 'white'), 
                     linewidth=lw, alpha=alpha, label=f"{labels.get(ticker)}: {sign}{last_val:.0f}%")

        # Linia 0% (Start)
        ax1.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        
        # Tytuł
        ax1.set_title("WOJNA TECHNOLOGICZNA: Kto dał zarobić? (2 Lata)", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Zwrot z inwestycji (%)', color='white')
        
        # Legenda
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- PRO TOOLS: CHIP WARS (INTC vs NVDA vs AMD) ---
    def get_chip_wars_data(self):
        """
        Pobiera dane tylko dla gigantów procesorów: Intel, Nvidia, AMD.
        Bez Bitcoina. Czysta walka o dominację w AI i PC.
        """
        try:
            # 2 lata (730 dni) to idealny okres, by zobaczyć wybuch AI
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            tickers = ['INTC', 'NVDA', 'AMD']
            
            # Pobieramy dane
            data = yf.download(tickers, start=start_date, progress=False)
            
            # --- FIX MULTIINDEX (Standardowa naprawa yfinance) ---
            if isinstance(data.columns, pd.MultiIndex):
                # Próbujemy znaleźć poziom z cenami zamknięcia 'Close'
                try:
                    df = data.xs('Close', axis=1, level=0, drop_level=True)
                except KeyError:
                    try:
                        df = data.xs('Close', axis=1, level=1, drop_level=True)
                    except KeyError:
                        return None
            else:
                df = data['Close']

            # Upewniamy się, że mamy te 3 kolumny
            # Jeśli brakuje którejś (np. błąd pobierania), to trudno, pokażemy co jest
            available_tickers = [t for t in tickers if t in df.columns]
            df = df[available_tickers].copy()
            df = df.ffill().dropna()

            # --- NORMALIZACJA (Start od 0%) ---
            # Dzięki temu widzimy, kto dał zarobić, a kto stracił
            df_normalized = (df / df.iloc[0] - 1) * 100
            
            return df_normalized

        except Exception as e:
            print(f"Błąd Chip Wars: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_chip_wars(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # Oficjalne kolory marek
        colors = {
            'INTC': '#0071c5',   # Intel Blue (Klasyczny niebieski)
            'AMD': '#ed1c24',    # AMD Red (Czerwony)
            'NVDA': '#76b900'    # Nvidia Green (Neonowa zieleń)
        }
        
        labels = {
            'INTC': 'Intel (Walczy o życie)',
            'AMD': 'AMD (Goni lidera)',
            'NVDA': 'Nvidia (Król AI)'
        }

        # Rysujemy linie
        for ticker in df.columns:
            # Nvidia grubsza, bo to lider
            lw = 3 if ticker == 'NVDA' else 2
            # Intel cieńszy, jeśli radzi sobie słabo
            alpha = 1.0
            
            last_val = df[ticker].iloc[-1]
            sign = "+" if last_val > 0 else ""
            
            c = colors.get(ticker, 'white')
            l = labels.get(ticker, ticker)
            
            ax1.plot(df.index, df[ticker], color=c, linewidth=lw, alpha=alpha, label=f"{l}: {sign}{last_val:.0f}%")
            
            # Dodajemy "Kropkę" na końcu linii z wartością
            ax1.scatter(df.index[-1], last_val, color=c, s=50, zorder=5)

        # Linia 0% (Start)
        ax1.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.3)
        ax1.text(df.index[0], 2, " START (0%)", color='white', fontsize=8, alpha=0.5)
        
        # Tytuł
        ax1.set_title("WOJNA PROCESORÓW (Chip Wars): Kto wygrywa?", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Zysk / Strata (%)', color='white')
        
        # Legenda
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- PRO TOOLS: SECTOR ROTATION (S&P 500 - Z Generałami) ---
    def get_sector_performance_data(self):
        """
        Pobiera wyniki sektorów S&P 500.
        Wersja z podglądem 'Generałów' (Top Holdings) w nazwie.
        """
        try:
            # Mapa tickerów na nazwy z liderami
            sectors = {
                'XLK': 'Technologia (NVDA, MSFT, AAPL)',
                'XLF': 'Finanse (JPM, V, MA)',
                'XLV': 'Zdrowie (LLY, UNH, JNJ)',
                'XLE': 'Energia (XOM, CVX, EOG)',
                'XLY': 'Konsumpcyjne Lux (AMZN, TSLA)',
                'XLP': 'Defensywne (PG, COST, WMT)',
                'XLI': 'Przemysł (GE, CAT, RTX)',
                'XLC': 'Komunikacja (GOOGL, META)',
                'XLB': 'Surowce (LIN, SHW, FCX)',
                'XLRE': 'Nieruchomości (PLD, AMT)',
                'XLU': 'Użyteczność (NEE, SO, DUK)'
            }
            
            # Pobieramy dane (5 dni)
            tickers = list(sectors.keys())
            data = yf.download(tickers, period="5d", progress=False)
            
            # --- FIX MULTIINDEX ---
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    df = data.xs('Close', axis=1, level=0, drop_level=True)
                except KeyError:
                    try:
                        df = data.xs('Close', axis=1, level=1, drop_level=True)
                    except KeyError:
                        return None
            else:
                df = data['Close']

            if df.empty: return None
            
            # Obliczamy zmianę %
            last_close = df.iloc[-1]
            prev_close = df.iloc[-2]
            change_pct = ((last_close - prev_close) / prev_close) * 100
            
            results = []
            for ticker, val in change_pct.items():
                if ticker in sectors:
                    results.append({
                        'Sector': sectors[ticker],
                        'Change': val,
                        'Ticker': ticker
                    })
            
            df_res = pd.DataFrame(results)
            df_res = df_res.sort_values(by='Change', ascending=True)
            
            return df_res

        except Exception as e:
            print(f"Błąd Sector Rotation: {e}")
            return None

    def plot_sector_rotation(self, df):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # Kolory: Zielony dla wzrostów, Czerwony dla spadków
        colors = ['#00ff55' if x >= 0 else '#ff0055' for x in df['Change']]
        
        # Wykres poziomy (barh)
        bars = ax1.barh(df['Sector'], df['Change'], color=colors, alpha=0.8)
        
        # Linia zero
        ax1.axvline(0, color=t['text'], linewidth=1)
        
        # Etykiety wartości przy słupkach
        for bar, val in zip(bars, df['Change']):
            width = bar.get_width()
            
            # Pozycjonowanie tekstu
            if val >= 0:
                label_pos = width + 0.05
                align = 'left'
            else:
                label_pos = width - 0.05
                align = 'right'
                
            ax1.text(label_pos, bar.get_y() + bar.get_height()/2, f"{val:+.2f}%", 
                     va='center', ha=align, color='white', fontweight='bold', fontsize=10)

        # Analiza Risk-On / Risk-Off (Prosta heurystyka)
        # Sprawdzamy XLK (Tech) vs XLU (Utilities)
        try:
            tech_change = df[df['Sector'].str.contains('Tech')]['Change'].values[0]
            util_change = df[df['Sector'].str.contains('Utilities')]['Change'].values[0]
            
            if tech_change > util_change:
                status = "RISK ON (Atak) 🐂"
                status_col = "#00ff55"
            else:
                status = "RISK OFF (Obrona) 🛡️"
                status_col = "#ff0055"
        except:
            status = "MIESZANY"
            status_col = t['text']

        ax1.set_title(f"ROTACJA SEKTORÓW (S&P 500): {status}", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_xlabel('Zmiana dzienna (%)', color=t['text'])
        
        # Formatowanie osi
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, axis='x', alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_visible(False) # Ukrywamy oś Y, bo są etykiety
        ax1.tick_params(colors=t['text'])
        ax1.tick_params(axis='y', labelsize=11) # Większe nazwy sektorów
        
        return fig

    # --- PRO TOOLS: SECTOR SNIPER (Mega Pack - Updated Keys) ---
    def get_sector_sniper_data(self):
        """
        Wersja 'Fat Pack' z zaktualizowanymi kluczami nazw sektorów.
        """
        # Mapa: Musi idealnie pasować do nazw z get_sector_performance_data
        sector_holdings = {
            'Technologia (NVDA, MSFT, AAPL)': [
                'NVDA', 'AAPL', 'MSFT', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'CSCO',
                'INTC', 'IBM', 'NOW', 'UBER', 'ABNB', 'PLTR', 'PANW', 'SNOW', 'CRWD', 'DELL',
                'HPQ', 'MU', 'LRCX', 'AMAT', 'KLAC', 'SMCI', 'ARM'
            ],
            'Finanse (JPM, V, MA)': [
                'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'AXP', 'BLK', 'C',
                'USB', 'PNC', 'CB', 'MMC', 'SCHW', 'BX', 'KKR', 'APO', 'COF', 'DFS',
                'PYPL', 'SQ', 'SOFI', 'HOOD', 'NU', 'AIG', 'MET'
            ],
            'Zdrowie (LLY, UNH, JNJ)': [
                'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'AMGN', 'PFE', 'ISRG', 'DHR',
                'BMY', 'CVS', 'GILD', 'REGN', 'VRTX', 'HCA', 'MCK', 'CI', 'HUM', 'MRNA',
                'BNTX', 'NVO', 'AZN', 'NVS', 'SYK', 'EW'
            ],
            'Energia (XOM, CVX, EOG)': [
                'XOM', 'CVX', 'EOG', 'COP', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
                'KMI', 'WMB', 'HAL', 'BKR', 'DVN', 'FANG', 'TRGP', 'CTRA', 'EQT', 'MRO',
                'PBR', 'SHEL', 'TTE', 'BP'
            ],
            'Konsumpcyjne Lux (AMZN, TSLA)': [
                'AMZN', 'TSLA', 'HD', 'MCD', 'COST', 'NFLX', 'LOW', 'SBUX', 'BKNG', 'TJX',
                'NKE', 'LULU', 'CMG', 'MAR', 'HLT', 'RCL', 'CCL', 'F', 'GM', 'TM',
                'HMC', 'SONY', 'ABNB', 'DHI', 'LEN', 'ROST'
            ],
            'Defensywne (PG, COST, WMT)': [
                'PG', 'WMT', 'KO', 'PEP', 'PM', 'MDLZ', 'CL', 'MO', 'TGT', 'KMB',
                'COST', 'EL', 'GIS', 'K', 'HSY', 'MNST', 'STZ', 'KR', 'SYY', 'ADM',
                'DEO', 'UL', 'BUD', 'BTI'
            ],
            'Przemysł (GE, CAT, RTX)': [
                'GE', 'CAT', 'RTX', 'HON', 'UNP', 'UPS', 'BA', 'LMT', 'DE', 'ADP',
                'ETN', 'ITW', 'WM', 'MMM', 'CSX', 'NSC', 'FDX', 'EMR', 'PH', 'GD',
                'NOC', 'LHX', 'TXT', 'AXON'
            ],
            'Komunikacja (GOOGL, META)': [
                'GOOGL', 'META', 'NFLX', 'DIS', 'T', 'VZ', 'CMCSA', 'TMUS', 'CHTR', 'WBD',
                'PARA', 'LYV', 'EA', 'TTWO', 'SPOT', 'DASH', 'SNAP', 'PINS', 'ROKU', 'OMC'
            ],
            'Surowce (LIN, SHW, FCX)': [
                'LIN', 'SHW', 'FCX', 'APD', 'ECL', 'NEM', 'CTVA', 'DOW', 'DD', 'PPG',
                'VMC', 'MLM', 'ALB', 'CF', 'MOS', 'NUE', 'STLD', 'SCCO', 'RIO', 'BHP',
                'VALE'
            ],
            'Nieruchomości (PLD, AMT)': [
                'PLD', 'AMT', 'EQIX', 'PSA', 'CCI', 'O', 'WELL', 'SPG', 'DLR', 'VICI',
                'AVB', 'EQR', 'CBRE', 'CSGP', 'INVH', 'MAA', 'ESS', 'UDR', 'KIM'
            ],
            'Użyteczność (NEE, SO, DUK)': [
                'NEE', 'SO', 'DUK', 'CEG', 'AEP', 'SRE', 'D', 'PEG', 'EXC', 'XEL',
                'ED', 'ES', 'PCG', 'WEC', 'AWK', 'ETR', 'DTE', 'FE', 'PPL'
            ]
        }

        try:
            # 1. Pobieramy lidera sektorów
            df_sectors = self.get_sector_performance_data()
            if df_sectors is None or df_sectors.empty: return None, None
            
            # Najlepszy sektor
            winner_row = df_sectors.iloc[-1] 
            winner_name = winner_row['Sector']
            # Teraz winner_name będzie pasować do klucza słownika powyżej
            winner_tickers = sector_holdings.get(winner_name, [])
            
            if not winner_tickers: return None, None

            # 2. Pobieramy dane (Batch)
            data = yf.download(winner_tickers, period="1y", progress=False)
            
            scores = []
            
            # Fix MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                closes = data['Close']
                volumes = data['Volume']
            else:
                return None, None

            for ticker in winner_tickers:
                if ticker not in closes.columns: continue
                
                price = closes[ticker].dropna()
                vol = volumes[ticker].dropna()
                
                if len(price) < 200: continue
                
                # --- ALGORYTM PUNKTACJI (Łowca Dołków) ---
                score = 0
                
                curr_price = price.iloc[-1]
                sma_50 = price.rolling(50).mean().iloc[-1]
                sma_200 = price.rolling(200).mean().iloc[-1]
                
                # A. TREND
                if curr_price > sma_200: score += 30 
                if curr_price > sma_50: score += 10  
                
                # B. RSI
                delta = price.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1]
                
                if 30 <= rsi_val <= 50: score += 40
                elif rsi_val < 30: score += 25
                elif 50 < rsi_val <= 70: score += 20
                else: score += 5
                
                # C. WOLUMEN
                vol_avg = vol.rolling(20).mean().iloc[-1]
                curr_vol = vol.iloc[-1]
                if curr_vol > vol_avg: score += 20 
                
                scores.append({'Ticker': ticker, 'Score': score, 'Price': curr_price, 'RSI': rsi_val})
            
            # Ranking
            df_scores = pd.DataFrame(scores)
            df_scores = df_scores.sort_values(by='Score', ascending=False)
            
            return df_scores.head(12), winner_name

        except Exception as e:
            print(f"Błąd Snajpera: {e}")
            return None, None

    def plot_sector_sniper(self, df, sector_name):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # --- ZMIANA: Łagodniejsze progi kolorów ---
        # > 70 pkt = Zielony (Okazja/Strong)
        # > 40 pkt = Żółty (Obiecujące)
        # < 40 pkt = Szary (Słabe)
        colors = []
        for s in df['Score']:
            if s >= 70: colors.append('#00ff55') # Zielony Neon
            elif s >= 40: colors.append('#ffd700') # Złoty
            else: colors.append('#555555') # Ciemny Szary
            
        # Wykres
        bars = ax.barh(df['Ticker'], df['Score'], color=colors, alpha=0.9, edgecolor=t['bg'])
        
        ax.invert_yaxis() # Najlepsze na górze
        
        # --- ZMIANA: Białe napisy dla czytelności ---
        for bar, score, price, rsi in zip(bars, df['Score'], df['Price'], df['RSI']):
            width = bar.get_width()
            
            # 1. Info o cenie i RSI (Wewnątrz paska)
            # Jeśli pasek jest bardzo krótki, tekst by nie wszedł, więc można dać warunek,
            # ale tutaj założymy kolor biały, który będzie widać na szarym/zielonym/zółtym.
            # Zmieniamy kolor na 'white' i dodajemy cień (opcjonalnie) lub po prostu white.
            
            # Tekst informacyjny (Cena | RSI)
            ax.text(2, bar.get_y() + bar.get_height()/2, f"${price:.2f} | RSI: {rsi:.0f}", 
                    va='center', ha='left', color='white', fontweight='bold', fontsize=10)
            
            # 2. Wynik punktowy (Za paskiem)
            ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{score} pkt", 
                    va='center', ha='left', color='white', fontweight='bold', fontsize=11)

        ax.set_title(f"TOP SPÓŁKI Z SEKTORA: {sector_name}", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Siła Okazji (Trend + Niskie RSI)', color=t['text'])
        ax.set_xlim(0, 115)
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        ax.tick_params(colors=t['text'])
        ax.grid(False)
        
        return fig

    # --- PRO TOOLS: VALUE ARCHITECT (Top 30) ---
    def get_value_architect_data(self):
        """
        Skanuje rynek i zwraca TOP 30 okazji (Value Investing).
        """
        # MEGA LISTA (Tech, Finanse, Zdrowie, Energia, Konsumpcja, Przemysł...)
        sector_holdings = {
            'Tech': [
                'NVDA', 'AAPL', 'MSFT', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'CSCO',
                'PLTR', 'SNOW', 'GOOGL', 'META', 'INTC', 'IBM', 'TXN', 'NOW', 'UBER', 'ABNB',
                'PANW', 'CRWD', 'MU', 'AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'ROP', 'ADSK'
            ],
            'Finanse': [
                'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'PYPL', 'HOOD',
                'AXP', 'C', 'USB', 'BX', 'KKR', 'COF', 'AIG', 'CB', 'MMC', 'SOFI',
                'PNC', 'SCHW', 'TROW', 'SPGI', 'MCO', 'DFS', 'ALL'
            ],
            'Zdrowie': [
                'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE', 'ISRG', 'CVS', 'TMO', 'DHR',
                'BMY', 'AMGN', 'GILD', 'REGN', 'VRTX', 'HCA', 'MCK', 'CI', 'HUM', 'SYK',
                'EW', 'BSX', 'BDX', 'ZTS', 'ILMN', 'MRNA', 'BNTX'
            ],
            'Energia': [
                'XOM', 'CVX', 'EOG', 'SLB', 'OXY', 'SHEL', 'BP', 'COP', 'MPC', 'PSX',
                'VLO', 'KMI', 'WMB', 'HAL', 'BKR', 'DVN', 'FANG', 'TRGP', 'CTRA', 'EQT'
            ],
            'Konsumpcja': [
                'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'F', 'GM', 'LOW', 'TJX',
                'BKNG', 'MAR', 'HLT', 'RCL', 'CCL', 'LULU', 'CMG', 'YUM', 'DRI', 'TGT',
                'ROST', 'ORLY', 'AZO', 'LEN', 'DHI', 'EBAY', 'ETSY'
            ],
            'Defensywne': [
                'PG', 'KO', 'PEP', 'WMT', 'COST', 'MO', 'PM', 'CL', 'MDLZ', 'KMB',
                'GIS', 'K', 'HSY', 'MNST', 'STZ', 'KR', 'SYY', 'ADM', 'EL', 'TSN'
            ],
            'Przemysł': [
                'GE', 'CAT', 'BA', 'LMT', 'DE', 'RTX', 'HON', 'UNP', 'UPS', 'FDX',
                'ETN', 'ITW', 'WM', 'MMM', 'CSX', 'NSC', 'EMR', 'PH', 'GD', 'NOC',
                'LHX', 'TXT', 'AXON', 'ADP', 'PAYX'
            ],
            'Komunikacja': [
                'NFLX', 'DIS', 'T', 'VZ', 'CMCSA', 'TMUS', 'CHTR', 'WBD', 'PARA', 'LYV',
                'EA', 'TTWO', 'SPOT', 'DASH', 'SNAP', 'PINS', 'ROKU', 'OMC'
            ],
            'Surowce': [
                'LIN', 'FCX', 'NEM', 'VALE', 'RIO', 'SHW', 'APD', 'ECL', 'CTVA', 'DOW',
                'DD', 'PPG', 'VMC', 'MLM', 'ALB', 'CF', 'MOS', 'NUE', 'STLD'
            ],
            'Nieruchomości': [
                'PLD', 'AMT', 'O', 'VICI', 'EQIX', 'PSA', 'CCI', 'WELL', 'SPG', 'DLR',
                'AVB', 'EQR', 'CBRE', 'CSGP', 'INVH', 'MAA', 'ESS'
            ],
            'Użyteczność': [
                'NEE', 'SO', 'DUK', 'CEG', 'AEP', 'SRE', 'D', 'PEG', 'EXC', 'XEL',
                'ED', 'ES', 'WEC', 'AWK'
            ]
        }
        
        all_tickers = []
        ticker_sector_map = {}
        for sec, tickers in sector_holdings.items():
            for t in tickers:
                if t not in all_tickers:
                    all_tickers.append(t)
                    ticker_sector_map[t] = sec

        try:
            data = yf.download(all_tickers, period="1y", progress=False)
            
            value_picks = []
            
            if isinstance(data.columns, pd.MultiIndex):
                try: closes = data.xs('Close', axis=1, level=0, drop_level=True)
                except: closes = data['Close']
            else:
                closes = data['Close']

            for ticker in all_tickers:
                if ticker not in closes.columns: continue
                
                price = closes[ticker].dropna()
                if len(price) < 200: continue
                
                curr_price = price.iloc[-1]
                high_52 = price.max()
                
                # Wskaźniki
                drawdown = ((curr_price - high_52) / high_52) * 100
                
                sma_200 = price.rolling(200).mean().iloc[-1]
                dist_to_sma = ((curr_price - sma_200) / sma_200) * 100
                
                delta = price.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1]
                
                # Scoring
                score = 0
                
                # Przecena
                if drawdown < -30: score += 40
                elif drawdown < -20: score += 25
                elif drawdown < -10: score += 10
                else: score -= 20
                
                # SMA 200
                if dist_to_sma < 0: score += 30
                elif dist_to_sma < 5: score += 15
                elif dist_to_sma > 20: score -= 15
                
                # RSI
                if rsi_val < 30: score += 30
                elif rsi_val < 45: score += 15
                elif rsi_val > 70: score -= 10
                
                if score > 30:
                    value_picks.append({
                        'Ticker': ticker,
                        'Sector': ticker_sector_map[ticker],
                        'Discount': drawdown,
                        'Price': curr_price,
                        'Score': score,
                        'RSI': rsi_val
                    })
            
            df_value = pd.DataFrame(value_picks)
            if not df_value.empty:
                df_value = df_value.sort_values(by='Score', ascending=False)
            
            return df_value.head(30) # ZMIANA: Zwracamy 30 pozycji

        except Exception as e:
            print(f"Błąd Architekta: {e}")
            return None

    def plot_value_architect(self, df):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        # ZMIANA: Wysoki wykres (14 cali), żeby pomieścić 30 pozycji
        fig = plt.figure(figsize=(13, 14))
        ax = fig.add_subplot(111)
        
        sec_colors = {
            'Tech': '#00e5ff', 'Finanse': '#00ff55', 'Zdrowie': '#ff0055',
            'Energia': '#ffd700', 'Konsumpcja': '#ff9100', 'Defensywne': '#d500f9',
            'Przemysł': '#ff5500', 'Surowce': '#00aaaa', 'Nieruchomości': '#aaaaaa',
            'Komunikacja': '#aa00ff', 'Użyteczność': '#76ff03'
        }
        colors = [sec_colors.get(x, 'white') for x in df['Sector']]
        
        # Paski (Długość = Score)
        bars = ax.barh(df['Ticker'], df['Score'], color=colors, alpha=0.85, edgecolor=t['bg'])
        
        ax.invert_yaxis()
        
        # Etykiety
        for bar, price, rsi, sect, disc, score in zip(bars, df['Price'], df['RSI'], df['Sector'], df['Discount'], df['Score']):
            
            # Wewnątrz paska (Ticker + Cena)
            ax.text(2, bar.get_y() + bar.get_height()/2, f"{sect} | ${price:.2f}", 
                    va='center', ha='left', color='black', fontsize=9, fontweight='bold', alpha=0.8)
            
            # Za paskiem (Punkty + Info)
            info_text = f"{score} pkt  (📉 {disc:.1f}%  |  RSI: {rsi:.0f})"
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, info_text, 
                    va='center', ha='left', color='white', fontsize=10, fontweight='bold')

        ax.set_title("ARCHITEKT WARTOŚCI: Top 30 Okazji (Score)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Score (Suma punktów za: Spadek + RSI + Średnią)', color=t['text'])
        ax.set_xlim(0, 140)
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        
        ax.tick_params(colors=t['text'], axis='y', labelsize=10)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # ax.grid(True, axis='x', alpha=0.1, color=t['grid'])
        ax.grid(False) # <--- TO WYŁĄCZA SIATKĘ SEABORNA
        
        return fig

    # --- PRO TOOLS: ALTCOIN GEM HUNTER (Force Top 15) ---
    def get_altcoin_gem_data(self):
        """
        Skanuje rynek Altcoinów.
        WERSJA BEZ FILTRA: Zwraca zawsze TOP 15 coinów, niezależnie od wyniku punktowego.
        """
        coins = [
            'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD',
            'DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'WIF-USD', 'BONK-USD', 'FLOKI-USD', 'NAKA-USD', 'ENA-USD',
            'RNDR-USD', 'FET-USD', 'TAO-USD', 'AGIX-USD', 'OCEAN-USD', 'NEAR-USD', 'SHRAP-USD', 'ASTER-USD',
            'ARB-USD', 'OP-USD', 'MATIC-USD', 'IMX-USD', 'STX-USD', 'INJ-USD', 'TIA-USD', 'VRA-USD',
            'UNI-USD', 'AAVE-USD', 'MKR-USD', 'SNX-USD', 'LDO-USD', 'PENDLE-USD', 'ONDO-USD', 'LUNC-USD',
            'APT-USD', 'SUI-USD', 'SEI-USD', 'ATOM-USD', 'ICP-USD', 'FIL-USD', 'HBAR-USD', 'KAS-USD',
            'FTM-USD', 'ALGO-USD', 'SAND-USD', 'MANA-USD', 'AXS-USD', 'GALA-USD', 'FLOW-USD', 'XLM-USD', 'JASMY-USD'
        ]

        try:
            data = yf.download(coins, period="max", progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                try: closes = data['Close']; volumes = data['Volume']
                except: return None
            else:
                return None

            gems = []

            for ticker in coins:
                if ticker not in closes.columns: continue
                
                series = closes[ticker].dropna()
                vol_series = volumes[ticker].dropna()
                
                if len(series) < 180: continue 
                
                curr_price = series.iloc[-1]
                ath = series.max()
                
                # Dane techniczne
                drawdown = ((curr_price - ath) / ath) * 100
                
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1]
                
                ema_21 = series.ewm(span=21, adjust=False).mean().iloc[-1]
                dist_ema = ((curr_price - ema_21) / ema_21) * 100
                
                vol_avg = vol_series.rolling(30).mean().iloc[-1]
                curr_vol = vol_series.iloc[-1]
                vol_ratio = curr_vol / vol_avg if vol_avg > 0 else 0
                
                # --- SCORING ---
                score = 0
                
                # Drawdown
                if -95 <= drawdown <= -70: score += 30
                elif -70 < drawdown <= -50: score += 15
                elif drawdown < -95: score -= 50
                
                # RSI
                if rsi_val < 35: score += 20
                elif rsi_val < 45: score += 10
                
                # EMA 21
                if -5 <= dist_ema <= 5: score += 30
                elif dist_ema > 0: score += 20
                elif dist_ema < -15: score -= 30
                
                # Wolumen
                if vol_ratio > 1.2: score += 20
                
                # Status słowny
                status = "AKUMULACJA" if -5 <= dist_ema <= 5 else "ODBICIE" if dist_ema > 0 else "DOŁEK"
                
                # Dodajemy WSZYSTKO (bez warunku if score > 30)
                gems.append({
                    'Coin': ticker.replace('-USD',''),
                    'Score': score,
                    'Drawdown': drawdown,
                    'RSI': rsi_val,
                    'DistEMA': dist_ema,
                    'Status': status
                })
            
            # Ranking
            df_gems = pd.DataFrame(gems)
            if not df_gems.empty:
                df_gems = df_gems.sort_values(by='Score', ascending=False)
            
            # Zwracamy TOP 15
            return df_gems.head(15)

        except Exception as e:
            print(f"Błąd Gem Hunter: {e}")
            return None

    def plot_altcoin_gems(self, df):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        # Wysokość 10 cali idealnie mieści 15 pozycji z odstępami
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        colors = []
        for s in df['Score']:
            if s >= 75: colors.append('#00ff55')
            elif s >= 50: colors.append('#00e5ff')
            else: colors.append('#ffd700')
            
        # height=0.4 to cienkie paski (standard to 0.8)
        bars = ax.barh(df['Coin'], df['Score'], color=colors, alpha=0.85, 
                       edgecolor=t['bg'], height=0.4)
        
        ax.invert_yaxis()
        
        # Etykiety
        for bar, dd, rsi, ema, status, score in zip(bars, df['Drawdown'], df['RSI'], df['DistEMA'], df['Status'], df['Score']):
            
            # Lewa strona (Dane) - Mniejsza czcionka (8)
            ax.text(2, bar.get_y() + bar.get_height()/2, f"ATH: {dd:.0f}% | EMA21: {ema:+.1f}% | {status}", 
                    va='center', ha='left', color='white', fontweight='bold', fontsize=8)
            
            # Prawa strona (Punkty)
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{score} pkt", 
                    va='center', ha='left', color='white', fontweight='bold', fontsize=9)

        ax.set_title("GEM HUNTER 2.0: Top 15 (Wszystkie Okazje)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Score Niezawodności', color=t['text'])
        ax.set_xlim(0, 115)
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        ax.tick_params(colors=t['text'])
        ax.grid(False)
        
        return fig

    # --- PRO TOOLS: BTC FUTURE PATH (Nostradamus - 8 Years / Double Cycle) ---
    def get_btc_projection_data(self):
        """
        Generuje projekcję ceny BTC na DWA PEŁNE CYKLE (2022 - 2030).
        Model "Fractal Repeat" z tłumieniem zmienności.
        """
        try:
            # Daty historycznych dołków
            b15 = pd.to_datetime('2015-01-14')
            b18 = pd.to_datetime('2018-12-15')
            b22 = pd.to_datetime('2022-11-21') # Start (Dno 2022)
            
            # 1. Pobieramy dane
            data = yf.download('BTC-USD', start='2014-01-01', progress=False)
            
            # Fix MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                try: df = data.xs('Close', axis=1, level=0, drop_level=True)
                except: df = data['Close']
            else:
                df = data['Close']
            
            if isinstance(df, pd.Series): df = df.to_frame(name='Price')
            else: df = df.iloc[:, 0].to_frame(name='Price')

            # --- OBLICZANIE WZORCA JEDNEGO CYKLU (4 lata) ---
            
            # Cykl 2015-2018
            c15 = df[(df.index >= b15) & (df.index < b18)].copy()
            c15['Days'] = (c15.index - b15).days
            m15 = c15.set_index('Days')['Price'] / c15['Price'].iloc[0]
            
            # Cykl 2018-2022
            c18 = df[(df.index >= b18) & (df.index < b22)].copy()
            c18['Days'] = (c18.index - b18).days
            m18 = c18.set_index('Days')['Price'] / c18['Price'].iloc[0]
            
            # Cykl Obecny (dane realne)
            c22 = df[df.index >= b22].copy()
            c22['Days'] = (c22.index - b22).days
            current_days = c22['Days'].max()
            current_price = c22['Price'].iloc[-1]
            base_price_22 = c22['Price'].iloc[0]

            # Tworzymy Wzorzec Cyklu (0-1460 dni)
            cycle_len = 1460
            days_range = range(cycle_len + 1)
            
            # Interpolacja do pełnych dni
            s15 = m15.reindex(days_range).interpolate(limit_direction='both').fillna(method='ffill')
            s18 = m18.reindex(days_range).interpolate(limit_direction='both').fillna(method='ffill')
            
            # Średnia Krzywa (Waga: 75% z 2018, 25% z 2015 + Tłumienie 0.85)
            base_curve = (s15 * 0.25 + s18 * 0.75) * 0.85
            
            # --- TWORZENIE DRUGIEGO CYKLU (2026-2030) ---
            # To jest "Fractal Repeat". Bierzemy base_curve i doklejamy ją na koniec.
            # Ale musimy ją spłaszczyć (Diminishing Returns), np. mnożnik zysku * 0.7
            
            # Wzorzec drugiego cyklu (Damping)
            # Wzór: Nowy_Mnożnik = 1 + (Stary_Mnożnik - 1) * 0.7
            # To sprawia, że wzrosty są mniejsze, a spadki łagodniejsze
            second_cycle_curve = 1 + (base_curve - 1) * 0.7
            
            # --- SKLEJANIE CAŁOŚCI (0 - 2920 dni) ---
            full_projection = []
            
            # Wartość końcowa pierwszego cyklu (to będzie baza dla drugiego)
            end_val_cycle_1 = base_curve.iloc[-1] 
            
            # Pętla po 2 cyklach
            for d in range(cycle_len * 2):
                if d <= cycle_len:
                    # Cykl 1
                    val = base_curve.get(d, base_curve.iloc[-1])
                else:
                    # Cykl 2
                    # Dni wewnątrz drugiego cyklu (0-1460)
                    d2 = d - cycle_len
                    multiplier_in_c2 = second_cycle_curve.get(d2, second_cycle_curve.iloc[-1])
                    # Wartość absolutna = Koniec Cyklu 1 * Mnożnik Cyklu 2
                    val = end_val_cycle_1 * multiplier_in_c2
                
                full_projection.append(val)
                
            # Konwersja na Series
            full_curve = pd.Series(full_projection, index=range(len(full_projection)))
            
            # --- SKALOWANIE DO DZIŚ ---
            # Dopasowujemy model do obecnej ceny
            model_val_today = full_curve.get(current_days, 1.0)
            if model_val_today == 0: model_val_today = 1.0
            
            scale_factor = current_price / (base_price_22 * model_val_today)
            
            # Obliczamy ceny ($)
            theoretical_path = (base_price_22 * full_curve) * scale_factor
            
            # Daty (od 2022 do 2030)
            date_index = [b22 + timedelta(days=x) for x in range(len(full_projection))]
            theoretical_path.index = date_index
            
            # Zwracamy tylko przyszłość (uciętą na dacie ostatniej świecy)
            # Ale do wizualizacji potrzebujemy całości w df_proj
            
            return c22['Price'].to_frame(), theoretical_path

        except Exception as e:
            print(f"Błąd Nostradamusa (8Y): {e}")
            return None, None

    def plot_btc_projection(self, df_real, df_proj):
        if df_real is None or df_proj is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        
        # 1. Historia
        ax.plot(df_real.index, df_real.iloc[:, 0], color='white', linewidth=2, label='Obecny Cykl', zorder=10)
        
        # 2. Przyszłość (Model 8 lat)
        last_date = df_real.index[-1]
        future_part = df_proj[df_proj.index > last_date]
        
        ax.plot(future_part.index, future_part, color='#00e5ff', linestyle='--', linewidth=2, label='Model 2022-2030', alpha=0.9)
        
        # --- ZAAWANSOWANE SZUKANIE PUNKTÓW ZWROTNYCH ---
        # Dzielimy przyszłość na dwa okresy (Cykl 1 i Cykl 2), żeby znaleźć 2 szczyty i 2 dołki
        # Punkt podziału to mniej więcej 2026-11 (ok. 1460 dni od startu)
        
        start_date = df_proj.index[0]
        mid_date = start_date + timedelta(days=1460) # Koniec 1 cyklu / Start 2 cyklu
        
        # --- CYKL 1 (2022-2026) ---
        c1_data = df_proj[df_proj.index < mid_date]
        # Łączymy z realnymi danymi dla pełnego obrazu szczytu
        combined_c1 = pd.concat([df_real.iloc[:, 0], c1_data[c1_data.index > last_date]])
        
        # Szczyt 1
        top1_date = combined_c1.idxmax()
        top1_price = combined_c1.max()
        
        # Dołek 1 (Po szczycie 1)
        correction_c1 = c1_data[c1_data.index > top1_date]
        if not correction_c1.empty:
            bot1_date = correction_c1.idxmin()
            bot1_price = correction_c1.min()
            
            # Rysujemy Dołek 1 (2026)
            ax.scatter([bot1_date], [bot1_price], color='#ff9900', s=100, zorder=15, edgecolors='white')
            ax.text(bot1_date, bot1_price * 0.70, f"DNO 2026\n${bot1_price:,.0f}", color='#ff9900', ha='center', fontsize=9, fontweight='bold')

        # Rysujemy Szczyt 1
        col_t1 = '#00ff55' if top1_date > last_date else '#ff0055'
        lbl_t1 = "SZCZYT 1"
        ax.scatter([top1_date], [top1_price], color=col_t1, s=120, zorder=15, edgecolors='white')
        ax.text(top1_date, top1_price * 1.05, f"{lbl_t1}\n${top1_price:,.0f}", color=col_t1, ha='center', fontweight='bold', fontsize=10)

        # --- CYKL 2 (2026-2030) ---
        c2_data = df_proj[df_proj.index >= mid_date]
        
        if not c2_data.empty:
            # Szczyt 2
            top2_date = c2_data.idxmax()
            top2_price = c2_data.max()
            
            # Dołek 2 (Po szczycie 2)
            correction_c2 = c2_data[c2_data.index > top2_date]
            if not correction_c2.empty:
                bot2_date = correction_c2.idxmin()
                bot2_price = correction_c2.min()
                
                # Rysujemy Dołek 2 (2030)
                ax.scatter([bot2_date], [bot2_price], color='#ff9900', s=100, zorder=15, edgecolors='white')
                ax.text(bot2_date, bot2_price * 0.85, f"DNO 2030\n${bot2_price:,.0f}", color='#ff9900', ha='center', fontsize=9, fontweight='bold')

            # Rysujemy Szczyt 2
            ax.scatter([top2_date], [top2_price], color='#00ff55', s=120, zorder=15, edgecolors='white')
            ax.text(top2_date, top2_price * 1.05, f"SZCZYT 2029\n${top2_price:,.0f}", color='#00ff55', ha='center', fontweight='bold', fontsize=10)

        # Stylizacja
        ax.set_title("NOSTRADAMUS: Projekcja na 2 Cykle (2022 - 2030)", fontsize=16, color=t['text'], fontweight='bold')
        
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        ax.grid(True, alpha=0.15, color=t['grid'])
        ax.legend(facecolor=t['bg'], labelcolor=t['text'], loc='upper left')
        
        return fig

    # --- PRO TOOLS: NOSTRADAMUS PRO (Fixed Weights 25/35/40) ---
    def get_btc_weighted_projection(self):
        """
        Generuje projekcję PRO (Fixed):
        Wagi: 25% (2015), 35% (2018), 40% (2022).
        Dla obecnego cyklu (który nie jest pełny), uzupełniamy brakujące dni średnią z historii,
        ale zachowujemy wagę 40% dla tego co już się wydarzyło.
        """
        try:
            # Daty dołków
            b15 = pd.to_datetime('2015-01-14')
            b18 = pd.to_datetime('2018-12-15')
            b22 = pd.to_datetime('2022-11-21') 
            
            # Pobieranie danych
            data = yf.download('BTC-USD', start='2014-01-01', progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                try: df = data.xs('Close', axis=1, level=0, drop_level=True)
                except: df = data['Close']
            else:
                df = data['Close']
            
            if isinstance(df, pd.Series): df = df.to_frame(name='Price')
            else: df = df.iloc[:, 0].to_frame(name='Price')

            # --- 1. PRZYGOTOWANIE CYKLI (0 - 1460 dni) ---
            cycle_len = 1460 # 4 lata
            days_idx = range(cycle_len + 1)

            # Cykl 2015 (Waga 0.25)
            c15 = df[(df.index >= b15) & (df.index < b18)].copy()
            c15['Days'] = (c15.index - b15).days
            m15 = c15.set_index('Days')['Price'] / c15['Price'].iloc[0]
            # Interpolacja do pełnych 1460 dni
            s15 = m15.reindex(days_idx).interpolate(limit_direction='both').fillna(method='ffill')

            # Cykl 2018 (Waga 0.35)
            c18 = df[(df.index >= b18) & (df.index < b22)].copy()
            c18['Days'] = (c18.index - b18).days
            m18 = c18.set_index('Days')['Price'] / c18['Price'].iloc[0]
            # Interpolacja
            s18 = m18.reindex(days_idx).interpolate(limit_direction='both').fillna(method='ffill')

            # Cykl 2022 (Realny - Waga 0.40)
            c22 = df[df.index >= b22].copy()
            c22['Days'] = (c22.index - b22).days
            m22_real = c22.set_index('Days')['Price'] / c22['Price'].iloc[0]
            
            # Tworzymy "Pełny Cykl 2022" (Hybryda: Realne dane + Prognoza na końcówkę)
            # Na razie wypełniamy średnią historyczną (żeby mieć co wstawić w przyszłość)
            avg_history = (s15 * 0.4 + s18 * 0.6) # Baza do uzupełnienia luki
            
            # Budujemy s22: Tam gdzie mamy dane, bierzemy realne. Tam gdzie nie, bierzemy historię.
            s22_full = avg_history.copy()
            s22_full.update(m22_real) # Nadpisujemy prawdziwymi danymi
            
            # Ale uwaga: Przejście z realnych danych na średnią może zrobić "schodek".
            # Wygładzamy to (opcjonalne, ale estetyczne), lub zostawiamy brutalnie.
            # Tutaj zostawiamy, bo waga 40% i tak to wygładzi.

            # --- 2. TWORZENIE KRZYWEJ BAZOWEJ (MASTER CURVE) ---
            # To jest Twój wzór: 25% c15 + 35% c18 + 40% c22
            master_curve = (s15 * 0.25) + (s18 * 0.35) + (s22_full * 0.40)

            # --- 3. PROJEKCJA NA 8 LAT (2 CYKLE) ---
            
            full_projection = []
            
            # Wartość końcowa pierwszego cyklu (baza dla drugiego)
            end_val_c1 = master_curve.iloc[-1]
            peak_val = master_curve.max()

            # Pętla po 2920 dniach (2 cykle)
            for d in range(cycle_len * 2 + 1):
                
                # --- PIERWSZY CYKL (2022 - 2026) ---
                if d <= cycle_len:
                    # Tutaj po prostu bierzemy naszą wyliczoną krzywą
                    val = master_curve.get(d, master_curve.iloc[-1])
                    full_projection.append(val)
                    
                # --- DRUGI CYKL (2026 - 2030) ---
                else:
                    d2 = d - cycle_len
                    # Bierzemy kształt master_curve, ale modyfikujemy go (Damping & Raising)
                    base_val = master_curve.get(d2, master_curve.iloc[-1])
                    
                    # LOGIKA MODYFIKATORÓW (Twoje życzenie):
                    # 1. Zduszenie szczytu o ~18%
                    # 2. Podniesienie dołka o ~20%
                    
                    # Czy jesteśmy wysoko? (Blisko szczytu)
                    if base_val > (peak_val * 0.6): 
                        # Im wyżej, tym mocniej tniemy (progresywnie do 0.82)
                        modifier = 0.82
                    # Czy jesteśmy nisko? (W dołku)
                    elif base_val < (peak_val * 0.3):
                        # Podnosimy dołek
                        modifier = 1.20
                    else:
                        # Środek - płynne przejście
                        modifier = 0.95
                        
                    # Obliczamy nową wartość dla drugiego cyklu
                    # Nowa_Cena = (Cena_Koniec_Cyklu_1) * (Wzrost_w_Cyklu_2 * Modyfikator)
                    # Musimy podzielić base_val przez start cyklu (który jest ~1.0), żeby dostać sam wzrost
                    growth_factor = base_val 
                    val = end_val_c1 * growth_factor * modifier
                    
                    full_projection.append(val)

            # --- 4. SKALOWANIE DO RZECZYWISTOŚCI ---
            full_curve = pd.Series(full_projection)
            
            # Znajdujemy dzisiejszy dzień w cyklu
            current_days = (c22.index[-1] - b22).days
            current_price = c22['Price'].iloc[-1]
            base_price_22 = c22['Price'].iloc[0]
            
            # Sprawdzamy, gdzie jest model w dniu dzisiejszym
            model_val_today = full_curve.get(current_days, 1.0)
            
            # Obliczamy czynnik skalujący, żeby wykres "kleił się" do dzisiejszej ceny
            scale_factor = current_price / (base_price_22 * model_val_today)
            
            # Generujemy ceny
            theoretical_path = (base_price_22 * full_curve) * scale_factor
            
            # Generujemy daty
            date_index = [b22 + timedelta(days=x) for x in range(len(full_projection))]
            theoretical_path.index = date_index

            return c22['Price'].to_frame(), theoretical_path

        except Exception as e:
            print(f"Błąd Nostradamusa Weighted: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_btc_weighted_projection(self, df_real, df_proj):
        if df_real is None or df_proj is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # 1. Historia
        ax.plot(df_real.index, df_real.iloc[:, 0], color='white', linewidth=2, label='Dane Rzeczywiste', zorder=10)
        
        # 2. Przyszłość
        last_date = df_real.index[-1]
        future_part = df_proj[df_proj.index > last_date]
        
        ax.plot(future_part.index, future_part, color='#00e5ff', linestyle='--', linewidth=2, label='Model Ważony (25/35/40)', alpha=0.9)
        
        # --- PUNKTY ZWROTNE ---
        # Szukamy Dna 2026 i Szczytu 2030
        
        # Okno czasowe na Dno 2026
        mask_2026 = (future_part.index > pd.to_datetime('2026-01-01')) & (future_part.index < pd.to_datetime('2027-12-31'))
        section_2026 = future_part[mask_2026]
        
        if not section_2026.empty:
            bot_date = section_2026.idxmin()
            bot_price = section_2026.min()
            ax.scatter([bot_date], [bot_price], color='#ff9900', s=130, zorder=15, edgecolors='white', linewidth=2)
            ax.text(bot_date, bot_price * 0.80, f"DNO 2026\n(+20%)\n${bot_price:,.0f}", color='#ff9900', ha='center', fontweight='bold', fontsize=9)

        # Okno czasowe na Szczyt 2029/2030
        mask_2030 = (future_part.index > pd.to_datetime('2028-01-01'))
        section_2030 = future_part[mask_2030]
        
        if not section_2030.empty:
            top_date = section_2030.idxmax()
            top_price = section_2030.max()
            ax.scatter([top_date], [top_price], color='#00ff55', s=130, zorder=15, edgecolors='white', linewidth=2)
            ax.text(top_date, top_price * 1.15, f"SZCZYT 2029/30\n(-18%)\n${top_price:,.0f}", color='#00ff55', ha='center', fontweight='bold', fontsize=9)

        # Tytuły
        ax.set_title("NOSTRADAMUS PRO: 25% (2015) | 35% (2018) | 40% (2022)", fontsize=16, color=t['text'], fontweight='bold')
        
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        ax.grid(True, alpha=0.15, color=t['grid'])
        ax.legend(facecolor=t['bg'], labelcolor=t['text'])
        
        return fig

    # --- PRO TOOLS: NOSTRADAMUS 4.0 (Logarithmic Probability) ---
    def get_btc_probability_projection(self):
        """
        Generuje projekcję opartą na PRAWDOPODOBIEŃSTWIE LOGARYTMICZNYM.
        Zamiast średniej arytmetycznej (która zawyża wyniki przez stare cykle),
        stosujemy średnią geometryczną ważoną i uwzględniamy "Market Cap Drag".
        
        To naturalnie sprowadza szczyty do realistycznych poziomów (ETF Era),
        bez sztucznego ucinania wyników.
        """
        try:
            # Daty dołków
            b15 = pd.to_datetime('2015-01-14')
            b18 = pd.to_datetime('2018-12-15')
            b22 = pd.to_datetime('2022-11-21') 
            
            # Pobieranie danych
            data = yf.download('BTC-USD', start='2014-01-01', progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                try: df = data.xs('Close', axis=1, level=0, drop_level=True)
                except: df = data['Close']
            else:
                df = data['Close']
            
            if isinstance(df, pd.Series): df = df.to_frame(name='Price')
            else: df = df.iloc[:, 0].to_frame(name='Price')

            # --- PRZYGOTOWANIE CYKLI (Log Returns) ---
            # Obliczamy "Logarytmiczny Mnożnik".
            # Logarytm naturalnie kompresuje gigantyczne wzrosty z przeszłości.
            
            cycle_len = 1460 # 4 lata
            days_idx = range(cycle_len + 1)

            # Pomocnicza funkcja do wyciągania znormalizowanej krzywej
            def get_log_curve(start_date, end_date):
                c = df[(df.index >= start_date) & (df.index < end_date)].copy()
                if c.empty: return None
                c['Days'] = (c.index - start_date).days
                # Mnożnik (ile razy urosło)
                mult = c.set_index('Days')['Price'] / c['Price'].iloc[0]
                # Zamiana na logarytm (np. 100x -> 4.6, 20x -> 3.0)
                log_mult = np.log(mult)
                # Interpolacja
                return log_mult.reindex(days_idx).interpolate(limit_direction='both').fillna(method='ffill')

            # Logarytmiczne krzywe historyczne
            log_c15 = get_log_curve(b15, b18)
            log_c18 = get_log_curve(b18, b22)
            
            # Obecny cykl (Realny)
            c22 = df[df.index >= b22].copy()
            c22['Days'] = (c22.index - b22).days
            log_c22_real = np.log(c22.set_index('Days')['Price'] / c22['Price'].iloc[0])
            
            # Uzupełnienie luki 2022 (czym wypełnić przyszłość do 2026?)
            # Używamy średniej logarytmicznej z historii, ale przeskalowanej do obecnej zmienności
            # Obliczamy "Energy Factor": Jak silny jest obecny cykl vs 2018?
            # Porównujemy max log_return z teraz do max log_return z 2018 w tym samym punkcie czasu
            current_day = c22['Days'].max()
            val_now = log_c22_real.iloc[-1]
            val_18_then = log_c18.get(current_day, 1.0)
            
            energy_ratio = val_now / val_18_then if val_18_then > 0 else 0.5
            # Ograniczamy, żeby nie wyszło szaleństwo
            if energy_ratio > 0.8: energy_ratio = 0.8 
            
            # Tworzymy syntetyczną końcówkę dla obecnego cyklu (bazując na kształcie 2018, ale z obecną energią)
            log_c22_synthetic = log_c18 * energy_ratio
            log_c22_full = log_c22_synthetic.copy()
            log_c22_full.update(log_c22_real) # Wklejamy prawdę tam gdzie jest

            # --- OBLICZANIE ŚREDNIEJ WAŻONEJ (W LOGARYTMACH) ---
            # To jest klucz! Średnia w logarytmach to Średnia Geometryczna w cenach.
            # To naturalnie "zabija" odstające wartości (outliers) z 2015 roku.
            
            # Wagi Użytkownika
            w15 = 0.25
            w18 = 0.35
            w22 = 0.40
            
            # Dodatkowy czynnik "Market Maturity" (Dojrzewanie Rynku)
            # Cykl 2015 musi ważyć mniej w "cenie", mimo że waży 25% w "czasie".
            # W tym modelu wagi wpływają głównie na KSZTAŁT fali (kiedy szczyt/dołek),
            # a mniej na amplitudę (cenę), bo logarytm to spłaszcza.
            
            master_log_curve = (log_c15 * w15 * 0.4) + (log_c18 * w18 * 0.7) + (log_c22_full * w22 * 1.0)
            # Wyjaśnienie mnożników (0.4, 0.7, 1.0):
            # To jest "Waga Energetyczna". Mówimy, że "siła" cyklu 2015 jest dziś relewantna tylko w 40%.
            
            # Konwersja z Logarytmu na Mnożnik Ceny (exp)
            master_multiplier = np.exp(master_log_curve)
            
            # --- PROJEKCJA NA 8 LAT (2 CYKLE) ---
            full_projection = []
            
            # Wartość końcowa cyklu 1
            end_mult_c1 = master_multiplier.iloc[-1]
            
            for d in range(cycle_len * 2 + 1):
                # Cykl 1 (2022-2026)
                if d <= cycle_len:
                    val = master_multiplier.get(d, master_multiplier.iloc[-1])
                    full_projection.append(val)
                
                # Cykl 2 (2026-2030)
                else:
                    d2 = d - cycle_len
                    # Bierzemy kształt z master_curve
                    base_shape = master_multiplier.get(d2, master_multiplier.iloc[-1])
                    
                    # Logika ETF/Instytucji dla drugiego cyklu:
                    # Mniejsza zmienność = mniejsze wzrosty, ale mniejsze spadki.
                    # Zamiast mnożyć, potęgujemy logarytm (kompresja).
                    # Wzrost ^ 0.7 (spłaszczenie)
                    
                    val_c2 = np.power(base_shape, 0.70)
                    
                    # Doklejamy do końca cyklu 1
                    # Uwaga: Musimy przeskalować start c2 do 1.0
                    rel_growth = val_c2 # base_shape startuje od 1.0 (bo exp(0)=1)
                    
                    full_projection.append(end_mult_c1 * rel_growth)

            # --- SKALOWANIE DO RZECZYWISTOŚCI ---
            full_curve = pd.Series(full_projection)
            
            # Punkt styku (DZIŚ)
            current_days = (c22.index[-1] - b22).days
            current_price = c22['Price'].iloc[-1]
            base_price_22 = c22['Price'].iloc[0] # $15.5k
            
            # Sprawdzamy mnożnik modelu na dziś
            model_val_today = full_curve.get(current_days, 1.0)
            
            # Kalibracja: Jeśli model mówi 3.0x a jest 4.0x, podnosimy cały wykres
            scale_factor = current_price / (base_price_22 * model_val_today)
            
            theoretical_path = (base_price_22 * full_curve) * scale_factor
            
            # Daty
            date_index = [b22 + timedelta(days=x) for x in range(len(full_projection))]
            theoretical_path.index = date_index
            
            return c22['Price'].to_frame(), theoretical_path

        except Exception as e:
            print(f"Błąd Nostradamusa 4.0: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_btc_probability_projection(self, df_real, df_proj):
        if df_real is None or df_proj is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # 1. Historia
        ax.plot(df_real.index, df_real.iloc[:, 0], color='white', linewidth=2, label='Dane Rzeczywiste', zorder=10)
        
        # 2. Przyszłość (Nostradamus 4.0)
        last_date = df_real.index[-1]
        future_part = df_proj[df_proj.index > last_date]
        
        ax.plot(future_part.index, future_part, color='#00e5ff', linestyle='--', linewidth=2, label='Model Prawdopodobieństwa (Log-Scale)', alpha=0.9)
        
        # --- ANALIZA PUNKTÓW ---
        
        # Szczyt 2025 (Koniec Hossy)
        # Ograniczamy szukanie do 2026 roku
        mask_2025 = (future_part.index < pd.to_datetime('2026-06-01'))
        bull_2025 = future_part[mask_2025]
        
        if not bull_2025.empty:
            top_date = bull_2025.idxmax()
            top_price = bull_2025.max()
            
            # Jeśli szczyt jest w przyszłości
            if top_date > last_date:
                ax.scatter([top_date], [top_price], color='#00ff55', s=130, zorder=15, edgecolors='white', linewidth=2)
                ax.text(top_date, top_price * 1.05, f"REALISTYCZNY SZCZYT\n${top_price:,.0f}", 
                        color='#00ff55', ha='center', fontweight='bold', fontsize=10)

        # Dno 2026/2027 (Miękkie Lądowanie)
        mask_2027 = (future_part.index >= pd.to_datetime('2026-01-01')) & (future_part.index < pd.to_datetime('2028-01-01'))
        bear_2027 = future_part[mask_2027]
        
        if not bear_2027.empty:
            bot_date = bear_2027.idxmin()
            bot_price = bear_2027.min()
            
            ax.scatter([bot_date], [bot_price], color='#ff9900', s=130, zorder=15, edgecolors='white', linewidth=2)
            ax.text(bot_date, bot_price * 0.90, f"STABILNE DNO\n${bot_price:,.0f}", 
                    color='#ff9900', ha='center', fontweight='bold', fontsize=10)

        # Szczyt 2029 (Dojrzały Rynek)
        mask_2030 = (future_part.index >= pd.to_datetime('2028-01-01'))
        bull_2030 = future_part[mask_2030]
        
        if not bull_2030.empty:
            top2_date = bull_2030.idxmax()
            top2_price = bull_2030.max()
            
            ax.scatter([top2_date], [top2_price], color='#00e5ff', s=130, zorder=15, edgecolors='white', linewidth=2)
            ax.text(top2_date, top2_price * 1.05, f"SZCZYT 2029/30\n${top2_price:,.0f}", 
                    color='#00e5ff', ha='center', fontweight='bold', fontsize=10)

        # Tytuły
        ax.set_title("NOSTRADAMUS 4.0: Model Prawdopodobieństwa (ETF Era)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Model oparty na logarytmicznym dojrzewaniu rynku (Market Cap Drag). Brak sztucznych limitów.', color='gray', fontsize=10)
        
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        ax.grid(True, alpha=0.15, color=t['grid'])
        ax.legend(facecolor=t['bg'], labelcolor=t['text'])
        
        return fig

    # --- PRO TOOLS: NOSTRADAMUS 5.0 (Fixed Anchor / Seamless) ---
    def get_btc_gold_projection(self):
        """
        Generuje projekcję HYBRYDOWĄ (BTC Cycles + Gold ETF).
        NAPRAWIONO: 'Anchoring'. Żółta linia startuje idealnie z końca białej.
        """
        try:
            # --- DANE ---
            b15 = pd.to_datetime('2015-01-14')
            b18 = pd.to_datetime('2018-12-15')
            b22 = pd.to_datetime('2022-11-21') 
            
            data = yf.download(['BTC-USD', 'GLD'], start='2004-01-01', progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                try: 
                    df_btc = data['Close']['BTC-USD'].to_frame(name='Price')
                    df_gld = data['Close']['GLD'].to_frame(name='Price')
                except:
                    df_btc = data.xs('BTC-USD', level=1, axis=1)['Close'].to_frame(name='Price')
                    df_gld = data.xs('GLD', level=1, axis=1)['Close'].to_frame(name='Price')
            else:
                return None, None

            df_btc = df_btc.dropna()
            df_gld = df_gld.dropna()

            # --- 1. MODEL CYKLICZNY (BAZA) ---
            cycle_len = 1460 * 2 
            days_idx = range(cycle_len + 1)
            
            def get_log_curve(df_in, start, end):
                c = df_in[(df_in.index >= start) & (df_in.index < end)].copy()
                if c.empty: return None
                c['Days'] = (c.index - start).days
                mult = c.set_index('Days')['Price'] / c['Price'].iloc[0]
                # FIX: Zamiast .fillna(method='ffill') dajemy .ffill()
                return np.log(mult).reindex(days_idx).interpolate(limit_direction='both').ffill()

            log_c15 = get_log_curve(df_btc, b15, b18)
            log_c18 = get_log_curve(df_btc, b18, b22)
            
            c22 = df_btc[df_btc.index >= b22].copy()
            c22['Days'] = (c22.index - b22).days
            log_c22_real = np.log(c22.set_index('Days')['Price'] / c22['Price'].iloc[0])
            
            log_c22_synthetic = (log_c15 * 0.4 + log_c18 * 0.6) * 0.9
            log_c22_full = log_c22_synthetic.copy()
            log_c22_full.update(log_c22_real)

            base_log_curve = (log_c15 * 0.25) + (log_c18 * 0.35) + (log_c22_full * 0.40)
            
            # Tłumienie drugiego cyklu
            final_base_curve = []
            end_c1 = base_log_curve.iloc[1460] if len(base_log_curve) > 1460 else base_log_curve.iloc[-1]
            
            for d in range(len(days_idx)):
                if d <= 1460:
                    val = base_log_curve.get(d, 0)
                else:
                    d2 = d - 1460
                    shape = base_log_curve.get(d2, 0)
                    val = end_c1 + (shape * 0.7) 
                final_base_curve.append(val)
            
            base_curve_exp = np.exp(final_base_curve)

            # --- 2. FRAKTAL ZŁOTA (GLD) ---
            gld_start = pd.to_datetime('2004-11-18')
            gld_etf_era = df_gld[df_gld.index >= gld_start].copy()
            gld_etf_era['Days'] = (gld_etf_era.index - gld_start).days
            gld_multiplier = gld_etf_era.set_index('Days')['Price'] / gld_etf_era['Price'].iloc[0]
            
            volatility_factor = 3.5 
            gld_scaled = (gld_multiplier - 1) * volatility_factor + 1
            # FIX: Zamiast .fillna(method='ffill') dajemy .ffill()
            gld_fractal = gld_scaled.reindex(days_idx).interpolate(limit_direction='both').ffill()
            
            shift_days = 415 
            gld_shifted = gld_fractal.shift(shift_days).fillna(1.0)
            
            # MIKS
            final_projection_mult = (base_curve_exp * 0.80) + (gld_shifted * 0.20)
            
            # --- 3. KALIBRACJA PUNKTU STYKU (THE GLUE) ---
            full_curve = pd.Series(final_projection_mult)
            
            current_days = (c22.index[-1] - b22).days
            current_price = c22['Price'].iloc[-1]
            base_price_22 = c22['Price'].iloc[0]
            
            # Wartość modelu w DNIU DZISIEJSZYM
            model_val_today = full_curve.get(current_days, 1.0)
            
            # Obliczamy cenę modelu na dziś (gdyby nie było korekty)
            model_price_today_raw = base_price_22 * model_val_today
            
            # Ratio korygujące: Jaka jest różnica między Prawdą a Modelem DZIŚ?
            anchor_ratio = current_price / model_price_today_raw
            
            # Aplikujemy to ratio do CAŁEJ krzywej przyszłości.
            # To przesuwa cały żółty wykres w górę/dół, żeby idealnie trafił w koniec białego.
            theoretical_path = (base_price_22 * full_curve) * anchor_ratio
            
            date_index = [b22 + timedelta(days=x) for x in range(len(final_projection_mult))]
            theoretical_path.index = date_index
            
            return c22['Price'].to_frame(), theoretical_path

        except Exception as e:
            print(f"Błąd Nostradamusa 5.0 (Anchor): {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_btc_gold_projection(self, df_real, df_proj):
        if df_real is None or df_proj is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # 1. Historia
        ax.plot(df_real.index, df_real.iloc[:, 0], color='white', linewidth=2, label='Dane Rzeczywiste', zorder=10)
        
        # 2. Przyszłość (ZSZYWANIE)
        last_real_date = df_real.index[-1]
        last_real_price = df_real.iloc[-1, 0]
        
        # Bierzemy przyszłość
        future_part = df_proj[df_proj.index > last_real_date]
        
        # Dodajemy ostatni punkt historii na początek przyszłości, żeby nie było dziury graficznej
        seed_point = pd.Series([last_real_price], index=[last_real_date])
        future_part_connected = pd.concat([seed_point, future_part])
        
        # Rysujemy
        ax.plot(future_part_connected.index, future_part_connected, color='#ffd700', linestyle='--', linewidth=2.5, label='Model "Złota Era"', alpha=1.0)
        
        # --- ZNACZNIKI ---
        # Szczyt
        mask_top = (future_part.index < pd.to_datetime('2026-06-01'))
        if not future_part[mask_top].empty:
            top_date = future_part[mask_top].idxmax()
            top_price = future_part[mask_top].max()
            
            if top_date > last_real_date:
                ax.scatter([top_date], [top_price], color='#00ff55', s=150, zorder=15, edgecolors='white', linewidth=2)
                ax.text(top_date, top_price * 1.05, f"SZCZYT ZŁOTEJ ERY\n${top_price:,.0f}", 
                        color='#00ff55', ha='center', fontweight='bold', fontsize=10)
                
        # Dołek (Stabilizacja ETF)
        # Szukamy dołka po szczycie
        mask_bot = (future_part.index > top_date) & (future_part.index < pd.to_datetime('2028-01-01'))
        bot_part = future_part[mask_bot]
        
        if not bot_part.empty:
            bot_date = bot_part.idxmin()
            bot_price = bot_part.min()
            
            ax.scatter([bot_date], [bot_price], color='#ff9900', s=120, zorder=15, edgecolors='white', linewidth=2)
            ax.text(bot_date, bot_price * 0.85, f"KOREKTA ETF\n(Płytka Bessa)\n${bot_price:,.0f}", 
                    color='#ff9900', ha='center', fontweight='bold', fontsize=9)


        # Tytuły
        ax.set_title("NOSTRADAMUS 5.0: The Golden Age (Połączone)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Model uwzględnia wydłużony cykl (Raoul Pal Theory) i wpływ ETF na stabilizację ceny.', color='gray', fontsize=10)
        
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        ax.grid(True, alpha=0.15, color=t['grid'])
        ax.legend(facecolor=t['bg'], labelcolor=t['text'])
        
        return fig

    # --- PRO TOOLS: GRAHAM'S GHOST (Bubble Detector - Expanded) ---
    def get_graham_valuation_data(self):
        """
        Oblicza Wycenę Grahama (Fair Value) dla szerokiego rynku.
        Rozszerzona lista: Tech, Finanse, Retail, Zdrowie, Krypto.
        """
        # Lista "Hype & Value" - Ponad 60 kluczowych spółek
        tickers = [
            # MAGNIFICENT 7 + AI & CHIPS
            'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA', 
            'AMD', 'AVGO', 'ORCL', 'QCOM', 'INTC', 'MU', 'TXN', 'SMCI',
            
            # SOFTWARE & CLOUD
            'ADBE', 'CRM', 'PLTR', 'SNOW', 'NOW', 'SAP', 'INTU', 'IBM', 'UBER',
            
            # FINANSE & KRYPTO
            'JPM', 'BAC', 'V', 'MA', 'GS', 'MS', 'BLK', 
            'PYPL', 'HOOD', 'COIN', 'MSTR', 'SQ',
            
            # KONSUMPCJA & RETAIL
            'NKE', 'MCD', 'SBUX', 'WMT', 'COST', 'TGT', 'HD', 'LOW', 
            'KO', 'PEP', 'PG', 'DIS', 'NFLX',
            
            # ZDROWIE (BIG PHARMA)
            'LLY', 'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'ISRG', 'CVS',
            
            # PRZEMYSŁ & ENERGIA & AUTO
            'XOM', 'CVX', 'CAT', 'DE', 'GE', 'BA', 'LMT', 
            'F', 'GM', 'TM',
            
            # INNE CIEKAWE (Chiny, Telekomy)
            'BABA', 'PDD', 'T', 'VZ'
        ]

        valuation_data = []

        try:
            # Pobieranie danych (Loop po tickerach)
            # Uwaga: Przy 60 spółkach to może potrwać ok. 10-15 sekund
            
            for t in tickers:
                try:
                    stock = yf.Ticker(t)
                    # fast_info jest szybsze dla ceny, ale info potrzebne dla EPS/BookValue
                    info = stock.info
                    
                    price = info.get('currentPrice', 0)
                    eps = info.get('trailingEps', 0)
                    bvps = info.get('bookValue', 0)
                    
                    if price == 0: continue

                    # --- FORMUŁA GRAHAMA ---
                    # Fair Value = Pierwiastek(22.5 * EPS * BVPS)
                    
                    graham_number = 0
                    if eps > 0 and bvps > 0:
                        graham_number = np.sqrt(22.5 * eps * bvps)
                    else:
                        graham_number = 0 # Firma przynosi straty lub ma ujemną wartość księgową
                    
                    # Obliczamy GAP (Lukę)
                    if graham_number > 0:
                        gap = ((price - graham_number) / graham_number) * 100
                    else:
                        gap = 999 # Oznaczenie dla firm "Bez Fundamentów" (Czysty Hype/Strata)

                    valuation_data.append({
                        'Ticker': t,
                        'Price': price,
                        'GrahamPrice': graham_number,
                        'Gap': gap,
                        'EPS': eps
                    })
                    
                except Exception:
                    continue
            
            df = pd.DataFrame(valuation_data)
            if not df.empty:
                # Sortujemy: Najtańsze (Niedowartościowane) na górze
                df = df.sort_values(by='Gap', ascending=True)
            
            return df

        except Exception as e:
            print(f"Błąd Ducha Grahama: {e}")
            return None

    def plot_graham_ghost(self, df):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        
        # Filtrowanie i Sortowanie
        df_plot = df.copy()
        
        # Odrzucamy "Śmieci" (Gap 999 - firmy bez zysków), żeby pokazać tylko te policzalne
        # Ewentualnie można je pokazać na końcu, ale Graham by je odrzucił od razu.
        df_plot = df_plot[df_plot['Gap'] != 999]
        
        # Sortujemy rosnąco wg Gap (Najbardziej ujemne = Największe okazje na górze)
        df_plot = df_plot.sort_values(by='Gap', ascending=True)
        
        # Bierzemy np. 25 najbardziej skrajnych (Najtańsze i Najdroższe)
        # Albo po prostu pierwsze 25 najtańszych, bo tego szukamy
        df_plot = df_plot.head(25)
        
        # Zwiększamy wysokość wykresu (z 10 na 14), bo mamy więcej pozycji
        fig = plt.figure(figsize=(13, 14))
        ax = fig.add_subplot(111)
        
        y_pos = np.arange(len(df_plot))
        
        for i, row in enumerate(df_plot.itertuples()):
            ticker = row.Ticker
            price = row.Price
            graham = row.GrahamPrice
            gap = row.Gap
            
            # Kolor: Zielony jeśli cena < graham, Czerwony jeśli odwrotnie
            color = '#00ff55' if price < graham else '#ff0055'
            
            # Most (Pasek)
            ax.plot([price, graham], [i, i], color=color, linewidth=4, alpha=0.7, zorder=1)
            
            # Kropka Ceny (Biała)
            ax.scatter(price, i, color='white', s=100, zorder=5, edgecolors=color, linewidth=2)
            
            # Kropka Grahama (Złoty Diament)
            ax.scatter(graham, i, color='#ffd700', s=100, marker='D', zorder=5)
            
            # Tekst %
            diff_text = f"{gap:+.1f}%"
            # Pozycjonowanie tekstu (zawsze trochę na prawo od prawej kropki)
            text_x = max(price, graham)
            
            ax.text(text_x * 1.1, i, diff_text, 
                    va='center', color=color, fontweight='bold', fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['Ticker'], fontsize=11, fontweight='bold', color='white')
        
        ax.set_title("DUCH GRAHAMA: Top 25 Okazji (Cena < Wartość)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Cena Akcji ($) - Skala Logarytmiczna', color=t['text'])
        
        # Legenda
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Cena Rynkowa', markerfacecolor='white', markersize=10),
            Line2D([0], [0], marker='D', color='w', label='Wartość Grahama', markerfacecolor='#ffd700', markersize=10),
            Line2D([0], [0], color='#00ff55', lw=4, label='Promocja (Okazja)'),
            Line2D([0], [0], color='#ff0055', lw=4, label='Hype (Drogo)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', facecolor=t['bg'], labelcolor=t['text'])
        
        # Oś X Logarytmiczna (bo porównujemy akcje po $20 i po $3000)
        ax.set_xscale('log')
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        ax.tick_params(colors=t['text'])
        # ax.grid(True, axis='x', alpha=0.15, color=t['grid'])
        ax.grid(False) # <--- TO WYŁĄCZA SIATKĘ SEABORNA
        
        return fig

    # --- PRO TOOLS: GRAHAM'S JUDGE (Data Logic) ---
    def get_graham_valuation_data(self, subset='all'):
        """
        Oblicza Wycenę Grahama.
        subset='dow': Skanuje TYLKO 30 spółek z indeksu Dow Jones (Elita).
        subset='all': Skanuje szeroki rynek (60+ spółek).
        """
        # 1. Definicja Listy DOW JONES 30 (Aktualny skład 2025/26)
        dow_30 = [
            'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON',
            'IBM', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA',
            'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WMT', 'DIS', 'SHW', 'AMZN'
        ]
        
        # 2. Definicja Reszty Rynku
        market_picks = [
            'GOOGL', 'META', 'TSLA', 'AMD', 'AVGO', 'ORCL', 'QCOM', 'INTC', 'MU', 'TXN',
            'ADBE', 'PLTR', 'SNOW', 'UBER', 'BAC', 'WFC', 'MS', 'BLK', 'PYPL', 'HOOD', 
            'COIN', 'MSTR', 'SBUX', 'COST', 'TGT', 'LOW', 'PEP', 'LLY', 'PFE', 'ABBV', 
            'CVS', 'DE', 'GE', 'F', 'GM', 'TM', 'BABA', 'T', 'PDD', 'SPOT'
        ]

        # Wybór koszyka na podstawie argumentu
        if subset == 'dow':
            target_tickers = dow_30
        else:
            target_tickers = list(set(dow_30 + market_picks))

        valuation_data = []

        try:
            # Pobieranie danych
            # (Pobieramy w pętli, bo potrzebujemy info o EPS i BookValue)
            for t in target_tickers:
                try:
                    stock = yf.Ticker(t)
                    info = stock.info
                    
                    price = info.get('currentPrice', 0)
                    eps = info.get('trailingEps', 0)
                    bvps = info.get('bookValue', 0)
                    
                    if price == 0: continue

                    # Formuła Grahama: Sqrt(22.5 * EPS * BVPS)
                    graham_number = 0
                    if eps > 0 and bvps > 0:
                        graham_number = np.sqrt(22.5 * eps * bvps)
                    
                    # Gap (Luka)
                    if graham_number > 0:
                        gap = ((price - graham_number) / graham_number) * 100
                    else:
                        gap = 999 

                    valuation_data.append({
                        'Ticker': t,
                        'Price': price,
                        'GrahamPrice': graham_number,
                        'Gap': gap,
                        'IsDow': t in dow_30
                    })
                    
                except Exception:
                    continue
            
            df = pd.DataFrame(valuation_data)
            if not df.empty:
                df = df.sort_values(by='Gap', ascending=True)
            
            return df

        except Exception as e:
            print(f"Błąd Grahama: {e}")
            return None

    # --- WIZUALIZACJA ---
    def plot_graham_ghost(self, df, title_suffix=""):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        
        # Filtrujemy, sortujemy i bierzemy Top 30
        df_plot = df[df['Gap'] != 999].copy()
        df_plot = df_plot.sort_values(by='Gap', ascending=True).head(30)
        
        fig = plt.figure(figsize=(13, 14))
        ax = fig.add_subplot(111)
        
        y_pos = np.arange(len(df_plot))
        
        for i, row in enumerate(df_plot.itertuples()):
            color = '#00ff55' if row.Price < row.GrahamPrice else '#ff0055'
            
            # Pasek
            ax.plot([row.Price, row.GrahamPrice], [i, i], color=color, linewidth=4, alpha=0.6, zorder=1)
            # Cena (Biała)
            ax.scatter(row.Price, i, color='white', s=130, zorder=5, edgecolors=color, linewidth=2)
            # Graham (Złoty)
            ax.scatter(row.GrahamPrice, i, color='#ffd700', s=130, marker='D', zorder=5)
            # Tekst
            ax.text(max(row.Price, row.GrahamPrice) * 1.15, i, f"{row.Gap:+.1f}%", va='center', color=color, fontweight='bold', fontsize=10)
            
        ax.set_yticks(y_pos)
        # Oznaczenie Diamentem dla spółek z Dow Jones
        labels = [f"💎 {row.Ticker}" if getattr(row, 'IsDow', False) else row.Ticker for row in df_plot.itertuples()]
        ax.set_yticklabels(labels, fontsize=11, fontweight='bold', color='white')
        
        full_title = f"SĘDZIA WARTOŚCI: {title_suffix}"
        ax.set_title(full_title, fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Cena Akcji ($) - Skala Logarytmiczna', color=t['text'])
        ax.set_xscale('log')
        
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        ax.tick_params(colors=t['text'])
        # ax.grid(True, axis='x', alpha=0.15, color=t['grid'])
        ax.grid(False) # <--- TO WYŁĄCZA SIATKĘ SEABORNA
        
        return fig

    # --- PRO TOOLS: DOW JONES ARCHITECT (PURE TECH) ---
    def get_dow_architect_data(self):
        """
        Skanuje 30 spółek Dow Jones (Aktualny Skład).
        Wersja CZYSTA TECHNICZNA (Bez Grahama).
        Szuka: Wyprzedania (RSI), Niskiej Ceny (vs SMA200) i Paniki (Drawdown).
        """
        # Aktualny skład Dow Jones (Bez Intela, z Nvidią)
        dow_30 = [
            'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON',
            'IBM', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA',
            'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WMT', 'DIS', 'SHW', 'AMZN'
        ]

        architect_data = []

        try:
            # Pobieramy dane (2 lata wstecz wystarczy dla SMA200)
            data = yf.download(dow_30, period="2y", progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                try: closes = data.xs('Close', axis=1, level=0, drop_level=True)
                except: closes = data['Close']
            else:
                closes = data['Close']

            for ticker in dow_30:
                if ticker not in closes.columns: continue
                
                price_series = closes[ticker].dropna()
                if len(price_series) < 200: continue
                
                curr_price = price_series.iloc[-1]
                high_52 = price_series.iloc[-252:].max()
                
                # 1. SMA 200 (Średnia roczna)
                sma_200 = price_series.rolling(200).mean().iloc[-1]
                dist_sma = ((curr_price - sma_200) / sma_200) * 100
                
                # 2. RSI 14
                delta = price_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
                
                # 3. Drawdown (Spadek od szczytu)
                drawdown = ((curr_price - high_52) / high_52) * 100
                
                # --- PUNKTACJA (SCORE) ---
                # Im więcej punktów, tym większa okazja techniczna ("Diament w błocie")
                score = 0
                
                # RSI (Max 40 pkt)
                if rsi_val < 30: score += 40      # Skrajne wyprzedanie
                elif rsi_val < 40: score += 25
                elif rsi_val < 50: score += 10
                elif rsi_val > 70: score -= 15    # Wykupienie
                
                # SMA 200 (Max 30 pkt)
                if dist_sma < -15: score += 30    # Głęboko pod średnią
                elif dist_sma < -5: score += 15
                elif dist_sma > 20: score -= 10   # Mocno nad średnią (Bańka)
                
                # Drawdown (Max 30 pkt)
                if drawdown < -30: score += 30    # Krach na spółce
                elif drawdown < -20: score += 20
                elif drawdown < -10: score += 10
                
                architect_data.append({
                    'Ticker': ticker,
                    'Price': curr_price,
                    'Score': score,
                    'RSI': rsi_val,
                    'DistSMA': dist_sma,
                    'Drawdown': drawdown
                })

            df = pd.DataFrame(architect_data)
            if not df.empty:
                # Sortujemy: Najlepsze okazje na górze
                df = df.sort_values(by='Score', ascending=False)
            
            return df

        except Exception as e:
            print(f"Błąd Architekta Dow: {e}")
            return None

    def plot_dow_architect(self, df):
        if df is None or df.empty: return None
        
        t = self.get_theme_colors()
        df_plot = df.copy()
        
        fig = plt.figure(figsize=(13, 14))
        ax = fig.add_subplot(111)
        
        # Kolory (Neonowy Zielony dla okazji, Czerwony dla przegrzanych)
        colors = []
        for s in df_plot['Score']:
            if s >= 60: colors.append('#00ff00')   # Super Okazja (RSI < 30, Cena < SMA)
            elif s >= 30: colors.append('#00e5ff') # Dobra okazja
            elif s >= 0: colors.append('#aaaaaa')  # Neutral
            else: colors.append('#ff0055')         # Przegrzane
            
        bars = ax.barh(df_plot['Ticker'], df_plot['Score'], color=colors, alpha=0.8)
        ax.invert_yaxis() # Najlepsze na górze
        
        for bar, price, rsi, sma, dd in zip(bars, df_plot['Price'], df_plot['RSI'], df_plot['DistSMA'], df_plot['Drawdown']):
            width = bar.get_width()
            
            # Cena wewnątrz paska
            ax.text(2 if width > 0 else -2, bar.get_y() + bar.get_height()/2, f"${price:.0f}", 
                    va='center', ha='left' if width > 0 else 'right', 
                    color='black' if width > 20 else 'white', fontweight='bold', fontsize=9)
            
            # Statystyki za paskiem
            stats = f"RSI: {rsi:.0f} | SMA200: {sma:+.0f}% | DD: {dd:.0f}%"
            ax.text(max(width, 0) + 2, bar.get_y() + bar.get_height()/2, stats,
                    va='center', ha='left', color='white', fontsize=10)

        ax.set_title("DOW JONES ARCHITECT: Ranking Techniczny (RSI + SMA)", fontsize=16, color=t['text'], fontweight='bold')
        ax.set_xlabel('Score Techniczny (Im więcej, tym bardziej wyprzedane)', color=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_visible(False)
        ax.tick_params(colors=t['text'], axis='y', labelsize=11)
        for label in ax.get_yticklabels(): label.set_fontweight('bold')
        # ax.grid(True, axis='x', alpha=0.15, color=t['grid'])
        ax.grid(False) # <--- TO WYŁĄCZA SIATKĘ SEABORNA
        
        return fig

    # --- PRO TOOLS: FED DATA LOGIC ---
    def get_fed_policy_data(self):
        """
        Pobiera rentowność obligacji 10-letnich (TNX) i określa sentyment Fedu.
        Zwraca słownik z danymi i statusem (Dovish/Hawkish).
        """
        try:
            # 1. Pobieramy rentowność 10-latek (Smart Money)
            bond = yf.Ticker("^TNX")
            hist = bond.history(period="1mo")
            
            if hist.empty: return None

            current_yield = hist['Close'].iloc[-1]
            prev_yield = hist['Close'].iloc[-2]
            month_ago_yield = hist['Close'].iloc[0]
            
            # 2. Obliczamy zmianę
            daily_change = current_yield - prev_yield
            monthly_change = current_yield - month_ago_yield
            
            # 3. Logika Oceny (Algorytm)
            # Spadek rentowności = Rynek gra pod luźną politykę (Drukarka)
            # Wzrost rentowności = Rynek gra pod twardą politykę (Zasysanie)
            
            if monthly_change < -0.05: # Trend spadkowy
                status_title = "🟢 DRUKARKA (Gołąb)"
                desc = "Rynek gra pod obniżki stóp."
                icon = "🕊️"
                color = "green"
                delta_color = "inverse" # Zielony przy spadku (dla nas dobrze)
            
            elif monthly_change > 0.15: # Wyraźny wzrost
                status_title = "🔴 ZACISKANIE (Jastrząb)"
                desc = "Rynek boi się inflacji/stóp."
                icon = "🦅"
                color = "red"
                delta_color = "inverse" # Czerwony przy wzroście (dla nas źle)
            
            else:
                status_title = "🟡 NEUTRALNIE"
                desc = "Czekanie na dane makro."
                icon = "⚖️"
                color = "orange"
                delta_color = "off"

            return {
                "current_yield": current_yield,
                "daily_change": daily_change,
                "status_title": status_title,
                "desc": desc,
                "icon": icon,
                "color": color,
                "delta_color": delta_color
            }

        except Exception as e:
            print(f"Błąd Fed Data: {e}")
            return None

    # --- PRO TOOLS: FED SIDEBAR DISPLAY (NAPRAWIONE KOLORY) ---
    def display_fed_monitor_sidebar(self):
        """
        Wyświetla widget w pasku bocznym.
        POPRAWKA: Usunięto jasne tła (st.success/error), zastosowano kolorowany tekst HTML
        dla lepszej czytelności w trybie ciemnym.
        """
        data = self.get_fed_policy_data()
        
        if data is None:
            st.sidebar.caption("Fed Monitor: Brak danych")
            return

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏛️ FED & MONEY MONITOR")
        
        # 1. Metryka (Liczby) - To wygląda dobrze w dark mode
        st.sidebar.metric(
            label=f"Rentowność 10Y (Sygnał)",
            value=f"{data['current_yield']:.2f}%",
            delta=f"{data['daily_change']:+.2f} pkt",
            delta_color=data['delta_color']
        )
        
        st.sidebar.markdown("") # Mały odstęp
        
        # 2. Werdykt (Tekst) - TUTAJ BYŁ PROBLEM
        # Zamiast st.success/error, używamy markdown z HTMLem, żeby kolorować TYLKO tekst.
        
        if data['color'] == "green":
            # Neonowa zieleń dla Gołębia
            html_text = f"""
            <div style='margin-top: 10px;'>
                <span style='font-size: 1.2em;'>{data['icon']}</span>
                <span style='color: #00ff55; font-weight: bold; font-size: 1.1em;'> {data['status_title']}</span>
            </div>
            """
            st.sidebar.markdown(html_text, unsafe_allow_html=True)
            st.sidebar.caption(data['desc'])

        elif data['color'] == "red":
            # Neonowa czerwień dla Jastrzębia
            html_text = f"""
            <div style='margin-top: 10px;'>
                <span style='font-size: 1.2em;'>{data['icon']}</span>
                <span style='color: #ff0055; font-weight: bold; font-size: 1.1em;'> {data['status_title']}</span>
            </div>
            """
            st.sidebar.markdown(html_text, unsafe_allow_html=True)
            st.sidebar.caption(data['desc'])

        else:
            # Pomarańczowy dla Neutralnego
            html_text = f"""
            <div style='margin-top: 10px;'>
                <span style='font-size: 1.2em;'>{data['icon']}</span>
                <span style='color: #ffaa00; font-weight: bold; font-size: 1.1em;'> {data['status_title']}</span>
            </div>
            """
            st.sidebar.markdown(html_text, unsafe_allow_html=True)
            st.sidebar.caption(data['desc'])

    def apply_custom_style(self):
        """
        STYLE V5 (SLIM BUTTONS): 
        1. Sidebar -> Tło Ciemne.
        2. Teksty -> Śnieżnobiałe.
        3. Inputy -> Białe tło, Czarny tekst.
        4. Scrollbar -> Biały po najechaniu.
        5. GUZIKI -> Cieńsza czcionka (font-weight: 400).
        """
        st.markdown("""
            <style>
                /* --- 1. GŁÓWNY KONTENER SIDEBARA --- */
                [data-testid="stSidebar"] {
                    background-color: #0E1117;
                    border-right: 1px solid #262730;
                }

                /* --- 2. TEKSTY W SIDEBARZE (BIAŁE) --- */
                [data-testid="stSidebar"] h1, 
                [data-testid="stSidebar"] h2, 
                [data-testid="stSidebar"] h3, 
                [data-testid="stSidebar"] p, 
                [data-testid="stSidebar"] span, 
                [data-testid="stSidebar"] label,
                [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
                    color: #ffffff !important;
                }

                [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
                    color: #ffffff !important;
                    font-weight: bold !important;
                }
                [data-testid="stSidebar"] [data-testid="stMetricValue"] {
                    color: #ffffff !important;
                }
                [data-testid="stSidebar"] .streamlit-expanderHeader p {
                    color: #ffffff !important;
                    font-weight: bold !important;
                }

                /* --- 3. INPUTY (SUWAKI) - BIAŁE TŁO, CZARNY TEKST --- */
                div[data-testid="stNumberInput"] label p {
                    color: #ffffff !important;
                    font-size: 16px !important;
                    font-weight: 900 !important;
                }
                div[data-testid="stNumberInput"] input {
                    background-color: #ffffff !important;
                    color: #000000 !important;
                    border: 2px solid #00ff55 !important;
                    font-weight: bold !important;
                }
                div[data-testid="stNumberInput"] button {
                    background-color: #e0e0e0 !important;
                    color: #000000 !important;
                    border: 1px solid #cccccc !important;
                }
                div[data-testid="stNumberInput"] button svg {
                    fill: #000000 !important;
                }
                div[data-testid="stNumberInput"] button:hover {
                    background-color: #00ff55 !important;
                    border-color: #00ff55 !important;
                }

                /* --- 4. POZIOME LINIE --- */
                [data-testid="stSidebar"] hr {
                    border-color: #555555 !important;
                }

                /* --- 5. SCROLLBAR --- */
                [data-testid="stSidebar"] ::-webkit-scrollbar {
                    width: 10px !important;
                    height: 10px !important;
                }
                [data-testid="stSidebar"] ::-webkit-scrollbar-track {
                    background: #0E1117 !important;
                }
                [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
                    background: #555555 !important;
                    border-radius: 5px !important;
                }
                [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
                    background: #ffffff !important;
                    border: 1px solid #00ff55 !important;
                }

                /* --- 6. ODCHUDZANIE GUZIKÓW (BUTTONS) --- */
                
                /* Styl ogólny guzika */
                div.stButton > button {
                    background-color: #1e1e1e !important;
                    color: white !important;
                    border: 1px solid #444 !important;
                }
                
                /* To tutaj zmieniamy grubość czcionki wewnątrz guzika */
                div.stButton > button p {
                    font-weight: 400 !important; /* <--- ZMIANA: 400 to 'normal', 700 to 'bold' */
                    font-size: 14px !important;  /* Opcjonalnie: lekko mniejsza czcionka */
                }

                /* Efekt najechania (Hover) */
                div.stButton > button:hover {
                    border-color: #00ff55 !important;
                    color: #00ff55 !important;
                }
                /* Tekst po najechaniu też ma być cienki, ale zielony */
                div.stButton > button:hover p {
                    color: #00ff55 !important;
                }

            </style>
        """, unsafe_allow_html=True)

    # --- PRO TOOLS: PORTFOLIO MANAGER (Visualizer Only) ---
    def display_portfolio_manager(self, current_equity):
        """
        Wyświetla status 'Na co mnie stać' na podstawie obliczonej wartości portfela.
        Nie pyta o kwotę - przyjmuje ją jako argument 'current_equity'.
        """
        # Zapisujemy w sesji, żeby inne moduły (np. Kelly, Monte Carlo) widziały tę kwotę
        st.session_state['equity'] = current_equity

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### 💰 WARTOŚĆ: ${current_equity:,.2f}")

        # LOGIKA "NA CO MNIE STAĆ"
        # Cel: Lambo Urus (~250,000 $)
        lambo_goal = 250000.0
        progress = min(current_equity / lambo_goal, 1.0)
        
        if current_equity < 50:
            level = "Żul spod Żabki"
            item = "Frytki z ketchupem (na spółę z gołębiem)"
            icon = "🍟"
            color = "#888888"
        elif current_equity < 500:
            level = "Student I roku"
            item = "Zupka Chińska 'Złoty Kurczak' (Premium)"
            icon = "🍜"
            color = "#aaffaa"
        elif current_equity < 2000:
            level = "Początkujący Marzyciel"
            item = "Rower Wigry 3 (lekka rdza na błotniku)"
            icon = "🚲"
            color = "#00ffaa"
        elif current_equity < 5000:
            level = "Król Wsi (Junior)"
            item = "Golf III 1.9 TDI (klimatyzacja korbotronic)"
            icon = "🚗"
            color = "#00ccff"
        elif current_equity < 10000:
            level = "Handlarz Mirek"
            item = "Passat B5 w Kombi (Niemiec płakał)"
            icon = "💨"
            color = "#0088ff"
        elif current_equity < 25000:
            level = "Prezes Spółdzielni"
            item = "Używana Toyota Yaris (od emeryta)"
            icon = "🚙"
            color = "#5555ff"
        elif current_equity < 50000:
            level = "Programista 15k"
            item = "Nowa Dacia Duster (wypas, skóry!)"
            icon = "😎"
            color = "#aa00ff"
        elif current_equity < 100000:
            level = "Kryptowalutowy Baron"
            item = "Kawalerka w Radomiu (stan deweloperski)"
            icon = "🏢"
            color = "#ff00ff"
        elif current_equity < 200000:
            level = "Rekin Giełdy"
            item = "Porsche Panamera (po dzwonie w USA)"
            icon = "🏎️"
            color = "#ff0088"
        elif current_equity < 249000:
            level = "Prawie Elon Musk"
            item = "Już czujesz zapach nowej skóry..."
            icon = "🤏"
            color = "#ff0044"
        else:
            level = "KRÓL ŻYCIA (IMPERATOR)"
            item = "LAMBORGHINI URUS (Salon Polska!)"
            icon = "🚀"
            color = "#00ff00"

        # Wyświetlanie
        st.sidebar.progress(progress)
        
        percent = (current_equity / lambo_goal) * 100
        
        html_status = f"""
        <div style='background-color: #1E1E1E; padding: 12px; border-radius: 8px; border: 1px solid #333; margin-top: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
            <div style='color: #aaa; font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px;'>Status Społeczny</div>
            <div style='color: {color}; font-size: 1.0em; font-weight: bold; margin-top: 2px;'>{level}</div>
            <hr style='border: 0; border-top: 1px solid #333; margin: 8px 0;'>
            <div style='color: #fff; font-size: 0.9em;'>{icon} Stać cię na:</div>
            <div style='color: #fff; font-size: 1.1em; font-weight: bold; font-style: italic;'>"{item}"</div>
            <div style='color: #666; font-size: 0.8em; margin-top: 8px; text-align: right;'>Do Lambo: {percent:.1f}%</div>
        </div>
        """
        st.sidebar.markdown(html_status, unsafe_allow_html=True)

    # --- PRO TOOLS: SEASONAL PATTERN (Average Year) ---
    def get_seasonal_stats(self, heatmap_data):
        """
        Oblicza statystyki z mapy cieplnej (Pivot Table):
        1. Średni zwrot dla każdego miesiąca.
        2. Win Rate (% zielonych lat) dla każdego miesiąca.
        """
        if heatmap_data is None or heatmap_data.empty: return None
        
        # heatmap_data ma miesiące jako kolumny (Jan, Feb...)
        
        # 1. Średnia
        avg_returns = heatmap_data.mean()
        
        # 2. Win Rate (Ile razy było > 0)
        # count() zlicza nie-puste, sum() zlicza True
        win_rate = (heatmap_data > 0).sum() / heatmap_data.count() * 100
        
        # 3. Skumulowany Trend (Symulacja roku)
        # Zakładamy start od 100%
        cum_trend = [100]
        for r in avg_returns:
            # Procent składany: poprzedni * (1 + zmiana%)
            new_val = cum_trend[-1] * (1 + r/100)
            cum_trend.append(new_val)
            
        # Usuwamy punkt startowy 100, zostawiamy 12 miesięcy
        cum_trend = pd.Series(cum_trend[1:], index=avg_returns.index)
        
        return pd.DataFrame({
            'Avg': avg_returns,
            'WinRate': win_rate,
            'Trend': cum_trend
        })

    def plot_seasonal_stats(self, stats_df):
        """
        Rysuje wykres słupkowy średnich zwrotów + linię trendu rocznego.
        """
        if stats_df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        
        months = stats_df.index
        
        # --- 1. SŁUPKI (Średnia zmiana miesięczna) ---
        colors = ['#00ff55' if x >= 0 else '#ff0055' for x in stats_df['Avg']]
        bars = ax1.bar(months, stats_df['Avg'], color=colors, alpha=0.6, label='Średnia Zmiana %')
        
        # Etykiety na słupkach
        for bar, val in zip(bars, stats_df['Avg']):
            y_pos = bar.get_height()
            offset = 1 if val >= 0 else -1.5
            ax1.text(bar.get_x() + bar.get_width()/2, y_pos + offset, f"{val:+.1f}%", 
                     ha='center', va='bottom' if val<0 else 'top', 
                     color='white', fontweight='bold', fontsize=9)

        ax1.set_ylabel('Średnia Zmiana Miesięczna (%)', color=t['text'])
        ax1.tick_params(axis='y', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        ax1.axhline(0, color=t['text'], linewidth=1)

        # --- 2. LINIA (Skumulowany Trend Roczny) - Prawa Oś ---
        ax2 = ax1.twinx()
        ax2.plot(months, stats_df['Trend'], color='#ffd700', linewidth=3, marker='o', label='Ścieżka "Idealnego Roku"')
        
        ax2.set_ylabel('Symulacja Roku (Start=100)', color='#ffd700', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#ffd700', colors=t['text'])
        
        # Dodajemy Win Rate nad punktami linii
        for i, (m, val) in enumerate(zip(months, stats_df['Trend'])):
            wr = stats_df['WinRate'].iloc[i]
            ax2.text(i, val + 2, f"WR:\n{wr:.0f}%", ha='center', color='#ffd700', fontsize=8)

        # Tytuł
        ax1.set_title("SEZONOWY WZORZEC BTC: Jak wygląda przeciętny rok?", fontsize=16, color=t['text'], fontweight='bold')
        
        # Legenda
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor=t['bg'], labelcolor=t['text'])

        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax2.spines['right'].set_color(t['text'])
        
        return fig

    def display_bottom_ticker(self):
        """
        Wyświetla pasek z reklamami na dole ekranu (wysokość 60px).
        Dodano klauzulę prawną (Not Financial Advice) dla bezpieczeństwa.
        """
        # Teksty reklamowe zmiksowane z zastrzeżeniem prawnym
        ad_text = (
            "🚀 <b>SONAI PREMIUM:</b> Odblokuj Zdrowie psychiczne z SonAi!  |  "
            "<span style='color: #ff0055;'>⚠️ <b>NOT FINANCIAL ADVICE:</b> Wszystkie dane mają charakter wyłącznie edukacyjny. </span>  |  "
            "💎 <b>ZASADA #1:</b> Nie tracimy pieniędzy.  |  "
            "🧠 <b>HEAL-TO-EARN:</b> SonAi-Zarabiaj krypto dbając o zdrowie (Już wkrótce!)  |  "
            "<span style='color: #ff0055;'>⚠️ <b>RYZYKO:</b> Inwestujesz na własną odpowiedzialność. </span>  |  "
            "📊 <b>NOWOŚĆ:</b> Sprawdź zakładkę 'Sezonowość' i zobacz idealny rok BTC.  |  "
            "📞 <b>REKLAMA TUTAJ:</b> Twoja firma widoczna dla inwestorów - kontakt: ads@sonai.com  |  "
            "🔥 <b>PRO TIP:</b> Kiedy wszyscy się boją, Ty szukaj okazji (zobacz Skaner Bólu). "
        )

        # Styl CSS i HTML paska (wysokość 60px)
        ticker_html = f"""
        <style>
            .ticker-container {{
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                height: 60px;
                background-color: #0e1117; 
                color: #00ff41; 
                z-index: 999999; 
                border-top: 2px solid #00ff41;
                display: flex;
                align-items: center;
                font-family: 'Courier New', monospace;
                font-size: 18px;
                box-shadow: 0px -5px 15px rgba(0, 255, 65, 0.2);
            }}
            
            marquee {{
                width: 100%;
                line-height: 60px;
            }}

            /* Ukrywamy standardową stopkę Streamlit */
            footer {{visibility: hidden;}}
            
            /* Margines dla głównej zawartości */
            .main .block-container {{
                padding-bottom: 100px !important;
            }}
        </style>

        <div class="ticker-container">
            <marquee loop="infinite" direction="left" scrollamount="10">
                {ad_text}
            </marquee>
        </div>
        """
        
        # Wyświetlamy ticker na dole
        st.markdown(ticker_html, unsafe_allow_html=True)
        
        # Automatyczne dodanie pełnego Expandera do Sidebaru (żebyś nie musiał pamiętać o wywołaniu)
        st.sidebar.markdown("---")
        with st.sidebar.expander("⚖️ DISCLAIMER (NOTA PRAWNA)", expanded=False):
            st.markdown(f"""
            <div style='font-size: 0.85em; text-align: justify; color: #aaa; line-height: 1.4;'>
                <b>Lambo czy Karton (v{APP_VERSION})</b> to oprogramowanie analityczne. 
                <br><br>
                1. <b>Brak Porady:</b> Żadna informacja tutaj zawarta nie jest "rekomendacją inwestycyjną".
                <br><br>
                2. <b>Ryzyko:</b> Rynek finansowy jest nieprzewidywalny. Możesz stracić wszystkie zainwestowane środki.
                <br><br>
                3. <b>Odpowiedzialność:</b> Autor nie ponosi odpowiedzialności za Twoje zyski ani straty.
            </div>
            """, unsafe_allow_html=True)

    # --- GOOGLE ANALYTICS (FIX: TRWAŁE ID UŻYTKOWNIKA) ---
    def setup_analytics(self):
        """
        Wysyła dane do GA4.
        NAPRAWA: Używa trwałego pliku 'user_id.json', aby rozpoznać tego samego użytkownika
        nawet po restarcie programu.
        """
        # Sprawdzamy, czy już wysłaliśmy sygnał w tej sesji (żeby nie dublować przy klikaniu)
        if 'analytics_sent' in st.session_state:
            return

        GA_ID = "G-D4BM5ZM6NB"
        id_filename = "user_id.json"
        cid = None

        # --- 1. LOGIKA TRWAŁEGO ID ---
        try:
            # Sprawdzamy czy plik z ID już istnieje
            if os.path.exists(id_filename):
                with open(id_filename, 'r') as f:
                    data = json.load(f)
                    cid = data.get('client_id')
            
            # Jeśli pliku nie ma lub jest uszkodzony, generujemy nowy ID
            if not cid:
                cid = str(uuid.uuid4())
                with open(id_filename, 'w') as f:
                    json.dump({'client_id': cid}, f)
                    
        except Exception as e:
            # Fallback w razie błędu zapisu (np. brak uprawnień)
            print(f"⚠️ Błąd pliku ID: {e}")
            cid = str(uuid.uuid4())

        # Zapisujemy ID w sesji na wszelki wypadek
        st.session_state['client_id'] = cid

        # --- 2. WYSYŁANIE DANYCH ---
        # Parametry
        payload = {
            'v': '2',                   # Wersja protokołu GA4
            'tid': GA_ID,               # Twój ID
            'cid': cid,                 # TRWAŁE ID UŻYTKOWNIKA
            'en': 'page_view',          # Zdarzenie: Wyświetlenie strony
            'dl': 'https://lambo-app.com/home', # Udajemy URL
            'dt': 'Lambo czy Karton',   # Tytuł strony
            'seg': '1',                 # Aktywna sesja
            '_p': cid                   # Debugging ID
        }

        # Wysyłamy dane po cichu w tle
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            requests.post(
                'https://www.google-analytics.com/g/collect', 
                params=payload, 
                headers=headers, 
                timeout=2
            )
            # print(f"✅ Analytics sent. User ID: {cid}")
            
            # Oznaczamy sukces w sesji
            st.session_state['analytics_sent'] = True
            
        except Exception as e:
            print(f"⚠️ Błąd wysyłania analityki: {e}")

    # --- SYSTEM AKTUALIZACJI (AUTO-UPDATER) ---
    def check_for_updates(self):
        """
        Sprawdza, czy na GitHubie jest nowsza wersja kodu.
        Jeśli tak, pozwala użytkownikowi zaktualizować aplikację jednym kliknięciem.
        """
        # TUTAJ WKLEISZ LINK DO SWOJEGO PLIKU NA GITHUBIE (Wersja RAW)
        # Na razie zostaw to puste lub wpisz ten przykład, wyjaśnię niżej jak go zdobyć:
        UPDATE_URL = "https://raw.githubusercontent.com/SonAi-ai/lambo-app/refs/heads/main/market_app.py"
        
        try:
            # 1. Pobieramy kod z internetu (tylko tekst)
            response = requests.get(UPDATE_URL, timeout=3)
            
            if response.status_code == 200:
                new_code = response.text
                
                # 2. Szukamy wersji w nowym kodzie za pomocą "Regex"
                # Szuka linijki: APP_VERSION = "x.x"
                match = re.search(r'APP_VERSION\s*=\s*"([0-9\.]+)"', new_code)
                
                if match:
                    remote_version = match.group(1)
                    
                    # 3. Porównujemy wersje (Lokalna vs Internetowa)
                    # Jeśli wersja w necie jest inna (większa) niż nasza
                    if remote_version != APP_VERSION:
                        st.warning(f"🚀 **DOSTĘPNA NOWA WERSJA!** (Twoja: {APP_VERSION} -> Nowa: {remote_version})")
                        st.info("Nowości: Ulepszenia algorytmu i poprawki błędów.")
                        
                        # 4. Przycisk aktualizacji
                        if st.button("📥 POBIERZ I ZAKTUALIZUJ TERAZ"):
                            with st.spinner("Pobieranie aktualizacji... Nie wyłączaj programu!"):
                                # Nadpisujemy bieżący plik nowym kodem
                                with open(__file__, "w", encoding="utf-8") as f:
                                    f.write(new_code)
                                
                                st.success("✅ Aktualizacja zakończona! Aplikacja zaraz się zrestartuje.")
                                import time
                                time.sleep(2)
                                st.rerun() # Restartuje aplikację
                else:
                    # Nie znaleziono wersji w kodzie online (błąd pliku)
                    pass
                    
        except Exception as e:
            # Jeśli nie ma internetu lub link jest zły, po prostu milczymy (żeby nie wkurzać usera)
            # print(f"Błąd sprawdzania aktualizacji: {e}") 
            pass

    # --- METODA 1: OBLICZENIA (GET) ---
    def get_altcoin_indicator_data(self):
        # Pobieramy dane ETH
        df = yf.download("ETH-USD", start="2019-01-01", progress=False)
        
        if df.empty:
            st.error("Nie udało się pobrać danych ETH.")
            return None, None

        # Fix dla yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Liczymy średnie
        df['SMA100'] = df['Close'].rolling(window=100).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        
        # Logika: SMA100 przecina EMA100 w dół
        df['Prev_SMA'] = df['SMA100'].shift(1)
        df['Prev_EMA'] = df['EMA100'].shift(1)
        df['Bearish_Cross'] = ((df['Prev_SMA'] >= df['Prev_EMA']) & (df['SMA100'] < df['EMA100']))
        
        signals = df[df['Bearish_Cross']]
        return df, signals

    # --- METODA 2: RYSOWANIE (PLOT) ---
    def plot_altcoin_indicator(self, df, signals):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Kolory
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')

        # Linie
        ax.plot(df.index, df['Close'], color='white', alpha=0.3, linewidth=1, label='Cena ETH')
        ax.plot(df.index, df['SMA100'], color='#00d4ff', linewidth=1.5, label='SMA 100')
        ax.plot(df.index, df['EMA100'], color='#ff9900', linewidth=1.5, label='EMA 100')

        # Sygnały
        for date, row in signals.iterrows():
            ax.axvline(x=date, color='#ff0055', linestyle='--', alpha=0.5)
            ax.scatter(date, row['SMA100'], color='#ff0055', s=100, zorder=5, marker='v')

        ax.set_title("Altcoin Cycle Indicator (SMA100 < EMA100)", color='white')
        ax.set_yscale('log')
        ax.legend(facecolor='#0e1117', labelcolor='white')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        
        return fig

    def display_legal_disclaimer(self):
        """
        Wyświetla profesjonalne zastrzeżenie prawne (Disclaimer) w sidebarze.
        Chroni autora przed odpowiedzialnością za decyzje inwestycyjne użytkowników.
        """
        st.sidebar.markdown("---")
        with st.sidebar.expander("⚖️ NOTA PRAWNA (DISCLAIMER)", expanded=False):
            st.markdown(f"""
            <div style='font-size: 0.8em; text-align: justify; color: #aaa;'>
                <b>Lambo czy Karton (v{APP_VERSION})</b> jest narzędziem wyłącznie edukacyjnym i informacyjnym. 
                <br><br>
                1. <b>To nie jest porada:</b> Żadne dane, analizy ani predykcje AI (Nostradamus, Oracle, itp.) 
                nie stanowią rekomendacji inwestycyjnej w rozumieniu przepisów prawa.
                <br><br>
                2. <b>Ryzyko:</b> Inwestowanie w kryptowaluty i akcje wiąże się z wysokim ryzykiem utraty kapitału. 
                Wyniki historyczne nie gwarantują przyszłych zysków.
                <br><br>
                3. <b>Brak odpowiedzialności:</b> Autor oprogramowania nie ponosi odpowiedzialności za jakiekolwiek 
                straty finansowe powstałe w wyniku korzystania z niniejszej aplikacji.
                <br><br>
                4. <b>Dane:</b> Dane pochodzą z zewnętrznych źródeł (Yahoo Finance, FRED). Nie gwarantujemy 
                ich 100% dokładności ani ciągłości przesyłu.
                <br><br>
                <i>Decyzje podejmujesz wyłącznie na własne ryzyko. Zawsze konsultuj się z licencjonowanym doradcą finansowym.</i>
            </div>
            """, unsafe_allow_html=True)

    # --- NOWOŚĆ: INSIDER TRADING TRACKER (Congress Copy-Trade) ---
    def get_congress_tracker_data(self):
        """
        Analizuje wyniki inwestycyjne Kongresu USA w porównaniu do zwykłych ludzi (S&P 500).
        Używa ETF-ów:
        - NANC (Demokraci - Nancy Pelosi Strategy)
        - KRUZ (Republikanie - Ted Cruz Strategy)
        - SPY (Benchmark - Rynek)
        """
        try:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            tickers = ['NANC', 'KRUZ', 'SPY']
            
            # Pobieramy dane
            data = yf.download(tickers, start=start_date, progress=False)
            
            # Fix MultiIndex (Standardowa procedura)
            if isinstance(data.columns, pd.MultiIndex):
                try: 
                    df = data.xs('Close', axis=1, level=0, drop_level=True)
                except KeyError:
                    try: 
                        df = data.xs('Close', axis=1, level=1, drop_level=True)
                    except: 
                        return None
            else:
                df = data['Close']

            # Upewniamy się, że mamy wszystkie kolumny
            # Czasem ETFy są nowe i mogą mieć braki, więc usuwamy NaN
            df = df.dropna()

            if df.empty: return None

            # Normalizacja (Start = 0%)
            # Dzięki temu widzimy czysty zysk/stratę od początku okresu
            df_norm = (df / df.iloc[0] - 1) * 100
            
            # Obliczamy "Insider Alpha" (Przewaga nad rynkiem)
            # Ile Pelosi zarobiła więcej niż zwykły Kowalski?
            current_nanc = df_norm['NANC'].iloc[-1]
            current_spy = df_norm['SPY'].iloc[-1]
            
            alpha = current_nanc - current_spy
            
            return df_norm, alpha

        except Exception as e:
            print(f"Błąd Congress Tracker: {e}")
            return None, 0

    def plot_congress_tracker(self, df, alpha):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # --- 1. RYSOWANIE LINII ---
        
        # DEMOKRACI (NANC) - Niebieski
        # To tutaj zazwyczaj siedzi "Big Tech Insider" (Nancy Pelosi kupuje NVDA, MSFT)
        ax1.plot(df.index, df['NANC'], color='#2979ff', linewidth=2.5, label='Demokraci (NANC ETF)')
        
        # REPUBLIKANIE (KRUZ) - Czerwony
        # Często więcej energii i przemysłu
        ax1.plot(df.index, df['KRUZ'], color='#ff1744', linewidth=2.5, label='Republikanie (KRUZ ETF)')
        
        # RYNEK (SPY) - Biały/Szary przerywany
        ax1.plot(df.index, df['SPY'], color='white', linewidth=1.5, linestyle='--', alpha=0.7, label='Zwykli Ludzie (S&P 500)')
        
        # --- 2. KOLOROWANIE PRZEWAGI (Insider Advantage) ---
        # Wypełniamy przestrzeń między Demokratami a Rynkiem
        # Jeśli Pelosi wygrywa -> Zielona poświata "Insider Profit"
        ax1.fill_between(df.index, df['NANC'], df['SPY'], where=(df['NANC'] > df['SPY']), 
                         color='#00ff55', alpha=0.1, label='INSIDER ADVANTAGE (Zysk ponad rynek)')
        
        # --- 3. METRYKI KOŃCOWE ---
        last_date = df.index[-1]
        nanc_ret = df['NANC'].iloc[-1]
        kruz_ret = df['KRUZ'].iloc[-1]
        spy_ret = df['SPY'].iloc[-1]
        
        # Kropki na końcu
        ax1.scatter(last_date, nanc_ret, color='#2979ff', s=100, zorder=10)
        ax1.scatter(last_date, kruz_ret, color='#ff1744', s=100, zorder=10)
        ax1.scatter(last_date, spy_ret, color='white', s=80, zorder=10)
        
        # Teksty przy kropkach
        ax1.text(last_date, nanc_ret + 1, f"DEMS: {nanc_ret:+.1f}%", color='#2979ff', fontweight='bold', fontsize=10)
        ax1.text(last_date, kruz_ret - 2, f"REPS: {kruz_ret:+.1f}%", color='#ff1744', fontweight='bold', fontsize=10)
        
        # Tytuł
        title_status = "KONGRES WYGRYWA Z RYNKIEM 🏛️💰" if alpha > 0 else "KONGRES TRACI (Dziwne... 🤔)"
        ax1.set_title(f"INSIDER TRADING TRACKER: {title_status}", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Zysk z inwestycji (%)', color=t['text'])
        
        # Linia zero
        ax1.axhline(0, color=t['text'], linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Legenda
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        # Stylizacja
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- NOWOŚĆ: HEDGING CALCULATOR (Defense Mode) ---
    def get_hedging_data(self):
        """
        Oblicza poziom defensywy (Hedging Score) na podstawie:
        1. VIX (Indeks Strachu)
        2. DXY (Trend Dolara)
        3. Gold/SPX Ratio (Ucieczka do bezpiecznych przystani)
        """
        try:
            # Pobieramy dane (1 rok)
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            # ^VIX = Volatility Index
            # DX-Y.NYB = US Dollar Index
            # GC=F = Gold Futures
            # ^GSPC = S&P 500
            tickers = ['^VIX', 'DX-Y.NYB', 'GC=F', '^GSPC']
            
            data = yf.download(tickers, start=start_date, progress=False)
            
            # Fix MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                try: df = data.xs('Close', axis=1, level=0, drop_level=True)
                except: 
                    try: df = data.xs('Close', axis=1, level=1, drop_level=True)
                    except: return None, 0
            else:
                df = data['Close']

            # Uzupełnianie braków
            df = df.ffill().dropna()
            
            if df.empty: return None, 0

            # --- OBLICZENIA WSKAŹNIKÓW ---
            
            # 1. Analiza VIX (Ostatnia wartość)
            last_vix = df['^VIX'].iloc[-1]
            
            # 2. Trend DXY (Cena vs SMA 50)
            dxy_series = df['DX-Y.NYB']
            dxy_sma50 = dxy_series.rolling(50).mean().iloc[-1]
            last_dxy = dxy_series.iloc[-1]
            dxy_trend_bullish = last_dxy > dxy_sma50
            
            # 3. Gold vs SPX Ratio (Czy kapitał ucieka z akcji do złota?)
            ratio = df['GC=F'] / df['^GSPC']
            ratio_sma50 = ratio.rolling(50).mean().iloc[-1]
            last_ratio = ratio.iloc[-1]
            ratio_trend_bullish = last_ratio > ratio_sma50
            
            # --- ALGORYTM "DEFENSE SCORE" (0-100%) ---
            # 0% = Full Risk (Lambo)
            # 100% = Full Cash (Bunker)
            
            score = 0
            reasons = []
            
            # A. Ocena VIX (Strach)
            if last_vix < 15:
                score += 0 # Spokój
                reasons.append("VIX nisko (Spokój)")
            elif 15 <= last_vix < 20:
                score += 10 # Lekki niepokój
            elif 20 <= last_vix < 25:
                score += 30 # Strach
                reasons.append(f"VIX wysoki ({last_vix:.1f})")
            elif 25 <= last_vix < 30:
                score += 50 # Panika
                reasons.append("VIX paniczny!")
            else: # > 30
                score += 80 # Krach
                reasons.append("VIX KRACH (>30)!")
                
            # B. Ocena DXY (Dolar)
            if dxy_trend_bullish:
                score += 20
                reasons.append("Dolar rośnie (Risk Off)")
            
            # C. Ocena Gold/SPX (Rotacja defensywna)
            if ratio_trend_bullish:
                score += 10
                reasons.append("Złoto bije Akcje")
                
            # Limit 100%
            score = min(score, 100)
            
            # Zwracamy dane do wykresu oraz wynik
            result_data = {
                'vix': df['^VIX'],
                'ratio': ratio,
                'reasons': reasons
            }
            
            return result_data, score

        except Exception as e:
            print(f"Błąd Hedging Calc: {e}")
            return None, 0

    def plot_hedging_cockpit(self, data, score):
        if data is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(10, 8))
        
        # Układ: Góra (Pasek Obronny), Dół (Wykres VIX vs Ratio)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # --- WYKRES 1: PASEK STANU (DEFENSE METER) ---
        
        # Kolor paska zależny od wyniku
        if score < 20: 
            bar_color = '#00ff55' # Zielony (Atak)
            status = "TRYB ATAKU (Risk On)"
        elif score < 50: 
            bar_color = '#ffd700' # Żółty (Ostrożność)
            status = "TRYB OSTROŻNY (Neutral)"
        else: 
            bar_color = '#ff0055' # Czerwony (Obrona)
            status = "TRYB BUNKRA (Risk Off)"
            
        # Rysujemy pasek tła i pasek wyniku
        ax1.barh(0, 100, color=t['text'], alpha=0.1, height=0.5) # Tło
        ax1.barh(0, score, color=bar_color, alpha=0.9, height=0.5) # Wynik
        
        # Teksty
        ax1.text(50, 0, f"Zalecana Gotówka/Hedging: {score}%", ha='center', va='center', 
                 color='white' if score > 50 else t['text'], fontweight='bold', fontsize=14)
        
        ax1.set_title(f"SYSTEM OBRONNY: {status}", fontsize=16, color=t['text'], fontweight='bold')
        ax1.axis('off') # Ukrywamy osie dla paska
        
        # --- WYKRES 2: VIX vs GOLD/SPX (Dlaczego?) ---
        
        # Oś lewa: VIX (Strach) - Czerwona linia
        vix = data['vix']
        ax2.plot(vix.index, vix, color='#ff0055', linewidth=2, label='VIX (Strach)')
        ax2.axhline(20, color='#ff0055', linestyle='--', alpha=0.5, linewidth=1) # Poziom ostrzegawczy
        ax2.fill_between(vix.index, vix, 20, where=(vix > 20), color='#ff0055', alpha=0.15)
        
        ax2.set_ylabel('Indeks Strachu (VIX)', color='#ff0055', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#ff0055', colors=t['text'])
        ax2.tick_params(axis='x', colors=t['text'])
        
        # Oś prawa: Gold/SPX Ratio (Ucieczka) - Złota linia
        ax3 = ax2.twinx()
        ratio = data['ratio']
        # Normalizujemy do widoku (żeby pasowało do VIX) lub po prostu rysujemy trend
        ax3.plot(ratio.index, ratio, color='#ffd700', linewidth=1.5, linestyle=':', label='Złoto vs S&P 500')
        
        ax3.set_ylabel('Siła Złota względem Akcji', color='#ffd700', fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='#ffd700', colors=t['text'])
        
        # Kosmetyka
        ax3.spines['top'].set_visible(False); ax3.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False); ax3.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_color(t['text']); ax2.spines['left'].set_color(t['text'])
        
        ax2.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        ax3.legend(loc='upper right', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg']); ax2.set_facecolor(t['bg'])
        ax2.grid(True, alpha=0.1, color=t['grid'])
        
        return fig

    # --- NOWOŚĆ: COMMODITY SUPERCYCLE (Papier vs Rzeczy) ---
    def get_commodity_supercycle_data(self):
        """
        Analizuje relację Surowców do Akcji (S&P 500).
        Tworzy własny indeks surowcowy (Hard Assets Index) składający się z:
        1. Energii (Ropa CL=F) - Krew gospodarki.
        2. Przemysłu (Miedź HG=F) - Dr. Copper (wskaźnik koniunktury).
        3. Pieniądza (Złoto GC=F) - Ochrona wartości.
        
        Porównuje to do S&P 500 (^GSPC).
        """
        try:
            # 10 lat to minimum, żeby zobaczyć cykl surowcowy
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            
            tickers = ['CL=F', 'HG=F', 'GC=F', '^GSPC']
            
            data = yf.download(tickers, start=start_date, progress=False)
            
            # Fix MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                try: df = data.xs('Close', axis=1, level=0, drop_level=True)
                except: 
                    try: df = data.xs('Close', axis=1, level=1, drop_level=True)
                    except: return None
            else:
                df = data['Close']

            # Czyszczenie danych
            df = df.ffill().dropna()
            if df.empty: return None

            # --- 1. BUDOWA INDEKSU SUROWCOWEGO ---
            # Normalizujemy każdy składnik do 100 na początku okresu, 
            # żeby Ropa (70$) nie była mniej ważna niż Złoto (2000$).
            
            norm_oil = (df['CL=F'] / df['CL=F'].iloc[0]) * 100
            norm_copper = (df['HG=F'] / df['HG=F'].iloc[0]) * 100
            norm_gold = (df['GC=F'] / df['GC=F'].iloc[0]) * 100
            
            # Nasz "Hard Asset Index" (Średnia z trzech)
            df['Hard_Assets'] = (norm_oil + norm_copper + norm_gold) / 3
            
            # --- 2. S&P 500 (PAPER ASSETS) ---
            df['Paper_Assets'] = (df['^GSPC'] / df['^GSPC'].iloc[0]) * 100
            
            # --- 3. RATIO (Klucz do cyklu) ---
            # Ratio > 1 (lub rosnące) = Surowce wygrywają
            # Ratio < 1 (lub spadające) = Akcje wygrywają
            df['Supercycle_Ratio'] = df['Hard_Assets'] / df['Paper_Assets']
            
            # Średnia 200-dniowa dla Ratio (Trend)
            df['Ratio_SMA200'] = df['Supercycle_Ratio'].rolling(window=200).mean()
            
            return df

        except Exception as e:
            print(f"Błąd Supercycle: {e}")
            return None

    def plot_commodity_supercycle(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        
        # --- WYKRES RATIO (Linia Główna) ---
        ratio = df['Supercycle_Ratio']
        sma = df['Ratio_SMA200']
        
        # Kolor linii: Złoty jeśli nad średnią (Surowce), Biały jeśli pod (Akcje)
        # Rysujemy po prostu jedną linię, a tło powie prawdę
        ax1.plot(ratio.index, ratio, color='#ff9900', linewidth=2, label='Ratio: Surowce / Akcje')
        ax1.plot(sma.index, sma, color='grey', linewidth=1, linestyle='--', alpha=0.7, label='Trend (200 SMA)')
        
        # --- KOLOROWANIE REŻIMU (REGIME CHANGE) ---
        
        # ERA SUROWCÓW (Ratio nad średnią) - Inflacja, Hard Assets
        ax1.fill_between(ratio.index, ratio, sma, where=(ratio > sma), 
                         color='#ff9900', alpha=0.2, label='ERA SUROWCÓW (Inflacja / Hard Assets)')
        
        # ERA PAPIERU (Ratio pod średnią) - Deflacja, Tech Stocks
        ax1.fill_between(ratio.index, ratio, sma, where=(ratio <= sma), 
                         color='#00e5ff', alpha=0.15, label='ERA PAPIERU (Tech / Stocks)')
        
        # --- OPISY ---
        ax1.set_ylabel('Siła Surowców względem S&P 500', color=t['text'], fontweight='bold')
        ax1.tick_params(colors=t['text'])
        
        # Sprawdzamy gdzie jesteśmy dzisiaj
        last_val = ratio.iloc[-1]
        last_sma = sma.iloc[-1]
        
        if last_val > last_sma:
            status = "SUPERCYKL SUROWCOWY 🏗️ (Kupuj Złoto/Ropę)"
            color_st = '#ff9900'
        else:
            status = "DOMINACJA AKCJI 📱 (Kupuj Tech/S&P)"
            color_st = '#00e5ff'
            
        ax1.set_title(f"SUPERCYCLE DETECTOR: {status}", fontsize=16, color=color_st, fontweight='bold')
        
        # Dodajemy adnotacje "Co robić?"
        ax1.text(df.index[int(len(df)*0.1)], ratio.max()*0.9, "ROŚNIE = Kupuj Rzeczy (Hard Assets)", color='#ff9900', fontsize=10, fontweight='bold')
        ax1.text(df.index[int(len(df)*0.1)], ratio.min()*1.1, "SPADA = Kupuj Papier (Tech/Crypto)", color='#00e5ff', fontsize=10, fontweight='bold')

        # Legenda
        ax1.legend(loc='upper right', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- NAPRAWA: SILVER TIME MACHINE (Robust Data Fetching) ---
    def get_silver_fractal_prediction(self):
        """
        Skanuje historię srebra (SI=F, SLV, XAG-USD).
        Wersja odporna na błędy Yahoo Finance - próbuje 3 różnych tickerów,
        aż znajdzie działające dane.
        """
        # Lista tickerów do sprawdzenia (w kolejności priorytetu)
        tickers_to_try = ['SI=F', 'SLV', 'XAG-USD']
        
        df_final = None
        
        # 1. PĘTLA POBIERANIA (KASKADA)
        for ticker in tickers_to_try:
            try:
                # Pobieramy MAX historię
                data = yf.download(ticker, period="max", progress=False)
                
                if data.empty: continue
                
                # --- UNIWERSALNY EKSTRAKTOR KOLUMNY 'CLOSE' ---
                # Radzi sobie z każdą strukturą, jaką wypluje yfinance
                temp = None
                
                if isinstance(data.columns, pd.MultiIndex):
                    # Sprawdzamy poziom 0 (np. Price)
                    if 'Close' in data.columns.get_level_values(0):
                        temp = data['Close']
                    # Sprawdzamy poziom 1 (np. Ticker)
                    elif 'Close' in data.columns.get_level_values(1):
                        temp = data.xs('Close', axis=1, level=1)
                else:
                    if 'Close' in data.columns:
                        temp = data['Close']
                
                if temp is None: continue

                # Upewniamy się, że to Series (jedna kolumna)
                if isinstance(temp, pd.DataFrame):
                    # Jeśli nadal DataFrame, bierzemy pierwszą kolumnę
                    temp = temp.iloc[:, 0]
                
                # Czyszczenie
                temp = temp.dropna()
                
                # Wymagamy minimum 500 dni historii do analizy
                if len(temp) > 500:
                    df_final = temp
                    # print(f"Sukces: Pobrano dane srebra z {ticker}")
                    break # Mamy dane, wychodzimy z pętli
                    
            except Exception as e:
                print(f"Błąd pobierania {ticker}: {e}")
                continue
        
        # Jeśli po pętli nadal nic nie mamy -> Zwracamy błąd
        if df_final is None: return None, None, None

        try:
            # --- 2. ALGORYTM FRAKTALNY (Bez zmian) ---
            df = df_final
            lookback = 250
            forecast = 90
            
            # Zabezpieczenie przed zbyt krótką historią
            if len(df) < lookback + forecast: return None, None, None
            
            current_pattern = df.iloc[-lookback:].values
            
            def normalize(arr):
                return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) if np.max(arr) != np.min(arr) else arr

            norm_current = normalize(current_pattern)
            
            correlations = []
            
            # Skanujemy historię (krok 5 dni)
            for i in range(0, len(df) - lookback - forecast, 5):
                hist_window = df.iloc[i : i + lookback].values
                norm_hist = normalize(hist_window)
                
                try:
                    corr = np.corrcoef(norm_current, norm_hist)[0, 1]
                    if not np.isnan(corr):
                        correlations.append({
                            'index': i,
                            'date': df.index[i + lookback],
                            'corr': corr
                        })
                except: continue
            
            if not correlations: return None, None, None
            
            # Top 3 dopasowania
            top_matches = sorted(correlations, key=lambda x: x['corr'], reverse=True)[:3]
            
            projections = []
            current_last_price = df.iloc[-1]
            
            for m in top_matches:
                idx = m['index'] + lookback
                future_prices = df.iloc[idx : idx + forecast].values
                base_price_hist = df.iloc[idx]
                
                if base_price_hist == 0: continue
                
                roi_curve = future_prices / base_price_hist
                projected_path = current_last_price * roi_curve
                projections.append(projected_path)
            
            if not projections: return None, None, None
            
            avg_projection = np.mean(projections, axis=0)
            
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=x) for x in range(1, forecast + 1)]
            
            return df.iloc[-lookback:], pd.Series(avg_projection, index=future_dates), top_matches

        except Exception as e:
            print(f"Krytyczny błąd obliczeń srebra: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def plot_silver_fractal(self, current_df, projection, matches):
        if current_df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
        
        # 1. Historia (Ostatni rok)
        ax1.plot(current_df.index, current_df.values, color='white', linewidth=2, label='XAGUSD (Teraz)')
        
        # 2. Projekcja (Przyszłość)
        if projection is not None:
            # Łącznik (żeby nie było dziury na wykresie)
            proj_connected = pd.concat([current_df.iloc[-1:], projection])
            
            # Kolor zależny od wyniku
            end_price = projection.iloc[-1]
            start_price = current_df.iloc[-1]
            col_proj = '#00ff55' if end_price > start_price else '#ff0055'
            
            ax1.plot(proj_connected.index, proj_connected.values, color=col_proj, 
                     linestyle='--', linewidth=2.5, label='Ścieżka Fraktalna (Avg Top 3)')
            
            # Kropka na końcu
            ax1.scatter(projection.index[-1], end_price, color=col_proj, s=150, zorder=10, edgecolors='white')
            chg = ((end_price - start_price) / start_price) * 100
            ax1.text(projection.index[-1], end_price, f"{chg:+.1f}%", color=col_proj, fontweight='bold', ha='left', va='center')

        # Tytuł i Daty
        dates_str = ", ".join([m['date'].strftime('%Y') for m in matches])
        ax1.set_title(f"SILVER TIME MACHINE: Powtórka z lat {dates_str}", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena Srebra ($)', color='silver', fontweight='bold')
        
        # Stylizacja
        ax1.tick_params(axis='y', labelcolor='silver', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.15, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- NOWOŚĆ: SILVER MACRO (5-Year Forecast) ---
    def get_silver_macro_projection(self):
        """
        Wersja długoterminowa (5 LAT).
        Analizuje 2 lata wstecz (kontekst) i szuka bliźniaków, 
        które mają co najmniej 5 lat historii "do przodu".
        """
        tickers_to_try = ['SI=F', 'SLV', 'XAG-USD']
        df_final = None
        
        # 1. Pobieranie danych (Kaskada)
        for ticker in tickers_to_try:
            try:
                data = yf.download(ticker, period="max", progress=False)
                if data.empty: continue
                
                temp = None
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.get_level_values(0): temp = data['Close']
                    elif 'Close' in data.columns.get_level_values(1): temp = data.xs('Close', axis=1, level=1)
                else:
                    if 'Close' in data.columns: temp = data['Close']
                
                if temp is None: continue
                if isinstance(temp, pd.DataFrame): temp = temp.iloc[:, 0]
                
                temp = temp.dropna()
                # Potrzebujemy dużo historii: 2 lata (lookback) + 5 lat (forecast) = min 7 lat (~2500 dni)
                if len(temp) > 2000:
                    df_final = temp
                    break
            except: continue
        
        if df_final is None: return None, None, None

        try:
            df = df_final
            # PARAMETRY MACRO
            lookback = 365 * 2  # 2 lata kontekstu (żeby złapać strukturę rynku)
            forecast = 365 * 5  # 5 lat prognozy
            
            if len(df) < lookback + forecast: return None, None, None
            
            current_pattern = df.iloc[-lookback:].values
            
            def normalize(arr):
                return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) if np.max(arr) != np.min(arr) else arr

            norm_current = normalize(current_pattern)
            correlations = []
            
            # Skanujemy (krok 10 dni dla szybkości przy dużej historii)
            # Kończymy tak, żeby mieć 5 lat danych "po" dopasowaniu
            end_scan_idx = len(df) - lookback - forecast
            
            for i in range(0, end_scan_idx, 10):
                hist_window = df.iloc[i : i + lookback].values
                norm_hist = normalize(hist_window)
                
                try:
                    corr = np.corrcoef(norm_current, norm_hist)[0, 1]
                    if not np.isnan(corr) and corr > 0.50: # Szukamy w miarę dobrych dopasowań
                        correlations.append({
                            'index': i,
                            'date': df.index[i + lookback],
                            'corr': corr
                        })
                except: continue
            
            if not correlations: return None, None, None
            
            # Bierzemy TOP 3 najlepsze historyczne dopasowania
            top_matches = sorted(correlations, key=lambda x: x['corr'], reverse=True)[:3]
            
            projections = []
            current_last_price = df.iloc[-1]
            
            for m in top_matches:
                idx = m['index'] + lookback
                # Pobieramy przyszłość (5 lat)
                future_prices = df.iloc[idx : idx + forecast].values
                base_price_hist = df.iloc[idx]
                
                if base_price_hist == 0: continue
                
                # Skalowanie ROI
                roi_curve = future_prices / base_price_hist
                
                # Aplikujemy do dzisiejszej ceny
                projected_path = current_last_price * roi_curve
                projections.append(projected_path)
            
            if not projections: return None, None, None
            
            # Średnia ścieżka
            avg_projection = np.mean(projections, axis=0)
            
            # Daty przyszłe (5 lat)
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=x) for x in range(1, len(avg_projection) + 1)]
            
            # Zwracamy Series z datami
            proj_series = pd.Series(avg_projection, index=future_dates)
            
            return df.iloc[-lookback:], proj_series, top_matches

        except Exception as e:
            print(f"Błąd Silver Macro: {e}")
            return None, None, None

    def plot_silver_macro_projection(self, current_df, projection, matches):
        if current_df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        
        # 1. Historia (Ostatnie 2 lata)
        ax1.plot(current_df.index, current_df.values, color='white', linewidth=2, label='XAGUSD (Ostatnie 2 lata)')
        
        # 2. Projekcja (5 Lat)
        if projection is not None:
            # Łącznik
            seed = pd.Series([current_df.iloc[-1]], index=[current_df.index[-1]])
            proj_connected = pd.concat([seed, projection])
            
            # Analiza trendu
            start_p = current_df.iloc[-1]
            end_p = projection.iloc[-1]
            max_p = projection.max()
            
            # Kolor linii: Złoty/Srebrny
            ax1.plot(proj_connected.index, proj_connected.values, color='#c0c0c0', 
                     linestyle='--', linewidth=2, label='Średnia Fraktalna (5 Lat)')
            
            # SZCZYT w przyszłości
            top_date = projection.idxmax()
            top_val = projection.max()
            
            # Oznaczenie szczytu
            ax1.scatter(top_date, top_val, color='#00ff55', s=150, zorder=10, edgecolors='white')
            ax1.text(top_date, top_val * 1.1, f"TARGET CYKLU\n${top_val:,.0f}", color='#00ff55', fontweight='bold', ha='center')
            
            # Oznaczenie końca (5 lat)
            ax1.scatter(projection.index[-1], end_p, color='#00d4ff', s=100, zorder=10)
            roi_5y = ((end_p - start_p) / start_p) * 100
            ax1.text(projection.index[-1], end_p * 0.9, f"Za 5 lat:\n{roi_5y:+.0f}%", color='#00d4ff', fontweight='bold', ha='center')

        # Opisy bliźniaków
        years = [str(m['date'].year) for m in matches]
        title_years = ", ".join(years)
        
        ax1.set_title(f"SILVER MACRO 5Y: Powtórka z {title_years}?", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Cena Srebra ($)', color='silver', fontweight='bold')
        
        ax1.tick_params(axis='y', labelcolor='silver', colors=t['text'])
        ax1.tick_params(axis='x', colors=t['text'])
        
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.15, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        
        return fig

    # --- NAPRAWA: TRUE INFLATION (Composite: M2 + Gold + Stocks) ---
    def get_true_inflation_data(self):
        """
        Tworzy "Indeks Rzeczywistości" (True Cost of Wealth).
        Zestawia oficjalne CPI z koszykiem aktywów (M2, Złoto, S&P 500).
        To pokazuje inflację "stylu życia" i majątku.
        """
        try:
            # 10 lat historii
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            end_date = datetime.now()
            
            # 1. DANE Z FRED (Miesięczne)
            # CPIAUCSL = CPI (Oficjalna)
            # M2SL = M2 (Podaż)
            fred_df = web.DataReader(['CPIAUCSL', 'M2SL'], 'fred', start_date, end_date)
            fred_df = fred_df.resample('ME').last().ffill() # Koniec miesiąca
            
            # 2. DANE Z YAHOO (Dla Złota i S&P 500)
            # Musimy je sprowadzić do miesięcznych, żeby pasowały do FRED
            tickers = ['GC=F', '^GSPC']
            y_data = yf.download(tickers, start=start_date, progress=False)['Close']
            
            # Obsługa MultiIndex (standardowy fix)
            if isinstance(y_data.columns, pd.MultiIndex):
                # Próbujemy spłaszczyć
                try: y_data = y_data.xs('Close', axis=1, level=0, drop_level=True)
                except: pass
            
            # Resampling do miesięcznych (ME) - bierzemy ostatnią cenę w miesiącu
            y_monthly = y_data.resample('ME').last().ffill()
            
            # 3. ŁĄCZENIE
            # Bierzemy część wspólną dat
            common_idx = fred_df.index.intersection(y_monthly.index)
            
            df = pd.concat([fred_df.loc[common_idx], y_monthly.loc[common_idx]], axis=1)
            
            # Upewniamy się że nie ma pustych (dropna)
            df = df.dropna()
            
            # --- NORMALIZACJA (Start = 0%) ---
            # Wszystko startuje z tego samego punktu
            df_norm = pd.DataFrame(index=df.index)
            
            # Oficjalna (CPI)
            df_norm['Official_CPI'] = ((df['CPIAUCSL'] / df['CPIAUCSL'].iloc[0]) - 1) * 100
            
            # Składniki "Prawdziwej Inflacji"
            m2_chg = ((df['M2SL'] / df['M2SL'].iloc[0]) - 1) * 100
            gold_chg = ((df['GC=F'] / df['GC=F'].iloc[0]) - 1) * 100
            spx_chg = ((df['^GSPC'] / df['^GSPC'].iloc[0]) - 1) * 100
            
            # --- COMPOSITE INDEX (Średnia z 3 składników) ---
            # To jest "Koszt Posiadania Majątku"
            df_norm['True_Wealth_Cost'] = (m2_chg + gold_chg + spx_chg) / 3
            
            # Gap
            df_norm['The_Gap'] = df_norm['True_Wealth_Cost'] - df_norm['Official_CPI']
            
            return df_norm

        except Exception as e:
            print(f"Błąd True Inflation Composite: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_true_inflation(self, df):
        if df is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        
        # 1. OFICJALNA INFLACJA (Niebieska - "Matrix")
        ax1.plot(df.index, df['Official_CPI'], color='#00e5ff', linewidth=2, label='Oficjalne CPI (Koszyk Rządowy)')
        
        # 2. INFLACJA MAJĄTKOWA (Czerwona - "Rzeczywistość")
        ax1.plot(df.index, df['True_Wealth_Cost'], color='#ff0055', linewidth=3, label='Prawdziwy Koszt (M2 + Złoto + S&P)')
        
        # 3. WYPEŁNIENIE (THE GAP)
        ax1.fill_between(df.index, df['Official_CPI'], df['True_Wealth_Cost'], color='#ff0055', alpha=0.15, label='Ukryta Utrata Siły Nabywczej')
        
        # Oznaczenia na końcu
        last_cpi = df['Official_CPI'].iloc[-1]
        last_real = df['True_Wealth_Cost'].iloc[-1]
        gap = df['The_Gap'].iloc[-1]
        
        # Kropki
        ax1.scatter(df.index[-1], last_cpi, color='#00e5ff', s=120, zorder=10, edgecolors='white')
        ax1.scatter(df.index[-1], last_real, color='#ff0055', s=120, zorder=10, edgecolors='white')
        
        # Teksty wartości
        ax1.text(df.index[-1], last_cpi - 15, f"CPI: +{last_cpi:.0f}%", color='#00e5ff', fontweight='bold', ha='right')
        ax1.text(df.index[-1], last_real + 10, f"MAJĄTEK: +{last_real:.0f}%", color='#ff0055', fontweight='bold', ha='right')
        
        # Strzałka GAP
        mid_idx = int(len(df)*0.85)
        mid_date = df.index[mid_idx]
        mid_real = df['True_Wealth_Cost'].iloc[mid_idx]
        mid_cpi = df['Official_CPI'].iloc[mid_idx]
        
        ax1.annotate(f"RÓŻNICA: {gap:.0f}%\n(Tyle zbiedniałeś bez aktywów)", 
                     xy=(mid_date, (mid_real + mid_cpi)/2), 
                     ha='center', color='white', fontsize=10, fontweight='bold',
                     bbox=dict(facecolor=t['bg'], edgecolor='#ff0055', alpha=0.8))

        ax1.set_title(f"CICHY ZŁODZIEJ: CPI vs Koszt Dobrego Życia (10 Lat)", fontsize=16, color=t['text'], fontweight='bold')
        ax1.set_ylabel('Skumulowany Wzrost (%)', color=t['text'])
        
        ax1.legend(loc='upper left', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax1.set_facecolor(t['bg'])
        ax1.grid(True, alpha=0.1, color=t['grid'])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(t['text']); ax1.spines['left'].set_color(t['text'])
        ax1.tick_params(colors=t['text'])
        
        return fig

    # --- NOWOŚĆ: VPVR (Volume Profile Visible Range) ---
    def get_vpvr_data(self):
        """
        Oblicza Profil Wolumenu (Gdzie handlowano najwięcej pieniędzy?).
        Dzieli zakres cenowy na 100 stref (bins) i sumuje wolumen w każdej z nich.
        """
        try:
            # 1. Pobieramy dane (1 rok, interwał godzinny dla precyzji)
            # interwał 1h daje nam dużo "próbek" cenowych, co tworzy ładny profil
            data = yf.download('BTC-USD', period="1y", interval="1h", progress=False)
            
            # Fix MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                try: df = data.xs('Close', axis=1, level=0, drop_level=True)
                except: df = data['Close']
                # Potrzebujemy też wolumenu
                try: vol = data.xs('Volume', axis=1, level=0, drop_level=True)
                except: vol = data['Volume']
            else:
                df = data['Close']
                vol = data['Volume']

            # Upewniamy się, że to Series
            if isinstance(df, pd.DataFrame): df = df.iloc[:, 0]
            if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]

            # 2. Tworzymy "Koszyki Cenowe" (Histogram)
            price_min = df.min()
            price_max = df.max()
            
            # Dzielimy zakres cenowy na 120 poziomów
            bins = np.linspace(price_min, price_max, 120)
            
            # 3. Sumujemy wolumen w każdym koszyku
            # digitize zwraca indeks koszyka dla każdej ceny
            indices = np.digitize(df, bins)
            
            # Sumujemy wolumen dla każdego indeksu
            volume_profile = np.zeros(len(bins))
            for i in range(len(vol)):
                idx = indices[i]
                if idx < len(volume_profile):
                    volume_profile[idx] += vol.iloc[i]
            
            # 4. Znajdujemy POC (Point of Control) - Cena z największym wolumenem
            max_vol_idx = np.argmax(volume_profile)
            poc_price = bins[max_vol_idx]
            
            # 5. Znajdujemy Value Area (70% wolumenu)
            total_vol = np.sum(volume_profile)
            value_area_vol = total_vol * 0.70
            
            # Sortujemy wolumeny malejąco, żeby zebrać 70%
            sorted_indices = np.argsort(volume_profile)[::-1]
            current_vol = 0
            va_indices = []
            
            for idx in sorted_indices:
                current_vol += volume_profile[idx]
                va_indices.append(idx)
                if current_vol >= value_area_vol:
                    break
            
            val_high = bins[max(va_indices)] # Value Area High
            val_low = bins[min(va_indices)]  # Value Area Low
            
            return bins, volume_profile, poc_price, val_high, val_low, df.iloc[-1]

        except Exception as e:
            print(f"Błąd VPVR: {e}")
            return None, None, None, None, None, None

    def plot_vpvr(self, bins, volume_profile, poc, val_high, val_low, current_price):
        if bins is None: return None
        
        t = self.get_theme_colors()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Orientacja pozioma: Y = Cena, X = Wolumen
        
        # Kolory słupków
        # Szary = Poza strefą wartości
        # Niebieski/Żółty = Wewnątrz Value Area (70%)
        # Czerwony = POC (Najważniejsza linia)
        
        colors = []
        for b in bins:
            if b == poc: colors.append('#ff0055') # POC
            elif val_low <= b <= val_high: colors.append('#00e5ff') # Value Area (Turkus)
            else: colors.append('#555555') # Poza strefą (Szary)
            
        # Rysujemy słupki poziome (barh)
        # Używamy height jako odstępu między binami
        height = (bins[-1] - bins[0]) / len(bins) * 0.9
        ax.barh(bins, volume_profile, height=height, color=colors, alpha=0.6, align='center')
        
        # --- LINIE KLUCZOWE ---
        
        # 1. POC (Point of Control)
        ax.axhline(poc, color='#ff0055', linewidth=2, label=f'POC (Magnes): ${poc:,.0f}')
        
        # 2. Aktualna Cena
        ax.axhline(current_price, color='#ffd700', linestyle='--', linewidth=2, label=f'Cena Teraz: ${current_price:,.0f}')
        
        # 3. Value Area High/Low
        ax.axhline(val_high, color='#00e5ff', linestyle=':', linewidth=1, label=f'VA High: ${val_high:,.0f}')
        ax.axhline(val_low, color='#00e5ff', linestyle=':', linewidth=1, label=f'VA Low: ${val_low:,.0f}')
        
        # Wypełnienie tła Value Area
        # ax.axhspan(val_low, val_high, color='#00e5ff', alpha=0.05)

        # Logika Statusu
        if current_price > val_high:
            status = "Wybicie W GÓRĘ (Szukaj longa na re-teście)"
            s_col = "#00ff55"
        elif current_price < val_low:
            status = "Wybicie W DÓŁ (Szukaj shorta na re-teście)"
            s_col = "#ff0055"
        else:
            status = "W KONSOLIDACJI (Ping Pong między bandami)"
            s_col = "#ffd700"
            
        # Tytuły
        ax.set_title(f"VPVR (Volume Profile): Gdzie leżą pieniądze?", fontsize=16, color=t['text'], fontweight='bold')
        ax.text(volume_profile.max() * 0.5, bins[-1], status, color=s_col, fontsize=12, fontweight='bold', ha='center')
        
        ax.set_ylabel('Poziomy Cenowe ($)', color=t['text'])
        ax.set_xlabel('Wolumen Obrotu (Ilość akcji)', color=t['text'])
        
        # Legenda
        ax.legend(loc='upper right', facecolor=t['bg'], labelcolor=t['text'])
        
        fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
        ax.grid(True, alpha=0.1, color=t['grid'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(t['text']); ax.spines['left'].set_color(t['text'])
        ax.tick_params(colors=t['text'])
        
        return fig

def show_ad_splash():
    if 'ad_shown' not in st.session_state: st.session_state['ad_shown'] = False
    if not st.session_state['ad_shown']:
        st.markdown("""
        <style>
        .splash-container { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background-color: black; display: flex; justify-content: center; align-items: center; z-index: 999999; }
        .ad-box { width: 80%; height: 80%; border: 5px solid red; box-shadow: 0 0 20px red, 0 0 40px darkred; display: flex; justify-content: center; align-items: center; color: white; font-size: 40px; font-family: 'Courier New', monospace; text-transform: uppercase; font-weight: bold; text-align: center; background: #110000; }
        </style>
        <div class="splash-container"><div class="ad-box">MIEJSCE NA TWOJA REKLAME<br>(Tu moze byc Twoj baner)</div></div>
        """, unsafe_allow_html=True)
        time.sleep(3); st.session_state['ad_shown'] = True; st.rerun()

# --- MAIN (v51 - PRZYWRÓCONA STREFA MAKRO & ON-CHAIN) ---
def main():
    plt.close('all')  # <--- TUTAJ
    plt.rcParams['figure.max_open_warning'] = 100
    show_ad_splash()
    
    # Inicjalizacja
    app = MarketProbabilityIndex()
    t = app.get_theme_colors()

    # Wyświetlamy Disclaimer w sidebarze
    app.display_legal_disclaimer()

    # --- NOWOŚĆ: URUCHOMIENIE ANALITYKI ---
    app.setup_analytics()

    # WYWOŁANIE AKTUALIZACJI:
    app.check_for_updates()

    # --- 1. NAPRAWA KOLORÓW (Najpierw malujemy tło) ---
    app.apply_custom_style()

    # To sprawi, że w pasku bocznym zawsze będzie widać "Pogodę Fedu"
    app.display_fed_monitor_sidebar()

    # --- PRZENIESIONE NA GÓRĘ: INPUTY PORTFELA ---
    wallet_data = app.load_wallet()
    st.sidebar.markdown("---")
    st.sidebar.header("💼 Twój Portfel (Ilość)")
    
    # Zapisujemy wartości do zmiennych, żeby użyć ich później w obliczeniach
    user_btc = st.sidebar.number_input("BTC", 0.0, step=0.01, format="%.4f", value=wallet_data.get('btc', 0.0), key='user_btc_input', on_change=app.save_wallet_callback)
    user_eth = st.sidebar.number_input("ETH", 0.0, step=0.1, format="%.2f", value=wallet_data.get('eth', 0.0), key='user_eth_input', on_change=app.save_wallet_callback)
    user_sol = st.sidebar.number_input("SOL", 0.0, step=1.0, format="%.2f", value=wallet_data.get('sol', 0.0), key='user_sol_input', on_change=app.save_wallet_callback)
    
    # Style CSS (Zaktualizowane: Rozjaśnione napisy)
    st.markdown(f"""
        <style>
        /* Podstawowe kolory aplikacji */
        .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
        header[data-testid="stHeader"] {{ background-color: {t['bg']} !important; }}
        [data-testid="stSidebar"] {{ background-color: {t['sidebar_bg']}; border-right: 1px solid {t['grid']}; }}
        
        /* 1. ZMNIEJSZENIE KONTENERA GUZIKA */
        div.stButton {{ 
            margin-top: 0px !important; 
            margin-bottom: 5px !important; 
            height: auto !important; 
            padding: 0px !important; 
        }}

        /* 2. STYL GUZIKA (TŁO I RAMKA) */
        div.stButton > button {{ 
            background-color: {t['sidebar_bg']} !important; 
            color: #ffffff !important; /* <--- TU WYMUSZAMY BIEL */
            border: 1px solid {t['grid']} !important; 
            width: 100%; 
            height: 32px !important; 
            min-height: 32px !important; 
            padding: 0px !important; 
        }}

        /* 3. TEKST W GUZIKU (ZWYKŁYM) */
        div.stButton > button p {{ 
            font-size: 13px !important; 
            margin: 0px !important; 
            padding: 0px !important; 
            line-height: 30px !important; 
            color: #ffffff !important; /* <--- TU TEŻ BIEL */
            font-weight: 400 !important; /* Trochę grubiej dla czytelności */
        }}
        
        /* Efekt najechania myszką na guzik */
        div.stButton > button:hover {{ 
            border-color: {t['accent']} !important; 
            color: {t['accent']} !important; 
        }}
        
        div[data-testid="stExpander"] {{ border: 1px solid {t['bear']} !important; }}
        div[data-testid="stVerticalBlock"] {{ gap: 0.2rem !important; }}

        /* --- 4. NOWOŚĆ: ROZJAŚNIANIE ZAKŁADEK (TABS) W CENTRUM ANALIZ --- */
        
        /* Tekst na nieaktywnej zakładce */
        button[data-baseweb="tab"] div p {{
            color: #cccccc !important; /* Jasny szary */
            font-weight: 600 !important;
        }}
        
        /* Tekst na AKTYWNEJ zakładce */
        button[data-baseweb="tab"][aria-selected="true"] div p {{
            color: #ffffff !important; /* Czysta biel */
            text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.5); /* Lekka poświata */
        }}
        
        /* Pasek pod aktywną zakładką (ten czerwony/kolorowy) */
        button[data-baseweb="tab"][aria-selected="true"] {{
            border-bottom-color: #ff0055 !important; /* Neonowy róż/czerwień */
        }}
        
        </style>
    """, unsafe_allow_html=True)
    
    sns.set_theme(style=t['sns_style'])
    st.title("🕸️ Lambo czy Karton? (v51 Full Restore)")
    
    if st.sidebar.button("🌞/🌚 Zmień Motyw"): app.toggle_theme(); st.rerun()
    # --- NOWA WERSJA (Z PAMIĘCIĄ SESJI) ---
    
    # 1. Guzik Resetu - Czyści pamięć, jeśli chcesz świeże dane
    if st.button("🔄 Odśwież Dane"): 
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    # 2. INTELIGENTNE ŁADOWANIE (Raz na sesję)
    # To wykonuje się TYLKO RAZ przy uruchomieniu.
    if 'data_loaded' not in st.session_state:
        with st.spinner("🚀 Start silników... Konfiguruję system..."):
            
            # A. POBIERANIE DANYCH (Z Internetu)
            df, vol = app.get_market_data()
            fng = app.get_crypto_fear_greed()
            eco_n, eco_h = app.analyze_news_sentiment('economy')
            cry_n, cry_h = app.analyze_news_sentiment('crypto')
            forecast = app.get_ai_prediction(df)

            # B. OBLICZENIA MATEMATYCZNE (To też robimy raz i zapamiętujemy!)
            if df is not None:
                mvrv, ism, _ = app.calculate_cycle_metrics(df)
                hist_indices = app.get_historical_indices(df)
                eco_res = app.analyze_economy(df, eco_n)
                cry_res = app.analyze_crypto(df, fng, cry_n)
                alerts = app.check_anomalies(df)
                
                # Zapisujemy ostatnie ceny dla portfela
                last_prices = {
                    'btc': df['btc'].iloc[-1],
                    'eth': df['eth'].iloc[-1],
                    'sol': df['sol'].iloc[-1],
                    'dxy': df['dxy'].iloc[-1]
                }
            else:
                mvrv, ism, hist_indices, eco_res, cry_res, alerts, last_prices = None, None, None, 0, 0, [], {}

            # C. ZAPIS WSZYSTKIEGO DO PAMIĘCI (SESSION STATE)
            st.session_state['df'] = df
            st.session_state['vol'] = vol
            st.session_state['fng'] = fng
            st.session_state['eco_n'] = eco_n
            st.session_state['eco_h'] = eco_h
            st.session_state['cry_n'] = cry_n
            st.session_state['cry_h'] = cry_h
            st.session_state['forecast'] = forecast
            
            # Zapisujemy wyniki obliczeń
            st.session_state['mvrv'] = mvrv
            st.session_state['ism'] = ism
            st.session_state['hist_indices'] = hist_indices
            st.session_state['eco_res'] = eco_res
            st.session_state['cry_res'] = cry_res
            st.session_state['alerts'] = alerts
            st.session_state['last_prices'] = last_prices

            # Oznaczamy flagę, że gotowe
            st.session_state['data_loaded'] = True

    # 3. BŁYSKAWICZNE WYPAKOWANIE Z PAMIĘCI (Z ZABEZPIECZENIEM)
    # Dodaliśmy wartości domyślne (np. 0 lub 50), żeby 'round' się nie wywalał
    df = st.session_state.get('df')
    vol = st.session_state.get('vol')
    
    # Bezpieczne pobieranie (z domyślnymi wartościami, jeśli pamięć jest pusta)
    fng = st.session_state.get('fng', 50) 
    eco_n = st.session_state.get('eco_n', 0.5)
    eco_h = st.session_state.get('eco_h', [])
    cry_n = st.session_state.get('cry_n', 0.5)
    cry_h = st.session_state.get('cry_h', [])
    forecast = st.session_state.get('forecast')
    
    mvrv = st.session_state.get('mvrv')
    ism = st.session_state.get('ism')
    hist_indices = st.session_state.get('hist_indices')
    
    # Tu był błąd (NoneType) - dajemy 0 jako wartość domyślną
    eco_res = st.session_state.get('eco_res', 0)
    cry_res = st.session_state.get('cry_res', 0)
    
    alerts = st.session_state.get('alerts', [])
    
    # Upewniamy się, że last_prices jest słownikiem
    last_prices = st.session_state.get('last_prices')
    if last_prices is None: last_prices = {}
    
    if df is not None:
        # --- 1. OBLICZENIA WARTOŚCI PORTFELA (AUTOMATYCZNE) ---
        # Ceny bierzemy z pamięci sesji
        price_btc = last_prices.get('btc', 0)
        price_eth = last_prices.get('eth', 0)
        price_sol = last_prices.get('sol', 0)
        
        # Liczymy wartość majątku (Ilość * Cena)
        total_usd = (user_btc * price_btc) + (user_eth * price_eth) + (user_sol * price_sol)
        
        # --- 2. WYŚWIETLENIE STATUSU (LAMBO METER) ---
        # Przekazujemy obliczoną kwotę do funkcji wyświetlającej w pasku bocznym
        app.display_portfolio_manager(total_usd)

        # --- 3. WATCHDOG (ALERTY) ---
        # Zmienna 'alerts' jest już gotowa (pobrana z session_state powyżej)
        if alerts:
            st.error("### ⚠️ ZAGROŻENIA:")
            for title, desc in alerts:
                with st.expander(title, expanded=True): st.markdown(desc)
        else:
            st.success("✅ Watchdog: Stabilnie.")

        # --- 4. GÓRNY PANEL (GOSPODARKA / KRYPTO) ---
        # Zmienne 'eco_res', 'cry_res', 'hist_indices' są już gotowe (pobrane z session_state)
        
        # Logowanie do pliku (używamy last_prices zamiast df, żeby było szybciej)
        app.save_log(eco_res, cry_res, last_prices.get('btc', 0), last_prices.get('dxy', 0), fng, cry_n)

        c1, c2 = st.columns(2)
        def stat(s): return "EUPHORIA 🚀" if s>.8 else "DOBRY 🐂" if s>.55 else "NEUTRAL 😐" if s>.45 else "SŁABY 🐻" if s>.2 else "KRYTYCZNY 💀"
        
        with c1:
            st.header(f"🏭 GOSPODARKA: {round(eco_res*100)}%")
            st.caption(stat(eco_res))
            st.progress(eco_res)
            st.pyplot(app.create_mini_chart(hist_indices['Eco_Index'].tail(365), "Trend (365d)", "#00ff00", fill=True))
            with st.expander("S&P 500"): st.pyplot(app.create_mini_chart(df['sp500'].tail(365), "Cena", t['text'], False))

        with c2:
            st.header(f"₿ KRYPTO: {round(cry_res*100)}%")
            st.caption(stat(cry_res))
            st.progress(cry_res)
            st.pyplot(app.create_mini_chart(hist_indices['Cry_Index'].tail(365), "Trend (365d)", "#00d4ff", fill=True))
            with st.expander("Bitcoin"): st.pyplot(app.create_mini_chart(df['btc'].tail(365), "Cena", t['text'], False))

        st.divider()
        
        # --- 5. CENTRUM ANALIZ (TABS) ---
        st.subheader("🌐 Centrum Analiz")
        tabs = st.tabs(["🕸️ MATRYCA", "⚖️ Godziwa Cena", "📡 Radar Siły", "📰 News AI", "🔔 Kanał Gaussa", "🎯 Snajper", "📐 Geometria", "🔮 Szklana Kula", "🌈 Rainbow Chart", "📊 Mapa Rynku"])
        
        # --- ZMIANA: ZMNIEJSZANIE WYKRESÓW I CZCIONEK ---
        
        with tabs[0]: 
            _, fair_val, fair_dev = app.get_fair_value_analysis(df)
            fig = app.get_market_matrix(df, mvrv, ism, fair_dev, cry_n)
            
            if fig: 
                fig.set_size_inches(2, 2) # Mały kwadrat
                ax = fig.gca()
                # Zmniejszamy czcionki dla małego wykresu
                ax.set_title(ax.get_title(), fontsize=8) 
                ax.tick_params(labelsize=6) 
            
            st.pyplot(fig)

        with tabs[1]: 
            f, p, d = app.get_fair_value_analysis(df)
            st.metric("Fair Value", f"${p:,.0f}", f"{d:+.2f}%", delta_color="inverse")
            
            if f: 
                f.set_size_inches(4, 2)
                ax = f.gca()
                ax.set_title("Fair Value Model", fontsize=8)
                
                # --- ZMIANA: Legenda w dolnym prawym rogu ---
                ax.legend(fontsize=6, loc='lower right') 
                # --------------------------------------------
                
                ax.tick_params(labelsize=6)
                
            st.pyplot(f)

        with tabs[2]: 
            fig = app.plot_relative_strength_radar(df)
            if fig: 
                fig.set_size_inches(6, 3.5)
                ax = fig.gca()
                ax.set_title("Radar Siły", fontsize=8)
                ax.tick_params(labelsize=6) # Mniejsze napisy na osiach
                
            st.pyplot(fig)

        with tabs[3]: 
            c_n1, c_n2 = st.columns(2)
            with c_n1: 
                st.subheader("Gospodarka")
                for ti, s, l in eco_h: st.markdown(f"[{ti}]({l})")
            with c_n2: 
                st.subheader("Krypto")
                for ti, s, l in cry_h: st.markdown(f"[{ti}]({l})")

        with tabs[4]: 
            fig = app.plot_gaussian_channel(df)
            if fig: 
                fig.set_size_inches(6, 3)
                ax = fig.gca()
                ax.set_title("Kanał Gaussa", fontsize=8)
                ax.legend(fontsize=6)
                ax.tick_params(labelsize=6)
                
            st.pyplot(fig)

        with tabs[5]: st.dataframe(app.get_sniper_signals(df)) 

        with tabs[6]: 
            fig, _ = app.plot_fibonacci_chart(df)
            if fig: 
                fig.set_size_inches(6, 3)
                ax = fig.gca()
                ax.set_title("Fibonacci Levels", fontsize=8)
                ax.tick_params(labelsize=6)
                
            st.pyplot(fig)

        with tabs[7]: 
            if PROPHET_AVAILABLE: 
                fig = app.plot_ai_forecast(forecast)
                if fig: 
                    fig.set_size_inches(6, 3)
                    ax = fig.gca()
                    ax.set_title("AI Forecast", fontsize=8)
                    ax.legend(fontsize=6)
                    ax.tick_params(labelsize=6)
                    
                st.pyplot(fig)
            else: st.warning("Brak Prophet")

        with tabs[8]: 
            fig = app.plot_rainbow_chart(df)
            if fig: 
                fig.set_size_inches(6, 3)
                ax = fig.gca()
                ax.set_title("Rainbow Chart", fontsize=8)
                ax.tick_params(labelsize=6)
                
            st.pyplot(fig)

        with tabs[9]: 
            # 1. AUTOMATYCZNE POBIERANIE (Zamiast guzika)
            # Sprawdzamy, czy dane są w pamięci. Jeśli nie - pobieramy je raz.
            if 'sector_data' not in st.session_state:
                with st.spinner("Ładuję Mapę Rynku..."):
                    st.session_state['sector_data'] = app.get_lazy_sector_data()
            
            # 2. Generowanie wykresu z pamięci
            fig = app.plot_sector_momentum(st.session_state['sector_data'])
            
            # 3. Formatowanie (Regułka wielkości)
            if fig: 
                fig.set_size_inches(6, 3)
                ax = fig.gca()
                ax.set_title("Sektory (Momentum)", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(False) # Wyłącz siatkę dla czytelności
                
            st.pyplot(fig)

        # --- 5. STREFA MAKRO & ON-CHAIN (POPRAWIONA) ---
        st.divider()
        st.subheader("🌍 Strefa Makro & On-Chain")
        
        # 1. Definicja zmiennych (Dodałem 'tab_season' na końcu)
        tab_whale, tab_liq, tab_alt, tab_cycle, tab_hist, tab_copper, tab_macro, tab_season, tab_pain, tab_rsi, tab_breadth = st.tabs([
            "🐋 Wieloryby", "⛽ Płynność", "💎 Altseason", "🚲 Zegar Cyklu", 
            "📜 Historia", "🏭 Miedź vs Złoto", "📉 Heatmapa", "📉 BTC Sezonowosc", 
            "🩸 Skaner Bólu", "🌡️ RSI Skaner", "🌊 Wariograf"
        ])
        
        # 2. Treść zakładek (Twoja stara treść + nowa)
        with tab_whale: 
            c_w1, c_w2 = st.columns(2)
            c_w1.pyplot(app.plot_etf_tracker(df, vol, 'ibit', 'IBIT (Bitcoin)'))
            c_w2.pyplot(app.plot_etf_tracker(df, vol, 'etha', 'ETHA (Ethereum)'))
            
        with tab_liq: st.pyplot(app.plot_liquidity_chart(df))
        with tab_alt: st.pyplot(app.plot_altseason_chart(df))
        with tab_cycle: st.pyplot(app.plot_cycle_clock(mvrv, ism))
        with tab_hist: st.pyplot(app.plot_history_truth(df, mvrv, ism))
        
        with tab_copper: 
            c_cop1, c_cop2 = st.columns(2)
            c_cop1.pyplot(app.create_mini_chart(df['copper'].tail(365), "Miedź", "#ff9900", False))
            c_cop2.pyplot(app.create_mini_chart(df['gold'].tail(365), "Złoto", "#ffd700", False))
            
        with tab_macro: st.pyplot(app.plot_correlation_heatmap(df))

        # --- NOWA 8. ZAKŁADKA: SEZONOWOŚĆ (AUTO) ---
        with tab_season:
            st.markdown("### 📅 Kalendarz Sezonowości")
            
            # Automatyczne ładowanie
            if 'season_data' not in st.session_state:
                with st.spinner("Analizuję historię..."):
                    s_data, s_sym = app.get_seasonality_data('BTC-USD')
                    st.session_state['season_data'] = s_data
                    st.session_state['season_sym'] = s_sym
            
            # Rysowanie z pamięci
            s_data = st.session_state.get('season_data')
            s_sym = st.session_state.get('season_sym')
            
            if s_data is not None:
                st.pyplot(app.plot_seasonality_heatmap(s_data, s_sym))
                st.caption("ℹ️ *Zielone = Zysk, Czerwone = Strata.*")

        # --- SKANER BÓLU (AUTO) ---
        with tab_pain:
            st.markdown("### 🩸 Skaner Bólu (Drawdown from ATH)")
            st.write("Ranking pokazuje, jak głęboko poniżej swojego historycznego szczytu znajduje się dany projekt.")
            
            # Automatyczne ładowanie
            if 'pain_data' not in st.session_state:
                with st.spinner("Szukam ofiar bessy..."):
                    st.session_state['pain_data'] = app.get_ath_drawdown_data()
            
            # Rysowanie
            dd_data = st.session_state.get('pain_data')
            if dd_data is not None:
                st.pyplot(app.plot_ath_drawdown(dd_data))
            else:
                st.error("Błąd pobierania danych ATH.")

        # --- RSI SCANNER (AUTO) ---
        with tab_rsi:
            st.markdown("### 🌡️ RSI Heatmap (Polowanie na dołki)")
            
            # Automatyczne ładowanie
            if 'rsi_data' not in st.session_state:
                with st.spinner("Mierzę temperaturę rynku..."):
                    st.session_state['rsi_data'] = app.get_rsi_heatmap_data()
            
            # Rysowanie
            rsi_data = st.session_state.get('rsi_data')
            if rsi_data is not None:
                # Logika sukcesu (tekstowa)
                oversold = rsi_data[rsi_data['rsi'] < 30]
                if not oversold.empty:
                    st.success(f"💡 **OKAZJE (RSI < 30):** {', '.join(oversold['coin'].tolist())}")
                
                st.pyplot(app.plot_rsi_heatmap_grid(rsi_data))
            else:
                st.error("Błąd danych RSI.")

        # --- WARIOGRAF (AUTO) ---
        with tab_breadth:
            st.markdown("### 🌊 Market Breadth (Wariograf + Cena BTC)")
            
            # Automatyczne ładowanie
            if 'breadth_data' not in st.session_state:
                with st.spinner("Przesłuchuję 30 coinów..."):
                    mb_data, mb_price = app.get_market_breadth_data()
                    st.session_state['breadth_data'] = mb_data
                    st.session_state['breadth_price'] = mb_price
            
            # Rysowanie
            mb_data = st.session_state.get('breadth_data')
            mb_price = st.session_state.get('breadth_price')
            
            if mb_data is not None:
                st.pyplot(app.plot_market_breadth(mb_data, mb_price))
            else:
                st.error("Błąd obliczeń szerokości rynku.")

        # --- 6. STREFA ON-DEMAND (Z SUWAKIEM) ---
    st.divider()
    st.subheader("🧪 Strefa Danych na Zadanie (On-Demand)")
    
    # Proporcje [2, 3] dają więcej miejsca na guziki
    c1, c2 = st.columns([2, 3]) 
    
    with c1:
        # Kontener z suwakiem (height=500)
        with st.container(height=500, border=True):
            
            st.markdown("### 💎 PANEL STEROWANIA")
            
            # 3 KOLUMNY GUZIKÓW
            b1, b2, b3 = st.columns(3)

            # --- KOLUMNA 1: MAKRO & WYCENA ---
            with b1:
                st.caption("🏛️ **MAKRO & CYKLE**")
                if st.button("🌐 Makro Context"): st.session_state['active_lazy_chart'] = 'macro_context'
                if st.button("🌐 Macro Mix"): st.session_state['active_lazy_chart'] = 'all_components'
                if st.button("🏛 TGA Monitor"): st.session_state['active_lazy_chart'] = 'tga_monitor'
                if st.button("📈 Leading Index"): st.session_state['active_lazy_chart'] = 'business_cycle'
                if st.button("🌏 Global Liquidity"): st.session_state['active_lazy_chart'] = 'global_liq'
                if st.button("📊 Global Liq (Net)"): st.session_state['active_lazy_chart'] = 'global_liquidity'
                if st.button("🌊 Fed YoY Wave"): st.session_state['active_lazy_chart'] = 'liquidity_wave'
                if st.button("💸 M2 vs BTC"): st.session_state['active_lazy_chart'] = 'm2_supply'
                if st.button("💣 FRED Detonator"): st.session_state['active_lazy_chart'] = 'macro_detonator'
                if st.button("🏦 Kredyt Bankowy"): st.session_state['active_lazy_chart'] = 'credit_conditions'
                if st.button("📉 Yield Curve", key="yield_curve_btn"): st.session_state['active_lazy_chart'] = 'yield_curve'
                if st.button("🚦 NFCI Index"): st.session_state['active_lazy_chart'] = 'nfci_conditions'
                if st.button("🏦 Credit Impulse"): st.session_state['active_lazy_chart'] = 'credit_impulse'
                if st.button("💸 Bank Stimulus"): st.session_state['active_lazy_chart'] = 'bank_stimulus'
                
                st.caption("💎 **WYCENA (PRO)**")
                if st.button("🦄 Altcoin Indicator"): st.session_state['active_lazy_chart'] = 'altcoin_bull'
                if st.button("📅 Sezonowość Stat"): st.session_state['active_lazy_chart'] = 'seasonal_stats'
                if st.button("🔮 BTC Future(ETF)", key="nostradamus_gold_btn"): st.session_state['active_lazy_chart'] = 'btc_nostradamus_gold'
                #if st.button("🔮 BTC Future 4.0", key="nostradamus_log_btn"): st.session_state['active_lazy_chart'] = 'btc_nostradamus_log'
                #if st.button("🔮 BTC Future PRO", key="nostradamus_pro_btn"): st.session_state['active_lazy_chart'] = 'btc_nostradamus_pro'
                #if st.button("🔮 BTC Future Path", key="nostradamus_btn"): st.session_state['active_lazy_chart'] = 'btc_nostradamus'
                if st.button("🐋 Whale Divergence", key="whale_div_btn"): st.session_state['active_lazy_chart'] = 'whale_divergence'
                if st.button("🩸 Liquidation Heatmap", key="liq_heat_btn"): st.session_state['active_lazy_chart'] = 'liquidation_heatmap'
                if st.button("📊 MVRV Z-Score"): st.session_state['active_lazy_chart'] = 'mvrv_z_score'
                if st.button("🔮 Power Law"): st.session_state['active_lazy_chart'] = 'power_law'
                if st.button("⛏️ Puell Multiple"): st.session_state['active_lazy_chart'] = 'puell_multiple'
                if st.button("📈 2-Year Multi"): st.session_state['active_lazy_chart'] = 'two_year_multiplier'
                if st.button("🎯 SuperTrend", key="supertrend_btn"): st.session_state['active_lazy_chart'] = 'supertrend'
                if st.button("🌟 Golden Ratio"): st.session_state['active_lazy_chart'] = 'golden_ratio'
                if st.button("🎯 Pi Cycle Top"): st.session_state['active_lazy_chart'] = 'pi_cycle'
                if st.button("💎 Mayer Multiple"): st.session_state['active_lazy_chart'] = 'mayer'

            # --- KOLUMNA 2: SENTYMENT & TECH ---
            with b2:
                st.caption("🧠 **SENTYMENT**")
                if st.button("⚖️ Sentyment Bar"): st.session_state['active_lazy_chart'] = 'sentiment_bars'
                if st.button("🍩 Fear/Greed"): st.session_state['active_lazy_chart'] = 'sentiment_donut'
                if st.button("🩸 Radar Likwidacji"): st.session_state['active_lazy_chart'] = 'liquidation'
                if st.button("📊 VPVR (Profil)"): st.session_state['active_lazy_chart'] = 'vpvr'
                #if st.button("🩸 Kill Zones (Map)"): st.session_state['active_lazy_chart'] = 'liq_map'
                if st.button("🫧 Crypto Bubbles"): st.session_state['active_lazy_chart'] = 'bubbles'
                if st.button("⚖️ ETH/BTC Ratio"): st.session_state['active_lazy_chart'] = 'crypto_rotation'
                if st.button("🕵 Smart Money"): st.session_state['active_lazy_chart'] = 'smart_money'
                
                st.caption("📊 **ANALIZA TECH**")
                if st.button("🔄 RRG Rotacja"): st.session_state['active_lazy_chart'] = 'rrg_chart'
                if st.button("📐 Hurst Exp"): st.session_state['active_lazy_chart'] = 'hurst'
                if st.button("🔮 Fourier Cycle"): st.session_state['active_lazy_chart'] = 'fourier'
                if st.button("🧬 Fraktale"): st.session_state['active_lazy_chart'] = 'fractals'
                if st.button("🧬 Genom (3D)"): st.session_state['active_lazy_chart'] = 'genome_3d'
                if st.button("💣 BTC Squeeze"): st.session_state['active_lazy_chart'] = 'squeeze'
                if st.button("💣 Alt Squeeze"): st.session_state['active_lazy_chart'] = 'alt_squeeze'
                if st.button("Ξ ETH Squeeze"): st.session_state['active_lazy_chart'] = 'eth_squeeze'
                
                st.caption("📊 **ANALIZA TECH Tradycja**")
                if st.button("🔥 Prawdziwa Inflacja"): st.session_state['active_lazy_chart'] = 'true_inflation'
                if st.button("🥈 Srebro (5 Lat)"): st.session_state['active_lazy_chart'] = 'silver_macro'
                if st.button("🥈 Srebro (Fraktale)"): st.session_state['active_lazy_chart'] = 'silver_fractal'
                if st.button("🏗️ Cykl Surowcowy"): st.session_state['active_lazy_chart'] = 'commodity_supercycle'
                if st.button("🛡️ Hedging Calc"): st.session_state['active_lazy_chart'] = 'hedging_calc'
                if st.button("🏛 Insiderzy (Congress)"): st.session_state['active_lazy_chart'] = 'congress_tracker'
                if st.button("🏛 Architekt Dow Jones", key="dow_architect_btn"): st.session_state['active_lazy_chart'] = 'dow_architect'
                if st.button("💎 Dow Jones Graham", key="graham_dow_btn"): st.session_state['active_lazy_chart'] = 'graham_dow'
                if st.button("👻 Duch Grahama", key="graham_btn"): st.session_state['active_lazy_chart'] = 'graham_ghost'
                if st.button("🏛 Architekt Czasu", key="architect_btn"): st.session_state['active_lazy_chart'] = 'value_architect'
                if st.button("🎯 Sektor Snajper", key="sniper_btn"): st.session_state['active_lazy_chart'] = 'sector_sniper'
                if st.button("📊 Sektory", key="sector_btn"): st.session_state['active_lazy_chart'] = 'sector_rotation'
                if st.button("⚔️ Tech War (AIvsBTC)", key="tech_war_btn"): st.session_state['active_lazy_chart'] = 'tech_war'
                if st.button("⚔️ Chip Wars", key="chip_wars_btn"): st.session_state['active_lazy_chart'] = 'chip_wars'

            # --- KOLUMNA 3: QUANT & AI & RISK ---
            with b3:
                st.caption("🤖 **MODELE AI**")
                if st.button("🧠 Oracle AI"): st.session_state['active_lazy_chart'] = 'oracle_ai'
                if st.button("⚡ ZEUS Model"): st.session_state['active_lazy_chart'] = 'zeus_model'
                if st.button("⏳ CRONOS Time"): st.session_state['active_lazy_chart'] = 'cronos_model'
                if st.button("🔄 Cykle Overlay"): st.session_state['active_lazy_chart'] = 'cycle_overlay'
                if st.button("🤖 AI Optimizer"): st.session_state['active_lazy_chart'] = 'portfolio_opt'

                st.caption("🧮 **QUANT & RISK**")
                if st.button("💎 Altcoin Gem Hunter", key="gem_btn"): st.session_state['active_lazy_chart'] = 'alt_gems'
                if st.button("🎲 Monte Carlo"): st.session_state['active_lazy_chart'] = 'monte_carlo'
                if st.button("🎰 Kelly Crit"): st.session_state['active_lazy_chart'] = 'kelly'
                if st.button("🛡️ VaR (Risk)"): st.session_state['active_lazy_chart'] = 'var_risk'
                if st.button("⚖️ Sharpe Ratio"): st.session_state['active_lazy_chart'] = 'sharpe'
                if st.button("🤝 Arbitraż"): st.session_state['active_lazy_chart'] = 'spread_arb'
                if st.button("⚖️ Sędzia"): st.session_state['active_lazy_chart'] = 'verdict'

        with c2:
            chart_type = st.session_state.get('active_lazy_chart')
            
            if chart_type == 'sentiment_bars':
                with st.spinner("Analizuję..."): st.pyplot(app.plot_sentiment_breakdown(app.get_sentiment_structure(df)))
            elif chart_type == 'sentiment_donut':
                with st.spinner("Obliczam..."): 
                    st.pyplot(app.plot_sentiment_donut_snapshot(df))
                    st.caption("ℹ️ *Estymacja algorytmiczna na podstawie zmienności.*")
            elif chart_type == 'global_liquidity':
                with st.spinner("Pobieram..."): st.pyplot(app.plot_global_liquidity_chart(app.get_lazy_liquidity_data()))
            elif chart_type == 'credit_impulse':
                with st.spinner("Analizuję..."): st.pyplot(app.plot_credit_impulse_chart(app.get_lazy_credit_impulse()))
            elif chart_type == 'all_components':
                with st.spinner("Analizuję..."): st.pyplot(app.plot_all_components_chart(app.get_lazy_all_components_data()))
            elif chart_type == 'bank_stimulus':
                with st.spinner("Analizuję..."): st.pyplot(app.plot_bank_stimulus_chart(app.get_lazy_bank_stimulus_data()))
            elif chart_type == 'oracle_ai':
                with st.spinner("Obliczam..."): st.pyplot(app.plot_oracle_forecast(app.get_oracle_prediction(df)))
            elif chart_type == 'zeus_model':
                with st.spinner("Obliczam..."): st.pyplot(app.plot_zeus_forecast(app.get_zeus_prediction(df), df))
            elif chart_type == 'cronos_model':
                with st.spinner("Obliczam..."): st.pyplot(app.plot_cronos_forecast(app.get_cronos_prediction(df), df))
            elif chart_type == 'cycle_overlay':
                with st.spinner("Nakładam..."): st.pyplot(app.plot_cycle_comparison(app.get_cycle_comparison_data(df)))
            elif chart_type == 'liquidation':
                with st.spinner("Szukam śladów krwi na wykresie..."):
                    liq_data = app.get_liquidation_proxy(df) # Tu pobierze sobie OHLC
                    st.pyplot(app.plot_liquidation_radar(liq_data))
                    st.caption("ℹ️ *Wykres pokazuje 'Pain Points'. Wysoki słupek = Świeca z długim cieniem na dużym wolumenie.*")
            elif chart_type == 'mayer':
                with st.spinner("Analizuję Mayer Multiple..."):
                    # Najpierw obliczamy dane (get), potem rysujemy (plot)
                    mayer_data = app.get_mayer_multiple_data(df)
                    st.pyplot(app.plot_mayer_multiple_bands(mayer_data))
                    st.caption("ℹ️ *Mayer Multiple > 2.4 to historyczny szczyt bańki. < 0.8 to historyczne dno.*")
            elif chart_type == 'squeeze':
                with st.spinner("Mierzę naciąg sprężyny..."):
                    # Pobieramy dane i próg
                    sq_data, sq_thresh = app.get_volatility_squeeze_data(df)
                    # Rysujemy
                    st.pyplot(app.plot_volatility_squeeze(sq_data, sq_thresh))
                    st.caption("ℹ️ *Wykres nie pokazuje KIERUNKU, tylko MOMENT. Gdy linia wchodzi w czerwone pole, w ciągu 24-48h następuje bardzo silny ruch ceny.*")
            elif chart_type == 'bubbles':
                with st.spinner("Nadmuchuję bąbelki..."):
                    b_data = app.get_crypto_bubbles_data()
                    st.pyplot(app.plot_crypto_bubbles(b_data))
                    st.caption("ℹ️ *Wielkość bąbla = Siła ruchu. Zielone = Wzrost, Czerwone = Spadek.*")
            elif chart_type == 'alt_squeeze':
                with st.spinner("Skanuję 15 największych altcoinów..."):
                    alt_data = app.get_altcoin_squeeze_rank()
                    st.pyplot(app.plot_altcoin_squeeze_radar(alt_data))
                    st.caption("ℹ️ *Im krótszy pasek (bliżej lewej strony), tym mocniej naciągnięta sprężyna. Czerwone paski = Ryzyko natychmiastowego ruchu.*")
            elif chart_type == 'eth_squeeze':
                with st.spinner("Badam puls Ethereum..."):
                    eth_data, eth_thresh = app.get_eth_volatility_squeeze_data()
                    st.pyplot(app.plot_eth_volatility_squeeze(eth_data, eth_thresh))
                    st.caption("ℹ️ *To dedykowany wykres dla Ethereum. Fioletowa linia pokazuje specyficzne dla ETH momenty wyciszenia zmienności.*")
            elif chart_type == 'rrg_chart':
                with st.spinner("Analizuję przepływ kapitału względem BTC..."):
                    rrg_df = app.get_rrg_data()
                    st.pyplot(app.plot_rrg_chart(rrg_df))
                    st.caption("ℹ️ *Jak czytać: Pieniądz płynie zgodnie z ruchem wskazówek zegara. Niebieski (Odbicie) -> Zielony (Lider) -> Żółty (Słabnie) -> Czerwony (Śmierć).*")
            elif chart_type == 'hurst':
                with st.spinner("Liczę wymiary fraktalne rynków..."):
                    hurst_data = app.get_hurst_ranking()
                    st.pyplot(app.plot_hurst_ranking(hurst_data))
                    st.caption("ℹ️ *H > 0.5: Rynek ma pamięć (Trend jest twoim przyjacielem). H < 0.5: Rynek jest jak sprężyna (Cena wraca do środka).*")
            elif chart_type == 'monte_carlo':
                with st.spinner("Symuluję 1000 alternatywnych przyszłości..."):
                    sim_data, last_p = app.get_monte_carlo_simulation()
                    st.pyplot(app.plot_monte_carlo(sim_data, last_p))
                    st.caption("ℹ️ *Każda linia to możliwa ścieżka ceny. Żółta przerywana to najbardziej prawdopodobny scenariusz.*")
            elif chart_type == 'sharpe':
                with st.spinner("Ocenuje ryzyko vs zysk..."):
                    s_data = app.get_sharpe_ranking()
                    st.pyplot(app.plot_sharpe_ranking(s_data))
                    st.caption("ℹ️ *Sharpe > 2 (Zielone) to wybitne aktywa. Dają duży zwrot przy małym stresie.*")
            elif chart_type == 'spread_arb':
                with st.spinner("Szukam anomalii BTC vs ETH..."):
                    z_data, _ = app.get_btc_eth_spread()
                    st.pyplot(app.plot_stat_arb_spread(z_data))
                    st.caption("ℹ️ *Jeśli wykres jest wysoko (>2), BTC jest nienaturalnie drogi względem ETH. Rynek zazwyczaj wraca do zera.*")
            elif chart_type == 'portfolio_opt':
                with st.spinner("Symuluję 5000 portfeli i szukam Złotego Środka..."):
                    sim_data, best_port, _ = app.get_portfolio_optimization()
                    
                    if sim_data is not None:
                        # Wyświetl tekstowe podsumowanie
                        ret_p = best_port['return'] * 100
                        vol_p = best_port['volatility'] * 100
                        st.success(f"🏆 **ZNALEZIONO IDEALNY PORTFEL!**\n\nPrzewidywany Zwrot: **{ret_p:.1f}%** | Ryzyko: **{vol_p:.1f}%**")
                        
                        st.pyplot(app.plot_efficient_frontier(sim_data, best_port))
                        st.caption("ℹ️ *Gwiazdka na wykresie to matematyczny ideał. Wykres kołowy pokazuje, ile procent kapitału włożyć w konkretne coiny, aby ten ideał osiągnąć.*")
                    else:
                        st.error("Błąd optymalizacji.")
            elif chart_type == 'var_risk':
                with st.spinner("Symuluję historyczne krachy (VaR)..."):
                    ret, v95, v99, es = app.get_var_data()
                    st.pyplot(app.plot_var_distribution(ret, v95, v99, es))
                    st.caption(f"ℹ️ **Interpretacja:** Z 95% pewnością Twoja dzienna strata nie przekroczy **{v95:.1%}**. Ale jeśli rynek runie (strefa czerwona), średnio stracisz **{es:.1%}**.")
            elif chart_type == 'verdict':
                with st.spinner("Przesłuchuję wskaźniki techniczne..."):
                    sent, sigs, price = app.get_technical_verdict()
                    
                    if sent is not None:
                        st.pyplot(app.plot_verdict_gauge(sent))
                        
                        # Lista dowodowa (Dlaczego taka decyzja?)
                        with st.expander("📜 Zobacz dowody (Lista sygnałów)"):
                            for s in sigs:
                                st.write(s)
                    else:
                        st.error("Błąd sędziego.")
            elif chart_type == 'kelly':
                with st.spinner("Licze optymalną stawkę..."):
                    fk, sk, win_p, odds = app.get_kelly_criterion()
                    st.pyplot(app.plot_kelly_gauge(fk, sk))
                    st.caption(f"ℹ️ **Statystyka:** Wygrywasz **{win_p*100:.0f}%** dni. Średnio wygrywasz **{odds:.2f}x** więcej niż tracisz. Matematycznie optymalny zakład to **{sk*100:.1f}%** portfela.")
            elif chart_type == 'smart_money':
                with st.spinner("Śledzę przepływy wielorybów..."):
                    p, ad = app.get_smart_money_data()
                    st.pyplot(app.plot_smart_money(p, ad))
                    st.caption("ℹ️ **Czerwona strefa:** Cena rośnie, a wolumen spada (Pułapka). **Zielona strefa:** Cena spada, a wolumen rośnie (Akumulacja przed wystrzałem).")
            elif chart_type == 'fourier':
                with st.spinner("Dekoduję Matrixa (Analiza Widmowa)..."):
                    # Pobieramy close, projekcję i liczbę dni
                    c, proj, days = app.get_fourier_projection()
                    
                    if c is not None:
                        st.pyplot(app.plot_fourier_cycles(c, proj, days))
                        st.caption("ℹ️ **Co to jest?** Algorytm usunął szum i zostawił tylko dominujące fale sinusoidalne. Zielona przerywana linia to matematyczna projekcja tych fal w przyszłość. To nie magia, to fizyka.")
                    else:
                        st.error("Błąd transformacji Fouriera.")
            elif chart_type == 'genome_3d':
                with st.spinner("Rekonstruuję czasoprzestrzeń rynkową..."):
                    d3, curr = app.get_phase_space_data()
                    
                    if d3 is not None:
                        st.pyplot(app.plot_phase_space_3d(d3, curr))
                        st.caption("ℹ️ **JAK TO CZYTAĆ?** To jest struktura rynku w 3D. \n1. **Ciasna kulka/supła:** Rynek śpi (akumulacja). \n2. **Rozciągnięta linia:** Silny trend. \n3. **Biała Gwiazda:** To my TERAZ. Zobacz, czy Gwiazda ucieka z 'supła' w pustą przestrzeń (Wybicie), czy wraca do środka.")
                    else:
                        st.error("Błąd generowania genomu.")
            elif chart_type == 'fractals':
                with st.spinner("Skanuję 10 lat historii w poszukiwaniu bliźniaków..."):
                    curr, matches, avg_fut, days = app.get_fractal_matches()
                    
                    if curr is not None and len(matches) > 0:
                        st.pyplot(app.plot_fractal_matches(curr, matches, avg_fut, days))
                        
                        # Budowanie opisu znalezionych dat
                        dates_str = ", ".join([m['date'].strftime('%Y-%m') for m in matches])
                        st.caption(f"ℹ️ **ZNALEZIONO BLIŹNIAKÓW:** Algorytm wykrył, że dzisiejszy rynek zachowuje się prawie identycznie jak w: **{dates_str}**. Kolorowa linia to średnia z tego, co stało się POTEM w tych latach.")
                    else:
                        st.warning("Nie znaleziono wystarczająco podobnych sytuacji w historii (Rynek jest unikalny).")
            elif chart_type == 'liquidity_wave':
                with st.spinner("Łączę się z bazą Rezerwy Federalnej (FRED)..."):
                    # Używamy nowej metody get_true_fed_liquidity
                    b_yoy, l_yoy = app.get_true_fed_liquidity()
                    
                    if b_yoy is not None:
                        st.pyplot(app.plot_true_fed_liquidity(b_yoy, l_yoy))
                        st.caption("ℹ️ **TO JEST PRAWDZIWY OBRAZ.** Wykres pokazuje zmianę płynności netto FED (Bilans - TGA - RRP). \n⚪ **Biała Linia (Fed):** Prawdziwy dopływ gotówki. Jeśli spada (jest pod zerem), Fed wysysa pieniądze z rynku. \n🟣 **Różowa Linia (BTC):** Reakcja ceny Bitcoina. Zauważ, że BTC teraz spada, bo Płynność Fed też jest płaska/spadkowa.")
                    else:
                        st.error("Błąd pobierania danych z FRED. Czy masz zainstalowane `pandas_datareader`?")
            elif chart_type == 'macro_detonator':
                with st.spinner("Pobieram dane o recesji i stopach procentowych..."):
                    macro_df = app.get_fred_macro_pack()
                    
                    if macro_df is not None:
                        st.pyplot(app.plot_macro_detonators(macro_df))
                        st.caption("ℹ️ **ANALIZA ZAGŁADY:**\n1. **Krzywa (Góra):** Jeśli czerwone pole znika i wracamy nad linię 0 -> **UCIEKAJ, RECESJA JEST TU.**\n2. **Realne Stopy (Środek):** Im wyżej, tym gorzej dla Bitcoina. Chcemy widzieć to nisko.\n3. **Stres (Dół):** Jeśli ta żółta linia wystrzeli w górę -> Rynek długu pękł (Lehman Brothers moment).")
                    else:
                        st.error("Błąd pobierania danych makro.")
            elif chart_type == 'm2_supply':
                with st.spinner("Licze dolary w obiegu..."):
                    btc, m2, m2_yoy = app.get_m2_supply_data()
                    
                    if btc is not None:
                        st.pyplot(app.plot_m2_vs_btc(btc, m2, m2_yoy))
                        st.caption("ℹ️ **O CO CHODZI?** \n🔵 **Niebieska Linia:** Ilość dolarów w gospodarce (M2). To jest 'poziom wody'. \n🟠 **Pomarańczowa Linia:** Cena Bitcoina. \nJeśli Bitcoin jest **nad** niebieską linią (zielone pole), to znaczy, że realnie się bogacisz. Jeśli jest pod, to tylko gonisz inflację.")
                    else:
                        st.error("Błąd danych M2.")
            elif chart_type == 'global_liq':
                with st.spinner("Sumuję bilanse FED, ECB i BOJ..."):
                    btc, gl_index = app.get_global_liquidity_index()
                    
                    if btc is not None:
                        st.pyplot(app.plot_global_liquidity_index(btc, gl_index))
                        st.caption("ℹ️ **THE EVERYTHING CHART.** \nTo jest suma bilansów trzech największych banków centralnych świata, przeliczona na dolary. \n🔵 **Niebieska fala:** Ilość pieniądza na świecie. \n🟠 **Bitcoin:** Zobacz, jak idealnie płynie na tej fali. Kiedy niebieska linia rośnie, BTC nie ma wyboru - musi rosnąć.")
                    else:
                        st.error("Błąd pobierania danych globalnych.")
            elif chart_type == 'credit_conditions':
                with st.spinner("Dzwonię do dyrektorów banków (SLOOS data)..."):
                    df_credit = app.get_credit_conditions()
                    
                    if df_credit is not None:
                        st.pyplot(app.plot_credit_conditions(df_credit))
                        st.caption("ℹ️ **JAK CZYTAĆ?** \n🔵 **Niebieskie Tło (Repo):** Banki mają nadmiar kasy i parkują ją bezpiecznie. \n🔴 **Czerwona Linia (Kredyty):** Jeśli idzie w GÓRĘ, to znaczy, że banki **nie chcą dawać kredytów** (zaostrzają wymogi). \n📉 **Wniosek:** Sprawdź swoją teorię. Czy jak Niebieskie (Repo) rośnie, to Czerwone (Trudność) też rośnie?")
                    else:
                        st.error("Błąd pobierania danych kredytowych.")
            elif chart_type == 'nfci_conditions':
                with st.spinner("Sprawdzam poziom stresu w systemie (Chicago Fed)..."):
                    btc, nfci = app.get_financial_conditions_index()
                    
                    if btc is not None:
                        st.pyplot(app.plot_financial_conditions(btc, nfci))
                        st.caption("ℹ️ **NAJWAŻNIEJSZY PRZEŁĄCZNIK RYNKU.** \nTen wykres to 'Sygnalizacja Świetlna' dla dużego kapitału. \n🟢 **Zielona Strefa (Góra):** Warunki są luźne. Banki chętnie ryzykują. Pieniądz jest tani. **Idealne środowisko dla Bitcoina.** \n🔴 **Czerwona Strefa (Dół):** Warunki są ciasne. Stres w systemie. Banki uciekają do gotówki. **Bitcoin zazwyczaj wtedy spada.**")
                    else:
                        st.error("Błąd pobierania indeksu NFCI.")
            elif chart_type == 'business_cycle':
                with st.spinner("Mierzę puls gospodarki USA..."):
                    b_yoy, lei_yoy = app.get_business_cycle_data()
                    
                    if b_yoy is not None:
                        st.pyplot(app.plot_business_cycle(b_yoy, lei_yoy))
                        st.caption("ℹ️ **THE BUSINESS CYCLE (Raoul Pal).** \nTo jest 'bicie serca' gospodarki. \n⚪ **Biała Linia (Gospodarka):** Jeśli jest pod zerem (czerwone tło) i zaczyna zawracać w górę -> **TO JEST MOMENT ZAKUPU ŻYCIA.** Wtedy Fed panikuje i drukuje. \n🟠 **Pomarańczowa (BTC):** Zobacz, że Bitcoin zawsze startuje, gdy Biała Linia odbija od dna.")
                    else:
                        st.error("Błąd pobierania danych cyklu (FRED).")
            elif chart_type == 'macro_context':
                with st.spinner("Pobieram dane z NASDAQ i rynku walutowego..."):
                    df_macro = app.get_macro_context_data()
                    
                    if df_macro is not None:
                        st.pyplot(app.plot_macro_context(df_macro))
                        st.caption("ℹ️ **JAK CZYTAĆ TEN WYKRES?** \n🟠 **BTC (Pomarańczowy):** Nasz król. \n🔵 **NASDAQ (Niebieski):** Jeśli BTC i NASDAQ idą razem w górę -> Hossa jest zdrowa (Technologiczna). \n🔴 **DXY (Czerwony przerywany, ODWRÓCONY):** To jest 'Wrecking Ball'. Jeśli ta linia idzie w DÓŁ (czyli Dolar rośnie w siłę), to miażdży Bitcoina. Chcemy, żeby Czerwona Linia szła w górę (Dolar słabł).")
                    else:
                        st.error("Błąd pobierania danych Makro.")
            elif chart_type == 'tga_monitor':
                with st.spinner("Zaglądam do skarbca Janet Yellen..."):
                    tga_data = app.get_tga_monitor_data()
                    
                    if tga_data is not None:
                        st.pyplot(app.plot_tga_monitor(tga_data))
                        st.caption("ℹ️ **SCENARIUSZ WYPŁAT:** \n📉 **Linia leci w dół:** Rząd wypłaca pieniądze (Luty/Marzec). To jest płynność dla giełdy. **Risk On.** \n📈 **Linia leci w górę:** Podatki wpływają (Kwiecień). Rząd zabiera pieniądze z rynku. **Risk Off (Rug Pull).** \n🎯 **Cel:** Obserwuj, czy linia zbliża się do czerwonej strefy (900 mld).")
                    else:
                        st.error("Błąd pobierania danych TGA.")
            elif chart_type == 'crypto_rotation':
                with st.spinner("Sprawdzam, czy królowa (ETH) obudziła się..."):
                    df_rot = app.get_crypto_rotation_data()
                    
                    if df_rot is not None:
                        st.pyplot(app.plot_crypto_rotation(df_rot))
                        st.caption("ℹ️ **BITCOIN CZY ALTY?** \nTen wykres mówi, co trzymać w portfelu. \n🟢 **Zielona Strefa:** ETH jest silniejsze od BTC. To czas na Altcoiny. \n🔴 **Czerwona Strefa:** BTC jest silniejszy. Alty krwawią. Trzymaj tylko BTC.")
                    else:
                        st.error("Błąd pobierania danych ETH/BTC.")
            elif chart_type == 'power_law':
                with st.spinner("Obliczam fizykę Bitcoina (Model Potęgowy)..."):
                    df_pl = app.get_bitcoin_power_law()
                    if df_pl is not None:
                        st.pyplot(app.plot_power_law(df_pl))
                        st.caption("ℹ️ **BITCOIN POWER LAW.** \nTo jest 'mapa drogowa' Bitcoina. \n🟢 **Zielona Linia:** Dno absolutne. Poniżej tej linii BTC praktycznie nie schodzi. \n🔴 **Czerwona Linia:** Bańka spekulacyjna. Jeśli dotkniemy tej linii, sprzedawaj wszystko. \n🔵 **Środek:** Fair Value. Cena zawsze wraca do środka.")
                    else:
                        st.error("Błąd obliczeń Power Law.")

            elif chart_type == 'pi_cycle':
                with st.spinner("Szukam szczytu hossy..."):
                    df_pi = app.get_pi_cycle_data()
                    if df_pi is not None:
                        st.pyplot(app.plot_pi_cycle(df_pi))
                        st.caption("ℹ️ **PI CYCLE TOP INDICATOR.** \nLegendarne narzędzie, które wyznaczyło szczyt w 2017 i 2021 z dokładnością do dni. \n⚠️ **ZASADA:** Jeśli **Pomarańczowa Linia (111 DMA)** przetnie **Zieloną Linię (350 DMA x2)** w górę -> **HOSSA SIĘ KOŃCZY.** Uciekaj z rynku.")
                    else:
                        st.error("Błąd danych Pi Cycle.")
            elif chart_type == 'puell_multiple':
                with st.spinner("Analizuję rentowność kopalni Bitcoin..."):
                    df_puell = app.get_puell_multiple_data()
                    if df_puell is not None:
                        st.pyplot(app.plot_puell_multiple(df_puell))
                        st.caption("ℹ️ **PUELL MULTIPLE (Wskaźnik Górników).** \nTen wskaźnik pokazuje, czy Bitcoin jest przewartościowany w stosunku do kosztów jego wydobycia. \n🟢 **Zielona Strefa (<0.5):** Górnicy krwawią. Historycznie to ZAWSZE był dołek cyklu. \n🔴 **Czerwona Strefa (>4.0):** Górnicy zarabiają fortunę. To zawsze był szczyt bańki.")
                    else:
                        st.error("Błąd obliczeń Puell Multiple.")
            elif chart_type == 'two_year_multiplier':
                with st.spinner("Szukam okazji życia..."):
                    df_2y = app.get_two_year_multiplier_data()
                    if df_2y is not None:
                        st.pyplot(app.plot_two_year_multiplier(df_2y))
                        st.caption("ℹ️ **THE INVESTOR TOOL.** \nNarzędzie dla 'leniwych milionerów'. \n🟢 **Poniżej Zielonej:** Kupujesz z zamkniętymi oczami (Dno Bessy). \n🔴 **Powyżej Czerwonej:** Sprzedajesz wszystko (Szczyt Hossy). \n⚪ **Pomiędzy:** Nic nie robisz (HODL).")
                    else:
                        st.error("Błąd obliczeń 2-Year Multiplier.")
            elif chart_type == 'golden_ratio':
                with st.spinner("Konsultuję się z ciągiem Fibonacciego..."):
                    df_gr = app.get_golden_ratio_data()
                    if df_gr is not None:
                        st.pyplot(app.plot_golden_ratio(df_gr))
                        st.caption("ℹ️ **THE GOLDEN RATIO MULTIPLIER.** \nBitcoin porusza się po szczeblach matematyki. \n🟡 **Linia Żółta (x1.6):** Pierwszy poważny opór. Często przystanek w hossie. \n🔴 **Linia Czerwona (x2.0):** Strefa realizowania zysków. \n🟣 **Linia Fioletowa (x3.0):** Historycznie absolutny szczyt cyklu. Jeśli tu jesteśmy -> Sprzedawaj wszystko.")
                    else:
                        st.error("Błąd obliczeń Golden Ratio.")
            elif chart_type == 'mvrv_z_score':
                with st.spinner("Badam odchylenie od normy rynkowej..."):
                    df_mvrv = app.get_mvrv_z_score_data()
                    if df_mvrv is not None:
                        st.pyplot(app.plot_mvrv_z_score(df_mvrv))
                        st.caption("ℹ️ **MVRV Z-SCORE.** \nNajpotężniejszy wskaźnik wykupienia rynku. \n🟢 **Strefa Zielona (Dół):** Cena jest poniżej wartości godziwej. To historycznie najlepszy moment na zakup. \n🔴 **Strefa Czerwona (Góra):** Cena odkleiła się od rzeczywistości (Bańka). Szczyt 2017 i 2021 był dokładnie w tej strefie.")
                    else:
                        st.error("Błąd obliczeń MVRV.")
            elif chart_type == 'yield_curve':
                with st.spinner("Sprawdzam przepowiednię recesji..."):
                    df_yc = app.get_yield_curve_data()
                    if df_yc is not None:
                        st.pyplot(app.plot_yield_curve(df_yc))
                        st.caption("ℹ️ **YIELD CURVE (10Y-2Y).** \nNajsłynniejszy wskaźnik makro na świecie. \n🔴 **Czerwone Tło (Poniżej 0):** Inwersja. Rynek obligacji krzyczy, że nadchodzi recesja. To się działo przed rokiem 2000, 2008, 2020 i... teraz. \n⚠️ **Uwaga:** Krach na giełdzie zazwyczaj następuje nie w trakcie inwersji, ale w momencie, gdy wykres gwałtownie wraca nad zero (Pivot Fedu).")
                    else:
                        st.error("Błąd pobierania Yield Curve.")
            elif chart_type == 'whale_divergence':
                with st.spinner("Śledzę przepływy dużego kapitału..."):
                    df_whale = app.get_whale_divergence_data()
                    if df_whale is not None:
                        st.pyplot(app.plot_whale_divergence(df_whale))
                        st.caption("ℹ️ **WHALE DIVERGENCE (RSI vs MFI).** \nPorównujemy emocje ulicy (RSI) z faktycznym przepływem pieniądza (MFI). \n🟢 **Zielone pola (MFI > RSI):** Wieloryby pompują pieniądze. Jeśli cena rośnie, a pole jest zielone -> Trend jest bezpieczny. \n🔴 **Czerwone pola (RSI > MFI):** Cena jest pompowana 'powietrzem' (bez wolumenu). Wieloryby po cichu wychodzą. To najsilniejszy sygnał ostrzegawczy przed spadkiem.")
                    else:
                        st.error("Błąd obliczeń Whale Divergence.")
            elif chart_type == 'liquidation_heatmap':
                with st.spinner("Skanuję rynek w poszukiwaniu ofiar..."):
                    # 1. Pobierz dane (zobaczymy błędy jeśli są)
                    df_liq = app.get_liquidation_heatmap_data()
                    
                    # 2. Rysuj tylko jeśli dane są OK
                    if df_liq is not None and not df_liq.empty:
                        fig = app.plot_liquidation_heatmap(df_liq)
                        if fig:
                            st.pyplot(fig)
                            st.caption("ℹ️ **LIQUIDATION LASERS.** Widzisz poziome wiązki ciągnące się od szczytów.")
                        else:
                            st.error("Błąd: Funkcja rysowania zwróciła pusty wykres.")
                    else:
                        st.warning("Nie udało się pobrać danych (zobacz komunikat błędu powyżej, jeśli wystąpił).")
            elif chart_type == 'tech_war':
                with st.spinner("Porównuję wydajność gigantów technologicznych..."):
                    df_tech = app.get_tech_war_data()
                    
                    if df_tech is not None:
                        st.pyplot(app.plot_tech_war(df_tech))
                        st.caption("""
                        ℹ️ **TECH WAR (Relative Performance).**
                        Wykres pokazuje, ile zarobiłbyś (lub stracił) w procentach, inwestując w te aktywa 2 lata temu.
                        \n🟢 **Nvidia (Zielony):** Lider rewolucji AI.
                        \n🔴 **AMD (Czerwony):** Główny konkurent.
                        \n🔵 **Intel (Niebieski):** Stary gigant walczący o przetrwanie.
                        \n🟠 **Bitcoin (Pomarańczowy):** Cyfrowe złoto jako punkt odniesienia.
                        \nJeśli linia jest pod zerem (przerywana linia), oznacza to stratę kapitału w tym okresie.
                        """)
                    else:
                        st.error("Błąd pobierania danych porównawczych.")
            elif chart_type == 'chip_wars':
                with st.spinner("Analizuję rynek półprzewodników..."):
                    df_chips = app.get_chip_wars_data()
                    
                    if df_chips is not None:
                        st.pyplot(app.plot_chip_wars(df_chips))
                        st.caption("""
                        ℹ️ **CHIP WARS (Wojna Procesorów).**
                        Porównanie stóp zwrotu z ostatnich 2 lat (start = 0%).
                        \n🟢 **Nvidia (Zielona):** Jeśli ta linia jest wysoko w chmurach, to znaczy, że AI napędza rynek.
                        \n🔴 **AMD (Czerwona):** Główny rywal. Zobacz, czy trzyma tempo, czy zostaje w tyle.
                        \n🔵 **Intel (Niebieska):** Dawny król. Jeśli linia jest pod zerem, oznacza to niszczenie wartości dla akcjonariuszy.
                        """)
                    else:
                        st.error("Błąd pobierania danych giełdowych.")
            elif chart_type == 'sector_rotation':
                with st.spinner("Skanuję sektory gospodarki USA..."):
                    df_sec = app.get_sector_performance_data()
                    
                    if df_sec is not None:
                        st.pyplot(app.plot_sector_rotation(df_sec))
                        st.caption("""
                        ℹ️ **ROTACJA SEKTORÓW (Gdzie płynie kapitał?).**
                        \n🐂 **Hossa (Risk On):** Gdy **Technologia (Tech)** i **Finanse** są na górze (zielone).
                        \n🐻 **Bessa (Risk Off):** Gdy **Defensywne (Staples)**, **Zdrowie** i **Użyteczność** są na górze.
                        \n📉 **Energia:** Często działa odwrotnie do reszty rynku (zależy od ropy).
                        """)
                    else:
                        st.error("Błąd pobierania danych sektorowych.")
            elif chart_type == 'sector_sniper':
                with st.spinner("Namierzam najlepsze spółki z najsilniejszego sektora..."):
                    # 1. Pobieramy ranking spółek z wygranego sektora
                    df_picks, winner = app.get_sector_sniper_data()
                    
                    if df_picks is not None:
                        st.pyplot(app.plot_sector_sniper(df_picks, winner))
                        
                        # Wyciągamy lidera
                        top_pick = df_picks.iloc[0]
                        st.success(f"🏆 **ZWYCIĘZCA:** {top_pick['Ticker']} (Score: {top_pick['Score']}/100)")
                        
                        st.caption("""
                        ℹ️ **JAK DZIAŁA SNAJPER?**
                        1. Program najpierw znalazł najsilniejszy dzisiaj sektor (np. Tech).
                        2. Pobrał 10 największych spółek z tego sektora.
                        3. Ocenił je punktowo:
                           * **Trend:** Cena nad średnią 200? (+25 pkt)
                           * **Momentum:** RSI 50-70? (+30 pkt)
                           * **Wolumen:** Większy niż zwykle? (+30 pkt)
                        \n🟢 **Zielone:** Silny trend, gotowe do wzrostu.
                        \n🟡 **Żółte:** Stabilne, ale bez fajerwerków.
                        \n⚫ **Szare:** Słabe, unikać.
                        """)
                    else:
                        st.error("Nie udało się pobrać danych dla spółek (błąd API lub brak lidera).")
            elif chart_type == 'value_architect':
                with st.spinner("Szukam pereł w błocie (Deep Value Scan)..."):
                    # 1. Pobieramy okazje
                    df_value = app.get_value_architect_data()
                    
                    if df_value is not None and not df_value.empty:
                        st.pyplot(app.plot_value_architect(df_value))
                        
                        # Wypisujemy najlepszą okazję
                        top = df_value.iloc[0]
                        st.success(f"💎 **NAJWIĘKSZA PROMOCJA:** {top['Ticker']} ({top['Sector']}) jest przeceniony o {top['Discount']:.1f}%!")
                        
                        st.caption("""
                        ℹ️ **ARCHITEKT WARTOŚCI (Strategia Warrena Buffetta).**
                        Ten algorytm ignoruje to, co jest "modne". Szuka tego, co jest **TANIE**.
                        \n📉 **Wykres:** Pokazuje solidne spółki, które spadły najmocniej od szczytu (Drawdown).
                        \n🎯 **Jak grać?** To są pozycje na miesiące/lata. Kupujesz strach, sprzedajesz euforię.
                        \n⚠️ **Uwaga:** Upewnij się, że spółka spadła przez sentyment, a nie przez bankructwo (sprawdź newsy).
                        """)
                    else:
                        st.warning("Brak wyraźnych okazji. Rynek jest drogi (Hossa).")
            elif chart_type == 'alt_gems':
                with st.spinner("Przeszukuję śmietnik historii w poszukiwaniu diamentów..."):
                    df_gems = app.get_altcoin_gem_data()
                    
                    if df_gems is not None and not df_gems.empty:
                        st.pyplot(app.plot_altcoin_gems(df_gems))
                        
                        top = df_gems.iloc[0]
                        st.success(f"💎 **NAJWIĘKSZA PERŁA:** {top['Coin']} (ATH: {top['Drawdown']:.1f}%, RSI: {top['RSI']:.0f})")
                        
                        st.caption("""
                        ℹ️ **GEM HUNTER (Strategia Vulture).**
                        Szukamy solidnych projektów, które zostały zmiażdżone cenowo, ale wykazują oznaki życia.
                        \n🟢 **Wysoki Score:** Oznacza projekt przeceniony o -80% do -90% (strefa akumulacji), z niskim RSI i rosnącym wolumenem.
                        \n⚠️ **Uwaga:** To łapanie spadających noży. Sprawdź, czy projekt nie jest martwy (GitHub, Twitter).
                        """)
                    else:
                        st.warning("Brak wyraźnych okazji. Albo rynek jest drogi, albo dane się nie pobrały.")
            elif chart_type == 'btc_nostradamus':
                with st.spinner("Konsultuję się z duchami poprzednich hoss..."):
                    df_real, df_proj = app.get_btc_projection_data()
                    
                    if df_real is not None:
                        st.pyplot(app.plot_btc_projection(df_real, df_proj))
                        
                        st.caption("""
                        ℹ️ **JAK TO DZIAŁA?**
                        To jest model fraktalny. Wziąłem zachowanie ceny z cyklu 2016 i 2020, uśredniłem je (z wagą większą dla 2020) i nałożyłem na obecny wykres od dnia Halvingu (Kwiecień 2024).
                        \n⚪ **Biała linia:** Gdzie jesteśmy teraz.
                        \n🔵 **Niebieska przerywana:** Gdzie powinniśmy być wg średniej historycznej.
                        \n🟢 **Zielony punkt:** Teoretyczny szczyt cyklu wg tego modelu.
                        \n⚠️ **To tylko matematyka, a nie jasnowidzenie.** Służy do orientacji, w którym miejscu cyklu jesteśmy.
                        """)
                    else:
                        st.error("Błąd obliczeń Nostradamusa.")
            elif chart_type == 'btc_nostradamus_pro':
                with st.spinner("Obliczam wagi: 25% (2015), 35% (2018), 40% (2022) + Tłumienie zmienności..."):
                    # Używamy nowej metody _weighted_
                    df_real, df_proj = app.get_btc_weighted_projection()
                    
                    if df_real is not None:
                        st.pyplot(app.plot_btc_weighted_projection(df_real, df_proj))
                        
                        st.caption("""
                        ℹ️ **NOSTRADAMUS PRO (Model Ważony).**
                        Ten model realizuje Twoją strategię "Dojrzewania Rynku".
                        \n📊 **Wagi Cykli:** Przyszłość jest budowana w 40% w oparciu o obecny cykl, w 35% o cykl 2018, a tylko w 25% o stary cykl 2015.
                        \n📉 **Zduszenie Szczytu:** Szczyt w 2030 roku został matematycznie obniżony o ~18% (trudniej o wzrosty).
                        \n📈 **Podniesienie Dołka:** Nadchodząca bessa (2026/2027) została spłycona o 20% (instytucje bronią ceny).
                        """)
                    else:
                        st.error("Błąd obliczeń Nostradamusa PRO.")
            elif chart_type == 'btc_nostradamus_log':
                with st.spinner("Obliczam wpływ ETF-ów i kapitalizacji rynkowej na przyszłe ceny..."):
                    df_real, df_proj = app.get_btc_probability_projection()
                    
                    if df_real is not None:
                        st.pyplot(app.plot_btc_probability_projection(df_real, df_proj))
                        
                        st.caption("""
                        ℹ️ **NOSTRADAMUS 4.0 (Matematyka Dojrzałego Rynku).**
                        Ten model naprawia błąd "szalonych cen" ($1M+).
                        \n🧠 **Jak to działa?** Zamiast uśredniać "procenty" (co daje kosmiczne wyniki), uśredniamy "energię logarytmiczną".
                        Oznacza to, że model "wie", iż poruszenie ceny o 10% przy cenie $100k kosztuje tyle samo energii (pieniędzy), co poruszenie o 1000% przy cenie $100.
                        \n📉 **Efekt:** Otrzymujesz krzywą, która uwzględnia, że Bitcoin jest teraz ciężkim aktywem (klasa tryliona dolarów) i zachowuje się stabilniej.
                        """)
                    else:
                        st.error("Błąd obliczeń Nostradamusa 4.0.")
            elif chart_type == 'btc_nostradamus_gold':
                with st.spinner("Pobieram dane o Złocie (GLD) od 2004 roku i miksuję z cyklami BTC..."):
                    df_real, df_proj = app.get_btc_gold_projection()
                    
                    if df_real is not None:
                        st.pyplot(app.plot_btc_gold_projection(df_real, df_proj))
                        
                        st.caption("""
                        ℹ️ **NOSTRADAMUS 5.0 (Złota Era).**
                        Najbardziej kompletny model.
                        \n🥇 **Czynnik Złota:** Uwzględnia, co stało się z ceną złota po wejściu ETF w 2004 r. Złoto rosło przez 8 lat.
                        \n📊 **Miks:** 80% to standardowe cykle Bitcoina (Halving), a 20% to "ścieżka złota" (skalowana zmiennością BTC).
                        \n🛡️ **Wynik:** Zazwyczaj pokazuje wyższe dno podczas bessy (bo instytucje trzymają ETF-y i nie sprzedają tak panicznie).
                        """)
                    else:
                        st.error("Błąd pobierania danych o Złocie.")
            elif chart_type == 'graham_ghost':
                with st.spinner("Wywołuję ducha Benjamina Grahama... To zajmie chwilę (Pobieram dane fundamentalne)..."):
                    df_graham = app.get_graham_valuation_data()
                    
                    if df_graham is not None:
                        st.pyplot(app.plot_graham_ghost(df_graham))
                        
                        st.caption("""
                        ℹ️ **JAK CZYTAĆ TEN WYKRES?**
                        To jest walka między **Rynkiem (Biała Kropka)** a **Matematyką (Złoty Diament)**.
                        \n🟢 **ZIELONY PASEK (Lewa strona):** Biała kropka jest PO LEWEJ od Diamentu.
                        Oznacza to, że akcja jest tańsza niż wynika to z jej zysków i majątku. **Potencjalna okazja.**
                        \n🔴 **CZERWONY PASEK (Prawa strona):** Biała kropka jest PO PRAWEJ od Diamentu.
                        Oznacza to, że płacisz "podatek od marzeń". Cena jest wyższa niż fundamenty (Bańka/Hype).
                        \n⚠️ **Uwaga:** Dla spółek technologicznych (NVDA, TSLA) Graham jest bardzo surowy. One często są "czerwone", bo rynek wycenia przyszłość, a Graham przeszłość. Ale jeśli pasek jest GIGANTYCZNY, uważaj!
                        """)
                    else:
                        st.error("Duch Grahama milczy (Błąd danych).")
            elif chart_type == 'graham_dow':
                with st.spinner("Sędzia ocenia 30 Diamentów (Dow Jones Industrial Average)..."):
                    # Wywołujemy metodę z parametrem subset='dow'
                    df_dow = app.get_graham_valuation_data(subset='dow')
                    
                    if df_dow is not None:
                        # Rysujemy z odpowiednim tytułem
                        st.pyplot(app.plot_graham_ghost(df_dow, "Dow Jones 30 (Najbezpieczniejsze Spółki)"))
                        
                        st.caption("""
                        ℹ️ **SĘDZIA DLA ARYSTOKRACJI (Dow Jones 30)**
                        Ten wykres pokazuje wycenę tylko dla 30 najważniejszych spółek w USA.
                        \n🟢 **ZIELONY:** Jeśli spółka z tej listy (np. Nike, Boeing, Coca-Cola) świeci na zielono, jest to rzadka okazja historyczna. Te firmy prawie nigdy nie bankrutują.
                        \n🔴 **CZERWONY:** Oznacza, że płacisz bardzo wysoką premię za "jakość" i bezpieczeństwo (np. Microsoft, Visa).
                        """)
                    else:
                        st.error("Błąd pobierania danych dla Dow Jones.")
            elif chart_type == 'dow_architect':
                with st.spinner("Skanuję technicznie 30 spółek Dow Jones (RSI + SMA200 + Drawdown)..."):
                    # Pobieramy dane (Wersja Czysta Techniczna - bez Grahama)
                    df_arch = app.get_dow_architect_data()
                    
                    if df_arch is not None:
                        st.pyplot(app.plot_dow_architect(df_arch))
                        
                        st.caption("""
                        ℹ️ **ARCHITEKT DOW JONES (Czysta Technika)**
                        Ten ranking ignoruje zyski i skupia się tylko na **cenie i psychologii tłumu**.
                        \n🟢 **ZIELONE PASKI (Góra):** Spółki "zmasakrowane" technicznie. Mają niskie RSI (wyprzedanie), są głęboko pod średnią roczną (SMA200) i zaliczyły duży spadek od szczytu. To tutaj szuka się "odbicia".
                        \n🔴 **CZERWONE PASKI (Dół):** Spółki rozgrzane do czerwoności. Wysokie RSI, cena daleko nad średnią. Ryzyko korekty.
                        \n💡 **Gdzie jest Intel (INTC) i Dow Inc (DOW)?**
                        Te spółki zostały **usunięte** z indeksu Dow Jones w listopadzie 2024 roku (zastąpiły je Nvidia i Sherwin-Williams). Aby je sprawdzić, użyj przycisku **"🏛 Architekt Czasu"** (tam jest szeroki rynek).
                        """)
                    else:
                        st.error("Błąd pobierania danych dla Architekta Dow Jones.")
            elif chart_type == 'seasonal_stats':
                with st.spinner("Analizuję cykle miesięczne..."):
                    # 1. Pobieramy dane
                    s_data, _ = app.get_seasonality_data('BTC-USD')
                    
                    if s_data is not None:
                        # 2. Liczymy statystyki
                        stats_df = app.get_seasonal_stats(s_data)
                        
                        # 3. Rysujemy Wykres
                        if stats_df is not None:
                            st.pyplot(app.plot_seasonal_stats(stats_df))
                            
                            # --- NOTATKA JAK CZYTAĆ (NOWOŚĆ) ---
                            with st.expander("📖 Jak czytać ten wykres? (Instrukcja)", expanded=True):
                                st.markdown("""
                                * 📊 **Kolorowe Słupki (Średnia Zmiana):** Pokazują, ile **średnio** % Bitcoin zyskiwał lub tracił w danym miesiącu w całej swojej historii.
                                    * 🔴 **Czerwony słupek:** Miesiąc historycznie spadkowy/korekcyjny (np. Wrzesień).
                                    * 🟢 **Zielony słupek:** Miesiąc historycznie wzrostowy (np. Październik).
                                * 🟡 **Żółta Linia (Idealny Rok):** To symulacja trendu. Pokazuje, jak zachowywałby się kurs, gdyby każdego miesiąca realizował dokładnie swoją średnią historyczną.
                                    * *Jeśli linia idzie w górę:* Historycznie najlepszy okres na trzymanie (HODL).
                                    * *Jeśli linia idzie w dół/płasko:* Historycznie okres akumulacji lub spadków.
                                * 🎯 **WR (Win Rate):** Liczba nad kropkami (np. 80%). Oznacza prawdopodobieństwo sukcesu (w ilu procentach lat ten miesiąc zamykał się na zielono).
                                """)
                            # -----------------------------------
                            
                        else:
                            st.error("Błąd obliczeń statystyk.")
                    else:
                        st.error("Błąd pobierania danych historycznych.")
            elif st.session_state.get('active_lazy_chart') == 'altcoin_bull':
                
                with st.spinner("Analizuję cykle Altcoinów (SMA100 vs EMA100 na ETH)..."):
                    # 1. Pobieranie danych (metoda GET)
                    df_alt, signals_alt = app.get_altcoin_indicator_data()
                    
                    if df_alt is not None:
                        # 2. Rysowanie wykresu (metoda PLOT)
                        fig = app.plot_altcoin_indicator(df_alt, signals_alt)
                        st.pyplot(fig)
                        
                        # 3. Opis i dodatki
                        st.caption("""
                        ℹ️ **ALTCOIN BULL MARKET INDICATOR (Teoria SMA100 < EMA100)**
                        \nWykres bazuje na Ethereum (ETH) jako reprezentancie rynku Altcoinów.
                        \n📉 **Zasada:** Historycznie wielkie hossy na Altcoinach rozpoczynały się, gdy średnia prosta 100-dniowa (SMA100 - niebieska) przecinała w dół średnią wykładniczą 100-dniową (EMA100 - pomarańczowa). Jest to tzw. "Bearish Cross", który paradoksalnie wyznaczał dołek przed parabolą.
                        """)
                        
                        # Wyświetlenie ostatniego sygnału
                        if not signals_alt.empty:
                            last_signal_date = signals_alt.index[-1].strftime('%Y-%m-%d')
                            st.success(f"📅 Ostatni sygnał startu hossy (SMA100 < EMA100): **{last_signal_date}**")
                    
                    else:
                        st.error("Błąd pobierania danych dla wskaźnika Altcoinów.")
            elif st.session_state.get('active_lazy_chart') == 'congress_tracker':
                
                with st.spinner("Przeszukuję raporty giełdowe Kongresu USA (Pelosi & Co)..."):
                    # 1. Pobieranie danych
                    c_df, c_alpha = app.get_congress_tracker_data()
                    
                    if c_df is not None:
                        # 2. Rysowanie wykresu
                        fig = app.plot_congress_tracker(c_df, c_alpha)
                        st.pyplot(fig)
                        
                        # 3. Interpretacja wyników (Sygnały)
                        if c_alpha > 5.0:
                            st.error(f"🚨 **ALARM:** Politycy biją rynek aż o **{c_alpha:.1f}%**! Wiedzą coś, czego my nie wiemy (Podejrzenie Insider Tradingu).")
                        elif c_alpha > 0.0:
                            st.warning(f"⚠️ **UWAGA:** Politycy zarabiają lepiej niż S&P 500 (+{c_alpha:.1f}%). Warto śledzić ich ruchy.")
                        else:
                            st.success(f"✅ **SPOKÓJ:** Kongresmenom w tym roku nie idzie. Zwykły rynek (SPY) wygrywa.")
                            
                        # 4. Opis i edukacja
                        st.caption("""
                        ℹ️ **INSIDER TRADING TRACKER (Kongres USA)**
                        \nWykres śledzi wyniki funduszy ETF, które automatycznie kopiują transakcje polityków:
                        \n🔵 **NANC (Demokraci):** Strategia "Nancy Pelosi". Często Big Tech i innowacje.
                        \n🔴 **KRUZ (Republikanie):** Strategia "Ted Cruz". Często Energia, Paliwa i Przemysł.
                        \n⚪ **SPY (S&P 500):** Benchmark zwykłego człowieka.
                        \n🕵️‍♂️ **Zasada:** Jeśli kolorowe linie gwałtownie odrywają się od białej w górę, oznacza to, że "ludzie przy władzy" agresywnie kupują zwycięskie sektory, zanim dowie się o nich ulica.
                        """)
                    else:
                        st.error("Błąd pobierania danych o Kongresie (ETF NANC/KRUZ).")
            elif st.session_state.get('active_lazy_chart') == 'hedging_calc':
                
                with st.spinner("Uruchamiam procedury obronne (Analiza VIX, DXY, Gold)..."):
                    # 1. Pobieranie danych
                    h_data, h_score = app.get_hedging_data()
                    
                    if h_data is not None:
                        # 2. Rysowanie wykresu
                        fig = app.plot_hedging_cockpit(h_data, h_score)
                        st.pyplot(fig)
                        
                        # 3. Interpretacja i Powody
                        reasons_text = ", ".join(h_data['reasons']) if h_data['reasons'] else "Rynek jest stabilny."
                        
                        if h_score > 50:
                            st.error(f"🛡️ **ZALECENIE:** Przejdź do defensywy! Sugerowana gotówka/złoto: **{h_score}%** portfela.\n\n**Powody:** {reasons_text}")
                        elif h_score > 20:
                            st.warning(f"⚠️ **ZALECENIE:** Zachowaj ostrożność. Sugerowana gotówka: **{h_score}%**.\n\n**Powody:** {reasons_text}")
                        else:
                            st.success(f"🚀 **ZALECENIE:** Pełny atak (Risk On). Sugerowana gotówka: **{h_score}%** (tylko na zakupy).")
                            
                        # 4. Opis
                        st.caption("""
                        ℹ️ **HEDGING CALCULATOR (Inteligentna Ochrona)**
                        \nAlgorytm analizuje trzy filary strachu:
                        \n1. **VIX:** Czy inwestorzy panikują na S&P 500?
                        \n2. **DXY:** Czy uciekają do dolara (gotówki)?
                        \n3. **Gold/SPX:** Czy uciekają do złota (zamiast akcji)?
                        \nIm wyższy wynik, tym więcej kapitału powinieneś trzymać w Stablecoinach lub Złocie, czekając na krach, aby odkupić taniej.
                        """)
                    else:
                        st.error("Błąd pobierania danych rynkowych (VIX/DXY).")
            elif st.session_state.get('active_lazy_chart') == 'commodity_supercycle':
                
                with st.spinner("Ważę Ropę, Miedź i Złoto przeciwko S&P 500..."):
                    # 1. Pobieranie danych
                    comm_df = app.get_commodity_supercycle_data()
                    
                    if comm_df is not None:
                        # 2. Rysowanie
                        fig = app.plot_commodity_supercycle(comm_df)
                        st.pyplot(fig)
                        
                        # 3. Metryki (Kto wygrywa 10-lecie?)
                        # Porównujemy zwrot z ostatnich 10 lat
                        ret_comm = comm_df['Hard_Assets'].iloc[-1] - 100
                        ret_paper = comm_df['Paper_Assets'].iloc[-1] - 100
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Surowce (Hard Assets)", f"{ret_comm:+.1f}%", "Energia + Metal")
                        c2.metric("Akcje (Paper Assets)", f"{ret_paper:+.1f}%", "S&P 500")
                        
                        # 4. Opis
                        st.caption("""
                        ℹ️ **COMMODITY SUPERCYCLE DETECTOR**
                        \nHistoria pokazuje, że kapitał krąży między "Rzeczami" a "Papierem".
                        \n🟠 **Wykres idzie w górę (Era Surowców):** Wysoka inflacja, wojny, braki podaży. Akcje technologiczne radzą sobie słabo. **BTC często zachowuje się wtedy jak cyfrowe złoto.**
                        \n🔵 **Wykres idzie w dół (Era Papieru):** Niska inflacja, globalizacja, pokój. Akcje Tech i Krypto (jako Tech) rosną parabolicznie.
                        """)
                    else:
                        st.error("Błąd pobierania danych surowcowych (Yahoo Finance Futures).")
            elif st.session_state.get('active_lazy_chart') == 'silver_fractal':
                
                with st.spinner("Przeszukuję archiwa srebra od lat 70-tych..."):
                    # 1. Pobieranie i Obliczanie
                    s_curr, s_proj, s_matches = app.get_silver_fractal_prediction()
                    
                    if s_curr is not None:
                        # 2. Wykres
                        fig = app.plot_silver_fractal(s_curr, s_proj, s_matches)
                        st.pyplot(fig)
                        
                        # 3. Informacje o bliźniakach
                        st.markdown("### 🧬 Wykryte Bliźniaki Historyczne:")
                        
                        cols = st.columns(3)
                        for i, match in enumerate(s_matches):
                            year = match['date'].year
                            similarity = match['corr'] * 100
                            cols[i].metric(f"Bliźniak #{i+1}", f"{year}", f"Zgodność: {similarity:.1f}%")
                        
                        # 4. Opis
                        st.caption("""
                        ℹ️ **SILVER TIME MACHINE (Fraktale).**
                        Srebro jest metalem spekulacyjnym, który często porusza się w identycznych schematach "Pump & Dump".
                        Algorytm porównał ostatnie 250 dni handlu z całą historią od 1970 roku.
                        \n⚪ **Biała linia:** Gdzie jesteśmy teraz.
                        \n🟢/🔴 **Kolorowa linia:** Co stało się w przeszłości po takim samym układzie wykresu (średnia z 3 najbardziej podobnych lat).
                        """)
                    else:
                        st.error("Błąd danych srebra (brak historii lub błąd Yahoo Finance).")
            elif st.session_state.get('active_lazy_chart') == 'silver_macro':
                
                with st.spinner("Symuluję rynek srebra do 2030 roku..."):
                    # 1. Obliczenia
                    s_curr, s_proj, s_matches = app.get_silver_macro_projection()
                    
                    if s_curr is not None:
                        # 2. Wykres
                        st.pyplot(app.plot_silver_macro_projection(s_curr, s_proj, s_matches))
                        
                        # 3. Lista bliźniaków
                        st.info(f"🧬 **GENETYKA RYNKU:** Obecna struktura (2 lata) jest najbardziej podobna do lat: **{', '.join([str(m['date'].year) for m in s_matches])}**.")
                        
                        st.caption("""
                        ℹ️ **SILVER MACRO (5-Year Forecast).**
                        Program odnalazł w historii momenty, w których srebro zachowywało się identycznie jak przez ostatnie 2 lata.
                        Następnie nałożył na wykres to, co stało się w tamtych latach przez kolejne 5 lat.
                        \nSzara przerywana linia to **Twoja mapa drogowa** na najbliższą dekadę.
                        """)
                    else:
                        st.error("Nie znaleziono dopasowań historycznych dla tak długiego okresu.")
            elif st.session_state.get('active_lazy_chart') == 'true_inflation':
                
                with st.spinner("Łączę M2, Złoto i S&P 500 w jeden indeks prawdy..."):
                    inf_df = app.get_true_inflation_data()
                    
                    if inf_df is not None:
                        st.pyplot(app.plot_true_inflation(inf_df))
                        
                        gap = inf_df['The_Gap'].iloc[-1]
                        
                        st.error(f"💸 **WNIOSKI:** Przez ostatnie 10 lat rządowe CPI wzrosło o X%, ale koszt utrzymania statusu majątkowego (M2+Gold+Stocks) wzrósł o Y%. Różnica to **{gap:.0f} punktów procentowych**.")
                        
                        # --- TUTAJ JEST DODANA NOTATKA, O KTÓRĄ PROSIŁEŚ ---
                        st.caption("""
                        ℹ️ **THE SILENT THIEF (Cichy Złodziej).**
                        \n🔵 **Niebieska Linia (CPI):** O ile wzrosły ceny "chleba i mleka" (koszyk rządowy).
                        \n🔴 **Czerwona Linia (Koszt Majątku):** Indeks złożony z **Podaży M2 + Złota + S&P 500**.
                        \nDlaczego tak? Bo sam M2 to tylko papier. Prawdziwa inflacja to utrata Twojej zdolności do kupna aktywów. Jeśli niebieska linia rośnie o 30%, a czerwona o 150%, to znaczy, że **realnie zbiedniałeś w relacji do majątku**, nawet jeśli stać Cię na chleb. To jest dowód na inflację aktywów (Asset Inflation).
                        """)
                    else:
                        st.error("Błąd pobierania danych inflacyjnych.")
            elif st.session_state.get('active_lazy_chart') == 'vpvr':
                
                with st.spinner("Skanuję wolumen na każdym poziomie cenowym (1 rok)..."):
                    bins, vol_prof, poc, vah, val, curr = app.get_vpvr_data()
                    
                    if bins is not None:
                        st.pyplot(app.plot_vpvr(bins, vol_prof, poc, vah, val, curr))
                        
                        st.caption("""
                        ℹ️ **VPVR (Volume Profile Visible Range).**
                        Ten wykres pokazuje, przy jakiej cenie odbył się największy handel w ciągu ostatniego roku.
                        \n🔴 **Czerwona Linia (POC):** Point of Control. To jest "cena sprawiedliwa". Rynek zawsze chce tu wracać. Działa jak najsilniejszy magnes.
                        \n🔵 **Niebieska Strefa (Value Area):** Tutaj odbyło się 70% handlu.
                        \n📉 **Jak grać?** 1. Jeśli cena jest **NAD** niebieską strefą -> Silne wsparcie.
                        2. Jeśli cena jest **POD** niebieską strefą -> Silny opór.
                        3. Jeśli cena jest w środku -> Szum/Bocznica (odbijanie się od band).
                        """)
                    else:
                        st.error("Błąd obliczeń VPVR.")
        # Pobieranie CSV
        if os.path.isfile("market_log.csv"):
            with open("market_log.csv", "rb") as f: st.download_button("📥 Pobierz CSV", f, "lambo.csv")

    app.display_bottom_ticker()

if __name__ == "__main__":
    main()
