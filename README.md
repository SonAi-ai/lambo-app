Witaj w Lambo czy Karton? – zaawansowanym, wielowątkowym terminalu wywiadowczym zbudowanym w Pythonie (Streamlit). Aplikacja śledzi globalną płynność makroekonomiczną, przepływy "Smart Money", sentyment oraz rotację kapitału, aby odpowiedzieć na ostateczne pytanie każdego inwestora: Czy w tej hossie kupujemy Lambo, czy rezerwujemy mokry karton?

🌟 Przegląd Projektu
Aplikacja działa jako potężny "Indeks Prawdopodobieństwa Rynkowego". Pobiera dane na żywo oraz historyczne z wielu niezależnych źródeł (Yahoo Finance, Rezerwa Federalna FRED, CME Group, API Binance, OpenSea), aby na bieżąco oceniać zdrowie tradycyjnej gospodarki, siłę rynku kryptowalut oraz obecną fazę cyklu finansowego.

Zamiast patrzeć tylko na cenę, terminal zagląda pod maskę rynków: analizuje twarde dane makro, przybliżenia wskaźników on-chain oraz fizyczne stany magazynowe metali szlachetnych.

🚀 Główne Funkcje
1. Globalne Centrum Dowodzenia
Gospodarka vs Krypto: Główne wskaźniki postępu podsumowujące kondycję tradycyjnego przemysłu (Eco_Index) w zestawieniu ze spekulacją cyfrową (Cry_Index).

Matryca Rynku (GPS): Wykres radarowy oceniający rynek w 5 wymiarach: Technika (RSI), On-Chain (MVRV), Makro (ISM), Wycena i Sentyment.

Monitor FED & Pieniędzy: Stały podgląd na rentowność 10-letnich obligacji USA (^TNX) w panelu bocznym, określający nastawienie Rezerwy Federalnej (Drukarka vs Zaciskanie).

Menedżer Portfela: "Lambo Meter", który przelicza wartość Twojego portfolio (BTC, ETH, SOL) i nadaje Ci odpowiedni "Status Społeczny" (od Żula pod Żabką po Imperatora w Urusie).

2. Zaawansowana Analityka Makroekonomiczna
Indeks Globalnej Płynności: Śledzi połączone bilanse największych banków centralnych (FED, EBC, BOJ) w zestawieniu z ceną Bitcoina.

Fed Net Liquidity: Oblicza faktyczną, czystą płynność w systemie USA (Aktywa - TGA - RRP).

Detonator Recesji: Analizuje inwersję krzywej dochodowości (10Y-2Y) nałożoną na oficjalne historyczne recesje NBER.

Supercykl Surowcowy: Autorski indeks badający rotację kapitału między "Rzeczami" (Ropa, Miedź, Złoto) a "Papierem" (S&P 500).

Prawdziwa Inflacja (Cichy Złodziej): Porównuje oficjalne CPI z kompozytowym wskaźnikiem kosztu utrzymania majątku (M2 + Złoto + SPX).

3. Projekcje AI i Cykle Kryptowalut
Wyrocznia AI: Wykorzystuje model uczenia maszynowego prophet (Facebook) do prognozowania ceny BTC na podstawie danych MVRV i ISM.

Nostradamus 5.0 (Złota Era): Hybrydowy model projekcyjny łączący halvingi Bitcoina z 8-letnim historycznym cyklem złota (po zatwierdzeniu ETF w 2004 roku).

Fibonacci Quad Sync: Analiza długoterminowego trendu przy użyciu wykładniczych średnich Fibonacciego (21, 55, 89, 233).

The Phoenix Cross: Autorski wskaźnik przecinania się EMA 89 z SMA 350, identyfikujący hossy generacyjne.

Echolokacja (Fourier): Dekodowanie fal sinusoidalnych z wykresu cenowego w celu przewidywania matematycznych punktów zwrotnych.

4. Skanery Rynku i "Gdzie jest kapusta?"
Wielowątkowe Skanery (Multi-Threading): Błyskawiczne pobieranie danych dla dziesiątek altcoinów z Binance API przy użyciu concurrent.futures.

Detektor Wybuchu (Squeeze): Wyłapuje ekstremalne zwężenia Wstęg Bollingera (dla BTC, ETH i Altcoinów), zwiastujące potężne ruchy cenowe.

Rotacja Sektorowa: Śledzi przepływ kapitału w S&P 500 oraz wykonuje "Deep Scan" wszystkich 500 spółek, szukając liderów w 3 koszykach (Mózg, Ciało, Schron).

Klastry Tematyczne: Gotowe analizy dla megatrendów:

Dolina Krzemowa (Mag 7) i Wojny Czipów

RoboCitizen & Deep Tech Materials (Baterie, Wodór)

Era Exascale (Superkomputery) i Wąskie Gardło AI (Prąd)

Space Domination & Exotic Propulsion (Nowy Kosmos)

Biotech Frontier, Longevity & Generation Zero (Medycyna przyszłości)

Insider Trading Tracker (Kopiowanie transakcji Kongresu USA).

5. Wojna o Fizyczny Metal (COMEX/NYMEX)
Bezpośredni dostęp do Giełd: Wykorzystuje bibliotekę curl_cffi do omijania zapór (firewalli) Wall Street i pobiera surowe raporty magazynowe .xls z CME Group.

Śledzenie Skarbców: Monitoruje na żywo metale (Złoto, Srebro, Platyna, Pallad, Miedź) z podziałem na statusy Registered i Eligible. Wbudowany system alarmowy reaguje na spadek/wzrost rezerw o 50% w ujęciu 30-dniowym.

Chiński Mur (SHFE): Eksperymentalny moduł scrapingowy (Selenium) próbujący wyciągać dane z giełdy w Szanghaju, wizualizujący zjawisko ukrywania danych ("Access Denied").

🛠️ Architektura Techniczna
UI/UX Framework: streamlit

Dane Finansowe: yfinance (Krypto/Akcje), pandas_datareader (dane makro z FRED), requests / curl_cffi (API i omijanie zabezpieczeń TLS), selenium (Scraping stron z renderowaniem JS).

Przetwarzanie Danych: pandas, numpy

Wizualizacja: matplotlib, seaborn

Machine Learning: prophet (Analiza szeregów czasowych)

Optymalizacja Wydajności: System działa na wątkach (concurrent.futures.ThreadPoolExecutor), skracając czas ładowania skanerów altcoinów z minut do kilku sekund.

OTA Updates (Over-The-Air): Wbudowany system automatycznej aktualizacji – skrypt sam pobiera swoją nowszą wersję i pliki zależne (logo, requirements) prosto z GitHuba.

📦 Instalacja i Uruchomienie

1. Pobierz program:

Kliknij zielony przycisk <> Code.

Wybierz Download ZIP.

Rozpakuj folder na pulpicie.

2. Zainstaluj Pythona (Tylko raz):

(Jeśli nie masz) Musisz zainstalować Pythona, najlepiej wersję  3.10.4 ze strony python.org.

WAŻNE: Podczas instalacji musisz zaznaczyć ptaszka przy: "Add Python to PATH".

3. Uruchom:

Wejdź do rozpakowanego folderu.

Kliknij dwa razy w plik start.bat

⚖️ Nota Prawna (Disclaimer)
Aplikacja Lambo czy Karton jest narzędziem o charakterze wyłącznie edukacyjnym i analitycznym. Żadne dane, analizy, sygnały ani predykcje modeli sztucznej inteligencji zawarte w programie nie stanowią porady inwestycyjnej ani rekomendacji finansowej. Rynek kryptowalut i akcji wiąże się z ekstremalnym ryzykiem utraty kapitału. Autor nie ponosi żadnej odpowiedzialności za ewentualne zyski lub straty wynikające z użytkowania oprogramowania. Każda decyzja finansowa jest podejmowana wyłącznie na własne ryzyko.

Stworzone przez Pawła, dla tych, którzy chcą jeździć Lambo, a nie mieszkać w Kartonie.
