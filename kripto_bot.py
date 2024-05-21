import tkinter as tk
from tkinter import ttk
import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from datetime import datetime, timedelta
import hashlib

current_index = 0
future_dates = None
future_predictions = None
users = {
    "user1": hashlib.sha256("password1".encode()).hexdigest(),
    "user2": hashlib.sha256("password2".encode()).hexdigest()
}

def login():
    username = username_entry.get()
    password = hashlib.sha256(password_entry.get().encode()).hexdigest()
    if username in users and users[username] == password:
        root.destroy()  
        open_main_app()  
    else:
        login_label.config(text="Hatalı kullanıcı adı veya şifre")

def open_main_app():
    root = tk.Tk()
    root.title("Binance Fiyat Takip Uygulaması")
    root.geometry("800x600")

    symbols = get_symbols()
    selected_symbol = tk.StringVar(root)
    selected_symbol.set(symbols[0])
    option_menu = ttk.Combobox(root, textvariable=selected_symbol, values=symbols)
    option_menu.grid(row=0, column=0, padx=10, pady=10)
    option_menu.bind("<<ComboboxSelected>>", on_symbol_select)

    symbol_price_label = tk.Label(root, text="", font=("Helvetica", 12))
    symbol_price_label.grid(row=0, column=1, padx=10, pady=10)

    label = tk.Label(root, text="Fiyat Yükleniyor...", font=("Helvetica", 16))
    label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    ma_label = tk.Label(root, text="", font=("Helvetica", 12))
    ma_label.grid(row=1, column=1, padx=10, pady=10, sticky="w")

    date_labels = []
    prediction_labels = []
    for i in range(7):
        date_label = tk.Label(root, text="", font=("Helvetica", 10))
        date_label.grid(row=2+i, column=1, padx=10, pady=5, sticky="e")
        date_labels.append(date_label)
        prediction_label = tk.Label(root, text="", font=("Helvetica", 10))
        prediction_label.grid(row=2+i, column=2, padx=10, pady=5, sticky="w")
        prediction_labels.append(prediction_label)

    previous_button = tk.Button(root, text="Geri", command=show_previous_day)
    previous_button.grid(row=9, column=1, padx=10, pady=5, sticky="e")
    next_button = tk.Button(root, text="İleri", command=show_next_day)
    next_button.grid(row=9, column=2, padx=10, pady=5, sticky="w")

    fig = plt.Figure(figsize=(8, 5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    update_price()

    root.mainloop()

def get_price(symbol):
    try:
        url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        return float(data['price'])
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
        print(f"API Error: {e}")
        return None

def get_historical_data(symbol):
    try:
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=100'
        response = requests.get(url)
        response.raise_for_status()  
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
        print(f"API Error: {e}")
        return pd.DataFrame()  

def get_symbols():
    try:
        url = 'https://api.binance.com/api/v3/exchangeInfo'
        response = requests.get(url)
        response.raise_for_status()  
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING' and s['symbol'].endswith('USDT')]
        return symbols
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
        print(f"API Error: {e}")
        return []

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_signals(df):
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df)
    df.dropna(inplace=True)
    buy_signals = []
    sell_signals = []
    flag = -1

    for i in range(len(df)):
        if df['MA20'].iloc[i] > df['MA50'].iloc[i]:
            if flag != 1:
                buy_signals.append(df.index[i])
                sell_signals.append(None)
                flag = 1
            else:
                buy_signals.append(None)
                sell_signals.append(None)
        elif df['MA20'].iloc[i] < df['MA50'].iloc[i]:
            if flag != 0:
                buy_signals.append(None)
                sell_signals.append(df.index[i])
                flag = 0
            else:
                buy_signals.append(None)
                sell_signals.append(None)
        else:
            buy_signals.append(None)
            sell_signals.append(None)
    
    df['Buy_Signal_Price'] = buy_signals
    df['Sell_Signal_Price'] = sell_signals
    return df

# Gelecek fiyat tahmini
def predict_future(df, days=7):
    df['timestamp'] = pd.to_numeric(df.index.values)
    X = df[['timestamp', 'MA20', 'MA50', 'RSI']].values
    y = df['close'].values
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    future_predictions = []
    future_dates = []
    for i in range(1, days+1):
        future_timestamp = np.array([df.index[-1].value + i * 86400000000000]).reshape(-1, 1)  # 86400000000000 ns = 1 day
        future_ma20 = df['MA20'].iloc[-1]
        future_ma50 = df['MA50'].iloc[-1]
        future_rsi = df['RSI'].iloc[-1]
        future_features = np.array([[future_timestamp[0, 0], future_ma20, future_ma50, future_rsi]])
        prediction = model.predict(future_features)
        future_predictions.append(prediction[0])
        future_dates.append((df.index[-1] + timedelta(days=i)).strftime('%Y-%m-%d'))
    return future_dates, future_predictions


# Fiyat güncelleme ve grafiği çizme
def update_price():
    global future_dates, future_predictions
    try:
        symbol = selected_symbol.get()
        price = get_price(symbol)
        if price is not None:
            df = get_historical_data(symbol)
            if not df.empty:
                df = get_signals(df)
                ma20 = df['MA20'].iloc[-1]
                ma50 = df['MA50'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                price_format = "{:,.8f}" if price < 1 else "{:,.2f}"
                label.config(text=f"{symbol} Fiyatı: ${price_format.format(price)}")
                ma_label.config(text=f"MA(20): {price_format.format(ma20)}\nMA(50): {price_format.format(ma50)}\nRSI: {rsi:.2f}")

                # Gelecek fiyat tahmini
                if future_dates is None or future_predictions is None:
                    future_dates, future_predictions = predict_future(df)
                for i in range(len(future_dates)):
                    date_labels[i].config(text=future_dates[i])
                    prediction_labels[i].config(text=f"${price_format.format(future_predictions[i])}")

                # Alım ve satım sinyallerini eklemek için yeni bir sütun oluşturma
                buy_signals = df[df['Buy_Signal_Price'].notna()].index
                sell_signals = df[df['Sell_Signal_Price'].notna()].index

                # Grafiği güncelleme
                fig.clear()
                ax_candle = fig.add_subplot(2, 1, 1)
                ax_volume = fig.add_subplot(2, 1, 2, sharex=ax_candle)
                mpf.plot(df, type='candle', mav=(20, 50), volume=ax_volume, ax=ax_candle,
                         show_nontrading=True, style='yahoo')
                ax_candle.scatter(buy_signals, df.loc[buy_signals]['close'], label='Alım Sinyali', marker='^', color='g', s=100)
                ax_candle.scatter(sell_signals, df.loc[sell_signals]['close'], label='Satım Sinyali', marker='v', color='r', s=100)
                ax_candle.legend()
                ax_candle.set_title(f'{symbol} Fiyat Grafiği')
                canvas.draw()
            else:
                label.config(text="Veri alınamadı")
        else:
            label.config(text="Veri alınamadı")
    except Exception as e:
        label.config(text=f"Bir hata oluştu: {e}")
    # Fiyatı her 5 saniyede bir güncelle
    root.after(5000, update_price)


# ... Diğer fonksiyonlar ve GUI kodu ...

def on_symbol_select(event):
    update_price()

def show_previous_day():
    # Geçmişteki fiyat tahminlerini gösterme işlevi
    pass

def show_next_day():
    # Gelecekteki fiyat tahminlerini gösterme işlevi
    pass

# Ana pencereyi oluşturma
root = tk.Tk()
root.title("Binance Fiyat Takip Uygulaması")
root.geometry("800x600")

# Sembol seçme
symbols = get_symbols()
selected_symbol = tk.StringVar(root)
selected_symbol.set(symbols[0])  # Varsayılan olarak ilk sembol seçili olacak
option_menu = ttk.Combobox(root, textvariable=selected_symbol, values=symbols)
option_menu.grid(row=0, column=0, padx=10, pady=10)
option_menu.bind("<<ComboboxSelected>>", on_symbol_select)


# Sembol fiyatı
symbol_price_label = tk.Label(root, text="", font=("Helvetica", 12))
symbol_price_label.grid(row=0, column=1, padx=10, pady=10)

# Fiyat etiketi
label = tk.Label(root, text="Fiyat Yükleniyor...", font=("Helvetica", 16))
label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

# MA ve RSI etiketi
ma_label = tk.Label(root, text="", font=("Helvetica", 12))
ma_label.grid(row=1, column=1, padx=10, pady=10, sticky="w")

# Gelecek fiyat tahmini etiketleri
date_labels = []
prediction_labels = []
for i in range(7):
    date_label = tk.Label(root, text="", font=("Helvetica", 10))
    date_label.grid(row=2+i, column=1, padx=10, pady=5, sticky="e")
    date_labels.append(date_label)
    prediction_label = tk.Label(root, text="", font=("Helvetica", 10))
    prediction_label.grid(row=2+i, column=2, padx=10, pady=5, sticky="w")
    prediction_labels.append(prediction_label)

# Geri ve İleri butonları
previous_button = tk.Button(root, text="Geri", command=show_previous_day)
previous_button.grid(row=9, column=1, padx=10, pady=5, sticky="e")
next_button = tk.Button(root, text="İleri", command=show_next_day)
next_button.grid(row=9, column=2, padx=10, pady=5, sticky="w")

# Grafik için figür oluşturma
fig = plt.Figure(figsize=(8, 5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Fiyatı güncelleme
update_price()

# Ana döngüyü başlatma
root.mainloop()

