# from flask import Flask, render_template, request
# import yfinance as yf
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.io import to_html
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model
# from datetime import datetime, timedelta
# import io
# import base64

# app = Flask(__name__,template_folder='templates',static_folder='static')

# # Load your pre-trained LSTM model
# model = load_model('model.h5')

# def preprocess_data(stock, start, end):
#     bit_coin_data = yf.download(stock, start, end)
#     Closing_price = bit_coin_data[['Close']]
#     Open_price = bit_coin_data[['Open']]
#     return Closing_price, Open_price, bit_coin_data

# def create_forecast(data, model, time_step=60, days=15):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)
#     last_sequence = scaled_data[-time_step:]
#     forecast = []

#     for _ in range(days):
#         last_sequence = last_sequence.reshape((1, time_step, 1))
#         next_price = model.predict(last_sequence)
#         next_price_with_noise = next_price + np.random.uniform(-0.03, 0.03)
#         forecast.append(next_price_with_noise[0, 0])
#         last_sequence = np.roll(last_sequence, -1, axis=1)
#         last_sequence[0, -1, 0] = next_price_with_noise

#     forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
#     return forecast

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/results', methods=['POST'])
# def results():
#     crypto = request.form['crypto']
#     num_days = int(request.form['num_days'])

#     end = datetime.now()
#     start = datetime(end.year - 2, end.month, end.day)

#     Closing_price, Open_price, bit_coin_data = preprocess_data(crypto, start, end)
#     data = Closing_price.values

#     forecast = create_forecast(data, model, days=num_days)

#     future_dates = [end + timedelta(days=i) for i in range(1, num_days + 1)]

#     # Generate plots
#     fig1 = go.Figure()
#     recent_days = bit_coin_data.tail(num_days)
#     fig1.add_trace(go.Bar(x=recent_days.index, y=recent_days['Open'], name='Open Price', marker_color='red'))
#     fig1.add_trace(go.Bar(x=recent_days.index, y=recent_days['Close'], name='Close Price', marker_color='blue'))
#     fig1.update_layout(title='Open and Close Prices for Last {} Days'.format(num_days), xaxis_title='Date', yaxis_title='Price', barmode='group')

#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=Closing_price.index, y=Closing_price['Close'], mode='lines', name='Close Price'))
#     fig2.update_layout(title='Bitcoin Close Price for Last 2 Years', xaxis_title='Date', yaxis_title='Price')

#     fig3 = go.Figure()
#     actual_days = Closing_price.tail(num_days)
#     predicted_days = forecast.flatten()
#     fig3.add_trace(go.Bar(x=actual_days.index, y=actual_days['Close'], name='Actual Prices', marker_color='green'))
#     fig3.add_trace(go.Bar(x=actual_days.index, y=predicted_days, name='Predicted Prices', marker_color='blue'))
#     fig3.update_layout(title='Actual vs Predicted Prices for Last {} Days'.format(num_days), xaxis_title='Date', yaxis_title='Price', barmode='group')

#     fig4 = go.Figure()
#     fig4.add_trace(go.Scatter(x=Closing_price.index, y=Closing_price['Close'], mode='lines', name='Historical Prices'))
#     fig4.add_trace(go.Scatter(x=future_dates, y=forecast.flatten(), mode='lines', name='Forecast Prices'))
#     fig4.update_layout(title='Bitcoin Price Forecast for Next {} Days'.format(num_days), xaxis_title='Date', yaxis_title='Price')

#     # Convert figures to HTML
#     plot_div1 = to_html(fig1, full_html=False)
#     plot_div2 = to_html(fig2, full_html=False)
#     plot_div3 = to_html(fig3, full_html=False)
#     plot_div4 = to_html(fig4, full_html=False)

#     return render_template('result.html', plot_div1=plot_div1, plot_div2=plot_div2, plot_div3=plot_div3, plot_div4=plot_div4)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load your pre-trained LSTM model
model = load_model('model.h5')

def preprocess_data(stock, start, end):
    bit_coin_data = yf.download(stock, start, end)
    Closing_price = bit_coin_data[['Close']]
    Open_price = bit_coin_data[['Open']]
    return Closing_price, Open_price, bit_coin_data

def create_forecast(data, model, time_step=60, days=15):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    last_sequence = scaled_data[-time_step:]
    forecast = []

    for _ in range(days):
        last_sequence = last_sequence.reshape((1, time_step, 1))
        next_price = model.predict(last_sequence)
        next_price_with_noise = next_price + np.random.uniform(-0.03, 0.03)
        forecast.append(next_price_with_noise[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_price_with_noise

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast

@app.route('/')
def main():
    return render_template('main.html')
    
@app.route('/prices')
def prices():
    return render_template('prices.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    crypto = request.form['crypto']
    num_days = int(request.form['num_days'])

    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)

    Closing_price, Open_price, bit_coin_data = preprocess_data(crypto, start, end)
    data = Closing_price.values

    forecast = create_forecast(data, model, days=num_days)

    future_dates = [end + timedelta(days=i) for i in range(1, num_days + 1)]

    # Generate plots
    fig1 = go.Figure()
    recent_days = bit_coin_data.tail(num_days)
    fig1.add_trace(go.Bar(x=recent_days.index, y=recent_days['Open'], name='Open Price', marker_color='red'))
    fig1.add_trace(go.Bar(x=recent_days.index, y=recent_days['Close'], name='Close Price', marker_color='blue'))
    fig1.update_layout(title='Open and Close Prices for Last {} Days'.format(num_days), xaxis_title='Date', yaxis_title='Price', barmode='group')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=Closing_price.index, y=Closing_price['Close'], mode='lines', name='Close Price'))
    fig2.update_layout(title='Bitcoin Close Price for Last 2 Years', xaxis_title='Date', yaxis_title='Price')

    fig3 = go.Figure()
    actual_days = Closing_price.tail(num_days)
    predicted_days = forecast.flatten()
    fig3.add_trace(go.Bar(x=actual_days.index, y=actual_days['Close'], name='Actual Prices', marker_color='green'))
    fig3.add_trace(go.Bar(x=actual_days.index, y=predicted_days, name='Predicted Prices', marker_color='blue'))
    fig3.update_layout(title='Actual vs Predicted Prices for Last {} Days'.format(num_days), xaxis_title='Date', yaxis_title='Price', barmode='group')

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=Closing_price.index, y=Closing_price['Close'], mode='lines', name='Historical Prices'))
    fig4.add_trace(go.Scatter(x=future_dates, y=forecast.flatten(), mode='lines', name='Forecast Prices'))
    fig4.update_layout(title='Bitcoin Price Forecast for Next {} Days'.format(num_days), xaxis_title='Date', yaxis_title='Price')

    # Convert figures to HTML
    plot_div1 = to_html(fig1, full_html=False)
    plot_div2 = to_html(fig2, full_html=False)
    plot_div3 = to_html(fig3, full_html=False)
    plot_div4 = to_html(fig4, full_html=False)

    return render_template('result.html', plot_div1=plot_div1, plot_div2=plot_div2, plot_div3=plot_div3, plot_div4=plot_div4)

if __name__ == '__main__':
    app.run(debug=True)
