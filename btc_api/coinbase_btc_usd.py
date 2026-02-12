# coinbase_btc_usd.py

import requests

url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"

response = requests.get(url, verify=False)
data = response.json()

print(data["data"]["amount"])
