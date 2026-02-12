# kraken_xbt_usd.py

import requests

url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"

response = requests.get(url, verify=False)
data = response.json()

pair = next(iter(data["result"].values()))

print(pair["c"][0])  # last trade price
