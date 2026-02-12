# bitstamp_btc_usd.py

import requests

url = "https://www.bitstamp.net/api/v2/ticker/btcusd/"

response = requests.get(url, verify=False)
data = response.json()

print(data["last"])
