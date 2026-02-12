# coingecko_btc_usd.py

import requests

url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

response = requests.get(url, verify=False)
data = response.json()

print(data["bitcoin"]["usd"])
