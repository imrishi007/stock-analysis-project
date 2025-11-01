from alpaca_trade_api.rest import REST

API_KEY = "PKJOI3ZMNW4T7SAV7C93"
SECRET_KEY = "598094fb-25a4-41a1-a8b2-99794593c1fb"
BASE_URL = "https://paper-api.alpaca.markets"


alpaca = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Test by fetching your account details
account = alpaca.get_account()
print(account)
