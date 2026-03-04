import requests
try:
    print("Testing Google...")
    r1 = requests.get("https://www.google.com", timeout=5)
    print(f"Google status: {r1.status_code}")
    
    print("\nTesting ExchangeRate-API v6 root...")
    url = "https://v6.exchangerate-api.com"
    r2 = requests.get(url, timeout=5)
    print(f"ExchangeRate-API status: {r2.status_code}")
    print(f"Response: {r2.text[:100]}")
except Exception as e:
    print(f"\nCaught error: {type(e).__name__}: {e}")
