import requests

url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24?api-key=579b464db66ec23bdd00000106ea7c593dbc4d9277b1913480d0862d&format=csv"

response = requests.get(url)

if response.status_code == 200:
    with open("market_prices.csv", "wb") as file:
        file.write(response.content)
    print("Data saved as market_prices.csv")
else:
    print(f"Error: {response.status_code}")
