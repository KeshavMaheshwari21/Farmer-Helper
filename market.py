import requests
import pandas as pd
from io import StringIO

# API URL
url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

# Request parameters
params = {
    "api-key": "579b464db66ec23bdd00000106ea7c593dbc4d9277b1913480d0862d",
    "format": "csv",
    "limit": 1000,  # Fetch more records
    "filters[State.keyword]": "Rajasthan",  # Filter for Rajasthan
    "filters[Arrival_Date]": "2024-03-25"  # Fetch latest data (modify as needed)
}

# API Request
response = requests.get(url, params=params)

# Save data if request is successful
if response.status_code == 200:
    csv_data = response.text
    df = pd.read_csv(StringIO(csv_data))

    # Save filtered data
    if not df.empty:
        df.to_csv("market_prices_rajasthan.csv", index=False)
        print("Filtered data saved as market_prices_rajasthan.csv")
    else:
        print("No data found for Rajasthan on the given date.")
else:
    print(f"Error: {response.status_code}")
