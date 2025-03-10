import requests

url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
params = {
    "api-key": "579b464db66ec23bdd00000106ea7c593dbc4d9277b1913480d0862d",
    "format": "csv"
}
headers = {
    "accept": "text/csv"
}

response = requests.get(url, params=params, headers=headers)

if response.status_code == 200:
    with open("data.csv", "wb") as file:
        file.write(response.content)
    print("CSV file saved as data.csv")
else:
    print(f"Error: {response.status_code}")
