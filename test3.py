import re, requests, bs4, json, time
from bs4 import BeautifulSoup
import googlemaps

def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        return key_f.readline().strip()

def osm_reverse_lookup(coord_tuple: tuple[float, float]) -> json:
        NOMINATIM_HEADERS = {
            "User-Agent": "Roadblocker/1.0 (daniellautc@gmail.com)"
        }

        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'format': 'jsonv2',
            'lat': coord_tuple[0],
            'lon': coord_tuple[1],
            'zoom': 18,
            'addressdetails': 1
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=NOMINATIM_HEADERS, timeout=10)
                if response.status_code == 429:
                    print("Rate limited by Nominatim. Please wait before making another request.")
                    return None
                if response.status_code != 200:
                    response.raise_for_status()
                    return None
                return response.json()

            except requests.exceptions.SSLError as e:
                print(f"SSL error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)

        print("Max retries exceeded for reverse lookup")
        return None

def gm_reverse_lookup(coord_tuple: tuple[float, float]) -> json:
        apikey = get_gm_api_key()
        gmaps = googlemaps.Client(key=apikey)
        lat, lon = coord_tuple
        reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
        if not reverse_geocode_result:
            return "No results found for these coordinates."
        return reverse_geocode_result

def main():
    coords = (22.26660866775255, 114.16237137746583)
    res = osm_reverse_lookup(coords)
    print(res)
    gres = gm_reverse_lookup(coords)
    print(gres)
     
if __name__ == "__main__":
    main()