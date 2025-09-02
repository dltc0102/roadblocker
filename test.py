import googlemaps, json, re, requests, time

lookup_coords: tuple[float, float] = (22.251172132041248, 114.17117263080621)
# Node: (GM) Corrected names: Sha Tau Kok Road (Lung Yeuk Tau) - (22.503284706386307, 114.14856045387968)
# Node: Corrected names: － Castle Peak Road – Tsing Lung Tau - (22.360005779960783, 114.04059982038386)
# Node: Corrected names: － Tuen Mun – Chek Lap Kok Tunnel - (22.32276571190256, 113.96192692351556)
# Node: (GM) Corrected names: 6 - (22.286979145515904, 114.1594365478507)
# Node: Corrected names: － Castle Peak Road – Lam Tei - (22.424892932088465, 113.98789061725248)
# Node: (GM) Corrected names: Castle Peak Road-Sham Tseng - (22.36731595211941, 114.06101151282067)
# Node: (GM) Corrected names: Xing Hua Yuan - (22.549038208374853, 114.15186893675946)
# Node: (GM) Corrected names: 6 - (22.286979145515904, 114.1594365478507)


def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        return key_f.readline().strip()

def get_lang_name(name_str: str) -> list[str, str]:
    chinese_pattern = r'[\u4e00-\u9fff]+'
    chinese_matches = re.findall(chinese_pattern, name_str)
    chinese_name = ' '.join(chinese_matches) if chinese_matches else ""
    english_name = re.sub(chinese_pattern, '', name_str).strip()
    return english_name, chinese_name

def contains_chinese(foo: str) -> bool:
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(foo))

def clean_name(name_str: str) -> str:
    if '-' in name_str:
        street, _ = name_str.split('-')
        return street.strip()
    else:
        return name_str.strip()

def osm_reverse_lookup(coord_tuple: tuple) -> json:
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
            time.sleep(2)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)

    print("Max retries exceeded for reverse lookup")
    return None

def is_valid_road(road_name: str) -> bool:
    return any(word in road_name for word in ["Street", "Road", "Drive", "Terrace", "Highway", "Tunnel", "Expressway", "Bypass", "Avenue", "Boulevard"])

def gm_reverse_lookup(coord_tuple: tuple) -> json:
    apikey = get_gm_api_key()
    gmaps = googlemaps.Client(key=apikey)
    lat, lon = coord_tuple
    gm_results = gmaps.reverse_geocode((lat, lon))
    if not gm_results:
        return "No results found for these coordinates."

    new_ename = None
    new_cname = None
    for gm_result in gm_results:
        address_components = gm_result["address_components"]
        possible_addresses = []
        for address_component in address_components:
            if len(address_component["types"]) == 1 and address_component["types"][0] == 'route' and not contains_chinese(address_component["long_name"]):
                long_name = address_component["long_name"]
                short_name = address_component["short_name"]
                possible_addresses.append((long_name, short_name))
                print(gm_results)

        for address in possible_addresses:
            long_name, short_name = address
            if is_valid_road(long_name):
                new_ename = long_name
                new_cname = short_name
        if new_ename:
            break
    return [new_ename, new_cname]

def main():
    gm_result = gm_reverse_lookup(lookup_coords)
    print(gm_result)
    osm_result = osm_reverse_lookup(lookup_coords)
    print(osm_result)

if __name__ == "__main__":
    main()
