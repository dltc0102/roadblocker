import googlemaps, json, re, requests, time

def get_elevation(coords_tuple: tuple[float, float]) -> float:
    lat, lon = coords_tuple
    format_lat: float = round(lat, 5)
    format_lon: float = round(lon, 5)
    format_coords = f"{format_lat}, {format_lon}"
    api_key = get_gm_api_key()
    params = {
        'locations': format_coords,
        'key': api_key
    }
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        'Accept': 'application/json, text/plain, */*',
    }
    res = requests.get(url=url, headers=headers, params=params)

    if res.status_code != 200:
        res.raise_for_status()
        return

    result = res.json()
    elevation = result['results'][0]['elevation']
    return elevation

# Node: (GM) Corrected names: Sha Tawu Kok Road (Lung Yeuk Tau) - (22.503284706386307, 114.14856045387968)
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

def regulate_dashes(name_str: str) -> str:
    new_name_str = name_str
    all_possible_dashes = ['\u002D', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2053', '\u207B', '\u208B', '\u2212', '\u2E17', '\u2E1A', '\u2E3A', '\u2E3B', '\uFE58', '\uFE63', '\uFF0D', '\uFF5E']
    for dash in all_possible_dashes:
        new_name_str = new_name_str.replace(dash, "-")

    return new_name_str

def is_valid_road_eng(name_str: str) -> bool:
    ROAD_TYPES = ["Street", "Road", "Drive", "Terrace", "Highway", "Tunnel", "Expressway", "Bypass", "Avenue", "Boulevard"]
    return any(word in name_str for word in ROAD_TYPES)

def is_valid_road_cn(name_str: str) -> bool:
    CHINESE_ROAD_TYPES = ["街道", "街", "隧道", "道", "公路"]
    return any(word in name_str for word in CHINESE_ROAD_TYPES)

def get_lang_names_v2(name_str: str) -> tuple[str, str]:
    dash_name_str = regulate_dashes(name_str)
    has_dash = '-' in dash_name_str
    if not has_dash:
        chinese_pattern = r'[\u4e00-\u9fff]+'
        chinese_matches = re.findall(chinese_pattern, dash_name_str)
        chinese_name = ' '.join(chinese_matches) if chinese_matches else ""
        english_name = re.sub(chinese_pattern, '', dash_name_str).strip()
        return english_name, chinese_name

    parts = dash_name_str.split()

    # english
    english_start_idx = next((i for i, part in enumerate(parts) if part and part[0].isascii() and part[0].isalpha()), 0)
    raw_english_parts = parts[english_start_idx:]
    dash_idx = raw_english_parts.index("-")
    english_parts = [' '.join(raw_english_parts[:dash_idx]), ' '.join(raw_english_parts[dash_idx+1:])]
    english_name = None
    for eng_part in english_parts:
        if is_valid_road_eng(eng_part):
            english_name = eng_part
            break

    # chinese
    chinese_full_name = parts[:english_start_idx][0].strip()
    chinese_parts = chinese_full_name.split("-")
    chinese_name = None
    for chi_part in chinese_parts:
        if is_valid_road_cn(chi_part):
            chinese_name = chi_part
            break

    return english_name, chinese_name

def extract_english_name(text: str) -> str:
    if not text: return ""
    if is_valid_road_eng(text): return text

    if '-' in text and len(text) > 5:
        parts = [part.strip() for part in text.split('-')]
        for part in parts:
            if is_valid_road_eng(part):
                return part
        return parts[0] if parts else text

    return text

def extract_chinese_name(text: str) -> str:
    if not text: return ""
    if is_valid_road_cn(text): return text

    if '-' in text and len(text) > 2:
        parts = [part.strip() for part in text.split('-')]
        for part in parts:
            if is_valid_road_cn(part):
                return part
        return parts[0] if parts else text

    return text

def get_lang_names_v3(name_str: str) -> tuple[str, str]:
    dash_name_str = regulate_dashes(name_str)

    if ',' in dash_name_str:
        parts = [part.strip() for part in dash_name_str.split(',', 1)]
        if len(parts) == 2:
            eng_part, cn_part = parts
            if eng_part and eng_part[0].isascii() and cn_part and '\u4e00' <= cn_part[0] <= '\u9fff':
                return extract_english_name(eng_part), extract_chinese_name(cn_part)
            elif cn_part and cn_part[0].isascii() and eng_part and '\u4e00' <= eng_part[0] <= '\u9fff':
                return extract_english_name(cn_part), extract_chinese_name(eng_part)

    dash_pos = dash_name_str.find('-')
    if dash_pos == -1:
        chinese_matches = re.findall(r'[\u4e00-\u9fff]+', dash_name_str)
        chinese_name = ' '.join(chinese_matches) if chinese_matches else ""
        english_name = re.sub(r'[\u4e00-\u9fff]+', '', dash_name_str).strip()
        return english_name, chinese_name

    left_part = dash_name_str[:dash_pos].strip()
    right_part = dash_name_str[dash_pos + 1:].strip()

    left_is_english = left_part and left_part[0].isascii()
    right_is_chinese = right_part and '\u4e00' <= right_part[0] <= '\u9fff'

    if left_is_english and right_is_chinese:
        return extract_english_name(left_part), extract_chinese_name(right_part)
    else:
        return extract_english_name(right_part), extract_chinese_name(left_part)

def extract_english_name(text: str) -> str:
    """Extract valid English road name from text"""
    if '-' in text:
        parts = text.split('-')
        for part in parts:
            if is_valid_road_eng(part.strip()):
                return part.strip()
        return parts[0].strip() if parts else ""
    return text.strip()

def extract_chinese_name(text: str) -> str:
    """Extract valid Chinese road name from text"""
    if '-' in text:
        parts = text.split('-')
        for part in parts:
            if is_valid_road_cn(part.strip()):
                return part.strip()
        return parts[0].strip() if parts else ""
    return text.strip()

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
            if len(address_component["types"]) == 1 and address_component["types"][0] == 'route' and not is_text_chinese(address_component["long_name"]):
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

def is_text_chinese(text: str) -> bool:
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def osm_address_lookup(request_address: str) -> json:
    NOMINATIM_HEADERS = {
        "User-Agent": "Roadblocker/1.0 (daniellautc@gmail.com)"
    }
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        'q': request_address,
        'format': 'jsonv2',
        'limit': 5,
        'addressdetails': 1,
        'country_code': 'cn'
    }

    response = requests.get(url, params=params, headers=NOMINATIM_HEADERS)
    if response.status_code == 429:
        print("Rate limited by Nominatim. Please wait before making another request.")
        return None

    if response.status_code != 200:
        response.raise_for_status()
        return None

    return response.json()

def get_chinese_address_name(self, address="") -> str:
    given_address = self.ename if address == "" else address
    eng_addr_lookup = osm_address_lookup(given_address)[0]
    eng_addr_result = eng_addr_lookup
    try:
        eng_addr_result = eng_addr_lookup["address"]["name"]
    except KeyError:
        eng_addr_result = eng_addr_lookup["name"]

    ename, cname = self.get_lang_names_v2(eng_addr_result)
    return cname

def gm_reverse_lookup(coord_tuple: tuple) -> json:
    apikey = get_gm_api_key()
    gmaps = googlemaps.Client(key=apikey)
    lat, lon = coord_tuple
    reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
    if not reverse_geocode_result:
        return "No results found for these coordinates."
    return reverse_geocode_result

class GMAddress:
    def __init__(self, long_name, short_name, road_type, category, coords):
        self.long_name = long_name
        self.short_name = short_name
        self.road_type = road_type
        self.category = category
        self.coords = coords
        self.lat = coords[0]
        self.lon = coords[1]
        self.elevation = None

        if self.coords:
            self.elevation = self.get_elevation(self.coords)

    def get_elevation(self, coords_tuple: tuple[float, float]) -> float:
        lat, lon = coords_tuple
        format_lat: float = round(lat, 5)
        format_lon: float = round(lon, 5)
        format_coords = f"{format_lat}, {format_lon}"
        api_key = get_gm_api_key()
        params = {
            'locations': format_coords,
            'key': api_key
        }
        url = "https://maps.googleapis.com/maps/api/elevation/json"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
            'Accept': 'application/json, text/plain, */*',
        }
        res = requests.get(url=url, headers=headers, params=params)

        if res.status_code != 200:
            res.raise_for_status()
            return

        result = res.json()
        elevation = result['results'][0]['elevation']
        return elevation

def get_gm_address(results: list, elevation: float, road_category: str) -> str | None:
    gm_address_nodes = []
    for result in results:
        address_components = result["address_components"]
        location = result["geometry"]["location"]
        lat = location["lat"]
        lon = location["lng"]
        coordinates = (lat, lon)
        for component in address_components:
            comp_type = component["types"]
            long_name = component["long_name"].upper()
            short_name = component["short_name"].upper()
            if len(comp_type) == 1 and comp_type[0] == "route" and not long_name.isdigit() and not short_name.isdigit():
                gm_address_nodes.append(GMAddress(long_name, short_name, comp_type, road_category, coordinates))

    valid_nodes = [node for node in gm_address_nodes if node.elevation is not None]

    if not valid_nodes:
        print("no gm address valid nodes.")
        return None

    if road_category == 'highway':
        return max(valid_nodes, key=lambda x: x.elevation)
    else:
        return min(valid_nodes, key=lambda x: x.elevation)

def correct_street_names(geometry_start_point, elevation):
    osm_reverse_result = osm_reverse_lookup(geometry_start_point)
    used_gmaps = False

    new_ename = None
    new_cname = None
    if osm_reverse_result:
        road_category = osm_reverse_result["category"]
        road_type = osm_reverse_result["addresstype"]
        try:
            address_name: str = osm_reverse_result["address"]["road"]
        except KeyError:
            address_name: str = osm_reverse_result["name"]

        new_ename, new_cname = get_lang_names_v2(address_name)

        if new_ename != "" and new_cname != "":
            ename = new_ename.upper()
            if is_text_chinese(new_cname):
                cname = new_cname
            else:
                cname = get_chinese_address_name(new_ename)
            return ename, cname

        else:
            gm_reverse_result = gm_reverse_lookup(geometry_start_point)
            gm_address = get_gm_address(gm_reverse_result, elevation, road_category)
            if gm_address:
                ename = gm_address.long_name

                if is_text_chinese(gm_address.short_name):
                    cname = gm_address.short_name
                else:
                    cname = get_chinese_address_name(gm_address.long_name)
                return ename, cname

def main():
    a1 = get_lang_names_v3("CASTLE PEAK ROAD - TSUEN WAN, 青山公路　-　荃灣段")
    a2 = get_lang_names_v3("青山公路　-　荃灣段, CASTLE PEAK ROAD - TSUEN WAN")
    print(a1)
    print(a2)

if __name__ == "__main__":
    main()
