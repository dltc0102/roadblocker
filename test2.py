import requests, bs4, json, re
from bs4 import BeautifulSoup

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

def get_chinese_address_name(eng_address: str) -> str:
    eng_addr_lookup = osm_address_lookup(eng_address)[0]
    eng_addr_result = eng_addr_lookup
    try:
        eng_addr_result = eng_addr_lookup["address"]["name"]
    except KeyError:
        eng_addr_result = eng_addr_lookup["name"]

    ename, cname = get_lang_names_v2(eng_addr_result)
    return cname

def main():
    res = get_chinese_address_name("ocean park road")
    print(res)

if __name__ == "__main__":
    main()