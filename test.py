import googlemaps, json, re

lookup_coords: tuple[float, float] = (22.280458812249023, 114.16778286509698)

def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        return key_f.readline().strip()

# Node: (GM) Corrected names: New Territories Circular Road - (22.50430307158854, 114.091346525785)
# Node: Corrected names: San Tin Highway - (22.498236136916386, 114.07886670996484)
# Node: (GM) Corrected names: Tim Mei Avenue - (22.280458812249023, 114.16778286509698)

def contains_chinese(foo: str) -> bool:
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(foo))

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
        for address_component in address_components:
            if len(address_component["types"]) == 1 and address_component["types"][0] == 'route' and not contains_chinese(address_component["long_name"]):
                print(gm_results)
                new_ename = address_component["long_name"]
                new_cname = address_component["short_name"]
                break
        if new_ename:
            break
    return [new_ename, new_cname]

def main():
    result = gm_reverse_lookup(lookup_coords)
    print(result)

if __name__ == "__main__":
    main()
