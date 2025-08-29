# Program goal: Find the top N fastest routes between two points

import os, zipfile, fiona, json, requests, re, time
from bs4 import BeautifulSoup
import urllib.request
import geopandas as gpd

def get_latest_road_network(input_dirpath: str) -> None:
    filename = "RdNet_IRNP.gdb"
    gdb_filepath = os.path.join(os.getcwd(), 'dataset', filename)
    if os.path.exists(gdb_filepath):
        print("path already exists, dont need to redownload.")
        return

    download_url: str = "https://static.data.gov.hk/td/road-network-v2/RdNet_IRNP.gdb.zip"
    os.makedirs(input_dirpath, exist_ok=True)

    dataset_zip_path: str = os.path.join(input_dirpath, "dataset.zip")
    urllib.request.urlretrieve(download_url, dataset_zip_path)
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_f:
        zip_f.extractall(input_dirpath)

    os.remove(dataset_zip_path)

def get_dataset_dirpath() -> str:
    return os.path.join(os.getcwd(), 'dataset')

def parse_gdb_files(input_dirpath: str) -> gpd.geodataframe.GeoDataFrame:
    gdb_dirs: list = [file for file in os.listdir(input_dirpath) if file.endswith('.gdb')]
    if not gdb_dirs:
        raise FileNotFoundError("No GDB directory found in the dataset folder")

    gdb_dir: str = gdb_dirs[0]
    gdb_filepath: str = os.path.join(input_dirpath, gdb_dir)

    layers              : list= fiona.listlayers(gdb_filepath)
    road_layer          : str = None
    road_network_name   : str = "CENTERLINE"
    for layer in layers:
        # print(f"layer name: {layer}")
        if layer == road_network_name:
            road_layer = layer
            break

    if not road_layer:
        print("Road layer not found")

    gdf: gpd.geodataframe.GeoDataFrame = gpd.read_file(gdb_filepath, layer=road_layer)
    print(f"Found {len(gdf)} road segments")
    return gdf

def remove_filepath(filepath: str) -> None:
    if os.path.exists(filepath):
        os.remove(filepath)

def get_segment_data(gdf: gpd.geodataframe.GeoDataFrame) -> tuple:
    all_street_data = {}

    # Group by street name
    grouped = gdf.groupby('STREET_ENAME')

    for street_name, group in grouped:
        all_street_data[street_name] = {
            "segments": {}
        }

        # Sort and process segments within each street by route id
        # st code has a chnace to be NaN
        sorted_group = group.sort_values('ROUTE_ID')
        for idx, (_, row) in enumerate(sorted_group.iterrows(), 1):
            segment_id = row["ROUTE_ID"]

            if segment_id not in all_street_data[street_name]:

                all_street_data[street_name][segment_id] = {
                    "segment_code": row["ST_CODE"],
                    "segment_direction": row["TRAVEL_DIRECTION"],
                    "segment_length": row["SHAPE_Length"],
                    "segment_geometry": row["geometry"],
                    "segment_number": idx
                }

    return gdf, all_street_data

def get_expressway_limits(headers: dict) -> dict | None:
    url = "https://en.wikipedia.org/wiki/List_of_streets_and_roads_in_Hong_Kong"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    if res.status_code != 200:
        print("url not 200.")
        return None

    soup = BeautifulSoup(res.content, "html.parser")
    expressway_table = soup.select_one("table.wikitable.collapsible")
    expressway_rows = expressway_table.find_all("tr")[1:]

    expressway_limits = {}
    for row in expressway_rows:
        cells = row.find_all(['td', 'th'])

        if len(cells) >= 3:
            # Get expressway name
            name_cell = cells[0]
            name_link = name_cell.find('a', title=True)
            expressway_name = name_link.get_text(strip=True) if name_link else name_cell.get_text(strip=True)

            # Get speed limit with regex to handle various formats
            speed_text = cells[2].get_text(strip=True)

            # Find all numbers in the speed limit text
            numbers = re.findall(r'\d+', speed_text)

            if numbers:
                if len(numbers) == 1:
                    speed_limit = int(numbers[0])
                else:
                    # Multiple numbers found, create tuple
                    speed_limit = tuple(int(num) for num in numbers)

                expressway_limits[expressway_name] = speed_limit

    return expressway_limits

def get_average_speed_by_street(gdf: gpd.geodataframe.GeoDataFrame, headers: dict) -> gpd.geodataframe.GeoDataFrame | None:
    new_gdf = gdf.copy()
    expressway_limits = get_expressway_limits(headers)
    if expressway_limits is None:
        print("url did not work")
        return None

    new_gdf["speed_limit"] = 50
    new_gdf["average_speed"] = 0.0

    for idx, row in new_gdf.iterrows():
        street_name = row["STREET_ENAME"]
        if street_name in expressway_limits:
            speed_limit_value = expressway_limits[street_name]

            # for now, take max value. try and get realtime value later
            if isinstance(speed_limit_value, tuple):
                speed_limit = max(speed_limit_value)
            else:
                speed_limit = speed_limit_value

            new_gdf.at[idx, 'speed_limit'] = speed_limit

        new_gdf.at[idx, 'average_speed'] = 0.9 * new_gdf.at[idx, 'speed_limit']

    return new_gdf

def convert_to_serializable(data: dict) -> dict:
    if hasattr(data, 'wkt'):
        return data.wkt
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return data

def serialise_segment_data_to_json(giv_data: dict) -> json:
    remove_filepath('parsed_data.json')
    print('removed filepath parsed_data.json')
    serializable_data: dict = convert_to_serializable(giv_data)
    with open('parsed_data.json', 'w', encoding='utf-8') as parsed_f:
        json.dump(serializable_data, parsed_f, indent=2, ensure_ascii=False)
    print('dump completed')

def get_estimated_time(distance: float, ave_speed: float) -> float:
    return float(distance / ave_speed)

def get_gdf_with_weight(gdf: gpd.geodataframe.GeoDataFrame) -> gpd.geodataframe.GeoDataFrame:
    new_gdf = gdf.copy()
    new_gdf["weight"] = 0.0

    for idx, street in new_gdf.iterrows():
        street_ave_speed = float(street["average_speed"]) # km/hr
        street_length_km = float(street["SHAPE_Length"] / 1000)
        weight: float = get_estimated_time(distance=street_length_km, ave_speed=street_ave_speed)
        new_gdf.at[idx, "weight"] = weight

    return new_gdf

def percent_encode_address(giv_address: str) -> str:
    encoded_address = requests.utils.quote(giv_address)
    return encoded_address

def address_lookup(headers: dict, request_address: str, lookup_num=200, tolerance=35, searching_mode=0) -> json:
    lookup_url = "https://www.als.gov.hk/lookup"
    encoded_address = percent_encode_address(request_address)

    lookup_params = {
        'q': encoded_address,
        'n': lookup_num,
        't': tolerance
    }

    if searching_mode != 0:
        lookup_params['b'] = 1

    res = requests.get(lookup_url, headers=headers, params=lookup_params)
    if res.status_code != 200:
        res.raise_for_status()
        return None

    lookup_json = res.json()
    return lookup_json

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
        'country_code': 'cn',
    }

    response = requests.get(url, params=params, headers=NOMINATIM_HEADERS)
    if response.status_code == 429:
        print("Rate limited by Nominatim. Please wait before making another request.")
        return None

    if response.status_code != 200:
        response.raise_for_status()
        return None

    data = response.json()
    return data

def get_fastest_routes(start_coords: tuple, end_coords: tuple, gdf: gpd.geodataframe.GeoDataFrame) -> list[dict, dict, dict]:
    route1 = {}
    route2 = {}
    route3 = {}
    return [route1, route2, route3]

def get_simplified_options(options: list | dict) -> list:
    if isinstance(options, list):
        simplified_options = []
        for option in options:
            option_details = {
                "place_id": option["place_id"],
                "display_name": option["display_name"],
                "category": option["category"],
                "type": option["type"],
                "lat": option["lat"],
                "lon": option["lon"],
            }
            simplified_options.append(option_details)
        return simplified_options
    elif isinstance(options, dict):
        return {
                "place_id": options["place_id"],
                "display_name": options["display_name"],
                "category": options["category"],
                "type": options["type"],
                "lat": options["lat"],
                "lon": options["lon"],
            }

def get_chosen_option(options: list) -> dict:
    simplified_options = get_simplified_options(options)
    chosen_idx = 1
    for idx, option in enumerate(simplified_options, 1):
        print(f"{idx}. {option["display_name"]} ({option["place_id"]})")
        for key, value in option.items():
            if key != "display_name" and key != "place_id":
                print(f" | {key}: {value}")
        print()

    print()
    while True:
        user_input = input(f"Which of the {len(simplified_options)} are you choosing? ")
        if user_input.isdigit():
            chosen_idx = int(user_input)
            result = options[chosen_idx]
            return options[chosen_idx]

def get_coordinate_details(start_details: dict, end_details: dict):
    return {
        "start": {
            "lat": start_details["lat"],
            "lon": start_details["lon"],
        },
        "end": {
            "lat": end_details["lat"],
            "lon": end_details["lon"]
        }
    }

def main():
    USER_HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    dataset_dirpath: str = get_dataset_dirpath()

    get_latest_road_network(input_dirpath=dataset_dirpath)

    parsed_gdf: gpd.geodataframe.GeoDataFrame = parse_gdb_files(input_dirpath=dataset_dirpath)

    gdf_after_getting_segment_data, segment_data = get_segment_data(parsed_gdf)

    gdf_with_speed = get_average_speed_by_street(gdf_after_getting_segment_data, headers=USER_HEADERS)

    gdf_with_weight = get_gdf_with_weight(gdf_with_speed)
    print(gdf_with_weight)

    # address look up
    # start_address: str = "2 lung pak street"
    # end_address: str = "89 pok fu lam road"
    # osm_start_options: json = osm_address_lookup(request_address=start_address)
    # osm_end_options: json = osm_address_lookup(request_address=end_address)
    # osm_start: dict = get_chosen_option(osm_start_options)
    # time.sleep(2)
    # osm_end: dict = get_chosen_option(osm_end_options)
    # coordinate_details: dict = get_coordinate_details(osm_start, osm_end)

    # print(coordinate_details)
if __name__ == "__main__":
    main()