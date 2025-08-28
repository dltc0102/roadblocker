# Program goal: Find the top N fastest routes between two points

import os, zipfile, fiona, json, requests, bs4, re
from bs4 import BeautifulSoup
import urllib.request
import geopandas as gpd

def get_latest_road_network(input_dirpath: str):
    filename = "RdNet_IRNP.gdb"
    gdb_filepath = os.path.join(os.getcwd(), 'dataset', filename)
    if os.path.exists(gdb_filepath):
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

def remove_filepath(filepath: str) -> str:
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

def get_expressway_limits():
    url = "https://en.wikipedia.org/wiki/List_of_streets_and_roads_in_Hong_Kong"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0"
    }
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    if res.status_code != 200:
        print("url not 200.")
        return {}

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

def get_average_speed_by_street(gdf: gpd.geodataframe.GeoDataFrame):
    new_gdf = gdf.copy()
    expressway_limits = get_expressway_limits()
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

def convert_to_serializable(data):
    if hasattr(data, 'wkt'):
        return data.wkt
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return data

def serialise_segment_data_to_json(giv_data):
    remove_filepath('parsed_data.json')
    print('removed filepath parsed_data.json')
    serializable_data = convert_to_serializable(giv_data)
    with open('parsed_data.json', 'w', encoding='utf-8') as parsed_f:
        json.dump(serializable_data, parsed_f, indent=2, ensure_ascii=False)
    print('dump completed')

def get_estimated_time(distance: float, ave_speed: float) -> float:
    return float(distance / ave_speed)

def get_gdf_with_weight(gdf: gpd.geodataframe.GeoDataFrame):
    new_gdf = gdf.copy()
    new_gdf["weight"] = 0.0

    for idx, street in new_gdf.iterrows():
        street_ave_speed = float(street["average_speed"]) # km/hr
        street_length_km = float(street["SHAPE_Length"] / 1000)
        weight: float = get_estimated_time(distance=street_length_km, ave_speed=street_ave_speed)
        new_gdf.at[idx, "weight"] = weight

    return new_gdf


def get_fastest_routes(start_coords: tuple, end_coords: tuple, gdf: gpd.geodataframe.GeoDataFrame) -> list[dict, dict, dict]:
    route1 = {}
    route2 = {}
    route3 = {}
    return [route1, route2, route3]

def main():
    dataset_dirpath: str = get_dataset_dirpath()
    get_latest_road_network(input_dirpath=dataset_dirpath)
    parsed_gdf: gpd.geodataframe.GeoDataFrame = parse_gdb_files(input_dirpath=dataset_dirpath)
    gdf_after_getting_segment_data, segment_data = get_segment_data(parsed_gdf)
    gdf_with_speed = get_average_speed_by_street(gdf_after_getting_segment_data)
    gdf_with_weight = get_gdf_with_weight(gdf_with_speed)

    c_start = (0, 0)
    c_end = (100, 100)
    fastest_routes = get_fastest_routes(c_start, c_end, gdf_with_weight)


if __name__ == "__main__":
    main()