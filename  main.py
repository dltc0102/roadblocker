# Program goal: Find the top N fastest routes between two points

import os, zipfile, fiona, json, requests, re, time, math
from pyproj import Transformer
from bs4 import BeautifulSoup
import urllib.request
import geopandas as gpd
from shapely import wkt
from scipy.spatial import cKDTree
import numpy as np
import heapq

# visualizer


"""
GDF
"""
def get_latest_road_network(input_dirpath: str) -> None:
    filename = "RdNet_IRNP.gdb"
    gdb_filepath = os.path.join(os.getcwd(), 'dataset', filename)
    if os.path.exists(gdb_filepath):
        print("Path already exists, redownload is not needed.")
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

def parse_gdb_files(input_dirpath: str, specified_name="CENTERLINE") -> gpd.geodataframe.GeoDataFrame:
    gdb_dirs: list = [file for file in os.listdir(input_dirpath) if file.endswith('.gdb')]
    if not gdb_dirs:
        raise FileNotFoundError("No GDB directory found in the dataset folder")

    gdb_dir: str = gdb_dirs[0]
    gdb_filepath: str = os.path.join(input_dirpath, gdb_dir)

    layers          : list = fiona.listlayers(gdb_filepath)
    layer_wanted    : str = None
    for layer in layers:
        if layer == specified_name:
            layer_wanted = layer

    gdf = gpd.read_file(gdb_filepath, layer=layer_wanted)
    print(f"Found {len(gdf)} road segments for gdf")
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

def convert_epsg_to_wgs84(gdf: gpd.geodataframe.GeoDataFrame):
    # GDF CRS: EPSG:2326
    # WGS84 CRS: EPSG:4326

    new_gdf = gdf.copy()
    crs_transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)

    coords_list = []
    for idx, street in new_gdf.iterrows():
        geom = street["geometry"]

        if geom.geom_type == 'Point':
            start_e, start_n = geom.coords[0]
        elif geom.geom_type in ['LineString', 'MultiLineString']:
            if geom.geom_type == 'MultiLineString':
                start_e, start_n = tuple((geom.geoms[0]).coords[0])
            else:
                start_e, start_n = geom.coords[0]
        else:
            start_e, start_n = (None, None)

        if start_e is not None and start_n is not None:
            lon, lat = crs_transformer.transform(start_e, start_n)
            coords_list.append((lat, lon))
        else:
            coords_list.append(None)

    new_gdf["coords"] = coords_list
    return new_gdf

class TrafficLightNode:
    def __init__(self, node_id, node_type, geometry, coords):
        self.node_id = node_id
        self.node_type = node_type
        self.coordinates = coords

    def _print(self):
        print(f"{self.node_id} ({self.node_type}): {self.coordinates}")

class Node:
    def __init__(self, ename, street_id, coords, heuristic):
        self.ename = ename
        self.street_id = street_id
        self.coordinates = coords # [lat, lon]
        self.heuristic = heuristic
        self.neighbours = []

    def _print(self):
        print(f"{self.ename} ({self.street_id}): {self.coordinates}")

def get_nodes_from_gdf(gdf: gpd.geodataframe.GeoDataFrame) -> list:
    new_gdf = gdf.copy()
    gdf_nodes = []

    for idx, street in new_gdf.iterrows():
        street_ename = street["STREET_ENAME"]
        street_id = street["ROUTE_ID"]
        street_coords = street["coords"]
        street_weight = street["weight"]
        street_node = Node(street_ename, street_id, street_coords, street_weight)
        gdf_nodes.append(street_node)

    return gdf_nodes

def parse_traffic_light_locations(gdf: gpd.geodataframe.GeoDataFrame) -> list[TrafficLightNode]:
    traffic_light_nodes = []
    for idx, street in gdf.iterrows():
        traffic_light_nodes.append(
            TrafficLightNode(
                node_id=street["FEATURE_ID"],
                node_type=street["FEATURE_TYPE"],
                geometry=street["geometry"],
                coords=street["coords"]
            )
        )
    return traffic_light_nodes


"""
COORDINATE STUFF
"""

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
        "start": (float(start_details["lat"]), float(start_details["lon"])),
        "end": (float(end_details["lat"]), float(end_details["lon"])),
    }


"""
ALGO
"""
class NodeKDTree:
    def __init__(self, nodes: list[Node]=None):
        self.tree = None
        self.nodes = nodes or []
        self.coords_array = None

        if nodes:
            self.build_tree(nodes)

    def build_tree(self, nodes: list[Node]):
        self.nodes = nodes
        self.coords_array = np.array([node.coordinates for node in nodes])  # Fixed: removed extra bracket
        self.tree = cKDTree(self.coords_array)

    def add_node(self, node: Node):
        self.nodes.append(node)
        self.build_tree(self.nodes)

    def find_nearest_neighbours(self, coord: list[float], k: int = 1):
        if not self.tree:
            raise ValueError("cKDTree not built yet.")
        distances, idxs = self.tree.query(coord, k=k)
        if k == 1:
            return self.nodes[idxs], distances
        else:
            return [self.nodes[i] for i in idxs], distances

    def get_all_nodes(self):
        return self.nodes

def reconstruct_path(came_from: dict[Node, Node], current: Node) -> list[Node]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def haversine_distance(node1, node2):
    lat1, lon1 = node1.coordinates
    lat2, lon2 = node2.coordinates
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371 #km
    return c * r

def a_star(start: Node, target: Node):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: start.heuristic}

    all_nodes = set()
    for neighbor, _ in start.neighbours:
        all_nodes.add(neighbor)
    for neighbor, _ in target.neighbours:
        all_nodes.add(neighbor)
    all_nodes.add(start)
    all_nodes.add(target)

    open_set_nodes = set([start])
    closed_set_nodes = set()

    while open_set:
        current_f, current = heapq.heappop(open_set)
        open_set_nodes.remove(current)

        if current == target:
            final_path = reconstruct_path(came_from, current)
            return final_path, g_score[target]

        closed_set_nodes.add(current)

        for neighbor, edge_cost in current.neighbours:
            if neighbor in closed_set_nodes:
                continue

            tentative_g = g_score[current] + edge_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + neighbor.heuristic

                if neighbor not in open_set_nodes:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_nodes.add(neighbor)

    return [], float('inf')

def get_top_n_routes(coordinate_details: dict, node_kdtree: NodeKDTree, top_n=3):
    start_coords = coordinate_details["start"]
    end_coords = coordinate_details["end"]

    nearby_start_nodes, _ = node_kdtree.find_nearest_neighbours(start_coords, k=top_n)
    nearby_end_nodes, _ = node_kdtree.find_nearest_neighbours(end_coords, k=top_n)

    routes = []
    for i, (start_node, end_node) in enumerate(zip(nearby_start_nodes, nearby_end_nodes)):
        print(f"Finding route {i+1}...")
        path, cost = a_star(start_node, end_node)
        if path:
            routes.append((path, cost))
            print(f"Route {i+1} found with cost: {cost:.2f}")

    return sorted(routes, key=lambda x: x[1])[:top_n]


"""
VISUALIZER
"""



"""
MAIN
"""
def main():

    # gdf stuff
    USER_HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    dataset_dirpath: str = get_dataset_dirpath()

    get_latest_road_network(input_dirpath=dataset_dirpath)

    parsed_gdf_road = parse_gdb_files(input_dirpath=dataset_dirpath, specified_name="CENTERLINE")
    parsed_gdf_traffic = parse_gdb_files(input_dirpath=dataset_dirpath, specified_name="TRAFFIC_FEATURES")

    gdf_traffic_with_wgs84 = convert_epsg_to_wgs84(parsed_gdf_traffic)
    traffic_light_locations: list = parse_traffic_light_locations(gdf_traffic_with_wgs84)
    for tl in traffic_light_locations:
        print(tl.node_id, tl.coordinates)

    # gdf_after_getting_segment_data, segment_data = get_segment_data(parsed_gdf_road)
    # print("GDF: Segment length column added.")

    # gdf_with_speed = get_average_speed_by_street(gdf_after_getting_segment_data, headers=USER_HEADERS)
    # print("GDF: Speed column added.")

    # gdf_with_weight = get_gdf_with_weight(gdf_with_speed)
    # print("GDF: Heuristics column added.")

    # gdf_with_wgs84 = convert_epsg_to_wgs84(gdf_with_weight)
    # print("GDF: EPSG:2326 converted to WGS84.")

    # nodes_from_gdf: list[Node] = get_nodes_from_gdf(gdf_with_wgs84)
    # print("GDF: Retrieved all nodes from gdf.")

    # node_kdtree = NodeKDTree(nodes_from_gdf)
    # print("cKDTree: Built nodes into kdtree.")

    # address look up
    start_address: str = "2 lung pak street"
    end_address: str = "89 pok fu lam road"
    # osm_start_options: json = osm_address_lookup(request_address=start_address)
    # osm_end_options: json = osm_address_lookup(request_address=end_address)
    # osm_start: dict = get_chosen_option(osm_start_options)
    # time.sleep(2)
    # osm_end: dict = get_chosen_option(osm_end_options)
    # coordinate_details: dict = get_coordinate_details(osm_start, osm_end)
    # print(coordinate_details)

    coordinate_details = {
        "start": [22.3642146, 114.1794265],
        "end": [22.2853336, 114.1330190]
    }
    # algorithm
    # top_n_routes = get_top_n_routes(coordinate_details, node_kdtree, top_n=3)
    # print("\nTop routes found:")
    # for i, (path, cost) in enumerate(top_n_routes, 1):
    #     print(f"Route {i}: Cost = {cost:.6f}")
    #     for node in path:
    #         print(f"  -> {node.ename} ({node.coordinates})")

if __name__ == "__main__":
    main()