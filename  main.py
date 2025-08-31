# Program goal: Find the top N fastest routes between two points

import os, zipfile, fiona, json, requests, re, time, math, googlemaps
from pyproj import Transformer
from bs4 import BeautifulSoup
import urllib.request
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
import heapq

# visualizer
"""----------
    UTILS
----------"""
def convert_mls_crs(mls_str) -> list[tuple[float, float]]:
    crs_transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
    all_coords = []

    if mls_str.geom_type == 'MultiLineString':
        for line_string in mls_str.geoms:
            for coord in line_string.coords:
                eing, ning = coord
                if eing is not None and ning is not None:
                    lon, lat = crs_transformer.transform(eing, ning)
                    all_coords.append((lat, lon))

    elif mls_str.geom_type == 'LineString':
        for coord in mls_str.coords:
            eing, ning = coord
            if eing is not None and ning is not None:
                lon, lat = crs_transformer.transform(eing, ning)
                all_coords.append((lat, lon))

    elif mls_str.geom_type == 'Point':
        eing, ning = mls_str.coords[0]
        if eing is not None and ning is not None:
            lon, lat = crs_transformer.transform(eing, ning)
            all_coords.append((lat, lon))

    return all_coords

def get_lang_name(name_str: str) -> list[str, str]:
    chinese_pattern = r'[\u4e00-\u9fff]+'
    chinese_matches = re.findall(chinese_pattern, name_str)
    chinese_name = ' '.join(chinese_matches) if chinese_matches else ""
    english_name = re.sub(chinese_pattern, '', name_str).strip()
    return english_name, chinese_name

def remove_filepath(filepath: str) -> None:
    if os.path.exists(filepath):
        os.remove(filepath)

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
    print(f"GDF: EPSG:2326 converted to WGS84 for gdf.")
    return new_gdf

def contains_chinese(foo: str) -> bool:
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(foo))


"""---------
  ROAD GDF
---------"""
def get_dataset_dirpath() -> str:
    return os.path.join(os.getcwd(), 'dataset')

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

def expand_gdf_cols(gdf: gpd.geodataframe.GeoDataFrame, headers: dict) -> gpd.geodataframe.GeoDataFrame:
    new_gdf = gdf.copy()
    expressway_limits = get_expressway_limits(headers)
    crs_transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
    if not expressway_limits:
        print('problem occurred in get_expressway_limits()')
        return None

    new_gdf["speed_limit"] = 50
    new_gdf["average_speed"] = 0.0
    new_gdf["weight"] = 0.0
    coords_list = []

    for idx, street in new_gdf.iterrows():
        street_ename = street["STREET_ENAME"]
        street_length_km = street["SHAPE_Length"] / 1000
        street_geom = street["geometry"]
        street_geom_coords = convert_mls_crs(street_geom)

        for street_geom_coord in street_geom_coords:
            print(street_geom_coord)
            eing, ning = street_geom_coord
            lon, lat = crs_transformer.transform(eing, ning)
            coords_list.append((lat, lon))

        if street_ename in expressway_limits:
            speed_limit_val = expressway_limits[street_ename]
            speed_limit = speed_limit_val
            if isinstance(speed_limit, tuple):
                speed_limit = max(speed_limit_val)

            new_gdf.at[idx, 'speed_limit'] = speed_limit

        new_gdf.at[idx, 'average_speed'] = 0.9 * new_gdf.at[idx, 'speed_limit']
        new_gdf.at[idx, 'weight'] = street_length_km / new_gdf.at[idx, 'average_speed']
        new_gdf.at[idx, 'coords'] = coords_list

    print("GDF: speed_limit column added.")
    print("GDF: average_speed column added")
    print("GDF: weight column added")
    print("GDF: EPSG:2326 converted to WGS84 for gdf.")
    return new_gdf

def parse_gdb_files(input_dirpath: str, headers: dict, specified_name="CENTERLINE") -> gpd.geodataframe.GeoDataFrame:
    gdb_dirs: list = [file for file in os.listdir(input_dirpath) if file.endswith('.gdb')]
    if not gdb_dirs:
        raise FileNotFoundError("No GDB directory found in the dataset folder")

    gdb_dir         : str = gdb_dirs[0]
    gdb_filepath    : str = os.path.join(input_dirpath, gdb_dir)
    layers          : list = fiona.listlayers(gdb_filepath)
    layer_wanted    : str = None
    for layer in layers:
        if layer == specified_name:
            layer_wanted = layer

    gdf: gpd.geodataframe.GeoDataFrame = gpd.read_file(gdb_filepath, layer=layer_wanted)
    print(f"Found {len(gdf)} road segments for gdf")
    return gdf

# def get_average_speed_by_street(gdf: gpd.geodataframe.GeoDataFrame, headers: dict) -> gpd.geodataframe.GeoDataFrame | None:
#     new_gdf = gdf.copy()
#     expressway_limits = get_expressway_limits(headers)
#     if expressway_limits is None:
#         print("url did not work")
#         return None

#     new_gdf["speed_limit"] = 50
#     new_gdf["average_speed"] = 0.0

#     for idx, row in new_gdf.iterrows():
#         street_name = row["STREET_ENAME"]
#         if street_name in expressway_limits:
#             speed_limit_value = expressway_limits[street_name]

#             # for now, take max value. try and get realtime value later
#             if isinstance(speed_limit_value, tuple):
#                 speed_limit = max(speed_limit_value)
#             else:
#                 speed_limit = speed_limit_value

#             new_gdf.at[idx, 'speed_limit'] = speed_limit

#         new_gdf.at[idx, 'average_speed'] = 0.9 * new_gdf.at[idx, 'speed_limit']

#     print("GDF: Speed column added.")
#     return new_gdf

# def get_gdf_with_weight(gdf: gpd.geodataframe.GeoDataFrame) -> gpd.geodataframe.GeoDataFrame:
    new_gdf = gdf.copy()
    new_gdf["weight"] = 0.0

    for idx, street in new_gdf.iterrows():
        street_ave_speed = float(street["average_speed"]) # km/hr
        street_length_km = float(street["SHAPE_Length"] / 1000)
        weight: float = street_length_km / street_ave_speed
        new_gdf.at[idx, "weight"] = weight

    print("GDF: Node's Weight column added. (time in hours)")
    return new_gdf

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

    print("GDF: Retrieved all nodes from gdf.")
    return gdf_nodes


"""--------------
   TRAFFIC GDF
--------------"""
class TrafficLightNode:
    def __init__(self, node_id, node_type, geometry, coords):
        self.node_id = node_id
        self.node_type = node_type
        self.coordinates = coords

    def _print(self):
        print(f"{self.node_id} ({self.node_type}): {self.coordinates}")

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



"""-------------------
   COORDINATE STUFF
-------------------"""
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

def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        return key_f.readline().strip()

def gm_reverse_lookup(coord_tuple: tuple) -> json:
    apikey = get_gm_api_key()
    gmaps = googlemaps.Client(key=apikey)
    lat, lon = coord_tuple
    reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
    if not reverse_geocode_result:
        return "No results found for these coordinates."
    return reverse_geocode_result

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


# class NodeKDTree:
#     def __init__(self, nodes: list[Node]=None):
#         self.tree = None
#         self.nodes = nodes or []
#         self.coords_array = None

#         if nodes:
#             self.build_tree(nodes)

#     def build_tree(self, nodes: list[Node]):
#         self.nodes = nodes
#         self.coords_array = np.array([node.coordinates for node in nodes])
#         self.tree = cKDTree(self.coords_array)
#         print("cKDTree: Built nodes into kdtree.")

#     def add_node(self, node: Node):
#         self.nodes.append(node)
#         self.build_tree(self.nodes)

#     def find_nearest_neighbors(self, coord: list[float], k: int = 1):
#         if not self.tree:
#             raise ValueError("cKDTree not built yet.")
#         distances, idxs = self.tree.query(coord, k=k)
#         if k == 1:
#             return self.nodes[idxs], distances
#         else:
#             return [self.nodes[i] for i in idxs], distances

#     def get_all_nodes(self):
        # return self.nodes

# def haversine_distance(node1: Node, node2: Node) -> float:
#     lat1, lon1 = node1.coordinates
#     lat2, lon2 = node2.coordinates
#     lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
#     c = 2 * math.asin(math.sqrt(a))

#     r = 6371 #km
#     return c * r

# def combined_heuristics(node: Node, target_coords: tuple[float, float]) -> float:
#     max_speed_kmh = 110
#     target_node = Node("TARGET", "TARGET", target_coords, 0)
#     geo_distance_km = haversine_distance(node, target_node)
#     return geo_distance_km / max_speed_kmh

# def build_neighbors_based_on_geometry(gdf: gpd.geodataframe.GeoDataFrame, nodes: list[Node], distance_threshold=1.0) -> list[Node]:
#     """Build neighbors based on actual geometric connections between road segments"""

#     # Create a spatial index for faster intersection checks
#     spatial_index = gdf.sindex
#     total_neighbors = 0

#     for idx, (_, row) in enumerate(gdf.iterrows()):
#         current_geom = row["geometry"]
#         current_node = nodes[idx]

#         # Use a buffer to find nearby segments (more efficient than checking all)
#         buffered_geom = current_geom.buffer(distance_threshold)  # 1 meter buffer
#         possible_matches_index = list(spatial_index.intersection(buffered_geom.bounds))

#         for match_idx in possible_matches_index:
#             if match_idx == idx:  # Skip self
#                 continue

#             match_row = gdf.iloc[match_idx]
#             match_geom = match_row["geometry"]
#             match_node = nodes[match_idx]

#             # Check if geometries intersect or are close
#             if (current_geom.intersects(match_geom) or
#                 current_geom.distance(match_geom) < distance_threshold):
#                 if match_node not in current_node.neighbors:
#                     current_node.neighbors.append(match_node)
#                     total_neighbors += 1

#     print(f"Built {total_neighbors} geometric neighbor relationships for {len(nodes)} nodes")
#     nodes_with_neighbors = sum(1 for node in nodes if node.neighbors)
#     print(f"{nodes_with_neighbors} nodes have at least one geometric neighbor")

#     return nodes

# def algo_a_star(start_node: Node, end_coords: tuple[float, float], timeout_seconds=10) -> tuple:
#     start_time = time.time()
#     open_set = []
#     heapq.heappush(open_set, (0, start_node))
#     came_from   : dict[Node, None] = {start_node: None}
#     g_score     : dict[Node, float] = {start_node: 0}
#     f_score     : dict[Node, float] = {start_node: combined_heuristics(start_node, end_coords)}
#     closed_set = set()
#     iterations = 0

#     while open_set:
#         iterations += 1
#         # Timeout check
#         if time.time() - start_time > timeout_seconds:
#             print(f"  A* timeout after {iterations} iterations")
#             return None, float('inf')

#         current_f, current = heapq.heappop(open_set)
#         if current in closed_set:
#             continue

#         closed_set.add(current)
#         target_node: Node = Node("TARGET", "TARGET", end_coords, 0)

#         # Check distance to target
#         distance_to_target = haversine_distance(current, target_node)
#         if distance_to_target < 0.1:  # 100 meters
#             path = []
#             total_time: float = g_score[current]
#             while current:
#                 path.append(current)
#                 current = came_from[current]
#             print(f"  Found route! Distance to target: {distance_to_target:.3f} km, Iterations: {iterations}")
#             return path[::-1], total_time

#         # Progress tracking
#         if iterations % 1000 == 0:
#             print(f"  Iteration {iterations}: Open set: {len(open_set)}, Closed set: {len(closed_set)}")

#         for neighbor in current.neighbors:
#             if neighbor in closed_set:
#                 continue

#             temp_g_score: float = g_score[current] + neighbor.weight
#             if neighbor not in g_score or temp_g_score < g_score[neighbor]:
#                 came_from[neighbor] = current
#                 g_score[neighbor] = temp_g_score
#                 f_score[neighbor] = temp_g_score + combined_heuristics(neighbor, end_coords)
#                 heapq.heappush(open_set, (f_score[neighbor], neighbor))

#     print(f"  No path found after {iterations} iterations")
#     return None, float('inf')

# def get_top_n_routes(coordinate_details: dict, node_kdtree: NodeKDTree, top_n=3):
    start_coords = coordinate_details["start"]
    end_coords = coordinate_details["end"]
    nearby_start_nodes, start_distances = node_kdtree.find_nearest_neighbors(start_coords, k=top_n)

    print(f"Found {len(nearby_start_nodes)} start nodes near {start_coords}")
    for i, (node, dist) in enumerate(zip(nearby_start_nodes, start_distances)):
        print(f"  Start node {i+1}: {node.ename} (ID: {node.street_id}) - Distance: {dist:.6f} km")
        print(f"    This node has {len(node.neighbors)} neighbors")
        if node.neighbors:
            for neighbor in node.neighbors[:3]:  # Show first 3 neighbors
                neighbor_dist = haversine_distance(node, neighbor)
                print(f"      Neighbor: {neighbor.ename} - Distance: {neighbor_dist:.6f} km")

    routes = []
    for i, start_node in enumerate(nearby_start_nodes):
        print(f"Running A* from start node {i+1}...")
        path, total_time = algo_a_star(start_node, end_coords, timeout_seconds=15)
        if path:
            print(f"  Found route with {len(path)} segments, time: {total_time*60:.2f} min")
            routes.append({
                'path': path,
                'total_time_hours': total_time,
                'start_node': start_node,
                'end_coords': end_coords
            })
        else:
            print("  No route found from this start node")

    routes.sort(key=lambda x: x['total_time_hours'])
    return routes[:top_n]




class GDFParser:
    def __init__(self, gdf):
        self.gdf = gdf

class Node:
    def __init__(self, street_data, ename, cname, elevation, st_code, exit_num, route_num, remarks, route_id, travel_direction, cre_date, last_upd_date_v, alias_ename, alias_cname, shape_length, geometry, speed_limit):
        self.street_data = street_data
        self.ename: str = ename
        self.cname: str = cname
        self.elevation: int = elevation
        self.st_code: float = st_code
        self.exit_num: int | None = exit_num
        self.route_num: int | None = route_num
        self.remarks: str | None = remarks
        self.route_id: int = route_id
        self.travel_direction: int = travel_direction
        self.cre_date: str = cre_date
        self.last_upd_date_v: str = last_upd_date_v
        self.alias_ename: str | None = alias_ename
        self.alias_cname: str | None = alias_cname
        self.shape_length: float = shape_length
        self.shape_length_km: float = self.shape_length / 1000
        self.geometry = geometry
        self.speed_limit: int = speed_limit
        self.average_speed: float = float(self.speed_limit * 0.9)
        self.wgs84_geometry: list[tuple] = []
        self.geometry_start_point = (0.0, 0.0)
        self.heuristic: float = self.shape_length_km / 110.0

        if self.ename == '-99':
            self._correct_street_names()

    def _print(self, label):
        print(f"{label}: {self.ename} - {self.shape_length}, h: {self.heuristic}")

    def _correct_street_names(self):
        self.wgs84_geometry: list[tuple] = convert_mls_crs(self.geometry)
        self.geometry_start_point = self.wgs84_geometry[0]
        reverse_lookup_result = osm_reverse_lookup(self.wgs84_geometry[0])
        used_gmaps = False
        if reverse_lookup_result:
            new_street_names = reverse_lookup_result["name"]
            new_ename, new_cname = get_lang_name(new_street_names)

            is_valid_road = any(road_word in new_ename for road_word in ["Street", "Road", "Drive", "Terrace", "Highway", "Tunnel", "Expressway", "Bypass", "Avenue"])

            unwanted_terms = any(word in new_ename for word in ["Flyover"])

            if (isinstance(new_ename, int) or
                new_ename == "" or
                new_ename == " " or
                not is_valid_road or
                unwanted_terms or
                ", " in new_ename or
                new_ename.strip().isdigit()):

                used_gmaps = True
                gm_results = gm_reverse_lookup(self.wgs84_geometry[0])

                new_ename = None
                new_cname = None
                for gm_result in gm_results:
                    address_components = gm_result["address_components"]
                    for address_component in address_components:
                        if len(address_component["types"]) == 1 and address_component["types"][0] == 'route' and not contains_chinese(address_component["long_name"]):
                            new_ename = address_component["long_name"]
                            new_cname = address_component["short_name"]
                            if new_cname.strip().isdigit():
                                new_cname = new_ename.replace("Road", "Rd")
                            break
                    if new_ename:
                        break

            self.ename = new_ename
            self.cname = new_cname
        add_gmaps_label = "(GM) " if used_gmaps else ""
        print(f"Node: {add_gmaps_label}Corrected names: {self.ename} - {self.geometry_start_point}")


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

    parsed_gdf_road = parse_gdb_files(dataset_dirpath, USER_HEADERS, "CENTERLINE")

    gdf_nodes = []
    for idx, street in parsed_gdf_road.iterrows():
        street_ename = street["STREET_ENAME"]
        expressway_limits = get_expressway_limits(USER_HEADERS)
        speed_limit = 50
        if street_ename in expressway_limits:
            speed_limit = expressway_limits[street_ename]

        gdf_nodes.append(Node(street, street_ename, street["STREET_CNAME"], street["ELEVATION"], street["ST_CODE"], street["EXIT_NUM"], street["ROUTE_NUM"], street["REMARKS"], street["ROUTE_ID"], street["TRAVEL_DIRECTION"], street["CRE_DATE"], street["LAST_UPD_DATE_V"], street["ALIAS_ENAME"], street["ALIAS_CNAME"], street["SHAPE_Length"], street["geometry"], speed_limit))

    remove_filepath('gdf_nodes.json')
    with open('gdf_nodes.json', 'w', encoding='utf-8') as nodes_f:
        json.dump(gdf_nodes, nodes_f, ensure_ascii=False, indent=2)


    # for idx, street in parsed_gdf_road.iterrows():
    #     print(street)


    # parsed_gdf_traffic = parse_gdb_files(input_dirpath=dataset_dirpath, specified_name="TRAFFIC_FEATURES")

    # gdf_traffic_with_wgs84 = convert_epsg_to_wgs84(parsed_gdf_traffic)
    # traffic_light_locations: list = parse_traffic_light_locations(gdf_traffic_with_wgs84)
    # for tl in traffic_light_locations:
    #     print(tl.node_id, tl.coordinates)

    # gdf_after_getting_segment_data, segment_data = get_segment_data(parsed_gdf_road)

    # gdf_with_speed = get_average_speed_by_street(gdf_after_getting_segment_data, headers=USER_HEADERS)

    # gdf_with_weight = get_gdf_with_weight(gdf_with_speed)

    # gdf_with_wgs84 = convert_epsg_to_wgs84(gdf_with_weight)

    # nodes_from_gdf: list[Node] = get_nodes_from_gdf(gdf_with_wgs84)
    # nodes_from_gdf = build_neighbors_based_on_geometry(gdf_with_wgs84, nodes_from_gdf)
    # node_kdtree = NodeKDTree(nodes_from_gdf)

    # address look up
    start_address: str = "2 lung pak street"
    end_address: str = "89 pok fu lam road"
    # osm_start_options: json = osm_address_lookup(request_address=start_address)
    # osm_end_options: json = osm_address_lookup(request_address=end_address)
    # osm_start: dict = get_chosen_option(osm_start_options)
    # time.sleep(2)
    # osm_end: dict = get_chosen_option(osm_end_options)
    # coordinate_details: dict = get_coordinate_details(osm_start, osm_end)

    # hardcode coordinate details for now
    coordinate_details = {
        "start": [22.3642146, 114.1794265],
        "end": [22.2853336, 114.1330190]
    }

    # algorithm
    # top_n_routes = get_top_n_routes(coordinate_details, node_kdtree, top_n=3)
    # for idx, route in enumerate(top_n_routes, 1):
    #     print(f"\nRoute #{idx}:")
    #     print(f"Total time: {route['total_time_hours'] * 60:.2f} minutes")
    #     print(f"Number of segments: {len(route['path'])}")
    #     print("Segments:")

    #     for jdx, node in enumerate(route['path'], 1):
    #         lat, lon = node.coordinates
    #         print(f"  {jdx}. {node.ename} (ID: {node.street_id}) - Lat: {lat:.6f}, Lon: {lon:.6f}")

    # figure out -99


if __name__ == "__main__":
    main()