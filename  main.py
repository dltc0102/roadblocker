# Program goal: Find the top N fastest routes between two points

import os, zipfile, fiona, json, requests, re, time, math, googlemaps, heapq
import shapely.geometry.multilinestring as shapely_mls
from pyproj import Transformer
from bs4 import BeautifulSoup
import urllib.request
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
from colorama import Fore as CFore
from colorama import Style as CStyle



"""----------
    UTILS
----------"""
def remove_filepath(filepath: str) -> None:
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Removed filepath '{filepath}'.")

def color_msg(color, msg):
    colors = {
        'red': CFore.RED,
        'blue': CFore.BLUE,
        'purple': CFore.MAGENTA,
        'cyan': CFore.CYAN,
        'green': CFore.GREEN,
        'yellow': CFore.YELLOW,
        'black': CFore.BLACK,
        'white': CFore.WHITE
    }
    return f"{colors[color.lower()]}{msg}{CStyle.RESET_ALL}"

def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        return key_f.readline().strip()



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

def get_all_gdf_data(input_dirpath: str, headers: dict) -> gpd.geodataframe.GeoDataFrame:
    gdb_dirs: list = [file for file in os.listdir(input_dirpath) if file.endswith('.gdb')]
    if not gdb_dirs:
        raise FileNotFoundError("No GDB directory found in the dataset folder")

    gdb_dir         : str = gdb_dirs[0]
    gdb_filepath    : str = os.path.join(input_dirpath, gdb_dir)
    layers          : list = fiona.listlayers(gdb_filepath)
    layer_wanted    : str = None
    # print(layers)

    gdf_data = {}
    for layer in layers:
        layer_gdf: gpd.geodataframe.GeoDataFrame = gpd.read_file(gdb_filepath, layer=layer)
        gdf_data[layer] = layer_gdf

    return gdf_data

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
    # new_gdf = gdf.copy()
    # new_gdf["weight"] = 0.0

    # for idx, street in new_gdf.iterrows():
    #     street_ave_speed = float(street["average_speed"]) # km/hr
    #     street_length_km = float(street["SHAPE_Length"] / 1000)
    #     weight: float = street_length_km / street_ave_speed
    #     new_gdf.at[idx, "weight"] = weight

    # print("GDF: Node's Weight column added. (time in hours)")
    # return new_gdf



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

class GMAddress:
    def __init__(self, api_key: str, long_name: str, short_name: str, road_type: str, category: str, coords: tuple[float, float]):
        self.api_key = api_key
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
        params = {
            'locations': format_coords,
            'key': self.api_key
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

class Node:
    def __init__(self, api_key: str, request_headers: dict, ename: str, cname: str, elevation: int, st_code: float, exit_num: int | None, route_num: int | None, remarks: str | None, route_id: int, travel_direction: int, cre_date: str, last_upd_date_v: str, alias_ename: str | None, alias_cname: str | None, shape_length: float, geometry: shapely_mls.MultiLineString, speed_limit: int):
        self.api_key                : str = api_key
        self.request_headers        : dict = request_headers
        self.ename                  : str = self._regulate_dashes(ename)
        self.cname                  : str = self._regulate_dashes(cname)
        self.elevation              : int = elevation
        self.st_code                : float = st_code
        self.exit_num               : int | None = exit_num
        self.route_num              : int | None = route_num
        self.remarks                : str | None = remarks
        self.route_id               : int = route_id
        self.travel_direction       : int = travel_direction
        # if travel_direction == 1: two way
        # if travel_direction == 3: one way
        self.cre_date               : str = cre_date
        self.last_upd_date_v        : str = last_upd_date_v
        self.alias_ename            : str | None = alias_ename
        self.alias_cname            : str | None = alias_cname
        self.shape_length           : float = shape_length
        self.shape_length_km        : float = self.shape_length / 1000
        self.geometry               : shapely_mls.MultiLineString = geometry
        self.speed_limit            : int = speed_limit
        self.average_speed          : float = None
        self.wgs84_geometry         : list[tuple] = self._convert_mls_crs()
        self.geometry_start_point   : tuple = self.wgs84_geometry[0]
        self.weight                 : float = self.shape_length_km / self.average_speed
        self.road_category          : str = ""
        self.road_type              : str = ""
        self.node_dict              : dict = None
        self.english_road_types     : list = ["Street", "Road", "Drive", "Terrace", "Highway", "Tunnel", "Expressway", "Bypass", "Avenue", "Boulevard"]
        self.chinese_road_types     : list = ["街道", "街", "隧道", "道", "公路"]

        if self.geometry_start_point:
            self.elevation = self.get_elevation(self.geometry_start_point)

        if (self.ename == '-99' or self.ename == None or '-' in self.ename) and self.elevation != 0:
            self.correct_street_names()
        else:
            self._print()

        if not self._is_valid_ename() and self.road_category != "" and self.road_type != '':
            self.create_node_dict()

        if self._is_valid_ename() and self.speed_limit is None:
            express_way_limits: dict = self.get_expressway_limits()
            if self.ename in express_way_limits:
                self.speed_limit = express_way_limits[self.ename]
            else:
                self.speed_limit = 50

        if self.speed_limit is not None:
            self.average_speed = float(self.speed_limit * 0.9)

    def _is_valid_ename(self) -> bool:
        return (self.ename != '-99' or self.ename != None or not '-' in self.ename or self.ename != '' or self.ename != ' ')

    def create_node_dict(self) -> dict:
        return {
            "ename": self.ename,
            "cname": self.cname,
            "elevation": self.elevation,
            "st_code": self.st_code,
            "exit_num": self.exit_num,
            "route_num": self.route_num,
            "remarks": self.remarks,
            "route_id": self.route_id,
            "travel_direction": self.travel_direction,
            "cre_date": self.cre_date,
            "last_upd_date_v": self.last_upd_date_v,
            "alias_ename": self.alias_ename,
            "alias_cname": self.alias_cname,
            "shape_length": self.shape_length,
            "shape_length_km": self.shape_length_km,
            "speed_limit": self.speed_limit,
            "average_speed": self.average_speed,
            "geometry_start_point": self.geometry_start_point,
            "weight": self.weight,
            "road_category": self.road_category,
            "road_type": self.road_type
        }

    def get_node_dict(self) -> dict:
        return self.node_dict

    def _print(self, color: str = "green") -> None:
        statement = color_msg(color, f"ename: {self.ename}, cname: {self.cname}, {self.geometry_start_point}")
        print(statement)

    def _convert_mls_crs(self) -> list[tuple[float, float]]:
        mls_str = self.geometry
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

    def _regulate_dashes(self, name_str: str) -> str:
        if not name_str: return name_str
        dash_chars = '\u002D\u2010\u2011\u2012\u2013\u2014\u2015\u2053\u207B\u208B\u2212\u2E17\u2E1A\u2E3A\u2E3B\uFE58\uFE63\uFF0D\uFF5E'
        if not any(char in name_str for char in dash_chars): return name_str
        dash_translation = str.maketrans(dash_chars, '-' * len(dash_chars))
        return name_str.translate(dash_translation)

    def _is_valid_road_eng(self, name_str: str) -> bool:
        return any(word in name_str for word in self.english_road_types)

    def _is_valid_road_cn(self, name_str: str) -> bool:
        return any(word in name_str for word in self.chinese_road_types)

    def _is_text_chinese(self, text: str) -> bool:
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(chinese_pattern.search(text))

    def _extract_english_name(self, text: str) -> str:
        if not text: return ""
        if '-' in text:
            parts = [part.strip() for part in text.split('-')]
            for part in parts:
                if any(road_type in part for road_type in self.english_road_types): return part
                if part and part[0].isascii(): return part
            return parts[0] if parts else ""
        return text.strip()

    def _extract_chinese_name(self, text: str) -> str:
        if not text: return ""
        if '-' in text:
            parts = [part.strip() for part in text.split('-')]
            for part in parts:
                if any(road_type in part for road_type in self.chinese_road_types): return part
                if any('\u4e00' <= char <= '\u9fff' for char in part): return part
            return parts[0] if parts else ""
        return text.strip()

    def get_expressway_limits(self) -> dict | None:
        url = "https://en.wikipedia.org/wiki/List_of_streets_and_roads_in_Hong_Kong"
        res = requests.get(url, headers=self.request_headers)
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
                name_cell = cells[0]
                name_link = name_cell.find('a', title=True)
                expressway_name = name_link.get_text(strip=True) if name_link else name_cell.get_text(strip=True)
                speed_text = cells[2].get_text(strip=True)
                numbers = re.findall(r'\d+', speed_text)
                if numbers:
                    if len(numbers) == 1:
                        speed_limit = int(numbers[0])
                    else:
                        speed_limit = tuple(int(num) for num in numbers)

                    expressway_limits[expressway_name] = speed_limit

        return expressway_limits

    def osm_reverse_lookup(self, coord_tuple: tuple[float, float]) -> json:
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

    def gm_reverse_lookup(self, coord_tuple: tuple[float, float]) -> json:
        apikey = get_gm_api_key()
        gmaps = googlemaps.Client(key=apikey)
        lat, lon = coord_tuple
        reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
        if not reverse_geocode_result:
            return "No results found for these coordinates."
        return reverse_geocode_result

    def gm_address_lookup(self) -> GMAddress | None:
        results = self.gm_reverse_lookup(self.geometry_start_point)
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
                    gm_address_nodes.append(
                        GMAddress(self.api_key, long_name, short_name, comp_type, self.road_category, coordinates)
                    )
        valid_nodes = [node for node in gm_address_nodes if node.elevation is not None]

        if not valid_nodes:
            print("no gm address valid nodes.")
            return None

        if self.road_category == 'highway':
            return max(valid_nodes, key=lambda x: x.elevation)
        else:
            return min(valid_nodes, key=lambda x: x.elevation)

    def get_lang_names_v4(self, name_str: str) -> tuple[str, str]:
        dash_name_str = self._regulate_dashes(name_str)
        dash_positions = [i for i, char in enumerate(dash_name_str) if char == '-']

        if not dash_positions:
            chinese_matches = re.findall(r'[\u4e00-\u9fff]+', dash_name_str)
            chinese_name = ' '.join(chinese_matches) if chinese_matches else ""
            english_name = re.sub(r'[\u4e00-\u9fff]+', '', dash_name_str).strip()
            return english_name, chinese_name

        space_positions = [i for i, char in enumerate(dash_name_str) if char == ' ']
        first_char = dash_name_str[0]
        left_is_chinese = self._is_text_chinese(first_char)

        english_part = None
        chinese_part = None
        if left_is_chinese:
            # right is english
            # 青山公路-青龍頭段 Castle Peak Road-Tsing Lung Tau
            chinese_part = dash_name_str[:space_positions[0]].strip()
            english_part = dash_name_str[space_positions[0]+1:].strip()
        else:
            # left is english
            # Castle Peak Road-Tsing Lung Tau 青山公路-青龍頭段
            chinese_part = dash_name_str[space_positions[-1]:].strip()
            english_part = dash_name_str[:space_positions[-1]].strip()

        return self._extract_english_name(english_part), self._extract_chinese_name(chinese_part)

    def get_chinese_address_name(self, address="") -> str:
        given_address = self.ename if address == "" else address
        eng_addr_lookup = osm_address_lookup(given_address)[0]
        eng_addr_result = eng_addr_lookup
        try:
            eng_addr_result = eng_addr_lookup["address"]["name"]
        except KeyError:
            eng_addr_result = eng_addr_lookup["name"]

        ename, cname = self.get_lang_names_v4(eng_addr_result)
        return cname

    def get_elevation(self, coords_tuple: tuple[float, float]) -> float:
        lat, lon = coords_tuple
        format_lat: float = round(lat, 5)
        format_lon: float = round(lon, 5)
        format_coords = f"{format_lat}, {format_lon}"
        params = {
            'locations': format_coords,
            'key': self.api_key
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

    def correct_street_names(self):
        print("using correct_street_names()")
        print(f"current names: {self.ename}, {self.cname}, {self.geometry_start_point}")
        osm_reverse_result = self.osm_reverse_lookup(self.geometry_start_point)
        used_gmaps = False

        new_ename = None
        new_cname = None
        if osm_reverse_result:
            print("using osm api")
            self.road_category = osm_reverse_result["category"]
            self.road_type = osm_reverse_result["addresstype"]
            try:
                address_name: str = osm_reverse_result["address"]["road"]
            except KeyError:
                address_name: str = osm_reverse_result["name"]

            print(f"address_name: {address_name}")
            new_ename, new_cname = self.get_lang_names_v4(address_name)
            print(f"osm api: {new_ename}, {new_cname}")
            if new_ename != "" and new_cname != "":
                self.ename = new_ename.upper()
                if self._is_text_chinese(new_cname):
                    self.cname = new_cname
                else:
                    self.cname = self.get_chinese_address_name(new_ename)
                self._print("red")
                return

            else:
                print("using gm api")
                gm_address = self.gm_address_lookup()
                if gm_address:
                    print(f"gm api: {gm_address.long_name}, {gm_address.short_name}")
                    self.ename = gm_address.long_name
                    self.cname = gm_address.short_name
                    self._print("yellow")
                    return
        print(color_msg("blue", "all streets corrected"))


"""--------------
   NODE KDTREE
--------------"""
class NODEKDTree:
    def __init__(self, ename, cname, shape_length_km, average_speed, speed_limit, node_weight, wgs84_coords):
        self.ename = ename
        self.cname = cname
        self.shape_length_km = shape_length_km
        self.average_speed = average_speed
        self.speed_limit = speed_limit
        self.node_weight = node_weight
        self.wgs84_coords = wgs84_coords

class GDF:
    def __init__(self, all_gdf_data, gm_api_key, request_headers):
        self.gm_api_key = gm_api_key
        self.all_gdf_data = all_gdf_data
        self.request_headers = request_headers

        self.vehicle_restriciton_layer = self.all_gdf_data.get("VEHICLE_RESTRICTION", gpd.GeoDataFrame())
        self.traffic_features_layer = self.all_gdf_data.get("TRAFFIC_FEATURES", gpd.GeoDataFrame())
        self.speed_limit_layer = self.all_gdf_data.get("SPEED_LIMIT", gpd.GeoDataFrame())
        self.run_in_out_layer = self.all_gdf_data.get("RUN_IN_OUT", gpd.GeoDataFrame())
        self.roundabout_layer = self.all_gdf_data.get("ROUNDABOUT", gpd.GeoDataFrame())
        self.prohibition_layer = self.all_gdf_data.get("PROHIBITION", gpd.GeoDataFrame())
        self.permit_layer = self.all_gdf_data.get("PERMIT", gpd.GeoDataFrame())
        self.pedestrian_zone_layer = self.all_gdf_data.get("PEDESTRIAN_ZONE", gpd.GeoDataFrame())
        self.nsr_layer = self.all_gdf_data.get("NSR", gpd.GeoDataFrame())
        self.bus_only_lane_layer = self.all_gdf_data.get("BUS_ONLY_LANE", gpd.GeoDataFrame())
        self.centerline_layer = self.all_gdf_data.get("CENTERLINE", gpd.GeoDataFrame())
        self.turn_layer = self.all_gdf_data.get("TURN", gpd.GeoDataFrame())
        self.intersection_layer = self.all_gdf_data.get("INTERSECTION", gpd.GeoDataFrame())
        self.tun_bridge_toll_layer = self.all_gdf_data.get("TUN_BRIDGE_TOLL", gpd.GeoDataFrame())
        self.onstreetpark_layer = self.all_gdf_data.get("ONSTREETPARK", gpd.GeoDataFrame())
        self.gisp_on_street_parking_layer = self.all_gdf_data.get("GISP_ON_STREET_PARKING", gpd.GeoDataFrame())
        self.tun_bridge_tv_toll_layer = self.all_gdf_data.get("TUN_BRIDGE_TV_TOLL", gpd.GeoDataFrame())

        self.centerline_nodes = []
        self.centerline_node_dicts = []
        if not self.centerline_layer.empty:
            for idx, street in self.centerline_layer.iterrows():
                street_rd_id = street["ROUTE_ID"]
                speed_limit = self.search_within_layer(
                    self.speed_limit_layer,
                    "ROAD_ROUTE_ID",
                    street_rd_id,
                    "SPEED_LIMIT"
                )
                self.centerline_nodes.append(
                    Node(
                        self.gm_api_key, self.request_headers, street["STREET_ENAME"], street["STREET_CNAME"], street["ELEVATION"], street["ST_CODE"], street["EXIT_NUM"], street["ROUTE_NUM"], street["REMARKS"], street["ROUTE_ID"], street["TRAVEL_DIRECTION"], street["CRE_DATE"], street["LAST_UPD_DATE_V"], street["ALIAS_ENAME"], street["ALIAS_CNAME"], street["SHAPE_Length"], street["geometry"], speed_limit
                    )
                )

        if not self.centerline_nodes == []:
            for node in self.centerline_nodes:
                self.centerline_node_dicts.append(node.get_node_dict())

        if not self.centerline_node_dicts == []:
            ctl_node_json = "centerline_nodes.json"
            remove_filepath(ctl_node_json)
            with open(ctl_node_json, 'w', encoding='utf-8') as nodes_f:
                json.dump(self.centerline_node_dicts, nodes_f, ensure_ascii=False, indent=2, default=str)

    def search_within_layer(self, gdf_layer, search_column, search_value, return_column=None):
        if gdf_layer.empty:
            return None

        matches = gdf_layer[gdf_layer[search_column] == search_value]
        if matches.empty:
            return None

        if return_column:
            return matches[return_column].iloc[0]
        else:
            return matches.iloc[0]

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
    gm_api_key: str = get_gm_api_key()
    get_latest_road_network(input_dirpath=dataset_dirpath)
    all_gdf_data = get_all_gdf_data(dataset_dirpath, USER_HEADERS)
    GDF(all_gdf_data, gm_api_key, USER_HEADERS)

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



if __name__ == "__main__":
    main()