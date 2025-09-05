# Program goal: Find the top N fastest routes between two points

import os, fiona, json, requests, re, time, googlemaps, heapq, shutil, ast, urllib.request, warnings, zipfile, glob
import shapely.geometry as geometry
from shapely.geometry import MultiLineString
from pyproj import Transformer
from bs4 import BeautifulSoup
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
from colorama import Fore as CFore
from colorama import Style as CStyle
import haversine.haversine as haversine
from tqdm import tqdm

start_time = time.time()
def timer_decorator(start_time):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"Time elapsed: {elapsed:.4f}s")
            return result
        return wrapper
    return decorator


# remove weird pyogrio warnings
warnings.filterwarnings(
    "ignore",
    message="Measured.*geometry types are not supported",
    category=UserWarning,
    module="pyogrio.raw"
)


"""----------
    UTILS
----------"""
@timer_decorator(start_time)
def remove_filepath(filepath: str) -> None:
    if os.path.exists(filepath):
        if os.path.isfile(filepath):
            os.remove(filepath)
            print(f"Removed filepath '{filepath}'.")
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
            print(f"Removed directory '{filepath}'.")

def color_msg(color: str, msg: str) -> str:
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

@timer_decorator(start_time)
def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        print("GM Api Key Received.")
        return key_f.readline().strip()


"""-------------------
   COORDINATE STUFF
-------------------"""
@timer_decorator(start_time)
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


"""----------
   OSM GDF
----------"""
warnings.filterwarnings("ignore", message="Non closed ring detected", category=RuntimeWarning, module="pyogrio.raw")

@timer_decorator(start_time)
def get_pbf_filepath() -> str:
    file_to_parse = None
    dataset_dirpath = os.path.join(os.getcwd(), 'dataset')
    for filepath in os.listdir(dataset_dirpath):
        if filepath.startswith('hong-kong') and filepath.endswith('.pbf'):
            file_to_parse = os.path.join(dataset_dirpath, filepath)
    print("Found PBF file to parse.")
    return file_to_parse

@timer_decorator(start_time)
def get_all_gdf_data(filepath: str, specified_layer=None) -> gpd.geodataframe.GeoDataFrame:
    if os.path.isdir(filepath):
        if specified_layer is not None:
            gdb_path = os.path.join(filepath, "RdNet_IRNP.gdb")
            if os.path.exists(gdb_path):
                filepath = gdb_path
        else:
            pbf_pattern = os.path.join(filepath, 'hong-kong-*.osm.pbf')
            pbf_files = glob.glob(pbf_pattern)
            filepath = pbf_files[0]
            print(f"Using OSM file: {os.path.basename(filepath)}")

    if specified_layer is not None:
        layer_gdf = gpd.read_file(filepath, layer=specified_layer)
        return layer_gdf

    layers = fiona.listlayers(filepath)
    gdf_data = {}
    for layer in layers:
        layer_gdf = gpd.read_file(filepath, layer=layer)
        gdf_data[layer] = layer_gdf
    print(f"All GDF data extracted. {len(layers)} layers found.")
    return gdf_data

class cKDTree:
    def __init__(self, osm_gdf, api_key, headers, gov_gdf, top_n):
        self.osm_gdf = osm_gdf
        self.gov_gdf = gov_gdf
        self.api_key = api_key
        self.headers = headers
        self.top_n = top_n

        self.tree = None
        self.fastest_routes = []

        if osm_gdf:
            self.tree = self.build_tree()
            

        # if self.tree:
        #     self.fastest_routes = self.find_fastest_routes()


    def build_tree(self):
        self.line_way_nodes: list[dict] = self.osm_gdf.get_line_nodes_ways()
        return

    def find_fastest_routes(self) -> list:
        # raise NotImplementedError("find_fastest_routes func() not implemented")
        return

    def astar_algorithm(self) -> list:
        # raise NotImplementedError("astar_algorithm func() not implemented")
        return

class OSM_GDF:
    def __init__(self, gdf_data, api_key, headers, gov_gdf, expressway_limits, start_time):
        self.gdf_data = gdf_data
        self.api_key = api_key
        self.headers = headers
        self.gov_gdf = gov_gdf
        self.expressway_limits = expressway_limits
        self.start_time = start_time
        self.crs_transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)

        # self.points_gdf = gdf_data['points']
        self.lines_gdf = gdf_data["lines"]
        # self.mls_gdf = gdf_data["multilinestrings"]
        # self.mpg_gdf = gdf_data["multipolygons"]
        # self.other_gdf = gdf_data["other_relations"]

        self.lines_nodes = []
        self.lines_nodes_ways = []
        if not self.lines_gdf.empty:
            self.create_line_nodes()

        print(f"# of line nodes {len(self.lines_nodes)}\n")

    @timer_decorator(start_time)
    def create_line_nodes(self):
        print("Creating Line Nodes...")
        for idx, line in tqdm(self.lines_gdf.iterrows(), total=len(self.lines_gdf), desc="Creating line nodes"):
            if line["name"] is not None:
                line_node = OSM_LineNode(
                    self.api_key, self.headers, self.gov_gdf, self.expressway_limits, self.crs_transformer, line["osm_id"], line["name"], line["highway"], line["waterway"], line["aerialway"], line["barrier"], line["man_made"], line["railway"], line["z_order"], line["other_tags"], line["geometry"]
                )
                self.lines_nodes.append(line_node.get_node_data())
                self.lines_nodes_ways.append(line_node.get_way_data())
        print("All Line Nodes Created.")

    def get_line_nodes_ways(self) -> list:
        """ returns a list of tuple(first_coord, last_coord) of each line_node """
        return self.lines_nodes_ways

    def get_line_nodes(self) -> list:
        """ Returns a list of line_nodes"""
        return self.lines_nodes
    # def gm_reverse_lookup(self, coord_tuple: tuple[float, float]) -> json:
    #     gmaps = googlemaps.Client(key=self.api_key)
    #     lat, lon = coord_tuple
    #     reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
    #     if not reverse_geocode_result:
    #         return "No results found for these coordinates."
    #     return reverse_geocode_result

    # def get_elevation(self, coords_tuple: tuple[float, float]) -> float:
    #     lat, lon = coords_tuple
    #     format_lat: float = round(lat, 5)
    #     format_lon: float = round(lon, 5)
    #     format_coords = f"{format_lat}, {format_lon}"
    #     params = {
    #         'locations': format_coords,
    #         'key': self.api_key
    #     }
    #     url = "https://maps.googleapis.com/maps/api/elevation/json"
    #     headers = {
    #         "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
    #         'Accept': 'application/json, text/plain, */*',
    #     }
    #     res = requests.get(url=url, headers=headers, params=params)

    #     if res.status_code != 200:
    #         res.raise_for_status()
    #         return

    #     result = res.json()
    #     elevation = result['results'][0]['elevation']
    #     return elevation

    # def _is_text_chinese(self, text: str) -> bool:
    #     chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    #     return bool(chinese_pattern.search(text))

class OSM_LineNode:
    def __init__(self, api_key, headers, gov_gdf, expressway_limits, crs_transformer, osm_id: int, name: str, highway: str, waterway: str, aerialway: str, barrier: str, man_made: str, railway: str, z_order: int, other_tags, geometry):
        self.api_key = api_key
        self.headers = headers
        self.gov_gdf = gov_gdf
        self.expressway_limits = expressway_limits
        self.crs_transformer = crs_transformer

        self.osm_id = osm_id
        self.name = name
        self.highway = highway
        self.waterway = waterway
        self.aerialway = aerialway
        self.barrier = barrier
        self.man_made = man_made
        self.railway = railway
        self.z_order = z_order
        self.other_tags = other_tags
        self.other_tags_data = {}
        self.geometry = geometry
        self.coord_lst = self.get_coords()
        self.first_coords = self.coord_lst[0]
        self.last_coords = self.coord_lst[-1]
        self.line_distance_km = haversine(self.first_coords, self.last_coords)
        self.maxspeed = 0.0
        self.avespeed = 0.0
        self.weight = 0.0

        if self.other_tags:
            self.other_tags_data = self.get_other_tags_dict()

        self.en_name = self.other_tags_data.get("name:en", None)
        self.zh_name = self.other_tags_data.get("name:zh", None)
        self.minspeed = self.other_tags_data.get("minspeed", None)
        self.maxspeed = self.other_tags_data.get("maxspeed", None)
        self.is_oneway = self.other_tags_data.get("oneway", None)
        self.is_street = True if self.other_tags_data.get("street") == 'yes' else False
        self.lanes = self.other_tags_data.get("lanes", None)

        if self.maxspeed == 0.0 or not self.maxspeed:
            self.maxspeed = self.get_expressway_speed()

        if not isinstance(self.maxspeed, float):
            self.maxspeed = float(self.maxspeed)
        else:
            self.avespeed = self.maxspeed * 0.9

        while self.first_coords == self.last_coords:
            self.line_distance_km = self.query_gov_gdf(self.name)

        if self.avespeed != 0.0 and self.line_distance_km is not None:
            self.weight = self.line_distance_km / self.avespeed

    def convert_mls(self, giv_geometry) -> list[tuple[float, float]]:
        """ Converts a MultiLineString of EPSG:2326 Coordinates into a list of coordinates in WGS84 Format """
        if isinstance(giv_geometry, MultiLineString):
            all_coords = []
            for line_string in giv_geometry.geoms:
                for coord in line_string.coords:
                    eing, ning = coord
                    if eing is not None and ning is not None:
                        lon, lat = self.crs_transformer.transform(eing, ning)
                        all_coords.append((lat, lon))
            return all_coords

    def query_gov_gdf(self, address: str) -> float:
        """ Find street in government data and return its length in km. """
        for idx, street in self.gov_gdf.iterrows():
            street_ename = street["STREET_ENAME"]
            if street_ename and address.lower() in street_ename.lower():
                shape_length = street["SHAPE_Length"]
                if shape_length is not None:
                    return float(shape_length / 1000)

                self.coord_lst = self.convert_mls(street["geometry"])
                self.first_coords = self.coord_lst[0]
                self.last_coords = self.coord_lst[-1]
                h_dis = haversine(self.first_coords, self.last_coords)
                return float(h_dis)

    def get_expressway_speed(self) -> float:
        """ Gets speed limit based on whether street's english name is in self.expressway_limits or not. """
        if self.en_name not in self.expressway_limits:
            return 50.0
        return self.expressway_limits[self.en_name]

    def get_other_tags_dict(self) -> dict:
        """ Parses the other_tags attribute into a dictionary for ease of access. """
        tag_content = self.other_tags.strip()
        tag_content = re.sub(r'\s+', ' ', tag_content)
        tag_dict = '{' + tag_content.replace('=>', ':') + '}'
        parsed_tag_dict = ast.literal_eval(tag_dict)

        result = {}
        for key, value in parsed_tag_dict.items():
            if isinstance(value, (int, float)):
                result[key] = float(value)
            else:
                result[key] = value
        return result

    def _print(self):
        print(f"\n{self.name} - [{self.first_coords} ~ {self.last_coords}]")
        print(f"distance: {self.line_distance_km}km")
        for key, value in self.other_tags_data.items():
            print(f"Tag '{key}': {value} {type(value)}")

    def get_node_data(self) -> dict:
        return {
            "osm_id": self.osm_id,
            "name": self.name,
            "en_name": self.en_name,
            "zh_name": self.zh_name,
            "is_highway": self.highway,
            "is_waterway": self.waterway,
            "is_aerialway": self.aerialway,
            "is_barrier": self.barrier,
            "is_man_made": self.man_made,
            "is_railway": self.railway,
            "is_oneway": self.is_oneway,
            "is_street": self.is_street,
            "z_order": self.z_order,
            "coords_lst": self.coord_lst,
            "first_coords": self.first_coords,
            "last_coords": self.last_coords,
            "node_length_km": self.line_distance_km,
            "lanes": self.lanes,
            "maxspeed": self.maxspeed,
            "minspeed": self.minspeed,
            "avespeed": self.avespeed,
            "weight": self.weight,
        }

        # print(f"{self.name} {self.first_coords} { self.last_coords} [{self.line_distance_km}]")

    def get_way_data(self) -> dict:
        """ Returns way data: {en_name: {first: first_coords, last: last_coords, weight: weight}} """
        return {self.en_name: {'first': self.first_coords, 'last': self.last_coords, 'weight': self.weight}}

    def get_coords(self) -> list[tuple[float, float]]:
        """ Takes geometry and converts it into a list of coordinates. (assumes geometry is already in wgs84 format) """
        coords = []
        for coord in self.geometry.coords:
            lon, lat = coord
            result = (lat, lon)
            coords.append(result)
        return coords


"""----------""
   GOV GDF
----------"""
@timer_decorator(start_time)
def get_latest_road_network(dataset_filepath: str) -> str:
    filename = "RdNet_IRNP.gdb"
    gdb_filepath = os.path.join(dataset_filepath, filename)

    if os.path.exists(gdb_filepath):
        print("Road network dataset already exists, redownload is not needed.")
        return gdb_filepath

    download_url: str = "https://static.data.gov.hk/td/road-network-v2/RdNet_IRNP.gdb.zip"
    os.makedirs(dataset_filepath, exist_ok=True)

    dataset_zip_path: str = os.path.join(dataset_filepath, "dataset.zip")
    urllib.request.urlretrieve(download_url, dataset_zip_path)

    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_f:
        zip_f.extractall(dataset_filepath)

    os.remove(dataset_zip_path)
    print("Road network dataset downloaded and extracted.")
    return gdb_filepath

@timer_decorator(start_time)
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
            name_cell = cells[0]
            name_link = name_cell.find('a', title=True)
            expressway_name = name_link.get_text(strip=True) if name_link else name_cell.get_text(strip=True)
            speed_text = cells[2].get_text(strip=True)
            numbers = re.findall(r'\d+', speed_text)
            if numbers:
                if len(numbers) == 1:
                    speed_limit = float(numbers[0])
                else:
                    speed_limit = float(max(numbers))

                expressway_limits[expressway_name] = speed_limit

    return expressway_limits

def main():
    USER_HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    gm_api_key: str = get_gm_api_key()
    output_fp: str = os.path.join(os.getcwd(), 'output')
    pbf_file_path: str = get_pbf_filepath()
    gdf_data = get_all_gdf_data(pbf_file_path)
    print("got gdf data for pbf file")

    dataset_fp: str = os.path.join(os.getcwd(), 'dataset')
    gov_gdf_path: str = get_latest_road_network(dataset_fp)
    gov_gdf = get_all_gdf_data(gov_gdf_path, "CENTERLINE")
    print('got gdf data for gov gdb')
    expressway_limits = get_expressway_limits(USER_HEADERS)
    print()
    print("-----")
    osm_gdf = OSM_GDF(gdf_data, gm_api_key, USER_HEADERS, gov_gdf, expressway_limits, start_time)
    line_nodes_ckdtree = cKDTree(osm_gdf, gm_api_key, USER_HEADERS, gov_gdf, top_n=5)


    # dev

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
    # coordinate_details = {
    #     "start": [22.3642146, 114.1794265],
    #     "end": [22.2853336, 114.1330190]
    # }

    # algorithm



if __name__ == "__main__":
    main()