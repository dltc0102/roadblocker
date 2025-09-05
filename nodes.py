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
import haversine.haversine as haversine_calc
from tqdm import tqdm

def get_nodes_json_fp():
    return os.path.join(os.getcwd(), 'output', 'line_way_nodes.json')

def read_json_file(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def haversine_coords(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    return haversine_calc(coord1, coord2)

def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        print("GM Api Key Received.")
        return key_f.readline().strip()

class node_cKDTree:
    def __init__(self, nodes_data, api_key, headers, top_n):
        self.nodes_data = nodes_data
        self.api_key            : str = api_key
        self.headers            : dict = headers
        self.top_n              : int = top_n
        self.start_point        : float[tuple, tuple] = (22.363929920293774, 114.17929436609448)
        self.target_point       : float[tuple, tuple] = (22.284566183373638, 114.13251488622221)

        self.way_info           : list = []
        self.midpoints          : list = []
        self.midpoints_array    : np.array = np.array([])
        self.kdtree             : cKDTree = None
        self.fastest_routes     : list = []

        if self.nodes_data:
            self.midpoints_array: np.array = self.get_midpoints()

        if self.midpoints_array.size > 0:
            self.kdtree: cKDTree = self.build_tree()

        # run find_fastest_routes()
            # use astar_algorithm()
            # use find_k_nearest_neighbors()

    def get_midpoints(self):
        self.way_info = []
        self.midpoints = []
        for way in self.nodes_data:
            for way_name, way_data in way.items():
                first_lat, first_lon = way_data["first"]
                last_lat, last_lon = way_data["last"]
                mid_lat = (first_lat + last_lat) / 2
                mid_lon = (first_lon + last_lon) / 2
                self.midpoints.append([mid_lat, mid_lon])
                self.way_info.append((way_name, way_data))
        return np.array(self.midpoints)

    def build_tree(self):
        return cKDTree(self.midpoints_array)

    def find_k_nearest_neighbors(self):
        distances, indices = self.kdtree.query(self.query_point, self.top_n)
        for i, (distance, index) in enumerate(zip(distances, indices)):
            way_name, way_data = self.way_info[index]
            print(f"{i+1}. {way_name} - Distance: {distance:.6f} km")

    def find_fastest_routes(self) -> list:
        self.astar_algorithm()
        return

    def astar_algorithm(self) -> list:
        self.find_k_nearest_neighbors()
        return

def main():
    USER_HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    gm_api_key: str = get_gm_api_key()
    nodes_json_fp: str = get_nodes_json_fp()
    nodes_data = read_json_file(nodes_json_fp)
    node_cKDTree(nodes_data, gm_api_key, USER_HEADERS, top_n=5)

if __name__ == "__main__":
    main()