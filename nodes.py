import os, fiona, json, requests, re, time, googlemaps, heapq, shutil, ast, urllib.request, warnings, zipfile, glob
import shapely.geometry as geometry
from shapely.geometry import MultiLineString
from pyproj import Transformer
from bs4 import BeautifulSoup
import geopandas as gpd
from scipy.spatial import KDTree
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

class AlgoKDTree:
    def __init__(self, gov_gdf, start_coords_lst: list, node_data: dict, k: int):
        self.gov_gdf            : gpd.geodataframe.GeoDataFrame = gov_gdf
        self.start_coords_lst   : list = start_coords_lst
        self.kdtree             : KDTree = KDTree(start_coords_lst)
        self.node_data          : dict = node_data
        self.k                  : int = k

        if self.node_data:
            print(len(self.node_data.keys()))

    def get_heuristic(self, query_point: np.array, target_point: np.array) -> float:
        query_lat, query_lon = tuple((float(thing) for thing in query_point))
        try:
            queried_data = self.node_data[(query_lat, query_lon)]
        except KeyError:
            new_data = self.query_gov_gdf(query_point)

        end_point = queried_data["last"]
        h_dist = haversine_calc(end_point, target_point)
        heuristic = h_dist / 110.0
        print(heuristic, queried_data["weight"])
        return heuristic

    def get_k_neighbors(self, query_point: tuple[float, float], radius=1.0) -> list:
        distances, indices = self.kdtree.query(query_point, k=self.k, distance_upper_bound=radius)
        neighbors = []
        for i, dist in enumerate(zip(distances, indices)):
            if dist <= radius and not np.isinf(dist):
                neighbors.append((self.points[indices[i]], dist))
        return neighbors

    def find_path(self, start_point: tuple[float, float], target_point: tuple[float, float]) -> float | None:
        start_np = np.array(start_point)
        target_np = np.array(target_point)
        max_search_radius: float = 2.0
        open_set = [(self.get_heuristic(start_np, target_np), tuple(start_np, 0), 0, [tuple(start_np)])]
        closed_set = set()
        g_scores = {tuple(start_np): 0}
        f_scores = {tuple(start_np): self.get_heuristic(start_np, target_np)}

        while open_set:
            curr_f, curr, curr_g, curr_path = heapq.heappop(open_set)
            curr_np = np.array(curr)

            if tuple(curr_np) in closed_set:
                continue

            closed_set.add(tuple(curr_np))

            if np.allclose(curr_np, target_np, atol=0.1):
                return curr_path, curr_g

            neighbors = self.get_k_neighbors(curr_np, max_search_radius)

            for neighbor, distance in neighbors:
                neighbor_tuple = tuple(neighbor)

                if neighbor_tuple in closed_set:
                    continue

                tentative_g = curr_g + distance

                if (neighbor_tuple not in g_scores or
                    tentative_g < g_scores[neighbor_tuple]):

                    g_scores[neighbor_tuple] = tentative_g
                    f_score = tentative_g + self.get_heuristic(neighbor, target_np)
                    f_scores[neighbor_tuple] = f_score

                    new_path = curr_path + [neighbor_tuple]
                    heapq.heappush(open_set, (f_score, neighbor_tuple, tentative_g, new_path))

        return None, float('inf')

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
    nodes_data_lst = read_json_file(nodes_json_fp)
    # node_cKDTree(nodes_data, gm_api_key, USER_HEADERS, top_n=5)

    start_point = (22.2822603, 114.1258876)
    start_np = np.array(start_point)
    target_point = (22.284566183373638, 114.13251488622221)
    target_np = np.array(target_point)

    nodes_data_dict = {}
    for node_data in nodes_data_lst:
        for key, value in node_data.items():
            first_lat, first_lon = value['first']
            last_lat, last_lon = value['last']
            weight = value['weight']
            nodes_data_dict[(first_lat, first_lon)] = {
                'last': (last_lat, last_lon),
                'weight': weight,
                'name': key
            }
    start_coords_lst = list(nodes_data_dict.keys())
    gov_gdf = {}
    algo_kdtree = AlgoKDTree(gov_gdf, start_coords_lst, nodes_data_dict, k=5)
    check = algo_kdtree.get_heuristic(start_np, target_np)
    # found_path = algo_kdtree.find_path(start_point, target_point)
    # print(found_path)



if __name__ == "__main__":
    main()