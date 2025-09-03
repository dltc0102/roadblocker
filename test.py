import os, zipfile, fiona, json, requests, re, time, math, googlemaps, heapq, shutil, glob
import shapely.geometry as geometry
import shapely.geometry.multilinestring as shapely_mls
from pyproj import Transformer
from bs4 import BeautifulSoup
import urllib.request
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
from colorama import Fore as CFore
from colorama import Style as CStyle
import warnings

warnings.filterwarnings("ignore", message="Non closed ring detected", category=RuntimeWarning, module="pyogrio.raw")

class GDF:
    def __init__(self, gdf_data, api_key, headers):
        self.gdf_data = gdf_data
        self.api_key = api_key
        self.headers = headers

        self.points_gdf = gdf_data['points']
        self.lines_gdf = gdf_data["lines"]
        self.mls_gdf = gdf_data["multilinestrings"]
        self.mpg_gdf = gdf_data["multipolygons"]
        self.other_gdf = gdf_data["other_relations"]

        self.point_nodes = []
        if not self.points_gdf.empty:
            for idx, point in self.points_gdf.iterrows():
                self.point_nodes.append(
                    PointNode(
                        point["osm_id"], point["name"], point["barrier"], point["highway"], point["ref"], point["address"], point["is_in"], point["place"], point["man_made"], point["other_tags"], point["geometry"]
                        )
                    )

class PointNode:
    def __init__(self, osm_id: int, road_name: str | None, barrier, highway, road_ref, address, is_in, place, man_made, other_tags, geometry):
        self.osm_id = osm_id
        self.road_name = road_name
        self.ename = None
        self.cname = None
        self.barrier = barrier
        self.highway = highway
        self.road_ref = road_ref
        self.address = address
        self.is_in = is_in
        self.place = place
        self.man_made =man_made
        self.other_tags = other_tags
        self.geometry = geometry
        self.wgs84_geometry = self._convert_mls_crs()

        if self.wgs84_geometry:
            self._print()

    def _print(self):
        print(f"{self.road_name} {self.wgs84_geometry[0]}")

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

def get_pbf_filepath() -> str:
    file_to_parse = None
    for filepath in os.listdir(os.getcwd()):
        if filepath.startswith('hong-kong') and filepath.endswith('.pbf'):
            file_to_parse = os.path.join(os.getcwd(), filepath)
    return file_to_parse

def get_all_gdf_data(filepath: str) -> gpd.geodataframe.GeoDataFrame:
    layers = fiona.listlayers(filepath)
    gdf_data = {}
    for layer in layers:
        layer_gdf = gpd.read_file(filepath, layer=layer)
        gdf_data[layer] = layer_gdf
    return gdf_data

def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        return key_f.readline().strip()

def main():
    USER_HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    gm_api_key = get_gm_api_key()
    pbf_file_path = get_pbf_filepath()
    gdf_data = get_all_gdf_data(pbf_file_path)
    GDF(gdf_data, gm_api_key, USER_HEADERS)


if __name__ == "__main__":
    main()


