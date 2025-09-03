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

def get_osm_shape_dirpath() -> str:
    matching_folders = glob.glob('planet_*-shp')
    for folder in matching_folders:
        if os.path.isdir(folder):
            return os.path.abspath(folder)

def get_shape_dir_files() -> list[str]:
    dirpath: str = get_osm_shape_dirpath()
    shape_dir: str = os.path.join(dirpath, 'shape')
    filepaths = []
    for filepath in os.listdir(shape_dir):
        new_filepath = os.path.join(shape_dir, filepath)
        filepaths.append(new_filepath)
    return filepaths

def get_shapefiles_by_extension(dir_filepaths: list[str], extension: str) -> list[gpd.geodataframe.GeoDataFrame]:
    gdf_lst = []
    for filepath in dir_filepaths:
        if filepath.endswith(extension):
            gdf = gpd.read_file(filepath)
            gdf_lst.append(gdf)

    return gdf_lst

class OSMData:
    def __init__(self, shape_type: str, shp_gdf, dbf_gdf):
        self.shape_type = shape_type
        self.shp_gdf = shp_gdf
        self.dbf_gdf = dbf_gdf

    def _print(self):
        print(f"self.shape_type: {self.shape_type}")

class RoadNode:
    def __init__(self, osm_id: int, road_name: str | None, road_type, road_width, road_geometry):
        self.osm_id = osm_id
        self.road_name = road_name
        self.road_type = road_type
        self.road_width = road_width
        self.geometry = road_geometry
        self.wgs84_geometry = self._convert_mls_crs()

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

def get_osmdata_by_shapetype(all_osm_data, shapetype):
    for osm_datapart in all_osm_data:
        if osm_datapart.shape_type == shapetype:
            return osm_datapart

def main():
    shape_dir_files = get_shape_dir_files()
    shape_types = ["Buildings", "Landuse", "Natural", "Places", "Points", "Railways", "Roads", "Waterways"]
    hk_shp_gdfs = get_shapefiles_by_extension(shape_dir_files, '.shp')
    hk_dbf_gdfs = get_shapefiles_by_extension(shape_dir_files, '.dbf')

    osm_datas = []
    for shape_type, shp_gdf, dbf_gdf in zip(shape_types, hk_shp_gdfs, hk_dbf_gdfs):
        osm_datas.append(
            OSMData(shape_type, shp_gdf, dbf_gdf)
        )
    roads_data = get_osmdata_by_shapetype(osm_datas, "Roads")
    roads_shp = roads_data.shp_gdf

    road_nodes = []
    for idx, road in roads_shp.iterrows():
        osm_id = road["osm_id"]
        road_name = road["name"]
        road_type = road["type"]
        road_width = road["width"]
        road_geometry = road["geometry"]
        road_nodes.append(
            RoadNode(osm_id, road_name, road_type, road_width, road_geometry)
        )

    for road_node in road_nodes:
        print(road_node.osm_id, len(road_node.wgs84_geometry))

if __name__ == "__main__":
    main()


