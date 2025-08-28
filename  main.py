# Program goal: Find the top N fastest routes between two points

import os, zipfile, fiona, json
import urllib.request
import geopandas as gpd

def get_latest_road_network(input_dirpath: str):
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

def write_gdf(gdf: gpd.geodataframe.GeoDataFrame, filepath: str) -> None:
    remove_filepath(filepath)
    gdf.to_json(filepath)

def main():
    dataset_dirpath: str = get_dataset_dirpath()
    get_latest_road_network(input_dirpath=dataset_dirpath)
    parsed_gdf: gpd.geodataframe.GeoDataFrame = parse_gdb_files(input_dirpath=dataset_dirpath)

    parsed_data_filepath: str = os.path.join(os.getcwd(), 'parsed_data.json')
    write_gdf(parsed_gdf, parsed_data_filepath)

if __name__ == "__main__":
    main()