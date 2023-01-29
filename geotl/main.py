"""Find the nearest point and find corresponding value from specified value uniary gdf.

Shapely’s nearest_points - function provides a nice and easy way of conducting the nearest neighbor analysis,
it can be quite slow.
Using it also requires taking the unary union of the point dataset where all the Points are merged into a single layer
This can be a real memory hungry and slow operation, that can cause problems with large point datasets

So looking at https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.radius_neighbors_graph
It suggest that
Algorithm used to compute the nearest neighbors:

    ‘ball_tree’ will use BallTree
        Space partitioning data structure for organizing points in a multi-dimensional space,
        used for nearest neighbor search.
        If the distance to the ball, db, is larger than distance to the currently closest neighbor,
        we can safely ignore the ball and all points within.
        The ball structure allows us to partition the data along an underlying manifold that our points are on,
        instead of repeatedly dissecting the entire feature space (as in KD-Trees).

        Has ability to calculate the distance between neighbors with various different distance metrics
        Can calculate euclidian distance between neighbors (good if your data is in metric crs)
        Can haversine distance which allows to determine Great Circle distances between locations
        In this solution will use haversine distance since we have lat and lon

        Slower than KD-Trees in low dimensions (d≤3)
         but a lot faster in high dimensions. Both are affected by the curse of dimensionality,
         but Ball-trees tend to still work if data exhibits local structure (e.g. lies on a low-dimensional manifold).

        Read more https://towardsdatascience.com/tree-algorithms-explained-ball-tree-algorithm-vs-kd-tree-vs-brute-force-9746debcd940


    ‘kd_tree’ will use KDTree
        Read https://medium.com/@schmidt.jerome/k-d-trees-and-nearest-neighbors-81b583860144
        KD-trees partition the feature space so we can rule out whole partitions that are further away than
         our closest k neighbors.
         However, the splits are axis aligned which does not extend well to higher dimensions.


    ‘brute’ will use a brute-force search.

    ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

Brute isn't an option
# TODO: Explore ball_tree vs kd_tree
# TODO: Explore cKDTree
https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/#Scaling-with-Leaf-Size
https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b
"""
import os
import geopandas as gpd
import pandas as pd
from sklearn.neighbors import BallTree
import h3
import zipfile
from shapely.geometry import Point
from uszipcode import SearchEngine
import numpy as np

os.environ['USE_PYGEOS'] = '0'


def get_centroid(ziploc):
    def get_point_from_zip(code):
        zipcode = engine.by_zipcode(code)
        return Point(zipcode.lat, zipcode.lng)

    if ziploc == "28":
        return get_point_from_zip(39762)
    elif len(ziploc) == 15:
        lat_lon = h3.h3_to_geo(ziploc)
        return Point(lat_lon[0], lat_lon[1])
    elif len(ziploc) == 5:
        return get_point_from_zip(ziploc)

    return None


def get_point(lat, lng):
    return Point(float(lat), float(lng))


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points

    Perform nearest neighbor search using BallTree function

    Initialize the BallTree object with the coordinate from point dataset that contains all the nearest neighbor candidates
        and Specify the distance metric to be haversine so we get the Great Circle Distances

        leaf_size parameter adjusts the tradeoff between the cost of BallTree node traversal and the cost of a brute-force distance estimate

        leaf size Benchmarking Nearest Neighbor Searches in Python - https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/

    Run the nearest neighbor query with tree.query(src_points, k=k_neighbors)
        src_points are the building-coordinates (as radians)
        k-parameter is the number of neighbors we want to calculate is 1 since we only want nearest 1 neighbour

    Then re-arrange the data back into a format in which the closest point indices
    and distances are in separate numpy arrays.

    """

    # TODO: Verify it's WGS84 projection as  Functions here assume that your input points are in WGS84 projection
    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def geo_radians(geom):
    try:
        geom.y * np.pi / 180, geom.x * np.pi / 180
    except Exception as e:
        print(f"Ex: {e}")
        print(geom)
        return None


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    Transforms the data from GeoDataFrame into numpy arrays - which is expected input type for BallTree function
    Convert the lat/lon coordinates into radians (and back),
        since scikit-learn’s haversine distance metric wants inputs as radians and also outputs the data as radians

    NOTE: Radian = Degree * PI / 180
    Since we convert to Radians, we get output distance in meters though input is in decimal degrees

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = 'centroid'
    right_geom_col = 'centroid'  # right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    # Notice: should be in Lat/Lon format
    left_radians = np.array(
        left_gdf[left_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    right_radians = np.array(
        right[right_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    # df_locations.apply(lambda x: geo_radians(x['centroid']), axis=1).to_list()

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius

    return closest_points


if __name__ == '__main__':
    # Data Source 1
    # CSV
    # ms_hinds_locations = pd.read_csv("data/ms_hinds_locations.csv")
    ms_hinds_locations = gpd.read_file("../data/ms_hinds_locations.csv")
    print(ms_hinds_locations.columns)

    ms_hinds_locations.head(2)

    # Data Source 4

    zf = zipfile.ZipFile("../data/ms_hinds_buildings.geojson.zip")
    # df = pd.read_csv(zf)
    # dfcsv = gpd.read_file("data/ms_hinds_buildings.geojson/ms_hinds_buildings_join_table.csv")
    # dfjson = gpd.read_file("data/ms_hinds_buildings.geojson/ms_hinds_buildings.json")

    ms_hinds_buildings = [gpd.read_file(zf.open(text_file.filename)) for text_file in zf.infolist() if
                          text_file.filename.endswith('.json')]
    df_ms_buildings = ms_hinds_buildings[0]

    df_locations = ms_hinds_locations
    df_locations.crs = 'epsg:4326'
    df_locations = df_locations.to_crs({'init': 'epsg:4326'})

    df_ms_buildings = df_ms_buildings.to_crs(crs={'init': 'epsg:4326'})
    df_ms_buildings['centroid'] = df_ms_buildings.centroid

    engine = SearchEngine(simple_or_comprehensive=SearchEngine.SimpleOrComprehensiveArgEnum.comprehensive)

    df_locations['f_lat'] = df_locations['f_lat'].astype(float)
    df_locations['f_lon'] = df_locations['f_lon'].astype(float)
    df_locations['centroid'] = df_locations.apply(
        lambda x: get_centroid(x['f_ziploc']) if pd.isnull(x['f_lat']) or pd.isnull(x['f_lon']) else get_point(
            x['f_lat'], x['f_lon']),
        axis=1)

    # Find closest public transport stop for each building and get also the distance based on haversine distance
    # Note: haversine distance which is implemented here is a bit slower than using e.g. 'euclidean' metric
    # but useful as we get the distance between points in meters

    df_ms_buildings.isna().sum()
    df_locations.isna().sum()

    # And the result looks like ..
    closest_building = nearest_neighbor(df_ms_buildings, df_locations, return_dist=True)

    # Rename the geometry of closest stops gdf so that we can easily identify it
    closest_building = closest_building.rename(columns={'centroid': 'closest_centroid', 'geometry': 'close_geometry'})

    # Merge the close building with building dataset
    #   The order of the closest_stops matches exactly the order in buildings,
    #   so we can easily merge the datasets based on index
    # Merge the datasets by index (for this, it is good to use '.join()' -function)
    buildings = df_ms_buildings.join(closest_building)
    # closest_building[['geometry']].isna().sum()

    buildings.head()
    # Now we should have exactly the same number of closest_building as we have buildings
    print(len(closest_building), '==', len(df_ms_buildings['centroid']))

    buildings.plot(column='distance', markersize=0.2, alpha=0.5, figsize=(10, 10), scheme='quantiles', k=4,
                   legend=True)
