import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import duckdb
import gzip
import zipfile
from shapely.ops import nearest_points


def get_nearest_values(row, other_gdf, point_column='geometry', value_column="geometry"):
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

    # Create an union of the other GeoDataFrame's geometries:
    other_points = other_gdf["geometry"].unary_union

    # Find the nearest points
    nearest_geoms = nearest_points(row[point_column], other_points)

    # Get corresponding values from the other df
    nearest_data = other_gdf.loc[other_gdf["geometry"] == nearest_geoms[1]]

    nearest_value = nearest_data[value_column].values[0]

    return nearest_value

if __name__ == '__main__':
    # Data Source 1
    # CSV
    # ms_hinds_locations = pd.read_csv("data/ms_hinds_locations.csv")
    ms_hinds_locations = gpd.read_file("data/ms_hinds_locations.csv")
    print(ms_hinds_locations.columns)
    # ms_hinds_locations.head(2)

    # Data Source 2
    # EXCEL
    ms_hinds_locations_xlsx = pd.read_excel("data/ms_hinds_locations.xlsx")
    print(ms_hinds_locations_xlsx.columns)
    # ms_hinds_locations_xlsx.head(2)

    # Data Source 3
    with gzip.open("data/ms_hinds_parcels.ndgeojson.gz", 'rb') as f_in:
        g_parcels = gpd.read_file(f_in)
        # g_parcels.head(2)

    # Data Source 4
    zf = zipfile.ZipFile("data/ms_hinds_buildings.geojson.zip")
    # df = pd.read_csv(zf)
    # dfcsv = gpd.read_file("data/ms_hinds_buildings.geojson/ms_hinds_buildings_join_table.csv")
    # dfjson = gpd.read_file("data/ms_hinds_buildings.geojson/ms_hinds_buildings.json")

    ms_hinds_buildings = [gpd.read_file(zf.open(text_file.filename)) for text_file in zf.infolist() if
                          text_file.filename.endswith('.json')]
    df_ms_buildings = ms_hinds_buildings[0]

    print(duckdb.query("""
             WITH tmp AS (SELECT LEN(f_ziploc) as len FROM ms_hinds_locations)
             SELECT len, COUNT(1) as num_records
             FROM tmp
             GROUP BY len
             """).to_df())

    df_locations = ms_hinds_locations
    df_locations.crs = 'epsg:4326'
    df_locations = df_locations.to_crs({'init': 'epsg:4326'})

    gdf_locations = gpd.GeoDataFrame(
        df_locations, geometry=gpd.points_from_xy(df_locations.f_lat, df_locations.f_lon, crs="EPSG:4326")
    )
    # TODO - FILL NA and cover edge cases
    # df_locations = ms_hinds_locations[['parcel_id', 'f_lat', 'f_lon', 'f_city', 'f_addr1', 'f_ziploc']]

    df_ms_buildings = df_ms_buildings.to_crs(crs={'init': 'epsg:4326'})
    df_ms_buildings['centroid'] = df_ms_buildings.centroid

    unary_union = gdf_locations.unary_union
    df_locations["nearest_roof"] = df_ms_buildings.apply(get_nearest_values, other_gdf=gdf_locations,
                                                         point_column="centroid", axis=1)



