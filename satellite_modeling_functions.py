import datetime
import satellite_access_functions as saf
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import skyfield

# satellite footprints in KM
satellite_parameters = {'SKYSAT': (6, 11),
                        'PNEO': (14, 50),
                        'PLEIADES': (20, 50),
                        'SPOT': (60, 50),
                        'NOVASAR': (50, 50),
                        'CAPELLA': (100, 10),
                        'ICEYE': (50, 50)}

def km_to_deg_lat(km):
    """Convert kilometers to degrees of latitude."""
    return km / 111.0

def km_to_deg_lon(km, lat):
    """Convert kilometers to degrees of longitude."""
    return km / (111.0 * np.cos(np.radians(lat)))

def rotate_points(center, angle, points):
    """Rotate points around a center point."""
    s, c = np.sin(np.radians(angle)), np.cos(np.radians(angle))
    rot_matrix = np.array([[c, -s], [s, c]])

    # Translate points to origin and apply rotation matrix
    translated_points = points - center
    rotated_points = np.dot(translated_points, rot_matrix.T)

    # Translate points back
    return rotated_points + center


def footprint_corners(lat, lon, foot_print_size, azimuth=False):
    """Calculate rectangle corners given center point, length, and width."""
    length_km, width_km = foot_print_size
    length_deg = km_to_deg_lat(length_km)
    width_deg = km_to_deg_lon(width_km, lat)

    # Define corners (top left, top right, bottom right, bottom left)
    corners = np.array([
        [lat + length_deg / 2, lon - width_deg / 2],
        [lat + length_deg / 2, lon + width_deg / 2],
        [lat - length_deg / 2, lon + width_deg / 2],
        [lat - length_deg / 2, lon - width_deg / 2]
    ])

    # Apply random rotation if azimuth isnt passed in
    if not azimuth:
        azimuth = np.random.uniform(0, 360)
    center = np.array([lat, lon])
    rotated_corners = rotate_points(center, azimuth, corners)
    return rotated_corners


def is_point_in_rectangle(corners, point):
    """
    Check if a point is inside a rectangle defined by its corners using Shapely.

    :param corners: List of (latitude, longitude) pairs representing rectangle corners.
    :param point: Tuple (latitude, longitude) representing the point to check.
    :return: True if the point is inside the rectangle, False otherwise.
    """
    polygon = Polygon(corners)
    point = Point(point)
    return polygon.contains(point)


def calculate_azimuth(position_1, position_2):
    # unpack the lat/lon for each position
    lat_1, lon_1 = [pos.radians for pos in position_1]
    lat_2, lon_2 = [pos.radians for pos in position_2]
    # compute the difference between the longitudes (in radians)
    diff_lon = lon_2 - lon_1

    # calculate the azimuth
    x = np.sin(diff_lon) * np.cos(lat_2)
    y = (np.cos(lat_1) * np.sin(lat_2)) - (np.sin(lat_1) * np.cos(lat_2) * np.cos(diff_lon))
    azimuth = np.arctan2(x, y)

    # convert azimuth to degrees
    azimuth = np.degrees(azimuth)
    return azimuth


def compute_satellite_access_azimuth(satellite, time):
    # load the satellite from the TLE file
    tle_file = 'ephemeral_data/ephemeral_data_2022_01_01_2023_12_31'
    sat = saf.load_most_accurate_satellite_tle_data(tle_file, satellite, time)
    # sat = sat_data[0]
    # create a follow_on time that's one minute after the desired satellite access window
    sat_times = saf.generate_sat_time_series(pd.to_datetime(time), pd.to_datetime(time) + datetime.timedelta(minutes=1))
    # get the orbital position of the satellite at T1
    sat_at_time_1 = sat.at(sat_times[0])
    # get the orbital position of the satellite at T2
    sat_at_time_2 = sat.at(sat_times[1])

    # get the geographic position of the sats: (lat,lon, altitude)
    geo_pos_1 = skyfield.toposlib.wgs84.geographic_position_of(sat_at_time_1)
    geo_pos_2 = skyfield.toposlib.wgs84.geographic_position_of(sat_at_time_2)
    # extract the lat/lon fields
    lat_lon_1 = (geo_pos_1.latitude, geo_pos_1.longitude)
    lat_lon_2 = (geo_pos_2.latitude, geo_pos_2.longitude)

    azimuth = calculate_azimuth(lat_lon_1, lat_lon_2)
    return azimuth


def find_closest_datetime(epoch, datetime_range):
    """
    Find the closest datetime from a given range to a specified epoch.

    Args:
    epoch (datetime): The epoch time to which we want to find the closest datetime.
    datetime_range (DatetimeIndex): A range of datetime values to search.

    Returns:
    datetime: The closest datetime from the datetime_range to the specified epoch.
    """
    # Compute absolute difference in seconds between the epoch and each datetime in the range
    abs_diff = abs((datetime_range - epoch).total_seconds())
    # Return the datetime that has the minimum difference (i.e., the closest one)
    return datetime_range[abs_diff.argmin()]


def map_and_reindex_ephemeral_data(single_sat_df, start_date, end_date):
    """
    Maps and reindexes satellite data to a specified datetime range.

    Args:
    single_sat_df (DataFrame): The DataFrame containing satellite data.
    start_date (str): The start date of the desired datetime range.
    end_date (str): The end date of the desired datetime range.

    Returns:
    DataFrame: A reindexed DataFrame with satellite data mapped to the specified datetime range. Every single minute will have the full TLE associated with it
    enabling you to compute the best satellite accesses for each minute
    """
    # Create a datetime range from start_date to end_date with a frequency of one minute
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='T')

    # Working with the provided satellite data DataFrame
    ephemeral_data = single_sat_df

    # Convert the 'EPOCH' column to datetime format for accurate comparison
    ephemeral_data['EPOCH'] = pd.to_datetime(ephemeral_data['EPOCH'])

    # Apply the find_closest_datetime function to map each EPOCH to the closest datetime
    ephemeral_data['Closest_Datetime'] = ephemeral_data['EPOCH'].apply(
        lambda x: find_closest_datetime(x, datetime_range))

    # Set 'Closest_Datetime' as the index for merging
    ephemeral_data.set_index('Closest_Datetime', inplace=True)

    # Create a DataFrame from the datetime range for merging
    time_range_df = pd.DataFrame({'Time': datetime_range})

    # Merge the time range DataFrame with the satellite data
    # This ensures all times are present and satellite data is aligned
    full_sat_data = pd.merge(left=time_range_df, right=ephemeral_data, left_on='Time', right_index=True, how='left')

    # Forward fill and backward fill to handle missing values
    full_sat_data.ffill(inplace=True)
    full_sat_data.bfill(inplace=True)

    return full_sat_data
