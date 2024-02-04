import numpy as np
import skyfield.api as skyfld
import pandas as pd
import datetime as dt

def load_satellite_data(file='', satellite_names=['SKYSAT']):
	"""
	Load satellite data from a CSV file and create Skyfield EarthSatellite objects.

	This function reads a CSV file containing satellite data, filters the data to include only
	specified satellites, and creates a Skyfield EarthSatellite object for each filtered satellite.

	Parameters:
	file (str): The file path to the CSV file containing satellite data.
	satellite_names (list of str): A list of satellite names to filter from the CSV file. If empty, all satellites are considered.

	Returns:
	pandas.Series: A series of Skyfield EarthSatellite objects indexed by the satellite names.
	"""

	# Load the satellite data from the CSV file
	sat_df = pd.read_csv(file)

	# Create a regular expression pattern from the list of satellite names
	pattern = '|'.join(satellite_names)

	# Filter the DataFrame to include only the specified satellites
	# If satellite_names is empty, all satellites will be included
	filtered_satellites = sat_df[sat_df['OBJECT_NAME'].str.contains(pattern)].groupby('OBJECT_NAME').first()

	# Create and return a series of EarthSatellite objects
	return filtered_satellites.apply(
		lambda row: skyfld.EarthSatellite(row['TLE_LINE1'], row['TLE_LINE2'], row.name, skyfld.load.timescale()),
		axis=1)


def load_most_accurate_satellite_tle_data(tle_file, satellite_name, target_time):
	sat_df = pd.read_csv(tle_file)
	sat_data = sat_df[sat_df['OBJECT_NAME'] == satellite_name]

	sat_data['EPOCH'] = pd.to_datetime(sat_data['EPOCH'])

	# Convert the input time to datetime
	input_time = pd.to_datetime(target_time)

	# Calculate the absolute difference with the input time
	sat_data['time_diff'] = (sat_data['EPOCH'] - input_time).abs()

	# Find the row with the smallest difference
	closest_row = sat_data.loc[sat_data['time_diff'].idxmin()]
	satellite_object = skyfld.EarthSatellite(closest_row['TLE_LINE1'], closest_row['TLE_LINE2'], closest_row['OBJECT_NAME'], skyfld.load.timescale())

	return satellite_object

def generate_sat_time_series(start=dt.datetime(2023, 12, 29), end=dt.datetime(2023, 12, 29), interval='1T'):
	"""
	Generate a series of Skyfield time objects for a specified date range and interval.

	This function creates a list of Skyfield time objects, each representing a specific
	point in time within the specified range. It is useful for generating time series data
	for satellite position calculations or similar time-sensitive operations in Skyfield.

	Parameters:
	start (datetime): The start of the time series (inclusive). Default is December 29, 2023.
	end (datetime): The end of the time series (inclusive). Default is December 29, 2023.
	interval (str): The interval between each time point in the series, formatted as a pandas frequency string.
					Default is '1T' (1 minute).

	Returns:
	list of Skyfield Time objects: A list of time objects at the specified intervals within the given range.
	"""

	# Load the Skyfield timescale
	ts = skyfld.load.timescale()

	# Generate a Pandas date range based on the start, end, and interval
	date_range = pd.date_range(start=start, end=end, freq=interval)

	# Convert the Pandas date range to a list of Skyfield time objects
	return [ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second) for time in date_range]


def is_earth_point_visible(satellite, earth_point, time):
	"""
	Determine if a point on Earth is visible from a satellite at a given time.

	This function calculates the visibility based on the angle between the position vectors
	of the Earth point and the satellite, both originating from the Earth's center.
	If this angle is less than 90 degrees, the Earth point is considered to be within the
	line of sight of the satellite.

	Parameters:
	satellite (Skyfield EarthSatellite object): The satellite from which visibility is being determined.
	earth_point (Skyfield Topos object): The point on Earth to check for visibility.
	time (Skyfield Time object): The specific time at which visibility is checked.

	Returns:
	bool: True if the Earth point is visible from the satellite at the specified time, False otherwise.
	"""

	# Calculate the position of the Earth point in ECEF coordinates (km) at the given time
	earth_point_position = earth_point.at(time).position.km

	# Calculate the relative position of the satellite with respect to the Earth point (km) at the given time
	sat_position = (satellite - earth_point).at(time).position.km

	# Assign the Earth point position vector; it originates from the Earth's center and points to the Earth point
	vector_to_earth_point = earth_point_position

	# Assign the satellite position vector; it originates from the Earth's center and points to the satellite
	vector_to_sat = sat_position

	# Normalize the vectors to unit vectors for the angle calculation
	unit_vector_to_earth_point = vector_to_earth_point / np.linalg.norm(vector_to_earth_point)
	unit_vector_to_sat = vector_to_sat / np.linalg.norm(vector_to_sat)

	# Calculate the dot product of the two unit vectors
	dot_product = np.dot(unit_vector_to_earth_point, unit_vector_to_sat)

	# Calculate the angle between the two vectors using the arccosine of the dot product
	angle = np.arccos(dot_product)

	# Check if the angle is less than 90 degrees (i.e., pi/2 radians)
	# This is "in view" in a pure sense. May need to adjust the angle to be smaller for a more accurate representation
	return angle < np.pi / 2

