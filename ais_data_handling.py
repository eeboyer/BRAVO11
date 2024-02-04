import pandas as pd
import numpy as np


def remove_stationary_ships(df):
	"""
	Removes instances where ships are stationary, defined as having the same LAT and LON in consecutive records.

	Parameters:
	df (pandas.DataFrame): DataFrame containing maritime traffic data.

	Returns:
	pandas.DataFrame: DataFrame after removing instances where ships are stationary.
	"""
	# Ensure the DataFrame has necessary columns
	if 'MMSI' not in df.columns or 'LAT' not in df.columns or 'LON' not in df.columns:
		raise ValueError("DataFrame must contain 'MMSI', 'LAT', and 'LON' columns.")

	# Sort by MMSI and BaseDateTime to ensure chronological order
	df = df.sort_values(by=['MMSI', 'BaseDateTime'])

	# Group by MMSI and check for consecutive records with same LAT and LON
	stationary = df.groupby('MMSI').apply(lambda x: x.duplicated(subset=['LAT', 'LON'], keep=False))

	# Flatten the grouped results and filter the original DataFrame
	stationary_flattened = stationary.reset_index(level=0, drop=True)

	# Keep rows where the ship is not stationary
	df_clean = df[~stationary_flattened]

	return df_clean

# Example usage:
# df = pd.read_csv('your_data.csv')  # Load your DataFrame
# df_clean = remove_stationary_ships(df)


def import_and_clean_AIS_data(file_name, num_rows=False):
	if num_rows:
		df = pd.read_csv(file_name, nrows=num_rows, parse_dates=['BaseDateTime'])
	else:
		df = pd.read_csv(file_name, parse_dates=['BaseDateTime'])
	# drop any rows with nan for lat or lon or MMSI
	df = df.dropna(subset=['LAT', 'LON', 'MMSI'])
	# fills nan vals with 0 -> Not available (default)
	df['VesselType'] = df['VesselType'].fillna(0)
	df['VesselType'] = df['VesselType'].astype(np.int8)

	# removes stationary ships (where lat/lon did not change between consecutive AIS reports
	df = remove_stationary_ships(df)

	# note that Heading of 511 means "Heading Unavailible"
	return df


def select_random_ship_data(df):
    """
    Selects all rows from the DataFrame corresponding to a randomly chosen MMSI using numpy.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing maritime traffic data.

    Returns:
    pandas.DataFrame: A DataFrame with all rows corresponding to the randomly selected MMSI.
    """
    # Ensure the DataFrame contains the 'MMSI' column
    if 'MMSI' not in df.columns:
        raise ValueError("DataFrame does not contain an 'MMSI' column.")

    # Randomly select an MMSI using numpy
    random_mmsi = np.random.choice(df['MMSI'].unique())

    # Select all rows with the chosen MMSI
    selected_data = df[df['MMSI'] == random_mmsi]

    return selected_data


def split_ship_data(df, time_dark, tracked_time, search_time):
	"""
	Splits data for a randomly selected ship into 'open', 'hidden', and 'search' sections using numpy for random choice.

	Parameters:
	df (pandas.DataFrame): DataFrame containing maritime traffic data.
	time_dark (int): Duration in hours for the 'hidden' section after the split time.
	tracked_time (int): Duration in hours for the 'open' section before the split time.
	search_time (int): Duration in hours for the 'search' section after the hidden section.

	Returns:
	tuple: Three DataFrames, the first for the 'open' section, the second for the 'hidden' section,
		   and the third for the 'search' section.
	"""
	# while loop in here is a little sketchy, but it ensure it only returns a valid ship (IE at least 2 points in the search_data
	valid_selection = False
	while not valid_selection:

		# Randomly select a ship using numpy
		random_mmsi = np.random.choice(df['MMSI'].unique())
		ship_data = df[df['MMSI'] == random_mmsi]

		# Ensure data is sorted by time
		ship_data = ship_data.sort_values(by='BaseDateTime')

		# Determine the latest possible split time
		latest_possible_split = ship_data['BaseDateTime'].iloc[-1] - pd.to_timedelta(time_dark, unit='h')

		# Randomly select a split time using numpy
		possible_split_times = ship_data[ship_data['BaseDateTime'] <= latest_possible_split]['BaseDateTime']
		split_time = np.random.choice(possible_split_times)

		# Create 'hidden' dataset
		hidden_start_time = split_time
		hidden_end_time = split_time + pd.to_timedelta(time_dark, unit='h')
		hidden_data = ship_data[
			(ship_data['BaseDateTime'] >= hidden_start_time) & (ship_data['BaseDateTime'] <= hidden_end_time)]

		# Create 'open' dataset
		open_start_time = split_time - pd.to_timedelta(tracked_time, unit='h')
		open_data = ship_data[
			(ship_data['BaseDateTime'] >= open_start_time) & (ship_data['BaseDateTime'] <= split_time)]

		# Create 'search' dataset
		search_end_time = hidden_end_time + pd.to_timedelta(search_time, unit='h')
		search_data = ship_data[
			(ship_data['BaseDateTime'] > hidden_end_time) & (ship_data['BaseDateTime'] <= search_end_time)]
		if len(search_data) > 1:
			valid_selection = True

	return open_data, hidden_data, search_data

# Example usage:
# df = pd.read_csv('your_data.csv')  # Load your DataFrame
# open_data, hidden_data, search_data = split_ship_data(df, time_dark=24, tracked_time=24, search_time=24)


