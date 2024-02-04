import satellite_access_functions as saf
import satellite_modeling_functions as smf
import datetime as dt
import numpy as np
import skyfield.api as skyfld
import pandas as pd

# boundaries for coordinates
east_bound = -150 # - 150
west_bound = -165 # - 165
north_bound = 30 # 30
south_bound = 15 # 15

# time range for sat data
start_time = dt.datetime(2023, 1, 1)
#end_time = dt.datetime(2023, 6, 1)
end_time = dt.datetime(2023, 1, 7) # smaller one just for testing before beefing this up

# names of satellites which will be queried. Total of 65 SAR/EO sats in this dataset
sats_to_query = ['SKYSAT', 'PNEO', 'PLEIADES 1', 'SPOT', 'NOVASAR', 'CAPELLA', 'ICEYE']

# generate the coordinate ranges
lons = np.arange(start=west_bound, stop=east_bound)
lats = np.arange(start=south_bound, stop=north_bound)

# generate tuples w/ format (GEO, lat, lon)
coordinates = [(f'{str(lat).rjust(2,"0").strip("-")}N{str(lon).rjust(3,"0").strip("-")}E', lat, lon) for lat in lats for lon in lons]

# generate time series data -> slice off last item because its equal to the end_time
time_data = saf.generate_sat_time_series(start_time, end_time, interval='1T')[:-1] # T for minutes

# load the skyfield timescale data
ts = skyfld.load.timescale()

# load the sat data for each satellite
tle_file = 'ephemeral_data/ephemeral_data_2022_01_01_2023_12_31.csv'
# load it as a dataframe with only the object name (satellite name, epoch, and TLE lines)
ephemeral_data = pd.read_csv(tle_file, parse_dates=['EPOCH'])[['OBJECT_NAME', 'EPOCH', 'TLE_LINE1', 'TLE_LINE2']]
# all the data gets loaded, then subset to the EPOCH data between the start and end time of the quered range
ephemeral_data = ephemeral_data[(ephemeral_data['EPOCH'] >= start_time) & (ephemeral_data['EPOCH'] < end_time)]

# create a dictionary with a key for each satellite and a value of the dataframe of ephemeral data corresponding to this satellite
ephemeral_dict = {sat: ephemeral_data[ephemeral_data.OBJECT_NAME==sat] for sat in ephemeral_data.OBJECT_NAME.unique()}

# vectorize the is_earth_point_visible function
# I just create a massive numpy array of objects and abuse vectorization to compute things faster
vectorized_sat_in_view = np.vectorize(saf.is_earth_point_visible)

#static time array -> gets passed around a lot for the skyfield functions
static_time_arr = np.array([t.utc_datetime() for t in time_data], dtype='<M8[ns]')

# adds a satellite object for each value in the ephemeral dict based off of the TLE for the corresponding row
for key in ephemeral_dict.keys():
	ephemeral_dict[key]['Satellite_Object'] = ephemeral_dict[key].apply(
		lambda row: skyfld.EarthSatellite(row['TLE_LINE1'], row['TLE_LINE2'], row.OBJECT_NAME, skyfld.load.timescale()),
		axis=1)

	ephemeral_dict[key] = ephemeral_dict[key][['OBJECT_NAME', 'Satellite_Object', 'EPOCH']]

# Now we expand out the df's in the dicts to include all time series.
# this whole function gets real ugly real quick and is a whore for RAM so be warned... I'm sorry
for key in ephemeral_dict.keys():
	ephemeral_dict[key] = smf.map_and_reindex_ephemeral_data(ephemeral_dict[key], str(start_time).split(' ')[0], str(end_time).split(' ')[0])
	# need to remove values that are greater than the end time. This happens sometimes if the map and reindex pulls data from the future
	ephemeral_dict[key] = ephemeral_dict[key][ephemeral_dict[key].Time < end_time].drop_duplicates(subset = ['Time'], keep='first')

for geo, lat, lon in coordinates:
	# initialize the earth point
	earth_point = skyfld.Topos(latitude_degrees=lat, longitude_degrees=lon)
	# create a big array of these earth points to allow vectorization of sat_in_view functionality... im sorry lil laptop
	big_point_array = np.full((len(time_data),), earth_point, dtype=object)

	# init data frame which will be written to CSV
	geo_accesses = pd.DataFrame()

	# name for csv that will be written
	file_name = f'access_{geo}_{start_time.isoformat().split("T")[0]}_{end_time.isoformat().split("T")[0]}.csv'
	path = 'access_data_hawaii_v2/'

	# iterate through each of the 65 satellites
    # we'll start with a single satellite and work up from there:
	for sat in ephemeral_dict.keys():
		# fields to export:
		sat_name = ephemeral_dict[sat]['OBJECT_NAME'][0]
		#sat_id = sat.OBJECT_ID

		# computational section:
		# grossly memory inefficient, but much faster -> needed to vectorize
		big_sat_array = ephemeral_dict[sat]['Satellite_Object']

		# apply the vectorized operation
		satellite_accesses = vectorized_sat_in_view(big_sat_array, big_point_array, np.array(time_data))
		v_df = pd.DataFrame({'Access': satellite_accesses, 'Time': static_time_arr})
		start_indices = np.where((v_df.Access * 1).diff() > 0)
		end_indices = np.where((v_df.Access * 1).diff() < 0)
		access_starts = v_df.iloc[start_indices]['Time'].values
		access_ends = v_df.iloc[end_indices]['Time'].values
		access_df = pd.DataFrame()
		# edge case for when window goes into the next day
		if len(access_starts) > len(access_ends):
			access_ends = np.append(access_ends, static_time_arr[-1])
		elif len(access_ends) > len(access_starts):
			access_starts = np.append(static_time_arr[-0], access_starts)
		access_df = pd.DataFrame({'ACCESS_START': access_starts, 'ACCESS_END': access_ends})
		#access_df['ACCESS_START'] = access_starts
		#access_df['ACCESS_END'] = access_ends
		access_df['DURATION'] = access_df['ACCESS_END'] - access_df['ACCESS_START']
		access_df['DURATION_SECONDS'] = access_df['DURATION'].dt.total_seconds()
		access_df['SAT_NAME'] = sat_name
		#access_df['SAT_ID'] = sat_id
		geo_accesses = pd.concat([geo_accesses, access_df])
		print(f'Completed {sat} for {geo}')

	geo_accesses['LAT'] = lat
	geo_accesses['LON'] = lon
	geo_accesses['GEO'] = geo
	geo_access = geo_accesses[['GEO', 'SAT_NAME', 'ACCESS_START', 'ACCESS_END', 'DURATION', 'DURATION_SECONDS', 'LAT', 'LON']]
	geo_access.to_csv(path + file_name, index=False)
	print(f'Completed GEO: {geo.strip("-")}\n')

