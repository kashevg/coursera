# functions for loading datasets
import numpy as np
import pandas as pd


class TripsLoader(object):
    def __init__(self):
        self.reg_cnt = None
        self.lon_min = self.lon_max = self.lon_div = None
        self.lat_min = self.lat_max = self.lat_div = None
        self.load_region()

    def load_region(self):
        # calculating regions
        reg = pd.read_csv('regions.csv', sep=';')
        self.reg_cnt = np.sqrt(reg.shape[0])

        # borders and dividers
        self.lon_min = reg.west.min()
        self.lon_max = reg.east.max()
        self.lon_div = (self.lon_max - self.lon_min) / (self.reg_cnt)

        self.lat_min = reg.south.min()
        self.lat_max = reg.north.max()
        self.lat_div = (self.lat_max - self.lat_min) / (self.reg_cnt)
        return

    # cleaning MACOS zips
    # for file in *; do   zip -d "$file" __MACOSX/\*; done

    def load_month(self, year=None, month=None, use_zip=True, file_name=None, drop_off=False):
        if file_name is None:
            l_file_name = 'zip/' if use_zip else 'data/'
            l_file_name += 'yellow_tripdata_' + str(year) + '-' + '%02d' % (month) + '.csv' + ('.zip' if use_zip else '')
        else:
            l_file_name = file_name
        # loading data from zipped csv file
        l_usecols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                     'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude']
        if drop_off:
            l_usecols += ['dropoff_longitude', 'dropoff_latitude']
        
        l_parse_dates = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
        l_dtype = {'passenger_count': np.int8,
                   'trip_distance': np.float32, 'pickup_longitude': np.float64, 'pickup_latitude': np.float64}
        df = pd.read_csv(l_file_name, compression=('zip' if use_zip else None),
                         usecols=l_usecols,
                         dtype=l_dtype,
                         parse_dates=l_parse_dates,
                         infer_datetime_format=True
                         )
        if drop_off:
            flt = ((df.passenger_count > 0) &
                    (df.trip_distance > 0.) &
                    (df.tpep_dropoff_datetime > df.tpep_pickup_datetime) &
                    (df.pickup_longitude.between(self.lon_min, self.lon_max)) &
                    (df.pickup_latitude.between(self.lat_min, self.lat_max)) &
                    (df.dropoff_longitude.between(self.lon_min, self.lon_max)) &
                    (df.dropoff_latitude.between(self.lat_min, self.lat_max)))
        else:
            flt = ((df.passenger_count > 0) &
                    (df.trip_distance > 0.) &
                    (df.tpep_dropoff_datetime > df.tpep_pickup_datetime) &
                    (df.pickup_longitude.between(self.lon_min, self.lon_max)) &
                    (df.pickup_latitude.between(self.lat_min, self.lat_max)))
                    
        # filtering
        df = df[flt]

        # flooring pickup datetime
        if drop_off:
            df.tpep_pickup_datetime = pd.to_datetime(df.tpep_dropoff_datetime).dt.floor('1H')
        else:
            df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime).dt.floor('1H')

        # calculating
        if drop_off:
            col_lat, col_long = df.dropoff_latitude, df.dropoff_longitude
        else:
            col_lat, col_long = df.pickup_latitude, df.pickup_longitude
        
        
        df['pickup_region'] = ((col_lat - self.lat_min) // self.lat_div + \
                               (col_long - self.lon_min) // self.lon_div * self.reg_cnt + 1.1).map(int)

        # dropping unnecessary columns
        col_to_drop = ['tpep_dropoff_datetime', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'trip_distance']
        if drop_off:
            col_to_drop += ['dropoff_longitude', 'dropoff_latitude']
            
        df.drop(col_to_drop, axis=1, inplace=True)

        # grouping by datetime and region
        df['trips'] = 0
        df_grp = df.groupby([df.tpep_pickup_datetime, df.pickup_region]).count()
        df_grp = df_grp.add_suffix('_count').reset_index()

        # creating a dummy dataset for good pivoting
        df0 = pd.DataFrame.from_dict(
            {'tpep_pickup_datetime': np.full((2500), np.datetime64('1900-01-01')),
             'pickup_region': np.arange(1, 2501),
             'trips_count': np.zeros(2500)}
        )

        df_test = pd.concat([df0, df_grp], sort=False)
        if drop_off:
            df_test.columns = ['tpep_dropoff_datetime', 'dropoff_region', 'trips_count'] + list(df_test.columns[3:])
        else:
            df_test.columns = ['tpep_pickup_datetime', 'pickup_region', 'trips_count'] + list(df_test.columns[3:])
        # pivoting and filtering out of dummy record
        df_pvt = pd.pivot_table(data=df_test,
                                values='trips_count',
                                columns=df_test.columns[1],
                                index=df_test.columns[0],
                                fill_value=0)
        df_pvt = df_pvt[df_pvt.index > '1900-01-01']
        return df_pvt

    def __call__(self, *args):
        return self.load_month(*args)

    def get_coords(self):
        return self.lon_min, self.lat_min, self.lon_max, self.lat_max
   