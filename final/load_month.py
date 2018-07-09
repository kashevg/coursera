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

    def load_month(self, year=None, month=None, use_zip=True, file_name=None):
        if file_name is None:
            l_file_name = 'zip/' if use_zip else 'data/'
            l_file_name += 'yellow_tripdata_' + str(year) + '-' + '%02d' % (month) + '.csv' + ('.zip' if use_zip else '')
        else:
            l_file_name = file_name
        # loading data from zipped csv file
        l_usecols = [u'tpep_pickup_datetime', u'tpep_dropoff_datetime',
                     u'passenger_count', u'trip_distance', u'pickup_longitude', u'pickup_latitude']
        l_parse_dates = [u'tpep_pickup_datetime', u'tpep_dropoff_datetime']
        l_dtype = {u'passenger_count': np.int8,
                   u'trip_distance': np.float32, u'pickup_longitude': np.float64, u'pickup_latitude': np.float64}
        df = pd.read_csv(l_file_name, compression=('zip' if use_zip else None),
                         usecols=l_usecols,
                         dtype=l_dtype,
                         parse_dates=l_parse_dates,
                         infer_datetime_format=True
                         )

        # filtering
        df = df[(df.passenger_count > 0) &
                (df.trip_distance > 0.) &
                (df.tpep_dropoff_datetime > df.tpep_pickup_datetime) &
                (df.pickup_longitude.between(self.lon_min, self.lon_max)) &
                (df.pickup_latitude.between(self.lat_min, self.lat_max))]

        # flooring pickup datetime
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime).dt.floor('1H')

        # calculating
        df['pickup_region'] = ((df.pickup_latitude - self.lat_min) // self.lat_div + \
                               (df.pickup_longitude - self.lon_min) // self.lon_div * self.reg_cnt + 1.1).map(int)

        # dropping unnecessary columns
        df.drop(['tpep_dropoff_datetime', 'passenger_count', 'passenger_count',
                 'pickup_latitude', 'pickup_longitude',
                 'trip_distance'
                 ], axis=1, inplace=True)

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

        # pivoting and filtering out of dummy record
        df_pvt = pd.pivot_table(data=df_test,
                                values='trips_count',
                                columns='pickup_region',
                                index='tpep_pickup_datetime',
                                fill_value=0)
        df_pvt = df_pvt[df_pvt.index > '1900-01-01']
        return df_pvt

    def __call__(self, *args):
        return self.load_month(*args)

    def get_coords(self):
        return self.lon_min, self.lat_min, self.lon_max, self.lat_max
   