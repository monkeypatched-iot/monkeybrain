import time
from influxdb_client_3 import InfluxDBClient3, Point
import pandas as pd

# Connect to influx db
TIME_SERIES_DB_TOKEN = "2e6W8x6yVzFPkehkbITD4-x9cOlQ4NSlEDKA5VuFITBC4NhLQgo__BGZasxeiiCauGylyobQ15H02SpgH6e95Q=="
TIME_SERIES_DB_ORG = "Monkeypatched"
TIME_SERIES_DB_HOST = "https://us-east-1-1.aws.cloud2.influxdata.com"
TIME_SERIES_DB_NAME = "imu_data"

class IMU:
    def __init__(self) -> None:
        self.client = InfluxDBClient3(host=TIME_SERIES_DB_HOST, token=TIME_SERIES_DB_TOKEN , org=TIME_SERIES_DB_ORG)

    def get_imu_data(self):
        # Define your Flux query
        query = f'''
            SELECT *
            FROM "imu_data"
            WHERE
            time >= now() - interval '30 days'
            '''

        # Execute the query
        table = self.client.query(query=query, database="imu_data", language='sql')

        # Convert to dataframe
        imu_df = table.to_pandas().sort_values(by="time")

        # return idf
        return imu_df

    def get_imu_data_by_timestamp(self,timestamp):

        # Define your Flux query
        query = f'''
            SELECT *
            FROM "imu_data"
            WHERE
            time = '{timestamp}'
            '''

        # Execute the query
        table = self.client.query(query=query, database="imu_data", language='sql')

        # Convert to dataframe
        imu_df = table.to_pandas().sort_values(by="time")

        # return idf
        return imu_df