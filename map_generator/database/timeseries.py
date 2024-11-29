import time
from influxdb_client_3 import InfluxDBClient3, Point
import pandas as pd

class TimeseriesDatabase:
    """
    A class to handle interactions with InfluxDB for IMU data storage and retrieval.
    """

    def __init__(self,host=None,token=None,org=None,database=None,file=None,):
        try:
            self.client = InfluxDBClient3(host=host, token=token, org=org)
            self.df = pd.read_csv(file)
            self.database_name = database
            self.write_data()
        except Exception as e:
            print(f"Initialization error: {e}")

    def write_data(self):
        """
        Writes IMU data from CSV to InfluxDB, recording each row as a separate point.
        """
        try:
            for index, row in self.df.iterrows():
                point = (
                    Point(self.database_name)
                    .time(row["timestamp"])
                    .field("orientation_x", row["orientation_x"])
                    .field("orientation_y", row["orientation_y"])
                    .field("orientation_z", row["orientation_z"])
                    .field("orientation_w", row["orientation_w"])
                    .field("angular_velocity_x", row["angular_velocity_x"])
                    .field("angular_velocity_y", row["angular_velocity_y"])
                    .field("angular_velocity_z", row["angular_velocity_z"])
                    .field("linear_acceleration_x", row["linear_acceleration_x"])
                    .field("linear_acceleration_y", row["linear_acceleration_y"])
                    .field("linear_acceleration_z", row["linear_acceleration_z"])
                )

                # Write the point to InfluxDB
                self.client.write(database=self.database_name, record=point)
                time.sleep(1)  # Optional: Separate points by 1 second

            print("Data write complete. Check the InfluxDB UI for the new data.")
        except Exception as e:
            print(f"Error writing data: {e}")
        finally:
            self.client.close()

    def get_data_by_timestamp(self, time_stamp):
        """
        Retrieves data from InfluxDB at a specific timestamp.

        Args:
            time_stamp (str): Timestamp to query for data.

        Returns:
            pd.DataFrame: DataFrame containing the queried data.
        """
        query = f'''
        SELECT *
        FROM "{self.database_name}"
        WHERE time == '{time_stamp}'
        '''
        try:
            table = self.client.query(query=query, database=self.database_name, language='sql')
            sorted_table = table.to_pandas().sort_values(by="time")
            return sorted_table
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
