import psycopg2
from psycopg2.errors import UniqueViolation

class SpatialDatabase:
    def __init__(self, db_name='places', user='user', password='password', host='localhost', port='5432'):
        # Initialize the PostgreSQL connection
        self.conn = psycopg2.connect(dbname=db_name, user=user, password=password, host=host, port=port)
        self.cursor = self.conn.cursor()

    def create_keyframe_table(self):
        # Create table for keyframes with spatial attributes
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS keyframes (
                id SERIAL PRIMARY KEY,
                keyframe_id INTEGER UNIQUE,
                timestamp TEXT,
                x REAL,
                y REAL,
                z REAL,
                orientation_x REAL,
                orientation_y REAL,
                orientation_z REAL,
                orientation_w REAL,
                features BYTEA
            );
        ''')
        self.conn.commit()

    def create_path_table(self):
        # Create table for paths with LINESTRING geometry type
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS paths (
                id SERIAL PRIMARY KEY,
                path_geometry GEOMETRY(LINESTRING, 4326)
            );
        ''')
        self.conn.commit()

    def insert_keyframe(self, keyframe_id, timestamp, x, y, z, orientation_x, orientation_y, orientation_z, orientation_w, features):
        try:
            # Attempt to insert the keyframe record into the table
            self.cursor.execute('''
                INSERT INTO keyframes (keyframe_id, timestamp, x, y, z, orientation_x, orientation_y, orientation_z, orientation_w, features)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (keyframe_id, timestamp, x, y, z, orientation_x, orientation_y, orientation_z, orientation_w, features))
            self.conn.commit()  # Commit the transaction
        except Exception as e:
            # Handle the case where the keyframe_id already exists
            if isinstance(e, UniqueViolation):  # Import UniqueViolation from psycopg2.errors if needed
                print(f"Keyframe with ID {keyframe_id} already exists. Skipping insertion.")
                self.conn.rollback()  # Rollback the transaction
            else:
                raise e  # Re-raise other exceptions

    def insert_path(self, path_coords):
        # Convert list of coordinates to LINESTRING (WKT format)
        line_string = ', '.join(f"{x} {y}" for x, y in path_coords)
        query = f'''
            INSERT INTO paths (path_geometry)
            VALUES (ST_GeomFromText('LINESTRING({line_string})', 4326));
        '''
        self.cursor.execute(query)
        self.conn.commit()

    def update_keyframe(self, keyframe_id, x=None, y=None, z=None, orientation=None, features=None):
        # Update a keyframe by its ID
        update_fields = []
        update_values = []

        if x is not None:
            update_fields.append("x = %s")
            update_values.append(x)
        if y is not None:
            update_fields.append("y = %s")
            update_values.append(y)
        if z is not None:
            update_fields.append("z = %s")
            update_values.append(z)
        if orientation is not None:
            update_fields.extend(["orientation_x = %s", "orientation_y = %s", "orientation_z = %s", "orientation_w = %s"])
            update_values.extend(orientation)
        if features is not None:
            update_fields.append("features = %s")
            update_values.append(features)

        update_values.append(keyframe_id)
        query = f"UPDATE keyframes SET {', '.join(update_fields)} WHERE keyframe_id = %s"
        self.cursor.execute(query, update_values)
        self.conn.commit()

    def delete_keyframe(self, keyframe_id):
        # Delete a keyframe by its ID
        self.cursor.execute("DELETE FROM keyframes WHERE keyframe_id = %s", (keyframe_id,))
        self.conn.commit()

    def get_keyframe(self, keyframe_id):
        # Retrieve a keyframe by its ID
        self.cursor.execute("SELECT * FROM keyframes WHERE keyframe_id = %s", (keyframe_id,))
        return self.cursor.fetchone()

    def get_path(self, path_id):
        # Retrieve a path by its ID
        self.cursor.execute("SELECT * FROM paths WHERE id = %s", (path_id,))
        return self.cursor.fetchone()

    def get_all_keyframes(self):
        # Retrieve all keyframes
        try:
            self.cursor.execute("SELECT * FROM keyframes")
            data = self.cursor.fetchall()
            # print("Retrieved data:", data)  # Debug print
            self.conn.commit()
            return data
        except Exception as e:
            print("Error:", e)  # Print any errors encountered
            return None

    def get_all_keyframe_locations(self):
        # Retrieve x and y locations for all keyframes
        self.cursor.execute("SELECT x, y FROM keyframes")
        data = self.cursor.fetchall()
        self.conn.commit()
        return data

    def get_all_paths(self):
        # Retrieve all paths
        self.cursor.execute("SELECT id, ST_AsText(path_geometry) FROM paths")
        return self.cursor.fetchall()

    def get_node_by_location(self, x, y):
        # Retrieve a keyframe by its location
        self.cursor.execute("SELECT id, keyframe_id, x, y FROM keyframes WHERE x = %s AND y = %s", (x, y))
        return self.cursor.fetchone()

    def get_node_by_location_by_id(self, keyframe_id):
        # Retrieve a keyframe by its ID and location
        self.cursor.execute("SELECT id, keyframe_id, x, y FROM keyframes WHERE keyframe_id = %s", (keyframe_id,))
        return self.cursor.fetchone()

    def get_points_along_path(self, path_id):
        # Extract points along a path
        query = '''
            SELECT ST_AsText(ST_PointN(path_geometry, generate_series(1, ST_NPoints(path_geometry))))
            FROM paths WHERE id = %s;
        '''
        self.cursor.execute(query, (path_id,))
        return self.cursor.fetchall()

    def close(self):
        # Close the database connection
        self.conn.commit()
        self.conn.close()
