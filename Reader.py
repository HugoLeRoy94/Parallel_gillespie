import tables as pt

class CustomHDF5Reader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None

    def open(self):
        """Open the HDF5 file."""
        try:
            self.file = pt.open_file(self.filepath, mode='r')
        except IOError as e:
            print(f"Failed to open file {self.filepath}: {e}")

    def close(self):
        """Close the HDF5 file."""
        if self.file is not None:
            self.file.close()

    def get_measurement_data(self, seed_hex, measurement_type):
        """
        Get data for a specific measurement type and seed.

        Parameters:
            seed_hex (str): Hexadecimal string representing the seed (e.g., '0xb711b').
            measurement_type (str): Type of measurement (e.g., 'Cluster', 'ISF', 'MSD', 'NRG', 'PCF').

        Returns:
            numpy.ndarray: The data array for the requested measurement, or None if not found.
        """
        try:
            node_path = f'/{seed_hex}/{measurement_type}'
            data_node = self.file.get_node(node_path)
            return data_node.read()
        except pt.NoSuchNodeError:
            print(f"No such node: {node_path}")
            return None

    def list_measurements(self, seed_hex):
        """
        List all measurements available for a given seed.

        Parameters:
            seed_hex (str): Hexadecimal string representing the seed.

        Returns:
            list: A list of available measurement types for the given seed.
        """
        measurements = []
        try:
            group_path = f'/{seed_hex}'
            for node in self.file.iter_nodes(group_path, classname='Array'):
                measurements.append(node._v_name)
        except pt.NoSuchNodeError:
            print(f"No such group: {group_path}")
        return measurements
    def list_groups(self):
        """
        List all the groups at the root of the HDF5 file, which correspond to seed_hex values.
        """
        try:
            for group in self.file.root._v_groups:
                print(f"Group/Seed Hex: {group}")
        except Exception as e:
            print(f"Error listing groups: {e}")
    def print_file_header(self):
        """
        Print the header (root attributes) of the HDF5 file.
        """
        try:
            # Access the root group of the HDF5 file
            root_group = self.file.root
            # Iterate through all attributes of the root group
            print("File Header:")
            for attr in root_group._v_attrs._f_list():
                value = root_group._v_attrs[attr]
                print(f"{attr}: {value}")
        except Exception as e:
            print(f"Error accessing file header: {e}")


# Example usage
#if __name__ == "__main__":
#    filepath = 'path_to_your_file.hdf5'
#    reader = CustomHDF5Reader(filepath)
#
#    reader.open()
#    try:
#        # Assuming you have a seed hex value and you're interested in 'MSD' measurements
#        seed_hex = '0xb711b'
#        measurement_type = 'MSD'
#        data = reader.get_measurement_data(seed_hex, measurement_type)
#        if data is not None:
#            print(f"Data for {measurement_type} with seed {seed_hex}: {data}")
#        
#        # List all measurements for a given seed
#        measurements = reader.list_measurements(seed_hex)
#        print(f"Available measurements for seed {seed_hex}: {measurements}")
#
#    finally:
#        reader.close()
