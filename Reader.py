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
        Return the names of all the groups at the root of the HDF5 file as a list of strings.
        Each group name corresponds to a seed_hex value.
        """
        group_names = []
        try:
            # Iterate through all groups under the root and collect their names
            for group in self.file.root._f_iter_nodes('Group'):
                group_names.append(group._v_name)
        except Exception as e:
            print(f"Error listing groups: {e}")
        return group_names
    def get_header_attributes(self):
        header_string = self.get_file_header()
        # Initialize an empty dictionary to store the extracted parameters
        parameters_dict = {}
        
        # Split the header string into lines
        lines = header_string.split('\n')
        
        # Iterate over each line
        for line in lines:
            # Check if the line contains an assignment (indicated by '=')
            if '=' in line:
                # Split the line into key and value parts
                key, value = line.split('=', 1)
                
                # Clean up whitespace and convert the value to the appropriate type
                key = key.strip()
                value = value.strip()
                
                # Attempt to evaluate numerical values and tuples
                try:
                    # This attempts to convert strings that represent numbers or tuples into actual numerical values or tuples
                    value = eval(value)
                except (NameError, SyntaxError):
                    # If eval fails (e.g., because the value is a string that doesn't represent a number or tuple), leave it as a string
                    pass
                
                # Store the key-value pair in the dictionary
                parameters_dict[key] = value
        
        return parameters_dict
    def get_file_header(self):
            """
            Return the header (root attributes) of the HDF5 file as a string.
            """
            header_str = "File Header:\n"
            try:
                # Access the root group of the HDF5 file
                root_group = self.file.root
                # Iterate through all attributes of the root group and concatenate them into the header string
                for attr in root_group._v_attrs._f_list():
                    value = root_group._v_attrs[attr]
                    header_str += f"{attr}: {value}\n"
            except Exception as e:
                header_str += f"Error accessing file header: {e}\n"
            return header_str