import json
from collections import OrderedDict

def load_category_names(file_path):
    """Load category names from a JSON file and return them in sorted order."""
    # Read the JSON content from the specified file
    with open(file_path, 'r') as file:
        category_mapping = json.load(file)

    # Convert the keys to integers and create a sorted dictionary
    category_mapping = {int(key): value for key, value in category_mapping.items()}
    sorted_category_mapping = OrderedDict(sorted(category_mapping.items()))

    return sorted_category_mapping
