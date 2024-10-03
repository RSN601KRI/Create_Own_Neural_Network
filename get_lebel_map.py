import collections
import json
def get_label_map(cat_name_file):
    # Open and load the JSON file
    with open(cat_name_file, 'r') as file:
        data = json.load(file)
    
    # Convert string keys to integers and store them in a new dictionary
    category_mapping = {int(key): value for key, value in data.items()}
    
    # Sort the dictionary by keys (integers) and return the sorted dictionary
    sorted_category_mapping = dict(sorted(category_mapping.items(), key=lambda item: item[0]))

    return sorted_category_mapping
