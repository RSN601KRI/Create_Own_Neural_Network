import collections
import json

def get_label_map(cat_name_file):
    with open(cat_name_file, 'r') as f:
        cat_to_name = json.load(f)

    cat_to_name = {int(k): v for k, v in cat_to_name.items()}
    sorted_cat_to_name = collections.OrderedDict(sorted(cat_to_name.items()))

    return sorted_cat_to_name
