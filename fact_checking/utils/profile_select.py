import os
import json
import random


def select_profile(profiles_name, num_profiles=10):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "utils", "config", profiles_name)
    with open(file_path, "r", encoding="utf-8") as f:
        all_profiles = json.load(f)
    return random.sample(all_profiles, min(num_profiles, len(all_profiles)))
