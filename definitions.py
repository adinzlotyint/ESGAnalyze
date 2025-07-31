import os
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(ROOT_DIR, 'base_model/')

with open(f"{ROOT_DIR}/config.json", 'r') as config:
  CONFIG = json.load(config)