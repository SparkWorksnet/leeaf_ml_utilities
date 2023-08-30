import json
import time
from urllib.request import urlopen

fields = ['latitude', 'longitude', 'altitude', 'speed', 'pause', 'heading', 'takePicture', 'gimbalAngle', 'yawAngle',
          'POITarget', 'roundCorners']

from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By

from image_utils import get_distance

options = webdriver.FirefoxOptions()
# options.add_argument("--disable-dev-shm-usage")
# options.add_argument('--disable-extensions')
options.headless = True


def load_json_from_url(url):
    response = urlopen(url)
    return json.loads(response.read())


def load_twinbase_home_point():
    with webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()),
                           options=options) as driver:  # modified
        form_url = "https://twinbase-leeaf.sparkworks.net/"
        driver.get(form_url)
        time.sleep(2)
        elems = driver.find_elements(By.TAG_NAME, "a")
        for elem in elems:
            if elem.text.startswith("Home Location"):
                print(elem.text)
                json_data = load_json_from_url(elem.get_attribute('href') + '/index.json')
                return {'latitude': json_data['geo:lat'], 'longitude': json_data['geo:long']}


def load_twinbase_tree_pois():
    tree_pois = []
    with webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()),
                           options=options) as driver:  # modified
        form_url = "https://twinbase-leeaf.sparkworks.net/"
        driver.get(form_url)
        time.sleep(2)
        elems = driver.find_elements(By.TAG_NAME, "a")
        for elem in elems:
            if elem.text.startswith("Olive Tree"):
                json_data = load_json_from_url(elem.get_attribute('href') + '/index.json')
                if 'geo:lat' in json_data and 'geo:long' in json_data:
                    # print([elem.text, [json_data['geo:lat'], json_data['geo:long']]])
                    tree_pois.append(
                        {
                            'index': int(elem.text.replace('Olive Tree', '')),
                            'name': elem.text,
                            'group': 'xrtku2qpxliuiuk9',
                            'uuid': json_data['dt-id'].split('/')[-1],
                            'latitude': json_data['geo:lat'],
                            'longitude': json_data['geo:long']
                        }
                    )

    tree_pois.sort(key=lambda x: x['index'])
    return tree_pois


def find_closest_tree(tree_pois, img_coordinates):
    # print(img_coordinates)
    minDistance = None
    minDistanceTree = None
    if img_coordinates is not None:
        for loc in tree_pois:
            tree_loc = [loc['latitude'], loc['longitude']]
            distance = get_distance(tree_loc, img_coordinates['coords']) * 1000
            # print(f"Distance From {loc[0]} is {distance} {loc[1]} {img_coordinates['coords']}")
            if minDistance is None or minDistance > distance:
                minDistance = distance
                minDistanceTree = loc
    else:
        minDistance = 0
        minDistanceTree = tree_pois[0]
    return (minDistanceTree, minDistance)
