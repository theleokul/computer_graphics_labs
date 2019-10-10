import requests
import re
import numpy as np


def __parse_ds(content):
    dataset = content.split('\n')
    parsed_dataset = []

    for data_piece in dataset:
        res = re.findall(r'^\[\d+\]\[\d+\]', data_piece)

        if len(res) > 0:
            parsed_dataset.append(data_piece.split(', ')[1:])

    return parsed_dataset


def get_ds(lat, lon, level):
    lons = [float(i) for i in range(361)]
    lats = [-90.0 + i for i in range(181)]
    levels = [
        50.0, 137.5, 212.5, 287.5, 362.5, 462.5, 587.5, 700.0, 800.0, 925.0,
        1075.0, 1225.0, 1375.0, 1525.0, 1675.0, 1825.0, 1975.0, 2125.0,
        2275.0, 2425.0, 2575.0, 2770.0
    ]

    lat_0 = lats.index(lat[0])
    lat_1 = lats.index(lat[1])

    lon_0 = lons.index(lon[0])
    lon_1 = lons.index(lon[1])

    level = levels.index(level)

    dataset_url = f'http://144.206.233.183/thredds/dodsC/Data/s362d.nc.ascii?\
                    data[{level}:1:{level}][{lon_0}:1:{lon_1}][{lat_0}:1:{lat_1}]'

    r = requests.get(dataset_url)
    r.raise_for_status()
    parsed_dataset = __parse_ds(r.content.decode())
    parsed_dataset = np.array(parsed_dataset).astype(float)
    parsed_dataset = np.transpose(parsed_dataset)

    return parsed_dataset
