import geopy.distance
from exif import Image


def decimal_coords(coords, ref):
    """
    Convert coordinates to decimal values.
    :param coords:
    :param ref:
    :return:
    """
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == 'W':
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def image_coordinates(image_path):
    """
    Get coordinates from image's exif values.
    :param image_path:
    :return:
    """
    with open(image_path, 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            coords = [decimal_coords(img.gps_latitude, img.gps_latitude_ref),
                      decimal_coords(img.gps_longitude, img.gps_longitude_ref)]
        except AttributeError:
            print('No Coordinates')
    else:
        print('The Image has no EXIF information')

    return {"time": img.datetime_original, "coords": coords}


def get_distance(coords_1, coords_2):
    """
    Get ground distance from lat,lon coordinates.
    :param coords_1:
    :param coords_2:
    :return:
    """
    return geopy.distance.geodesic(coords_1, coords_2).km
