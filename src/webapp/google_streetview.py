import cv2
import numpy as np
import urllib

API_KEY = 'AIzaSyCW38Y73-X6r2IVg6GHexRgu-X07uIlHGQ'
IMAGE_SIZE = '640x480'

def get_streetview_image(location, heading, pitch, fov):
    url = 'https://maps.googleapis.com/maps/api/streetview?key=%s&size=%s&location=%s&heading=%s&pitch=%s&fov=%s' % (API_KEY, IMAGE_SIZE, location, heading, pitch, fov);
    
    # Download the image, convert it to a numpy array
    response = urllib.urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    _, imagedata = cv2.imencode('.jpg', image)
    return imagedata
