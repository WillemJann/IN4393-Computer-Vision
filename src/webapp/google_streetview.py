import cv2
import glob
import numpy as np
import os
import urllib

API_KEY = 'AIzaSyCW38Y73-X6r2IVg6GHexRgu-X07uIlHGQ'
IMAGE_SIZE = '640x480'
IMAGE_FOLDER = '../../data/streetview_images/'

def download_image(location, heading, pitch, fov):
    url = 'https://maps.googleapis.com/maps/api/streetview?key=%s&size=%s&location=%s&heading=%s&pitch=%s&fov=%s' % (API_KEY, IMAGE_SIZE, location, heading, pitch, fov);
    
    # Download the image, convert it to a numpy array
    response = urllib.urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    return image

def get_image(location, heading, pitch, fov):
    image = download_image(location, heading, pitch, fov)
    
    # Return image data
    _, imagedata = cv2.imencode('.jpg', image)
    return imagedata

def save_image(location, heading, pitch, fov):
    image = download_image(location, heading, pitch, fov)
    
    # Determine image number
    image_number = 1
    files = glob.glob(IMAGE_FOLDER + '*.jpg')
    
    if len(files) > 0:
        last_file = max(files)
        file_name = os.path.basename(last_file)
        file_number = int(os.path.splitext(file_name)[0])
        image_number += file_number
    
    # Save image
    cv2.imwrite(IMAGE_FOLDER + '%06d.jpg' % image_number, image)
    
    return True
