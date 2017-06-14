import glob
import os
import skimage.io
import io

API_KEY = 'AIzaSyCW38Y73-X6r2IVg6GHexRgu-X07uIlHGQ'
IMAGE_SIZE = '640x480'
IMAGE_FOLDER = '../../data/streetview_images/'

def download_image(location, heading, pitch, fov):
    url = 'https://maps.googleapis.com/maps/api/streetview?key=%s&size=%s&location=%s&heading=%s&pitch=%s&fov=%s' % (API_KEY, IMAGE_SIZE, location, heading, pitch, fov);
    image = skimage.io.imread(url)
    
    return image

def get_image(location, heading, pitch, fov):
    image = download_image(location, heading, pitch, fov)
    
    # Return image data
    imagedata = io.BytesIO()
    skimage.io.imsave(imagedata, image, plugin='pil', format_str='jpeg')
    imagedata.seek(0)
    
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
    skimage.io.imsave(IMAGE_FOLDER + '%06d.jpg' % image_number, image)
    
    return True
