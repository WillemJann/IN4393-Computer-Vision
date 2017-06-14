import io
import skimage.io

def detect_signs(image):
    # TODO: implement sign detection
    
    # Return results
    output = io.BytesIO()
    skimage.io.imsave(output, image, plugin='pil', format_str='jpeg')
    output.seek(0)
    
    recognized_signs = {}
    
    return output, recognized_signs
