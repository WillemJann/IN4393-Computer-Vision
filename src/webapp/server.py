from flask import Flask, request, jsonify
from flask.helpers import send_file

from exception import WebAppException, MISSING_ARGUMENTS
import google_streetview

from detection.sign_detection import detect_signs

# The main application:
app = Flask(__name__, static_url_path="", static_folder="")

# Register an error handler on the WebAppException
@app.errorhandler(WebAppException)
def handle_exception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

# Main webpage
@app.route('/')
def index():
    return app.send_static_file('index.html')

# API endpoint for getting images from Google Streetview
@app.route('/api/get_streetview_image')
def get_streetview_image():
    try:
        location = request.args['location']
        heading = request.args['heading']
        pitch = request.args['pitch']
        fov = request.args['fov']
    except:
        raise WebAppException(error_code=MISSING_ARGUMENTS)
    
    imagedata = google_streetview.get_image(location, heading, pitch, fov)
    
    return send_file(imagedata, 'image/jpeg')

@app.route('/api/save_streetview_image')
def save_streetview_image():
    try:
        location = request.args['location']
        heading = request.args['heading']
        pitch = request.args['pitch']
        fov = request.args['fov']
    except:
        raise WebAppException(error_code=MISSING_ARGUMENTS)
    
    google_streetview.save_image(location, heading, pitch, fov)
    
    return ''

# API endpoint for road sign detection on Google Streetview
@app.route('/api/roadsign_detection')
def roadsign_detection():
    try:
        location = request.args['location']
        heading = request.args['heading']
        pitch = request.args['pitch']
        fov = request.args['fov']
    except:
        raise WebAppException(error_code=MISSING_ARGUMENTS)
    
    image = google_streetview.download_image(location, heading, pitch, fov)
    
    result_image, _ = detect_signs(image)
    
    return send_file(result_image, 'image/jpeg')


# Run application
if __name__ == "__main__":
    app.run(debug = True)
