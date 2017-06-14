from io import BytesIO
from flask import Flask, request, jsonify
from flask.helpers import send_file

from exception import WebAppException, MISSING_ARGUMENTS
from google_streetview import get_streetview_image

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
@app.route('/api/streetview_image')
def streetview_image():
    try:
        location = request.args['location']
        heading = request.args['heading']
        pitch = request.args['pitch']
        fov = request.args['fov']
    except:
        raise WebAppException(error_code=MISSING_ARGUMENTS)
    
    imagedata = get_streetview_image(location, heading, pitch, fov)
    
    return send_file(BytesIO(imagedata), 'image/jpeg')

# API endpoint for road sign detection on Google Streetview
# TODO