import base64
import csv

from flask import Flask, request, jsonify
from flask.helpers import send_file

from exception import WebAppException, MISSING_ARGUMENTS
import google_streetview

from sign_detection import detect_signs

# The main application:
app = Flask(__name__, static_url_path="", static_folder="")

# Register an error handler on the WebAppException
@app.errorhandler(WebAppException)
def handle_exception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

# Serve main webpage
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
    
    output, recognized_signs = detect_signs(image)
    
    return jsonify(image = base64.b64encode(output.read()), classification = recognized_signs)

# API endpoint for saving a Google Streetview image
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

# API endpoint for retrieving the list of pre-defined locations
@app.route('/api/load_predefined_locations')
def load_predefined_locations():
    with open('../../data/predefined_locations.csv') as file:
        reader = csv.DictReader(file, quoting=csv.QUOTE_NONNUMERIC, fieldnames=('lat', 'lng', 'heading', 'pitch', 'zoom'))
        locations = [row for row in reader]
        
    return jsonify(locations)

# API endpoint for saving a new pre-defined location
@app.route('/api/save_predefined_location')
def save_predefined_location():
    try:
        lat = request.args['lat']
        lng = request.args['lng']
        heading = request.args['heading']
        pitch = request.args['pitch']
        zoom = request.args['zoom']
    except:
        raise WebAppException(error_code=MISSING_ARGUMENTS)
    
    with open('../../data/predefined_locations.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([lat, lng, heading, pitch, zoom])
    
    return ''

# Run application
if __name__ == "__main__":
    app.run(debug = True, static_files={'/dataset': '../../data/dataset_clean'})
