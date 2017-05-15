import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QDesktopWidget,QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

# Import google maps api key
from google_services import api_key

def main():
	lat = '51.99032'
	lng = '4.37358'

	app = QApplication(sys.argv)
	window = QMainWindow()
	window.setWindowTitle('Road Sign Recognition')
	
	# Define window position
	dw = QDesktopWidget()
	width=dw.width()*0.9
	height=dw.height()*0.9
	x = (dw.width()- width) / 2
	y = (dw.height() - height) / 2
	window.setGeometry(x, y, width, height)

	# Define GUI layout
	layout = QVBoxLayout()
	widget = QWidget()
	widget.setLayout(layout)
	window.setCentralWidget(widget)

	# Create Combined Maps and Street View
	map_street_view = QWebEngineView()
	html = get_combined_view(lat,lng)
	map_street_view.setHtml(html)
	layout.addWidget(map_street_view)

	# Create and Add Google Maps View
	map_view = QWebEngineView()
	html = get_map(lat,lng)
	map_view.setHtml(html)
	#layout.addWidget(map_view)

	# Create and Add Google Street View
	street_view = QWebEngineView()
	street_view.load(get_street_view(lat,lng))
	layout.addWidget(street_view)

	# Center the window on the screen and show the window
	window.setCentralWidget(widget)
	window.show()
	sys.exit(app.exec_())

def get_map(lat, lng):
	#api_key='AIzaSyADNx3kLm_coFlLIwPc7TA3cQuOxGFTKDg'
	html = '''
	<!DOCTYPE html>
<html>
  <head>
    <style>
       #map {
        height: 400px;
        width: 100%;
       }
    </style>
  </head>
  <body>
    <div id="map"></div>
    <script>
      function initMap() {
        var delft = {lat: '''+lat+''', lng: '''+lng+'''};
        var map = new google.maps.Map(document.getElementById('map'), {
          zoom: 14,
          center: delft
        });
        var marker = new google.maps.Marker({
          position: delft,
          map: map
        });
      }	
    </script>
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key='''+api_key+'''&callback=initMap">
    </script>
  </body>
</html>
	'''
	return html

def get_street_view(lat, lng):
	url = QUrl('https://maps.googleapis.com/maps/api/streetview?size=600x300&location='+lat+','+lng+'&heading=151.78&pitch=-0.76&key='+api_key)
	return url

def get_combined_view(lat, lng):
	html = '''
	<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Street View side-by-side</title>
    <style>
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      #map, #pano {
        float: left;
        height: 100%;
        width: 50%;
      }
    </style>
  </head>
  <body>
    <div id="map"></div>
    <div id="pano"></div>
    <script>

      function initialize() {
        var fenway = {lat: '''+lat+''', lng: '''+lng+'''};
        var map = new google.maps.Map(document.getElementById('map'), {
          center: fenway,
          zoom: 14
        });
        var panorama = new google.maps.StreetViewPanorama(
            document.getElementById('pano'), {
              position: fenway,
              pov: {
                heading: 34,
                pitch: 10
              }
            });
        map.setStreetView(panorama);
      }
    </script>
    <script async defer
        src="https://maps.googleapis.com/maps/api/js?key='''+api_key+'''&callback=initialize">
    </script>
  </body>
</html>
'''
	return html
if __name__ == '__main__':
	main()