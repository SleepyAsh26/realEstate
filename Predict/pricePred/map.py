#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 04:14:10 2019

@author: sleepyash
"""
#AIzaSyDhw9EWzXaeWfpi1fMfZiRR7Y6tS-fKIUw 
import googlemaps
import pandas as pd
import gmaps
from ipywidgets.embed import embed_minimal_html
from shapely.geometry import Point, Polygon
import json
api_key = 'AIzaSyB47j8P2l_IEm_8mg-sUltopgKIya_PMt8'

Gmaps = googlemaps.Client(key=api_key)
dataset = pd.read_excel('MAPS.xlsx')
data = dataset["Address"]
#data = dataset["Address"] + dataset["District"]
data = list(data)
# Geocoding an address
coordinates = []
for i, place in enumerate(data):
    if i < 100 and place != 'nan':
        try:
            geocode_result = Gmaps.geocode(place)
            coor = geocode_result[0]
            coor = coor["geometry"]
            coor = coor['location']
            lat = coor['lat']
            lng = coor['lng']
            print(lat,lng)
            coordinates.append([lat,lng])
        except:
            pass
json1_file = open('TbilisiPly.json')
json1_str = json1_file.read()
json1_data = json.loads(json1_str)
json1_data = json1_data['geometries'][0]
TbilisiPLY = json1_data['coordinates'][0][0]

gmaps.configure(api_key=api_key)
TbilisiPLY = [(41.816420, 44.680124),
              (41.831395, 44.876235),
              (41.772717, 44.977507),
              (41.673009, 45.007487),
              (41.656818, 44.627870)]
DidiDigomiPLY = [(41.801125, 44.726993), 
                 (41.821502, 44.776487),
                 (41.814003, 44.784271), 
                 (41.787966, 44.784622),
                 (41.787665, 44.770549), 
                 (41.760436, 44.768482),
                 (41.766687, 44.709715)]
DidiDigomi = []
poly = Polygon(DidiDigomiPLY)
for Clat, Clng in coordinates:
    if poly.contains(Point(Clat,Clng)) == True:
        DidiDigomi.append([Clat,Clng])

fig = gmaps.figure()
fig.add_layer(gmaps.drawing_layer(features=[gmaps.Polygon(TbilisiPLY, stroke_color='red', fill_color=(255, 0, 132)) ]))
fig.add_layer(gmaps.drawing_layer(features=[gmaps.Polygon(DidiDigomiPLY, stroke_color='blue', fill_color=(255, 0, 132)) ]))

heatmap_layer = gmaps.heatmap_layer(DidiDigomi)
fig.add_layer(heatmap_layer)
markers = gmaps.symbol_layer(DidiDigomi, fill_color='green', stroke_color='green', scale=2)
fig.add_layer(markers)
embed_minimal_html('export.html', views=[fig])