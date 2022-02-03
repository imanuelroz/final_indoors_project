import json
import os


def get_country_predictions_geojson(country_id_dict, predictions):
    ''' predictions is a list of predictions (length 183)'''
    ''' country_id_dict is a dictionary built as follows: country:id'''

    with open('static/countries.geojson', 'r') as f:
        data = json.load(f)
        for feature in data['features']:
            properties = feature['properties']
            if properties['admin'] in country_id_dict.keys():
                feature['properties']['prediction'] = str(predictions[country_id_dict[properties['admin']]])
            else:
                feature['properties']['prediction'] = 0

        with open('static/tmp/predictions.geojson', 'w') as f:
            json.dump(data, f, indent=4)


'''if __name__ == "__main__":
    countries_dict = {"Italy":1, "United States":0, "Canada":2, "Australia":3}
    predictions = [0.9,0.8,0.5,0.26]
    get_country_predictions_geojson(countries_dict, predictions)'''
