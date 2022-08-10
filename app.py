from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import yaml
import joblib

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, 'static')
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


class NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def convert_obj_to_int(data_list):
    for index_ in range(0, len(data_list[0])):
        if type(data_list[0][index_]) == str:
            data_list[0][index_] = hash(data_list[0][index_])
    return data_list


def predict(data, param_path):
    config = read_params(param_path)
    model_dir_path = config['model_webapp_dir']
    model = joblib.load(model_dir_path)
    data = convert_obj_to_int(data)
    prediction = model.predict(data).tolist()[0]
    return prediction


def validate_input(dict_request):
    types = {'C1': int,
             'banner_pos': int,
             'device_type': int,
             'device_conn_type': int,
             'C14': int,
             'C15': int,
             'C16': int,
             'C17': int,
             'C18': int,
             'C19': int,
             'C20': int,
             'C21': int,
             'site_id': str,
             'site_domain': str,
             'site_category': str,
             'app_id': str,
             'app_domain': str,
             'app_category': str,
             'device_id': str,
             'device_ip': str,
             'device_model': str
             }
    for key, val in dict_request.items():
        try:
            d_type = types[key]
            if not d_type == type(val):
                dict_request[key] = d_type(val)
        except Exception as e:
            raise NotANumber
    return True


def form_response(dict_request, param_path):
    try:
        if validate_input(dict_request):
            data = dict_request.values()
            data = [[i for i in data]]
            response = predict(data, param_path)
            return response
    except NotANumber as e:
        response = str(e)
        return response


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req, params_path)
                return render_template("index.html", response=response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {'error': e}
            return render_template("404.html", error=error)

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
