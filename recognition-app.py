from flask      import Flask, Blueprint, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Blueprint('api', __name__)

@api.route('/predict', methods=['POST'])
def handle_predict():
    if request.method == "POST":
        j_obj         = request.get_json()
        request_time  = j_obj["request_time"]
        input_type    = j_obj["input_type"]
        file_id       = j_obj["file_id"]
        input_strokes = j_obj["input_strokes"]

        print(f"Got request with file_id: {file_id}")

        #
        # Do something here.
        #

        # Return results.
        return jsonify({
            "latex"  : "hello",
            "mathml" : "world"
        })


app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5050)
