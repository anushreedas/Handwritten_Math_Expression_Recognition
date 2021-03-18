import numpy as np
import matplotlib.pyplot as plt

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
        # print(input_strokes)

        # loop through elements of input_strokes[]
        for elem in input_strokes:
            # get list of coords
            ls = elem['points']
            # store the coords in a 2D array
            coords = None
            for subls in ls.split(','):
                # print(subls)
                data = subls.split()
                data = [int(x) for x in data]
                if coords is None:
                    coords = np.array(data)
                else:
                    coords = np.vstack((coords, data))
            x, y = zip(*coords)

        #     plt.plot(x, y, linewidth=2, c='black')
        # output_path = 'outputimg'
        # plt.savefig(output_path + '.png', bbox_inches='tight', dpi=100)
        # plt.gcf().clear()

        # call classifier here
        # Return results.
        return jsonify({
            "latex"  : "x",
            "mathml" : "x"
        })


app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5050)
