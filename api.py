from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from api_helpers import preprocess, create_grams, tokenize

app = Flask(__name__)
api = Api(app)

mod = pickle.load(open("api_models/svm.sav", 'rb'))
vec = pickle.load(open("api_models/svm_vec.sav", 'rb'))

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictReview(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        tt = tokenize(user_query)
        text = " ".join(tt)

        dat = vec.transform([text])
        pred = mod.predict(dat)[0]

        if pred == 1:
            return_text = "5-Star!"
        else:
            return_text = "Not quite.. (0 - 4 stars)"

        output = {'prediction': return_text}
        
        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictReview, '/')


if __name__ == '__main__':
    app.run(debug=True, port=9999)