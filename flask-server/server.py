from flask import Flask
from flask_restful import Api
from Controllers.extract_data import ExtractData
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
api = Api(app)

# register endpoints
api.add_resource(ExtractData, '/ExtractData')

app.run(debug=True)
