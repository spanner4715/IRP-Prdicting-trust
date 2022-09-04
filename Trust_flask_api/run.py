import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import model

app = Flask(__name__)
CORS(app)

#Test connection
@app.route('/')
def index():
    return 'hello'

@app.route('/predict', methods = ['POST'])
def postinput():
    insertValues = request.get_json()
    x1 = insertValues['Base_HR']
    x2 = insertValues['Base_EDA']
    x3 = insertValues['HR']
    x4 = insertValues['EDA']
    x5 = insertValues['Neutral']
    x6 = insertValues['Happy']
    x7 = insertValues['Sad']
    x8 = insertValues['Angry']
    x9 = insertValues['Surprised']
    x10 = insertValues['Scared']
    x11 = insertValues['Disgusted']
    x12 = insertValues['EDA_difference']
    x13 = insertValues['HR_difference']
    input = np.array([[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]])

    result = model.predict(input)


    return jsonify({'Trust classification':str(result)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
