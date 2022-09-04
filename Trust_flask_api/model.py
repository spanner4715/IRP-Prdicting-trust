import pickle
import gzip

with gzip.open('RFC.pgz','r') as f:
    rfc = pickle.load(f)


def predict(input):
    pred = rfc.predict(input)[0]
    print(pred)
    return pred