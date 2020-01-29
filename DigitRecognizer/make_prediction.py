import pandas as pd
import numpy as np
import sys
import csv

from train import lenet_5
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    df_test = pd.read_csv('data/test.csv')
    X_test = np.array(df_test)
    print(X_test.shape)

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # load model
    model = lenet_5()
    model.load_weights(sys.argv[1])

    y_preds = model.predict(X_test)
    y_preds = np.argmax(y_preds, axis=1)

    print(y_preds.shape)
    with open('submission.csv', mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['ImageId', 'label'])
        for idx, row in enumerate(y_preds):
            output_writer.writerow([idx+1, row])
