import sys
import re
import csv
import numpy as np
from train import standard_cnn
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


class ImageWithNames(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filenames_np = np.array(self.filenames)
        self.class_mode = None # so that we only get the images back

    def _get_batches_of_transformed_samples(self, index_array):
        return (super()._get_batches_of_transformed_samples(index_array),
                self.filenames_np[index_array])


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == "__main__":
    # load model
    model = standard_cnn()
    model.load_weights(sys.argv[1])

    # create the iterator to load all test images.
    test_folder = "/workspace/project/data/test"

    imagegen = ImageDataGenerator(rescale=1./255)
    test_iter = ImageWithNames(test_folder, imagegen, target_size=(32, 32), batch_size=1024, classes=["test"])

    # for idx in range(len(test_iter)):
    with open('initial_submission.csv', mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['id', 'label'])
        for idx in range(len(test_iter)):

            (test_images, test_filenames) =next(test_iter)
            probs = model.predict(test_images)

            predicted_class_indices = np.argmax(probs, axis=1)

            labels = load_label_names()

            predictions = [labels[k] for k in predicted_class_indices]

            for (filename, prediction) in zip(test_filenames, predictions):
                name = re.findall(r"^.*\/(.*).png$", filename)
                print(name[0], prediction)
                output_writer.writerow([name[0], prediction])

    print("Make prediction completed!")
