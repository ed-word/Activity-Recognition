import tensorflow as tf
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = tf.contrib.keras.applications.inception_v3.InceptionV3(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = tf.contrib.keras.models.Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = tf.contrib.keras.models.load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = tf.contrib.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))

        x = tf.contrib.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.contrib.keras.applications.inception_v3.preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features