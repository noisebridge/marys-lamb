import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import cv2
import numpy as np

def create_np_mask(pred_mask):
  pred_mask = np.argmax(pred_mask, axis=-1)
  return pred_mask[0]

class UNetWrapper:
    def __init__(self, output_channels=2):
        base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        self._down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        self._down_stack.trainable = False

        self._up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]

        self._output_channels = output_channels
        self._model = None

    def generate_model(self, ckpt_dir=None):
      inputs = tf.keras.layers.Input(shape=[128, 128, 3])

      # Downsampling through the model
      skips = self._down_stack(inputs)
      x = skips[-1]
      skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections
      for up, skip in zip(self._up_stack, skips):
          x = up(x)
          concat = tf.keras.layers.Concatenate()
          x = concat([x, skip])

      # This is the last layer of the model
      last = tf.keras.layers.Conv2DTranspose(
          self._output_channels, 3, strides=2,
          padding='same')  #64x64 -> 128x128

      x = last(x)

      self._model = tf.keras.Model(inputs=inputs, outputs=x)
      if ckpt_dir is not None:
        self._model.load_weights(ckpt_dir)

      return self._model

    def convert_to_tf_lite(self):
      converter  = tf.lite.TFLiteConverter.from_keras_model(self._model_lite)
      self._model_lite = converter.convert()
      

    def train(self):
      model = unet_model(2)
      model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  loss_weights=[0.5, 3.5],
                  metrics=['accuracy'])
      EPOCHS = 5
      VAL_SUBSPLITS = 5
      VALIDATION_STEPS = n_test_files//BATCH_SIZE//VAL_SUBSPLITS

      model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS)

    # Run a prediction on a single input image
    def predict(self, image):

        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]

        if np.max(image) > 1:
            image = image / 256.
        pred_mask = self._model.predict(image[np.newaxis, :, :, :])
        pred_mask =  create_np_mask(pred_mask)[:,:,np.newaxis]
        return np.array(tf.keras.preprocessing.image.array_to_img(pred_mask))


