from tensorflow import lite as tflite
from tensorflow import keras
#from tensorflow_examples.models.pix2pix import pix2pix

import signal

import cv2
import numpy as np

def convert_keras_to_tf_lite(keras_path, tflite_filepath):
  # Convert the model.
  converter = tflite.TFLiteConverter.from_saved_model(keras_path)
  model_tflite = converter.convert()
  # Save the model.
  with open(tflite_filepath, 'wb') as f:
    f.write(tflite_model)
  

def create_np_mask(pred_mask):
  pred_mask = np.argmax(pred_mask, axis=-1)
  return pred_mask[0]

class UNetWrapper:
    def __init__(self, output_channels=2, init_model=True):
        if init_model:
            import tensorflow as tf
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

    # TODO: Make TF Lite interpreter a separate class.
    def load_tf_lite_model(self, model_path):

      # FROM TFLITE MODEL
      print("loading tf lite model")
      # Load the TFLite model and allocate tensors.
      self._interpreter = tflite.Interpreter(model_path=model_path)
      print("created interpreter")
      self._input_details = self._interpreter.get_input_details()
      self._interpreter.resize_tensor_input(self._input_details[0]["index"], [1, 128, 128, 3])
      self._interpreter.allocate_tensors()
      print("allocated tensors")
      # Get input and output tensors.
      self._input_details = self._interpreter.get_input_details()
      self._interpreter.resize_tensor_input(self._input_details[0]["index"], [1, 128, 128, 3])
      self._interpreter.allocate_tensors()
      self._input_details = self._interpreter.get_input_details()
      self._output_details = self._interpreter.get_output_details()
      print("determined input and output")
        
      print(self._output_details)
      print("loaded tf lite model")

    '''
      NOTE: this takes the normalized / resized version of the input, and should not be run directly (run through the main predict() function).
    ''' 
    def _run_tf_lite_model(self, model_input):

      # Test the model on random input data.
      input_shape = self._input_details[0]['shape']
      self._interpreter.set_tensor(self._input_details[0]['index'], model_input)

      self._interpreter.invoke()

      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor.
      output_tensor = self._interpreter.get_tensor(self._output_details[0]['index'])

      # Convert to Numpy array for output
      return output_tensor[0]
      
    def save_keras_model(self):
      tf.saved_model.save(self._model, 'keras_model')

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

        if image.shape[0] > 128:
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]

        if np.max(image) > 1:
            image = image / 256.
        if self._model is None:
            image = image.astype(np.float32)
            pred_mask = self._run_tf_lite_model(image[np.newaxis, :, :, :])
            output = np.argmax(pred_mask, axis=2)
        else: 
            pred_mask = self._model.predict(image[np.newaxis, :, :, :])
            output = np.array(keras.preprocessing.image.array_to_img(pred_mask))
        return output


