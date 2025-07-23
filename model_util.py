

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image as process_image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model
import tensorflow as tf

# Optimize TensorFlow for faster inference
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

class FastDeepModel():
    '''Optimized MobileNet deep model for faster inference.'''
    def __init__(self):
        self._model = self._define_model()
        print('🚀 Loading Optimized MobileNet...')
        print()

    @staticmethod
    def _define_model(output_layer=-1):
        '''Define a pre-trained MobileNet model with optimizations.'''
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = base_model.layers[output_layer].output
        output = GlobalAveragePooling2D()(output)
        model = Model(inputs=base_model.input, outputs=output)
        
        # Optimize model for inference
        model.compile(optimizer='adam')  # Pre-compile for faster inference
        return model

    @staticmethod
    def preprocess_image_fast(path):
        '''Fast image processing with optimizations.'''
        try:
            if path.startswith('http'):
                # Download with timeout and smaller buffer
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(path, headers=headers, timeout=10, stream=True)
                response.raise_for_status()
                
                # Use PIL for faster processing
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Convert to numpy
                x = np.array(img, dtype=np.float32)
                x = preprocess_input(x)
                return x
            else:
                # Local file processing
                img = process_image.load_img(path, target_size=(224, 224))
                x = process_image.img_to_array(img)
                x = preprocess_input(x)
                return x
                
        except Exception as e:
            print(f"⚠️  Error processing image {path}: {e}")
            return None

    @staticmethod
    def preprocess_image(path):
        '''Backward compatibility wrapper.'''
        return FastDeepModel.preprocess_image_fast(path)

    @staticmethod
    def cosine_distance_fast(input1, input2):
        '''Optimized cosine distance calculation.'''
        # Normalize inputs for faster computation
        input1_norm = input1 / np.linalg.norm(input1, axis=1, keepdims=True)
        input2_norm = input2 / np.linalg.norm(input2, axis=1, keepdims=True)
        
        # Compute cosine similarity
        return np.dot(input1_norm, input2_norm.T)

    @staticmethod
    def cosine_distance(input1, input2):
        '''Backward compatibility wrapper.'''
        return FastDeepModel.cosine_distance_fast(input1, input2)

    def extract_feature(self, generator):
        '''Extract deep features with batch processing optimization.'''
        # Use predict with optimized batch size
        features = self._model.predict(generator, verbose=1, batch_size=32)
        return features

# Maintain backward compatibility
class DeepModel(FastDeepModel):
    '''Backward compatible DeepModel class.'''
    pass

class FastDataSequence(Sequence):
    '''Optimized predict generator with faster processing.'''
    def __init__(self, paras, generation, batch_size=32):
        self.list_of_label_fields = []
        self.list_of_paras = paras
        self.data_generation = generation
        self.batch_size = batch_size
        self.__idx = 0

    def __len__(self):
        '''The number of batches per epoch.'''
        return max(1, int(np.ceil(len(self.list_of_paras) / self.batch_size)))

    def __getitem__(self, idx):
        '''Generate one batch of data with optimizations.'''
        paras = self.list_of_paras[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x, batch_fields = self.data_generation(paras)

        # Filter out None values (failed images)
        valid_pairs = [(x, f) for x, f in zip(batch_x, batch_fields) if x is not None]
        
        if not valid_pairs:
            return np.zeros((0, 224, 224, 3), dtype=np.float32)
        
        valid_x, valid_fields = zip(*valid_pairs)

        if idx == self.__idx:
            self.list_of_label_fields.extend(valid_fields)
            self.__idx += 1

        return np.array(valid_x, dtype=np.float32)

# Maintain backward compatibility
class DataSequence(FastDataSequence):
    '''Backward compatible DataSequence class.'''
    pass
