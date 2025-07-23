


'''Optimized Image similarity using deep features.

Recommendation: the threshold of the `DeepModel.cosine_distance` can be set as the following values.
    0.84 = greater matches amount
    0.845 = balance, default
    0.85 = better accuracy
'''

from io import BytesIO
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import os
import datetime
import numpy as np
import requests
import h5py

from model_util import DeepModel, DataSequence, FastDeepModel

class OptimizedImageSimilarity():
    '''Optimized Image similarity with faster processing.'''
    def __init__(self):
        self._tmp_dir = './__generated__'
        self._batch_size = 32  # Increased batch size
        self._num_processes = 8  # Increased processes
        self._model = None
        self._title = []
        self._session = requests.Session()  # Reuse connections
        
        # Configure session for better performance
        adapter = requests.adapters.HTTPAdapter(
            max_retries=2,
            pool_connections=20,
            pool_maxsize=20
        )
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_processes(self):
        return self._num_processes

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @num_processes.setter
    def num_processes(self, num_processes):
        self._num_processes = num_processes

    def _data_generation_fast(self, args):
        '''Fast data generation with threading.'''
        batch_x, batch_fields = [], []
        
        # Use ThreadPoolExecutor for I/O bound operations (image downloads)
        with ThreadPoolExecutor(max_workers=min(self._num_processes, len(args))) as executor:
            future_to_arg = {executor.submit(self._sub_process_fast, arg): arg for arg in args}
            
            for future in as_completed(future_to_arg):
                x, fields = future.result()
                if x is not None:
                    batch_x.append(x)
                    batch_fields.append(fields)

        return batch_x, batch_fields

    def _predict_generator_fast(self, paras):
        '''Build a fast predict generator.'''
        return DataSequence(paras, self._data_generation_fast, batch_size=self._batch_size)

    def _sub_process_fast(self, para):
        '''Optimized sub-process with better error handling and timeouts.'''
        path, fields = para['path'], para['fields']
        try:
            if path.startswith('http'):
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                response = self._session.get(path, headers=headers, timeout=8)
                response.raise_for_status()
                feature = FastDeepModel.preprocess_image_fast(path)
            else:
                feature = FastDeepModel.preprocess_image_fast(path)
                
            return feature, fields

        except requests.exceptions.Timeout:
            print(f'⏰ Timeout downloading {fields[0]}')
        except requests.exceptions.RequestException as e:
            print(f'🌐 Network error downloading {fields[0]}: {e}')
        except Exception as e:
            print(f'⚠️  Error processing {fields[0]}: {e}')

        return None, None

    # Keep original methods for backward compatibility
    def _data_generation(self, args):
        return self._data_generation_fast(args)

    def _predict_generator(self, paras):
        return self._predict_generator_fast(paras)

    @staticmethod
    def _sub_process(para):
        '''Original sub-process method for compatibility.'''
        path, fields = para['path'], para['fields']
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
            res = requests.get(path, headers=headers, timeout=10)
            res.raise_for_status()
            feature = DeepModel.preprocess_image(BytesIO(res.content))
            return feature, fields

        except Exception as e:
            print('Error downloading %s: %s' % (fields[0], e))

        return None, None

    @staticmethod
    def load_data_csv(fname, delimiter=None, include_header=True, cols=None):
        '''Load `.csv` file with optimizations.'''
        assert delimiter is not None, 'Delimiter is required.'

        if include_header:
            usecols = None
            skip_header = 1
            if cols:
                with open(fname, 'r', encoding='utf-8') as f:
                    csv_head = f.readline().strip().split(delimiter)
                usecols = [csv_head.index(col) for col in cols]
        else:
            usecols = None
            skip_header = 0

        data = np.genfromtxt(
            fname,
            dtype=str,
            comments=None,
            delimiter=delimiter,
            encoding='utf-8',
            invalid_raise=False,
            usecols=usecols,
            skip_header=skip_header
        )

        return data if len(data.shape) > 1 else data.reshape(1, -1)

    @staticmethod
    def load_data_h5(fname):
        '''Load `.h5` file.'''
        with h5py.File(fname, 'r') as h:
            data = np.array(h['data'])
        return data

    def save_data_fast(self, title, lines):
        '''Fast save with optimized processing.'''
        # Load model
        if self._model is None:
            self._model = FastDeepModel()

        print(f'🚀 {title}: Fast download starts.')
        start = datetime.datetime.now()

        args = [{'path': line[-1], 'fields': line} for line in lines]

        # Fast prediction
        generator = self._predict_generator_fast(args)
        features = self._model.extract_feature(generator)

        # Save files
        if len(self._title) == 2:
            self._title = []
        self._title.append(title)

        if not os.path.isdir(self._tmp_dir):
            os.mkdir(self._tmp_dir)

        fname_feature = os.path.join(self._tmp_dir, '_' + title + '_feature.h5')
        with h5py.File(fname_feature, mode='w') as h:
            h.create_dataset('data', data=features)
        print(f'💾 {title}: features saved to `{fname_feature}`.')

        fname_fields = os.path.join(self._tmp_dir, '_' + title + '_fields.csv')
        np.savetxt(fname_fields, generator.list_of_label_fields, delimiter='\t', fmt='%s', encoding='utf-8')
        print(f'📄 {title}: fields saved to `{fname_fields}`.')

        print(f'✅ {title}: processing completed!')
        print(f'📊 Amount: {len(generator.list_of_label_fields)}')
        print(f'⏱️  Time consumed: {datetime.datetime.now()-start}')
        print()

    # Keep original method for compatibility
    def save_data(self, title, lines):
        return self.save_data_fast(title, lines)

    def iteration_fast(self, save_header, thresh=0.845, title1=None, title2=None):
        '''Fast iteration with optimized similarity calculation.'''
        if title1 and title2:
            self._title = [title1, title2]

        assert len(self._title) == 2, 'Two inputs are required.'

        feature1 = self.load_data_h5(os.path.join(self._tmp_dir, '_' + self._title[0] + '_feature.h5'))
        feature2 = self.load_data_h5(os.path.join(self._tmp_dir, '_' + self._title[1] + '_feature.h5'))

        fields1 = self.load_data_csv(os.path.join(self._tmp_dir, '_' + self._title[0] + '_fields.csv'), delimiter='\t', include_header=False)
        fields2 = self.load_data_csv(os.path.join(self._tmp_dir, '_' + self._title[1] + '_fields.csv'), delimiter='\t', include_header=False)

        print(f'📊 {self._title[0]}: features loaded, shape {feature1.shape}')
        print(f'📊 {self._title[1]}: features loaded, shape {feature2.shape}')

        print('🧮 Fast iteration starts...')
        start = datetime.datetime.now()

        distances = FastDeepModel.cosine_distance_fast(feature1, feature2)
        indexes = np.argmax(distances, axis=1)

        result = [save_header + ['similarity']]

        for x, y in enumerate(indexes):
            dis = distances[x][y]
            if dis >= thresh:
                result.append(np.concatenate((fields1[x], fields2[y], np.array(['%.5f' % dis])), axis=0))

        if len(result) > 1:
            np.savetxt('result_similarity.csv', result, fmt='%s', delimiter='\t', encoding='utf-8')
            print(f'💾 Results saved to `result_similarity.csv`.')
        else:
            print('⚠️  No matches found above threshold.')

        print(f'✅ Fast iteration completed!')
        print(f'📊 Processed: {len(fields1)*len(fields2):,} comparisons ({len(fields1)} * {len(fields2)})')
        print(f'⏱️  Time consumed: {datetime.datetime.now()-start}')
        print()

        return distances

    # Keep original method for compatibility  
    def iteration(self, save_header, thresh=0.845, title1=None, title2=None):
        return self.iteration_fast(save_header, thresh, title1, title2)

# Maintain backward compatibility
class ImageSimilarity(OptimizedImageSimilarity):
    '''Backward compatible ImageSimilarity class.'''
    pass

if __name__ == '__main__':
    similarity = OptimizedImageSimilarity()

    '''Setup'''
    similarity.batch_size = 32
    similarity.num_processes = 8

    '''Load source data'''
    test1 = similarity.load_data_csv('./demo/test1.csv', delimiter=',')
    test2 = similarity.load_data_csv('./demo/test2.csv', delimiter=',', cols=['id', 'url'])

    '''Save features and fields'''
    similarity.save_data('test1', test1)
    similarity.save_data('test2', test2)

    '''Calculate similarities'''
    result = similarity.iteration(['test1_id', 'test1_url', 'test2_id', 'test2_url'], thresh=0.845)
    print('Row for source file 1, and column for source file 2.')
    print(result)
