import os
import fnmatch
import shutil
import glob
import re

from PIL import Image

import numpy as np

from skimage import io
from sklearn.cross_validation import train_test_split

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import to_categorical

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn import DNN

import scipy

class Trainer(object):
    def __init__(self):
        self.image_size = 64
        self.all_files = []
        self.total_images_count = 0
        self.labels = []

        self.tf_data_counter = 0
        self.tf_image_data = None
        self.tf_image_labels = None
        self.tf_x = None
        self.tf_x_test = None
        self.tf_y = None
        self.tf_y_test = None
        
        self.tf_img_prep = None
        self.tf_img_aug = None
        self.tf_network = None

    def resize(self, image_path, output_path):
        print('Resizing')
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        SIZE = self.image_size, self.image_size
        os.chdir(image_path)
        for root, dirnames, filenames in os.walk('.'):
            counter = 1
            for filename in filenames:
                if not filename.endswith(('.jpg', '.jpeg', '.png', '.PNG')):
                    continue

                label = root.replace('./', '')
                
                oldFile = os.path.join(root, filename)
                im = Image.open(oldFile)
                im = im.resize(SIZE, Image.ANTIALIAS)
                
                newFile = os.path.join(output_path, "{}.{}.jpg".format(label, counter))
                im.save(newFile, "JPEG", quality=70)

                counter = counter + 1

    def build_image_filenames_list(self, train_images_path):
        print('Build Image List')
        all_files = glob.glob('{}/*.jpg'.format(train_images_path))
        all_files.extend(glob.glob('{}/*.jpeg'.format(train_images_path)))
        all_files.extend(glob.glob('{}/*.png'.format(train_images_path)))
        all_files.extend(glob.glob('{}/*.PNG'.format(train_images_path)))

        self.all_files = all_files
        self.total_images_count = len(all_files)
        # print('{}:{} processed from {}'.format(len(self.positive_files), len(self.negative_files), train_images_path))

    def build_labels(self):
        print('Extracting Label List')
        labels = list(map(lambda x: re.findall("^([^\.]+)\..*", x.rsplit('/', 1)[-1])[0], self.all_files))
        self.labels = reduce(lambda x,y: x+[y] if not y in x else x, labels,[])

    def init_np_variables(self):
        print('Init Numpy Variables')
        self.tf_image_data = np.zeros((self.total_images_count, self.image_size, self.image_size, 3), dtype='float64')
        self.tf_image_labels = np.zeros(self.total_images_count)

    def add_tf_dataset(self, list_images, labelIndex):
        print('Add TensorFlow Data Set for {}'.format(self.labels[labelIndex]))
        for image_file in list_images:
            try:
                img = io.imread(image_file)
                self.tf_image_data[self.tf_data_counter] = np.array(img)
                self.tf_image_labels[self.tf_data_counter] = labelIndex
                self.tf_data_counter += 1
            except:
                continue

    def process_tf_dataset(self):
        print('Process TensorFlow Data Set')
        # split our tf set in a test and training part
        self.tf_x, self.tf_x_test, self.tf_y, self.tf_y_test = train_test_split(
            self.tf_image_data, self.tf_image_labels, test_size=0.1, random_state=42)

        # encode our labels
        self.tf_y = to_categorical(self.tf_y, len(self.labels))
        self.tf_y_test = to_categorical(self.tf_y_test, len(self.labels))

    def setup_image_preprocessing(self):
        print('Setup Image Preprocessing')
        # normalization of images
        self.tf_img_prep = ImagePreprocessing()
        self.tf_img_prep.add_featurewise_zero_center()
        self.tf_img_prep.add_featurewise_stdnorm()

        # Randomly create extra image data by rotating and flipping images
        self.tf_img_aug = ImageAugmentation()
        #self.tf_img_aug.add_random_flip_leftright()
        self.tf_img_aug.add_random_blur (sigma_max=5.0)
        self.tf_img_aug.add_random_rotation(max_angle=10.)

    def setup_nn_network(self):
        print('Setup neural network structure')

        # our input is an image of 32 pixels high and wide with 3 channels (RGB)
        # we will also preprocess and create synthetic images
        self.tf_network = input_data(shape=[None, self.image_size, self.image_size, 3],
                                    data_preprocessing=self.tf_img_prep,
                                    data_augmentation=self.tf_img_aug)

        # layer 1: convolution layer with 32 filters (each being 3x3x3)
        layer_conv_1 = conv_2d(self.tf_network, 32, 3, activation='relu', name='conv_1')

        # layer 2: max pooling layer
        self.tf_network = max_pool_2d(layer_conv_1, len(self.labels))

        # layer 3: convolution layer with 64 filters
        layer_conv_2 = conv_2d(self.tf_network, 64, 3, activation='relu', name='conv_2')

        # layer 4: Another convolution layer with 64 filters
        layer_conv_3 = conv_2d(layer_conv_2, 64, 3, activation='relu', name='conv_3')

        # layer 5: Max pooling layer
        self.tf_network = max_pool_2d(layer_conv_3, len(self.labels))

        # layer 6: Fully connected 512 node layer
        self.tf_network = fully_connected(self.tf_network, 512, activation='relu')

        # layer 7: Dropout layer (removes neurons randomly to combat overfitting)
        self.tf_network = dropout(self.tf_network, 0.5)

        # layer 8: Fully connected layer with two outputs (pass or fail)
        self.tf_network = fully_connected(self.tf_network, len(self.labels), activation='softmax')

        # define how we will be training our network
        accuracy = Accuracy(name="Accuracy")
        self.tf_network = regression(self.tf_network, optimizer='adam',
                                    loss='categorical_crossentropy',
                                    learning_rate=0.0005, metric=accuracy)

    def train(self, train_images_path):
        print('Train...')
        self.build_image_filenames_list(train_images_path)
        self.build_labels()
        self.init_np_variables()
        for i, label in enumerate(self.labels):
            files = [fn for fn in self.all_files if os.path.basename(fn).startswith(label)]
            if len(files) == 0:
                continue;
            self.add_tf_dataset(files, i)
        
        self.process_tf_dataset()
        self.setup_image_preprocessing()
        self.setup_nn_network()
        
        self.tf_model = DNN(self.tf_network,
                       tensorboard_verbose=3,
                       checkpoint_path='/checkpoint/model_bowling.tfl.ckpt')

        self.tf_model.fit(self.tf_x, self.tf_y, n_epoch=100, shuffle=True,
                     validation_set=(self.tf_x_test, self.tf_y_test),
                     show_metric=True, batch_size=96,
                     snapshot_epoch=True,
                     run_id='model_bowling')

        #todo: save labels
        self.tf_model.save('/model/model_bowling.tflearn')

    def load_model(self, model_path):
        self.init_np_variables()
        self.setup_image_preprocessing()
        self.setup_nn_network()
        self.tf_model = DNN(self.tf_network, tensorboard_verbose=0)        
        #todo: load labels
        self.tf_model.load(model_path)

    def predict_image(self, image_path):
        img = scipy.ndimage.imread(image_path, mode="RGB")
        img = scipy.misc.imresize(img, (self.image_size, self.image_size), interp="bicubic").astype(np.float32, casting='unsafe')
        return self.tf_model.predict([img])