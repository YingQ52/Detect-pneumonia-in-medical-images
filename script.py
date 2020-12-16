import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import pickle
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import glob
import pydicom
from copy import deepcopy
import importlib
from keras.layers import Input, concatenate, Conv2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization


# config
_dir = 'data/'
det_class_path = _dir+'stage_1_detailed_class_info.csv'
bbox_path = _dir+'stage_1_train_labels.csv'
dicom_dir = _dir+'stage_1_train_images/'
test_dicom_dir = _dir+'stage_1_test_images/'
input_dir = _dir+'model.h5'

output_dir = 'experiment/'
output_weights_path = 'experiment/weights.h5'
submission = 'submission.csv'

class_names = ['Lung Opacity', 'No Lung Opacity / Not Normal']
MAX_LENGTH = 1024.0
HEATMAP_SIZE = (16, 16)
DECODE_SIZE = (16, 16)
LOCATION_SIZE = (14, 14)
BATCH_SIZE = 256
LR = 0.0000001

input_names = ['image', 'auxiliary', 'location']
output_names = [
    'classification', 
    'heatmap', 
    #'decode',
]

# Read info
class PatientInfo:
    """
    Thread-safe image generator with imgaug support
    For more information of imgaug see: https://github.com/aleju/imgaug
    """
    
    """
    directory: 'stage_1_train_images/*.dcm'
    """
    def __init__(self, img_dir, train=True, data=None, class_info=None, box_info=None):
        self.img_dir = img_dir
        self.train = train
        if data is None:
            if train:
                self.load_train(img_dir, class_info, box_info)
            else:
                self.load_test(img_dir)
        else:
            self.df = deepcopy(data)
        
    def extract_patientId(self, img_dir):
        dcm_fps = glob.glob(img_dir+'*')
        patientIds = []
        flag = True
        for dcm_fp in dcm_fps:
            if flag:
                flag = False
                i = -1
                while not dcm_fp[i] == '.':
                    i -= 1
                self.img_format = dcm_fp[i:]
            _, tail = os.path.split(dcm_fp[:-len(self.img_format)])
            patientIds.append(tail)
        patientIds = pd.DataFrame(data={'patientId': patientIds}) # np.array(patientIds)
        return patientIds
    
    
    def load_train(self, img_dir, class_info, box_info):
        self.df = self.extract_patientId(img_dir)
        self.df['Lung Opacity'] = None
        self.df['No Lung Opacity / Not Normal'] = None
        self.df['box'] = None
        self.df.set_index('patientId', inplace=True)
        
        df_class = pd.read_csv(class_info).drop_duplicates()
        df_box = pd.read_csv(box_info)
        
        for _, clf in df_class.iterrows():
            self.df.loc[clf['patientId']]['Lung Opacity'] = 1 if clf['class'] == 'Lung Opacity' else 0
            self.df.loc[clf['patientId']]['No Lung Opacity / Not Normal'] = 1 \
                if clf['class'] == 'No Lung Opacity / Not Normal' else 0
            self.df.loc[clf['patientId']]['box'] = [] 
                
        for _, box in df_box.iterrows():
            if box['Target'] > 0 and self.df.loc[box['patientId']]['Lung Opacity'] == 1:
                try:
                    self.df.loc[box['patientId']]['box'].append\
                        ((box['x'], box['y'], box['width'], box['height'], box['Target']))
                except:
                    print("unable to append box at", box['patientId'])
                    return
        
    def load_test(self, img_dir):
        self.df = self.extract_patientId(img_dir)
        self.df['Lung Opacity'] = None
        self.df['No Lung Opacity / Not Normal'] = None
        self.df['box'] = None
        self.df.set_index('patientId', inplace=True)
    
    def split(self, test_size=0.25, random_state=None, shuffle=True):
        train_data, test_data = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle)
        TrainInfo = PatientInfo(self.img_dir, train=self.train, data=train_data)
        TrainInfo.img_format = self.img_format
        TestInfo = PatientInfo(self.img_dir, train=self.train, data=test_data)
        TestInfo.img_format = self.img_format
        return TrainInfo, TestInfo
        
        
    def filterout(self, condition):
        pass
    
# get meta
def get_age_meta(Info):
    '''
    input:
        - Info: output of PatientInfo
    '''
    patientIds = Info.df.index.values
    img_dir = Info.img_dir[:-5]
    img_format = Info.img_format
    
    ages = np.array([float(pydicom.read_file(os.path.join(img_dir, xid+img_format))\
            [pydicom.tag.Tag(16, 16 + 16**3)].value) for xid in patientIds])
    return ages.mean(), ages.std()


# generator
class AugmentedLabelSequence(Sequence):
    def __init__(self, PatientInfo, 
                 batch_size=16, 
                 target_size=(224, 224), 
                 location_size=LOCATION_SIZE,
                 heatmap_size=HEATMAP_SIZE,
                 decode_size=DECODE_SIZE,
                 input_names=('image', 'auxiliary', 'location'),
                 output_names=('classification', 'heatmap', 'decode'),
                 augmenter=None, 
                 verbose=0, 
                 steps=None,
                 shuffle_on_epoch_end=True, 
                 random_state=1):  
        
        self.df = PatientInfo.df
        self.train = PatientInfo.train
        self.img_dir = PatientInfo.img_dir[:-5]
        self.img_format = PatientInfo.img_format
        
        self.batch_size = batch_size
        self.location_size = location_size
        self.target_size = target_size
        self.heatmap_size = heatmap_size
        self.decode_size = decode_size
        self.input_names = input_names
        self.output_names = output_names
        
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.Xid) / float(self.batch_size)))
        else:
            self.steps = int(steps)
            
    def __bool__(self):
        return True

    def __len__(self):
        return self.steps
        
    def __getitem__(self, idx):
        batch_x = self.extract_x(
            idx * self.batch_size, 
            min((idx + 1) * self.batch_size, len(self.Xid)),
        )
        if self.train:
            batch_y = self.extract_y(
                idx * self.batch_size, 
                min((idx + 1) * self.batch_size, len(self.Xid))
            )
            return batch_x, batch_y
        else:
            return batch_x
        
        
    def extract_image(self, id_start, id_end):
        batch_x_id = self.Xid[id_start : id_end]
        batch_image = np.asarray([self.load_image(xid) for xid in batch_x_id])
        batch_image = self.transform_batch_images(batch_image, batch_x_id)
        return batch_image
    
    def extract_dcm(self, dcm):
        gender = dcm[pydicom.tag.Tag(16, 64)].value
        age = int(dcm[pydicom.tag.Tag(16, 16 + 16**3)].value)
        age = (age - ages_mean) / ages_std
        return [1 if gender=='M' else -1 if gender=='F' else 0, age]
    
    def extract_auxiliary(self, id_start, id_end):
        batch_x_id = self.Xid[id_start : id_end]
        return np.array([self.extract_dcm(pydicom.read_file(os.path.join(self.img_dir, xid+self.img_format))) \
                         for xid in batch_x_id])
    
    def extract_location(self, id_start, id_end):
        x = np.array([[[(float(i) / self.location_size[0]) * 2 - 1, \
                        (float(j) / self.location_size[1]) * 2 - 1] \
                      for j in range(self.location_size[1])] \
                      for i in range(self.location_size[0])
                     ])
        return np.array([x] * (id_end - id_start))
    
    def extract_x(self, id_start, id_end):
        extract_functions = {
            'image': self.extract_image,
            'auxiliary': self.extract_auxiliary,
            'location': self.extract_location,
        }
        x = dict((k, extract_functions[k](id_start, id_end)) for k in self.input_names)
        return x

    def extract_classification(self, id_start, id_end):
        return self.y[id_start: id_end, 0:2]
        
    def BBox2heatmap(self, bboxs):
        # TODO: generate elipse heatmap
        heatmap = np.zeros(self.heatmap_size)
        for bbox in bboxs:
            x0 = int(np.floor(bbox[0] / MAX_LENGTH * (self.heatmap_size[0]))) - 1
            y0 = int(np.floor(bbox[1] / MAX_LENGTH * (self.heatmap_size[1]))) - 1
            x1 = int(np.floor((bbox[0] + bbox[2]) / MAX_LENGTH * (self.heatmap_size[0]))) - 1
            y1 = int(np.floor((bbox[1] + bbox[3]) / MAX_LENGTH * (self.heatmap_size[0]))) - 1
            heatmap[x0:x1+1, y0:y1+1] = 1
        return heatmap.reshape(self.heatmap_size[0] * self.heatmap_size[1])
        
    def extract_heatmap(self, id_start, id_end):
        return np.array([self.BBox2heatmap(self.y[i, 2]) for i in range(id_start, id_end)])
    
    def original_img(self, xid):
        image_path = os.path.join(self.img_dir, xid+self.img_format)
        img = pydicom.read_file(image_path).pixel_array
        img = img/255.
        img = resize(img, self.decode_size)
        return img.reshape(self.decode_size[0] * self.decode_size[1])
    
    def extract_decode(self, id_start, id_end):
        return np.array([self.original_img(self.Xid[i]) for i in range(id_start, id_end)])
        
    def extract_y(self, id_start, id_end):
        extract_functions = {
            'classification': self.extract_classification,
            'heatmap': self.extract_heatmap,
            'decode': self.extract_decode,
        }
        y = dict((k, extract_functions[k](id_start, min(id_end,len(self.Xid)) for k in self.output_names)
        return y
    
    def load_image(self, xid):
        image_path = os.path.join(self.img_dir, xid+self.img_format)
        img = pydicom.read_file(image_path).pixel_array
        img = img/255.
        img = np.stack((img,)*3, -1)
        img = resize(img, self.target_size)
        return img

    def transform_batch_images(self, batch_x, batch_x_id):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        try:
            batch_x = (batch_x - imagenet_mean) / imagenet_std
        except:
            print('batch transform failed at id={}'.format(batch_x_id))
        return batch_x
    
    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.
        """
        if not self.train:
            return None
        return self.extract_y(0, self.steps*self.batch_size)
    
    def prepare_dataset(self):
        df = self.df.sample(frac=1., random_state=self.random_state)
        self.Xid, self.y = np.array(df.index), df[['Lung Opacity', 'No Lung Opacity / Not Normal', 'box']].as_matrix()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()
            

 
# model
class ModelFactory:
    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layer="mixed10",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7b_ac",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260",
            ),
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]
    
    def add_inputs(self, base_model, aux_input, location_input):
        # freeze layers
        for layer in base_model.layers:
            layer.trainable = False
            
        #location_input = Input(shape=(14, 14, 2), name='location')
        x = base_model.get_layer(name='pool4_relu').output
        x = concatenate([x, location_input], axis=3, name='pool4_concatenate_mdf')
        x = Conv2D(512, (1, 1), name='pool4_conv_mdf', bias=False, trainable=True)(x)
        x = base_model.get_layer(name='pool4_pool')(x)
        
        for block in range(1, 17):
            y = x
            for i in range(2):
                x = base_model.get_layer(name='conv5_block{}_{}_bn'.format(block,i))(x)
                x = base_model.get_layer(name='conv5_block{}_{}_relu'.format(block,i))(x)
                x = base_model.get_layer(name='conv5_block{}_{}_conv'.format(block,i+1))(x)
            x = base_model.get_layer(name='conv5_block{}_concat'.format(block))([y, x])
        
        x = base_model.get_layer(name='bn')(x)
        x = base_model.get_layer(name='relu')(x)
        x = base_model.get_layer(name='avg_pool')(x)
        
        x = concatenate([x, aux_input], name='concatenate_aux')
        model = Model(inputs=base_model.inputs+[location_input, aux_input], output=x)
        return model
    
    def add_outputs(self, base_model, class_num, heatmap_size, 
                    activations=("sigmoid", "sigmoid"), names=("classification", "heatmap")):
        classification = Dense(class_num, activation=activations[0], name=names[0])(
            base_model.output
        )
        heatmap = Dense(heatmap_size[0] * heatmap_size[1], activation=activations[1], name=names[1])(
            base_model.output
        )
        model = Model(inputs=base_model.inputs, outputs=[classification, heatmap])
        return model
    
    def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
                  weights_path=None, input_shape=None, 
                  heatmap_size=HEATMAP_SIZE, decode_size=DECODE_SIZE):

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                "keras.applications.{}".format(self.models_[model_name]['module_name'])
            ),
            model_name)

        # inputs:
        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]

        img_input = Input(shape=input_shape, name='image')
        aux_input = Input(shape=(2,), name='auxiliary')
        loc_input = Input(shape=(LOCATION_SIZE[0], LOCATION_SIZE[1], 2), name='location')

        # base model:
        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        # add inputs and outputs:
        model = self.add_inputs(base_model, aux_input, loc_input)
        model = self.add_outputs(base_model, len(class_names), heatmap_size)
        
        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            print("load model weights_path: {}".format(weights_path))
            model.load_weights(weights_path)
        return model


# callback
import json
import keras.backend as kb
import numpy as np
import os
import shutil
import warnings
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score


class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """
    def __init__(self, sequence, class_names, weights_path, stats=None, workers=1):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.workers = workers
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_{}".format(os.path.split(weights_path)[1]),
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.

        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print("current learning rate: {}".format(self.stats['lr']))

        """
        y_hat shape: (#samples, len(class_names))
        y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
        """
        y_hat = self.model.predict_generator(self.sequence, workers=self.workers)
        y = self.sequence.get_y_true()

        print("*** epoch#{} dev auroc ***".format(epoch + 1))
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y['classification'][:, i], y_hat['classification'][:, i])\
                + 0
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print("{}. {}: {}".format(i+1, self.class_names[i], score))
        print("*********************************")

        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print("mean auroc: {}".format(mean_auroc))
        if mean_auroc > self.stats["best_mean_auroc"]:
            print("update best auroc from {} to {}".format(self.stats['best_mean_auroc'], mean_auroc))

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print("update log file: {}".format(self.best_auroc_log_path))
            with open(self.best_auroc_log_path, "a") as f:
                f.write("(epoch#{}) auroc: {}, lr: {}\n".format(epoch + 1, mean_auroc, self.stats['lr']))

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print("update model file: {} -> {}".format(self.weights_path, self.best_weights_path))
            self.stats["best_mean_auroc"] = mean_auroc
            print("*********************************")
        return

class MultiGPUModelCheckpoint(Callback):
    """
    Checkpointing callback for multi_gpu_model
    copy from https://github.com/keras-team/keras/issues/8463
    """
    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(Callback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


# optimizer and layer freezing
from keras.optimizers import Adam

adam = Adam(lr=LR)#, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

def compile_model(model):
    model.compile(
        optimizer=adam, 
        loss={
            'classification': 'binary_crossentropy', 
            'heatmap': 'binary_crossentropy',
            #'decode': 'mean_squared_error',
        },
        loss_weights={
            'classification': 1.0, 
            'heatmap': 5.0,
            #'decode': 0.1,
        },
    )
    return model

def unfreeze(model, conv_and_block):
    prefixes = []
    for prefix in conv_and_block:
        prefixes.append('conv{}_block{}'.format(prefix[0], prefix[1]))
    for layer in model.layers:
        for prefix in prefixes:
            if len(layer.name) >= len(prefix) and layer.name[:len(prefix)] == prefix:
                layer.trainable = True
    return model

def freeze_compile(my_model, freeze, start_layer=0, end_layer=-1):
    for layer in my_model.layers[start_layer:end_layer]:
        # print(layer.name)
        layer.trainable = not freeze
    my_model.compile(
        optimizer='adam', 
        loss={
            'classification': 'binary_crossentropy', 
            'heatmap': 'binary_crossentropy',
            #'decode': 'mean_squared_error',
        },
        loss_weights={
            'classification': 1.0, 
            'heatmap': 3.0,
            #'decode': 0.1,
        },
    )
    return my_model
    
# GPU settings
from keras.utils import multi_gpu_model
gpus = 1
if gpus > 1:
    my_model = multi_gpu_model(my_model, gpus)
    checkpoint = MultiGPUModelCheckpoint(
        filepath=output_weights_path,
        base_model=my_model,
    )
else:
    checkpoint = ModelCheckpoint(
        output_weights_path,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )
    

# Balanced weights
import random

# classification sample counts:
def get_classification_sample_counts(df, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts

# heatmap sample counts
def get_heatmap_sample_counts(data_generator, sample=None):
    if sample is None:
        sample = int(data_generator.steps / 10)
    sample_steps = random.sample(range(data_generator.steps), sample)
    heatmap_total_pixels = data_generator.heatmap_size[0] * data_generator.heatmap_size[1]
    ratio_sum = 0
    for step in sample_steps:
        _, batch_y = data_generator[step]
        positive = sum(sum(batch_y['heatmap']))
        ratio = positive * 1.0 / heatmap_total_pixels / len(batch_y['heatmap'])
        ratio_sum += ratio
        # print(step, ratio)
    return sample, {'heat_ratio': ratio_sum}

def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training

    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 

    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = [x for x in class_positive_counts.keys()]
    label_counts = np.array([x for x in class_positive_counts.values()])
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights


################## RUN ##################
PatientInfo_train = PatientInfo(dicom_dir+'*.dcm', train=True, data=None, class_info=det_class_path, box_info=bbox_path)
ages_mean, ages_std = get_age_meta(PatientInfo_train)
Label_generator = AugmentedLabelSequence(PatientInfo_train, batch_size=BATCH_SIZE)
'''my_model = ModelFactory().get_model(
    class_names,
    heatmap_size=HEATMAP_SIZE,
)
my_model.summary()'''
my_model = pickle.load(open(input_dir, 'rb'))
my_model = unfreeze(my_model, [(3, x) for x in range(1, 17)])
my_model = compile_model(my_model)

clf_positive_weights_multiply = 1
clf_total_count, clf_class_positive_counts = \
    get_classification_sample_counts(PatientInfo_train.df, class_names)
print(clf_total_count, clf_class_positive_counts)
classification_class_weights = \
    get_class_weights(clf_total_count, clf_class_positive_counts, clf_positive_weights_multiply)
classification_class_weights

heatmap_class_weights = [{0: 0.03924560546875, 1: 0.96075439453125}]
heatmap_class_weights *= HEATMAP_SIZE[0] * HEATMAP_SIZE[1]


train_steps = 'auto'
#train_steps = 2
if train_steps == 'auto':
    train_steps = Label_generator.steps
epochs = 1
generator_workers = 8

my_model = freeze_compile(my_model, True, start_layer=0, end_layer=-6)

# callback settings
training_stats = {}
generator_workers = 8  # worker number of the image generators
batch_size = 32
# patience parameter used for ReduceLROnPlateau callback
# If val_loss doesn't decrease for x epochs, learning rate will be reduced by factor of 10.
patience_reduce_lr=1
# minimun learning rate
min_lr=1e-8

auroc = MultipleClassAUROC(
    sequence=Label_generator,
    class_names=class_names,
    weights_path=output_weights_path,
    stats=training_stats,
    workers=generator_workers,
)
        
callbacks = [
    checkpoint,
    TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_reduce_lr,
                      verbose=1, mode="min", min_lr=min_lr),
    auroc,
]


history = my_model.fit_generator(
    generator=Label_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    callbacks=callbacks,
    class_weight={
        'classification': classification_class_weights,
        'heatmap': heatmap_class_weights,
        #'decode': {0: 0.5, 1:0.5},
    },
    workers=generator_workers,
    shuffle=True,
)

pickle.dump(my_model, open( "model.h5", "wb" ))
pickle.dump(history, open( "history.h5", "wb" ))
