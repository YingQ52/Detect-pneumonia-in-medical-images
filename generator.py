import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import glob
import pydicom
from copy import deepcopy

# rsna generator



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
        '''
        dcm_fps = glob.glob(img_dir)
        patients = {}
        for dcm_fp in dcm_fps:
            _, tail = os.path.split(dcm_fp[:-5])
            patientIds.append(tail)
        '''
        #dcm_img = [pydicom.read_file(name).pixel_array for name in dcm_fps]
    
    
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
    
    
class AugmentedLabelSequence(Sequence):
    def __init__(self, PatientInfo, 
                 batch_size=16, 
                 target_size=(224, 224), 
                 augmenter=None, 
                 verbose=0, 
                 steps=None,
                 shuffle_on_epoch_end=True, 
                 random_state=1):  
        
        self.df = PatientInfo.df
        self.train = PatientInfo.train
        self.img_dir = PatientInfo.img_dir
        self.img_format = PatientInfo.img_format
        
        self.batch_size = batch_size
        self.target_size = target_size
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
        batch_x_id = self.Xid[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(xid) for xid in batch_x_id])
        batch_x = self.transform_batch_images(batch_x, batch_x_id)
        if self.train:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y
        else:
            return batch_x

    
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
        '''
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        '''
        return self.y[:self.steps*self.batch_size, :]
    
    def prepare_dataset(self):
        df = self.df.sample(frac=1., random_state=self.random_state)
        self.Xid, self.y = np.array(df.index), df[['Lung Opacity', 'No Lung Opacity / Not Normal', 'box']].as_matrix()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()
            
            
class AugmentedLabelSequence_clf(AugmentedLabelSequence):
    def __init__(self, PatientInfo, 
                 batch_size=16, 
                 target_size=(224, 224), 
                 augmenter=None, 
                 verbose=0, 
                 steps=None,
                 shuffle_on_epoch_end=True, 
                 random_state=1):  
        AugmentedLabelSequence.__init__(self, PatientInfo, 
                 batch_size, 
                 target_size, 
                 augmenter, 
                 verbose, 
                 steps,
                 shuffle_on_epoch_end, 
                 random_state)
        
    def prepare_dataset(self):
        df = self.df.sample(frac=1., random_state=self.random_state)
        self.Xid, self.y = np.array(df.index), df[['Lung Opacity', 'No Lung Opacity / Not Normal']].as_matrix()