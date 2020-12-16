import sys
sys.path.append('../CheXNet-Keras')

import json
import shutil
import os
import pickle
from callback import MultipleClassAUROC, MultiGPUModelCheckpoint
from configparser import ConfigParser
from generator import AugmentedLabelSequence  # AugmentedImageSequence
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import concatenate
from models.keras import ModelFactory
from utility import get_sample_counts
from weights import get_class_weights
from augmenter import augmenter


# load original chexnet model
def load_ori_model(config_file="./config.ini"):
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "test", class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError(f"""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print(f"** test_steps: {test_steps} **")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    # load CheXNet model:
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)
    return model


# load model from chexnet and output intermediate layer
def load_intermediate_layer(config_file="./config.ini", index=-2):
    model = load_ori_model(config_file)
    input_layer = model.get_layer(index=0)
    output_layer = model.get_layer(index=index)
    model = Model(inputs=[input_layer.input], outputs=[output_layer.output])
    return model
    

def load_model(config_file="./config.ini", change_arch=False, compile_=True):
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "test", class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError(f"""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print(f"** test_steps: {test_steps} **")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    # load CheXNet model:
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)
    if change_arch:
        #return model
        # input layer, output layer:
        input_layer = model.get_layer(index=0)
        chex_output = model.get_layer(index=-1)
        # add second last layer:
        intermediate_layer = model.get_layer(index=-2)
        rsna_add_layer = Dense(10, activation='relu', name='rsna_add_layer')(intermediate_layer.output)  # params are tentative
        rsna_clf_output = Dense(3, activation='softmax', name='rsna_clf_output')(
            concatenate([rsna_add_layer, chex_output.output]))
        model = Model(inputs=[input_layer.input], outputs=[rsna_clf_output])
        losses = {'rsna_clf_output': 'categorical_crossentropy'}
        if compile_:
            print('** compile **')
            model.compile(optimizer='rmsprop', loss=losses,loss_weights=[1.])
    else:
        if compile_:
            print('** compile **')
            model.compile(optimizer='rmsprop', loss=losses,loss_weights=[1.])
    return model