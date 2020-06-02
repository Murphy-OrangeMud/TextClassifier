import tensorflow as tf
import json

from data_utils import Data
from charcnn_origin import CharCNNZhang
'''import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"'''

#设置定量的GPU使用量

'''config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
session = tf.Session(config=config)

#设置最小的GPU使用量

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)'''

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("config.json"))
    # Load training data
    training_data = Data(data_source=config["data"]["training_data_source"],
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()
    # Load validation data
    validation_data = Data(data_source=config["data"]["validation_data_source"],
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()

    model = CharCNNZhang(input_size=config["data"]["input_size"],
                         alphabet_size=config["data"]["alphabet_size"],
                         embedding_size=config["char_cnn_zhang"]["embedding_size"],
                         conv_layers=config["char_cnn_zhang"]["conv_layers"],
                         fully_connected_layers=config["char_cnn_zhang"]["fully_connected_layers"],
                         num_of_classes=config["data"]["num_of_classes"],
                         threshold=config["char_cnn_zhang"]["threshold"],
                         dropout_p=config["char_cnn_zhang"]["dropout_p"],
                         optimizer=config["char_cnn_zhang"]["optimizer"],
                         loss=config["char_cnn_zhang"]["loss"])

    model.test(testing_inputs=validation_inputs,
               testing_labels=validation_labels,
               batch_size=config["training"]["batch_size"])
    # Train model
    '''model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])'''
