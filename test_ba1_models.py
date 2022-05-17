import argparse

import tensorflow as tf
from model.custom_model import load_old_ba1_model
from model.dataset import  load_data_for_evaluation

parser = argparse.ArgumentParser(description='test')
parser.add_argument(
    '--model-dir',
    type=str,
    help='path to model to load')
parser.add_argument(
    '--weight-dir',
    type=str,
    help='path to model weights to load')
parser.add_argument(
    '--data-dir',
    type=str,
    help='path to data on which classification is to be performed')
parser.add_argument(
    '--train',
    default=False,
    type=bool,
    help='whether to use test or train data')

args = parser.parse_args()
if __name__ == '__main__':
    # get available devices and set memory growth
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # load model
    model = load_old_ba1_model(args.model_dir, args.weight_dir)

    test_data = load_data_for_evaluation(args.data_dir, args.train)

    loss, accuracy, recall, precision, auc = model.evaluate(test_data)

    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0
        print('precision or recall is zero')

    print('\n')
    print('-' * 80)
    print(f"loss: {loss}, \n"
          f"accuracy: {accuracy}, \n"
          f"recall: {recall}, \n"
          f"precision: {precision}, \n"
          f"auc: {auc}, \n"
          f"F1: {f1}")
