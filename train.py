import argparse
import os.path
import time
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

import model.dataset as dataset
from model.custom_model import \
    build_model, save_model_data, train_model, compile_model, set_layers_trainable
from report_builder.report_generator import ReportGenerator

# use random seed to reproduce results
np.random.seed(42)
tf.random.set_seed(42)
log = {}

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    metavar='BZ',
    help='batch size (default: 64)')
parser.add_argument(
    '--pre-learning-rate',
    type=float,
    default=1e-4,
    metavar='PLR',
    help='learning rate for pretraining (default: 0.0001)')
parser.add_argument(
    '--learning-rate',
    type=float,
    default=1e-4,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--beta-1',
    type=float,
    default=0.9,
    help='beta 1 for Adam (default: 150)')
parser.add_argument(
    '--beta-2',
    type=float,
    default=0.999,
    help='beta 2 for Adam (default: 150)')
parser.add_argument(
    '--dropout-rate',
    type=float,
    default=0.25,
    help='dropout rate (default: 0.25)')
parser.add_argument(
    '--pre-epochs',
    type=int,
    default=5,
    help='epochs for pretraining (default: 5)')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='epochs (default: 100)')
parser.add_argument(
    '--early-stopping',
    type=int,
    default=20,
    help='patience value for early stopping (default: 20)')
parser.add_argument(
    '--data-dir',
    default='data',
    metavar='DD',
    help='data dir')

args = parser.parse_args()

if __name__ == '__main__':
    # load data
    preprocess_config = dataset.preprocess_config()

    train_data, val_data, test_data = dataset.load_dataset(args.data_dir, target_size=(224, 224),
                                                           batch_size=args.batch_size, class_mode='binary',
                                                           configuration=preprocess_config)

    dataset_size = train_data.samples + val_data.samples + test_data.samples
    print(dataset_size)

    dataset_name = args.data_dir.split('/')[-3] if args.data_dir.split('/')[-2] == 'balanced_data' else \
        args.data_dir.split('/')[-2]

    # initialize the trainings report_builder
    report_generator = ReportGenerator('train', 'report_builder/templates', './')

    # build the model
    model = build_model(dropout_rate=args.dropout_rate)
    model = compile_model(model, args.pre_learning_rate, args.beta_1, args.beta_2,
                          ['accuracy', 'Recall', 'Precision', 'AUC'])

    time_of_start = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    train_start = time.time()
    set_layers_trainable(model, False)
    history = train_model(model, train_data, val_data, epochs=args.pre_epochs)
    set_layers_trainable(model, True)

    # callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stopping)

    # train the model
    model = compile_model(model, args.learning_rate, args.beta_1, args.beta_2,
                          ['accuracy', 'Recall', 'Precision', 'AUC'])
    history = train_model(model, train_data, val_data, epochs=args.epochs,
                          callbacks=[early_stopping])

    train_end = time.time()
    total_time = timedelta(seconds=(train_end - train_start))

    report_generator.generate_folder_structure()

    loss, accuracy, recall, precision, auc = model.evaluate(test_data)

    save_model_data(model, file_path='saved_models', date=time_of_start, model_name='vgg16', dataset_name=dataset_name,
                    dataset_size=dataset_size, train_data_size=train_data.samples, val_data_size=val_data.samples,
                    test_data_size=test_data.samples)

    f1 = 2 * ((precision * recall) / (precision + recall))

    try:
        report_generator.save_model_architecture(model)
    except ImportError:
        print("Could not save model architecture. Make sure graphviz is installed.")
    report_generator.save_train_figures_in_folder(history)
    report_generator.save_example_img(args.data_dir)
    report_generator.save_kernel_img(model)

    report_generator.generate_info_page(date=time.asctime(time.localtime(time.time())),
                                        learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2,
                                        split_ratio='0.7-0.15-0.15', dataset_name=dataset_name,
                                        dataset_size=str(dataset_size), train_data_size=str(train_data.samples),
                                        val_data_size=str(val_data.samples), test_data_size=str(test_data.samples),
                                        epochs=args.epochs, batch_size=args.batch_size,
                                        early_stopping_patience=args.early_stopping,
                                        )

    report_generator.generate_results_page(execution_time=total_time, epochs=len(history.history['loss']), loss=loss,
                                           accuracy=accuracy, recall=recall, precision=precision, auc=auc, f1=f1
                                           )

    report_generator.generate_visual_page()

    print('-' * 80)
    print('\n')
    print('-' * 80)
    print('Model trained successfully!')
    print('Total time: {}'.format(total_time))

    print(f"loss: {loss}, \n"
          f"accuracy: {accuracy}, \n"
          f"recall: {recall}, \n"
          f"precision: {precision}, \n"
          f"auc: {auc}, \n"
          f"F1: {f1}")
    print('Report was generated and can be found in the target folder.')
    print('\n')
