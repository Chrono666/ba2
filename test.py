import argparse

import tensorflow as tf
from model.custom_model import load_model_with_metadata, get_predictions_from_model
from model.dataset import load_classify_data, load_img_for_feature_maps, load_data_for_evaluation
from report_builder.report_generator import ReportGenerator

parser = argparse.ArgumentParser(description='test')
parser.add_argument(
    '--model-dir',
    default='saved_models/2022-04-08-08-54-49',
    metavar='MP',
    help='path to model to load')
parser.add_argument(
    '--data-dir',
    default='data_for_classification',
    metavar='DD',
    help='path to data on which classification is to be performed')

args = parser.parse_args()

if __name__ == '__main__':
    # get available devices and set memory growth
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # initialize the trainings report_builder
    report_generator = ReportGenerator('test', 'report_builder/templates', './')

    model, metadata = load_model_with_metadata(args.model_dir)

    images_for_prediction, file_list = load_classify_data(args.data_dir)

    true_positives, true_negatives, false_positives, false_negatives = get_predictions_from_model(model,
                                                                                                  images_for_prediction,
                                                                                                  file_list)

    true_positives_paths = [el[1] for el in true_positives if len(true_positives) > 0]
    true_negatives_paths = [el[1] for el in true_negatives if len(true_negatives) > 0]
    false_positives_paths = [el[1] for el in false_positives if len(false_positives) > 0]
    false_negatives_paths = [el[1] for el in false_negatives if len(false_negatives) > 0]

    report_generator.generate_folder_structure()

    try:
        report_generator.save_model_architecture(model)
    except ImportError:
        print("Could not save model architecture. Make sure graphviz is installed.")

    try:
        report_generator.save_grad_cam_img(model=model, image_type='tp', file_list=true_positives_paths)
        report_generator.save_grad_cam_img(model=model, image_type='tn', file_list=true_negatives_paths)
        report_generator.save_grad_cam_img(model=model, image_type='fp', file_list=false_positives_paths)
        report_generator.save_grad_cam_img(model=model, image_type='fn', file_list=false_negatives_paths)
    except:
        print("Something went wrong while saving grad cam images.")

    img_for_fm = load_img_for_feature_maps(args.data_dir)
    report_generator.save_feature_maps(model=model, images=img_for_fm)
    report_generator.save_classified_images(false_negatives_paths, img_prefix='false_negatives')
    report_generator.save_classified_images(false_positives_paths, img_prefix='false_positives')
    report_generator.save_classified_images(true_negatives_paths, img_prefix='true_negatives')
    report_generator.save_classified_images(true_positives_paths, img_prefix='true_positives')

    test_data = load_data_for_evaluation(args.data_dir)

    loss, accuracy, recall, precision, auc = model.evaluate(test_data)

    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0
        print('precision or recall is zero')

    report_generator.generate_test_info_page(model_name=metadata['model_name'], dataset_name=metadata['dataset_name'],
                                             dataset_size=metadata['dataset_size'][0],
                                             classified_image_size=len(images_for_prediction),
                                             true_positives=len(true_positives_paths),
                                             true_negatives=len(true_negatives_paths),
                                             false_positives=len(false_positives_paths),
                                             false_negatives=len(false_negatives_paths),
                                             loss=loss, accuracy=accuracy, recall=recall, precision=precision, auc=auc,
                                             f1=f1)

    report_generator.generate_feature_map_page()
    report_generator.generate_grad_cam_page()

    print('\n')
    print('-' * 80)
    print('Test report generated. Check the target folder for the report.')
    print('\n')
