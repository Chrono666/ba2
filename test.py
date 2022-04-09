import argparse

from model.custom_model import load_model_with_metadata, classify_with_model, get_predictions_from_model
from model.dataset import load_classify_data, load_images_for_grad_cam
from report_builder.report_generator import ReportGenerator

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '--model-path',
    default='saved_models/2022-04-08-08-54-49',
    metavar='MP',
    help='path to model to load')
parser.add_argument(
    '--data-dir',
    default='data_for_classification',
    metavar='DD',
    help='path to data on which classification is to be performed')

args = parser.parse_args()

# initialize the trainings report_builder
report_generator = ReportGenerator('test', 'report_builder/templates', './')

model, metadata = load_model_with_metadata(args.model_path)

images_for_prediction, _ = load_classify_data(args.data_dir)
images_for_heat_map, images = load_images_for_grad_cam(args.data_dir)

(true_positive, true_negative), (false_positive, false_negative) = get_predictions_from_model(model, args.data_dir)
print('positive_ok: ', true_positive)

report_generator.generate_folder_structure()
try:
    report_generator.save_model_architecture(model)
except:
    print("Could not save model architecture. Make sure graphviz is installed.")

report_generator.save_grad_cam_img(model=model, images_of_heatmap=images_for_heat_map, images=images)
report_generator.save_feature_maps(model=model, images=images_for_heat_map)

report_generator.generate_test_info_page(model_name=metadata['model_name'], dataset_name=metadata['dataset_name'],
                                         dataset_size=22, classified_image_size=len(images_for_prediction),
                                         classified_ok=2, classified_def=1)

report_generator.generate_feature_map_page()
