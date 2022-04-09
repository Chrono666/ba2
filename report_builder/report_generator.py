import math
import os
import time
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from model.dataset import save_example_images

from utils.figure_plot import plot_train_figures, plot_grad_cams, plot_kernels, plot_model_architecture, \
    extract_feature_maps_from_conv_layers, plot_classified_images


class ReportGenerator:
    def __init__(self, report_type, template_path, root_directory):
        self.env = Environment(loader=FileSystemLoader(template_path))
        self.report_type = report_type
        self.target_folder_path = os.path.join(root_directory, 'target')
        self.new_report_folder_path = os.path.join(self.target_folder_path, self.__get_datetime_for_folder())
        self.html_folder_path = os.path.join(self.new_report_folder_path, 'html')
        self.image_folder_path = os.path.join(self.new_report_folder_path, 'images')
        if self.report_type == 'train':
            self.train_figure_path = os.path.join(self.image_folder_path, 'train_figures')
            self.example_img_path = os.path.join(self.image_folder_path, 'examples')
            self.kernel_img_path = os.path.join(self.image_folder_path, 'kernels')
        if self.report_type == 'test':
            self.grad_cam_img_path = os.path.join(self.image_folder_path, 'grad_cams')
            self.feature_map_img_path = os.path.join(self.image_folder_path, 'feature_maps')
            self.false_positive_path = os.path.join(self.image_folder_path, 'false_positives')
            self.false_negative_path = os.path.join(self.image_folder_path, 'false_negatives')
            self.true_positive_path = os.path.join(self.image_folder_path, 'true_positives')
            self.true_negative_path = os.path.join(self.image_folder_path, 'true_negatives')
        self.__initialize_templates()

    def __initialize_templates(self):
        if self.report_type == 'train':
            self.info_template = self.env.get_template('trainings-report/info-template.html')
            self.results_template = self.env.get_template('trainings-report/results-template.html')
            self.visual_template = self.env.get_template('trainings-report/visual-template.html')
        if self.report_type == 'test':
            self.test_info_template = self.env.get_template('test-report/test-info-template.html')
            self.grad_cam_template = self.env.get_template('test-report/grad-cam-template.html')
            self.feature_map_template = self.env.get_template('test-report/feature-map-template.html')

    def generate_folder_structure(self):
        if not os.path.isdir(self.target_folder_path):
            os.mkdir(self.target_folder_path)
        os.mkdir(self.new_report_folder_path)
        os.mkdir(self.image_folder_path)
        if self.report_type == 'train':
            os.mkdir(self.train_figure_path)
            os.mkdir(self.example_img_path)
            os.mkdir(self.kernel_img_path)
        if self.report_type == 'test':
            os.mkdir(self.grad_cam_img_path)
            os.mkdir(self.feature_map_img_path)
            os.mkdir(self.false_positive_path)
            os.mkdir(self.false_negative_path)
            os.mkdir(self.true_positive_path)
            os.mkdir(self.true_negative_path)
        os.mkdir(self.html_folder_path)

    @staticmethod
    def __get_datetime_for_folder():
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def save_train_figures_in_folder(self, history):
        plot_train_figures(history, self.train_figure_path)

    def save_example_img(self, input_path):
        save_example_images(input_path, self.example_img_path)

    def save_kernel_img(self, model):
        plot_kernels(path=self.kernel_img_path, model=model)

    def save_model_architecture(self, model):
        plot_model_architecture(model=model, path=self.image_folder_path, file_name='model_architecture.png')

    def save_grad_cam_img(self, model, images_of_heatmap, images, conv_layer_name='block5_conv3'):
        plot_grad_cams(model=model, path=self.grad_cam_img_path, images_for_heatmap=images_of_heatmap, images=images,
                       conv_layer_name=conv_layer_name)

    def save_feature_maps(self, model, images):
        extract_feature_maps_from_conv_layers(model=model, path=self.feature_map_img_path, images=images)

    def save_classified_images(self, img_paths, img_prefix):
        if img_prefix == 'false_negatives':
            plot_classified_images(img_paths=img_paths, output_path=self.false_negative_path, img_prefix=img_prefix)
        elif img_prefix == 'false_positives':
            plot_classified_images(img_paths=img_paths, output_path=self.false_positive_path, img_prefix=img_prefix)
        elif img_prefix == 'true_negatives':
            plot_classified_images(img_paths=img_paths, output_path=self.true_negative_path, img_prefix=img_prefix)
        elif img_prefix == 'true_positives':
            plot_classified_images(img_paths=img_paths, output_path=self.true_positive_path, img_prefix=img_prefix)

    def generate_info_page(self, date=time.asctime(time.localtime(time.time())), optimizer='Adam', learning_rate=0.001,
                           beta_1=0.9, beta_2=0.999, batch_size=64, epochs=100, early_stopping_patience=20,
                           dataset_name='None',
                           dataset_size=0, split_ratio='08-01-01', train_data_size=0, val_data_size=0,
                           test_data_size=0, model_summary=None):
        info_page = self.info_template.render(document_title='Overall Information',
                                              date=date,
                                              optimizer=optimizer,
                                              learning_rate=learning_rate,
                                              beta_1=beta_1,
                                              beta_2=beta_2,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              early_stopping_patience=early_stopping_patience,
                                              dataset_name=dataset_name,
                                              dataset_size=dataset_size,
                                              split_ratio=split_ratio,
                                              train_data_size=train_data_size,
                                              val_data_size=val_data_size,
                                              test_data_size=test_data_size,
                                              model_summary=model_summary
                                              )
        with open(os.path.join(self.html_folder_path, 'info.html'), 'w') as f:
            f.write(info_page)

    def generate_results_page(self, execution_time, epochs, loss, accuracy, recall, precision, auc, f1=0):
        result_page = self.results_template.render(document_title='Results',
                                                   execution_time=execution_time,
                                                   time_per_epoch=execution_time / epochs,
                                                   epochs=epochs,
                                                   loss=loss,
                                                   accuracy=accuracy,
                                                   recall=recall,
                                                   precision=precision,
                                                   auc=auc,
                                                   f1=f1
                                                   )
        with open(os.path.join(self.html_folder_path, 'results.html'), 'w') as f:
            f.write(result_page)

    def generate_visual_page(self):
        visual_page = self.visual_template.render(document_title='Visualizations')
        with open(os.path.join(self.html_folder_path, 'visual.html'), 'w') as f:
            f.write(visual_page)

    def generate_test_info_page(self, model_name, dataset_name, dataset_size, classified_image_size, true_positives,
                                false_positives, false_negatives, true_negatives):
        test_info_page = self.test_info_template.render(document_title='Overall Information',
                                                        model_name=model_name,
                                                        dataset_name=dataset_name,
                                                        dataset_size=dataset_size,
                                                        classified_image_size=classified_image_size,
                                                        true_positives=true_positives,
                                                        false_positives=false_positives,
                                                        false_negatives=false_negatives,
                                                        true_negatives=true_negatives
                                                        )
        with open(os.path.join(self.html_folder_path, 'info.html'), 'w') as f:
            f.write(test_info_page)

    def generate_grad_cam_page(self):
        cam_page = self.grad_cam_template.render(document_title='Grad Cam Images')
        with open(os.path.join(self.html_folder_path, 'grad-cam.html'), 'w') as f:
            f.write(cam_page)

    def generate_feature_map_page(self):
        map_page = self.feature_map_template.render(document_title='Feature Maps')
        with open(os.path.join(self.html_folder_path, 'feature-map.html'), 'w') as f:
            f.write(map_page)
