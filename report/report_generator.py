import os
import time
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from model.dataset import save_example_images
from model.figure_plot import plot_train_figures, plot_kernels, plot_model_architecture


class ReportGenerator:
    def __init__(self, report_type, template_path, root_directory):
        self.env = Environment(loader=FileSystemLoader(template_path))
        self.report_type = report_type
        self.target_folder_path = os.path.join(root_directory, 'target')
        self.new_report_folder_path = os.path.join(self.target_folder_path, self.__get_datetime_for_folder())
        self.html_folder_path = os.path.join(self.new_report_folder_path, 'html')
        self.image_folder_path = os.path.join(self.new_report_folder_path, 'images')
        self.__initialize_templates()

    def __initialize_templates(self):
        if self.report_type == 'train':
            self.info_template = self.env.get_template('trainings-report/info-template.html')
            self.results_template = self.env.get_template('trainings-report/results-template.html')
            self.visual_template = self.env.get_template('trainings-report/visual-template.html')
        if self.report_type == 'test':
            self.test_info_template = self.env.get_template('trainings-report/info-template.html')
            self.grad_cam_template = self.env.get_template('trainings-report/grad-cam-template.html')
            self.feature_map_template = self.env.get_template('trainings-report/feature-map-template.html')

    def generate_folder_structure(self):
        if not os.path.isdir(self.target_folder_path):
            os.mkdir(self.target_folder_path)
        os.mkdir(self.new_report_folder_path)
        os.mkdir(self.image_folder_path)
        os.mkdir(self.html_folder_path)

    @staticmethod
    def __get_datetime_for_folder():
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def save_train_figures_in_folder(self, history):
        plot_train_figures(history, history.history, self.image_folder_path)

    def save_example_img(self, input_path):
        save_example_images(input_path, self.image_folder_path)

    def save_kernel_img(self, model):
        plot_kernels(path=self.image_folder_path, model=model)

    def save_model_architecture(self, model):
        plot_model_architecture(model=model, path=self.image_folder_path, file_name='model_architecture.png')

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

    def render_test_info_template(self):
        self.test_info_template.render(document_title='Overall Information')

    def render_grad_cam_template(self):
        self.grad_cam_template.render(document_title='Grad Cam Images')

    def render_feature_map_template(self):
        self.feature_map_template.render(document_title='Feature Maps')
