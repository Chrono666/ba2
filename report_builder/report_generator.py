import os
import time
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from model.dataset import save_example_images
from utils.figure_plot import plot_train_figures, plot_grad_cams, plot_kernels, plot_model_architecture, \
    extract_feature_maps_from_conv_layers, plot_classified_images


class ReportGenerator:
    def __init__(self, report_type, template_path, root_directory):
        """ Initialize the report generator.

        Arguments:
            report_type {str} -- type of report to generate.
            template_path {str} -- path to the template folder.
            root_directory {str} -- path to the root directory.
        """
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
            self.feature_map_img_path = os.path.join(self.image_folder_path, 'feature_maps')
            self.false_positive_path = os.path.join(self.image_folder_path, 'false_positives')
            self.fp_grad_cam_img_path = os.path.join(self.false_positive_path, 'grad_cams')
            self.false_negative_path = os.path.join(self.image_folder_path, 'false_negatives')
            self.fn_grad_cam_img_path = os.path.join(self.false_negative_path, 'grad_cams')
            self.true_positive_path = os.path.join(self.image_folder_path, 'true_positives')
            self.tp_grad_cam_img_path = os.path.join(self.true_positive_path, 'grad_cams')
            self.true_negative_path = os.path.join(self.image_folder_path, 'true_negatives')
            self.tn_grad_cam_img_path = os.path.join(self.true_negative_path, 'grad_cams')
        self.__initialize_templates()

    def __initialize_templates(self):
        """ Initialize the templates based on the report type."""
        if self.report_type == 'train':
            self.info_template = self.env.get_template('trainings-report/info-template.html')
            self.results_template = self.env.get_template('trainings-report/results-template.html')
            self.visual_template = self.env.get_template('trainings-report/visual-template.html')
        if self.report_type == 'test':
            self.test_info_template = self.env.get_template('test-report/test-info-template.html')
            self.grad_cam_template = self.env.get_template('test-report/grad-cam-template.html')
            self.feature_map_template = self.env.get_template('test-report/feature-map-template.html')

    def generate_folder_structure(self):
        """ Generate the folder structure for the report to be saved to."""
        if not os.path.isdir(self.target_folder_path):
            os.mkdir(self.target_folder_path)
        os.mkdir(self.new_report_folder_path)
        os.mkdir(self.image_folder_path)
        if self.report_type == 'train':
            os.mkdir(self.train_figure_path)
            os.mkdir(self.example_img_path)
            os.mkdir(self.kernel_img_path)
        if self.report_type == 'test':
            os.mkdir(self.feature_map_img_path)
            os.mkdir(self.false_positive_path)
            os.mkdir(self.false_negative_path)
            os.mkdir(self.true_positive_path)
            os.mkdir(self.true_negative_path)
            os.mkdir(self.fp_grad_cam_img_path)
            os.mkdir(self.fn_grad_cam_img_path)
            os.mkdir(self.tp_grad_cam_img_path)
            os.mkdir(self.tn_grad_cam_img_path)
        os.mkdir(self.html_folder_path)

    @staticmethod
    def __get_datetime_for_folder():
        """ Get the current datetime for the folder name."""
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def save_train_figures_in_folder(self, history):
        """ Save the training figures in the folder.

        Arguments:
            history (keras model history): The history of the training.
        """
        plot_train_figures(history, self.train_figure_path)

    def save_example_img(self, input_path):
        """ Save the example image in the folder.

        Arguments:
            input_path (str): The path to the image to be saved.
        """
        save_example_images(input_path, self.example_img_path)

    def save_kernel_img(self, model):
        """ Save the kernel images in the folder.

        Arguments:
            model (keras model): The model to be used to save the kernel images.
        """
        plot_kernels(path=self.kernel_img_path, model=model)

    def save_model_architecture(self, model):
        """ Save the model architecture in the folder.

        Arguments:
            model (keras model): The model to be used to save the kernel images.
        """
        plot_model_architecture(model=model, path=self.image_folder_path, file_name='model_architecture.png')

    def save_grad_cam_img(self, model, image_type, file_list, conv_layer_name='block5_conv3'):
        """ Save the grad cam images in the folder.

        Arguments:
            model (keras model): The model to be used for the grad cam.
            image_type (str): The type of the images (false positives, etc.).
            file_list (list): The list of the images to be used for the grad cam.
            conv_layer_name (str): The name of the last convolutional layer.
        """
        if image_type == 'fp':
            path = self.fp_grad_cam_img_path
        elif image_type == 'fn':
            path = self.fn_grad_cam_img_path
        elif image_type == 'tp':
            path = self.tp_grad_cam_img_path
        elif image_type == 'tn':
            path = self.tn_grad_cam_img_path
        else:
            raise ValueError('The image type is not valid.')

        plot_grad_cams(model=model, path=path, file_list=file_list, conv_layer_name=conv_layer_name)

    def save_feature_maps(self, model, images):
        """ Save the feature maps in the folder.

        Arguments:
            model (keras model): The model used for generating the feature maps.
            images (list): The images used for generating the feature maps.
        """
        extract_feature_maps_from_conv_layers(model=model, path=self.feature_map_img_path, images=images)

    def save_classified_images(self, img_paths, img_prefix):
        """ Save classified images in their corresponding folders.

        Arguments:
            img_paths (list): The paths to the images to be classified.
            img_prefix (str): The prefix of the images.
        """
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
        """ Generates the info page of the training report.

        Arguments:
            date (str): The date of the training.
            optimizer (str): The optimizer used for the training.
            learning_rate (float): The learning rate used for the training.
            beta_1 (float): The beta_1 used for the training.
            beta_2 (float): The beta_2 used for the training.
            batch_size (int): The batch size used for the training.
            epochs (str): The number of epochs used for the training.
            early_stopping_patience (str): The patience used for the early stopping.
            dataset_name (str): The name of the dataset used for the training.
            dataset_size (str): The size of the dataset used for the training.
            split_ratio (str): The split ratio used for the training.
            train_data_size (str): The size of the training data used for the training.
            val_data_size (str): The size of the validation data used for the training.
            test_data_size (str): The size of the test data used for the training.
            model_summary (str): The summary of the model used for the training.
        """
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

    def generate_results_page(self, execution_time, epochs, loss, accuracy, recall, precision, auc,
                              date=time.asctime(time.localtime(time.time())), f1=0):
        """ Generates the results page of the training report.

        Arguments:
            execution_time (timedelta): The time the model took to train.
            epochs (int): The number of epochs the model trained.
            loss (float): The loss of the model.
            accuracy (float): The accuracy of the model.
            recall (float): The recall of the model.
            precision (float): The precision of the model.
            auc (float): The auc of the model.
            date (str): The date the model was trained.
            f1 (float): The f1 of the model.
        """
        result_page = self.results_template.render(document_title='Results',
                                                   execution_time=execution_time,
                                                   date=date,
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

    def generate_visual_page(self, date=time.asctime(time.localtime(time.time()))):
        """ Generates the visual page of the training report.

        Arguments:
            date (str): The date the model was trained.
        """
        visual_page = self.visual_template.render(document_title='Visualizations', date=date)
        with open(os.path.join(self.html_folder_path, 'visual.html'), 'w') as f:
            f.write(visual_page)

    def generate_test_info_page(self, model_name, dataset_name, dataset_size, classified_image_size, true_positives,
                                false_positives, false_negatives, true_negatives, loss, accuracy, recall, precision,
                                auc, f1, date=time.asctime(time.localtime(time.time()))):
        """ Generates the info page of the test report.

        Arguments:
            model_name (str): The name of the model.
            dataset_name (str): The name of the dataset which was used to train the model.
            dataset_size (int): The size of the dataset.
            classified_image_size (int): The number of the classified images.
            true_positives (int): The number of true positives.
            false_positives (int): The number of false positives.
            false_negatives (int): The number of false negatives.
            true_negatives (int): The number of true negatives.
            date (str): The date the classification was done.
            loss (float): The loss of the model.
            accuracy (float): The accuracy of the model.
            recall (float): The recall of the model.
            precision (float): The precision of the model.
            auc (float): The auc of the model.
            f1 (float): The f1 of the model.
        """
        test_info_page = self.test_info_template.render(document_title='Overall Information',
                                                        model_name=model_name,
                                                        date=date,
                                                        dataset_name=dataset_name,
                                                        dataset_size=dataset_size,
                                                        classified_image_size=classified_image_size,
                                                        true_positives=true_positives,
                                                        false_positives=false_positives,
                                                        false_negatives=false_negatives,
                                                        true_negatives=true_negatives,
                                                        loss=loss,
                                                        accuracy=accuracy,
                                                        recall=recall,
                                                        precision=precision,
                                                        auc=auc,
                                                        f1=f1
                                                        )
        with open(os.path.join(self.html_folder_path, 'info.html'), 'w') as f:
            f.write(test_info_page)

    def generate_grad_cam_page(self, date=time.asctime(time.localtime(time.time()))):
        """ Generates the grad-cam page of the test report.

        Arguments:
            date (str): The date the classification was done.
        """
        cam_page = self.grad_cam_template.render(document_title='Grad Cam Images', date=date)
        with open(os.path.join(self.html_folder_path, 'grad-cam.html'), 'w') as f:
            f.write(cam_page)

    def generate_feature_map_page(self, date=time.asctime(time.localtime(time.time()))):
        """ Generates the feature map page of the test report.

        Arguments:
            date (str): The date the classification was done.
        """
        map_page = self.feature_map_template.render(document_title='Feature Maps', date=date)
        with open(os.path.join(self.html_folder_path, 'feature-map.html'), 'w') as f:
            f.write(map_page)
