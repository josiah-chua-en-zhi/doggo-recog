"""
Util function for preprocessing
"""
import os
import io
from io import BytesIO
import numpy as np
from PIL import Image
from utils_envvar import EnvVar


class PreprocessingUtils:

    @staticmethod
    def convert_image_from_bytes_to_numpy(image_data: bytes) -> np.ndarray:
        """
        This function converts image data in bytes format to a numpy array format.

        Args:
            image_data (bytes): The image data in bytes format to be converted to numpy array format.

        Returns:
            np.ndarray: The image data in numpy array format.

        Raises:
            TypeError: If the input `image_data` is not of type bytes.
            OSError: If there is an error while opening or loading the image.
        """
        if not isinstance(image_data, bytes):
            raise TypeError("Input image data must be of type bytes.")

        try:

            # Load the image blob data as a numpy array
            image = Image.open(io.BytesIO(image_data))
            image_array = np.asarray(image)

            image_array = np.copy(image_array)
            image_array.flags.writeable = True

            return image_array

        except OSError as _e:

            raise OSError("An error occurred while opening or loading the image.") from _e
        
    @staticmethod
    def convert_image_from_numpy_to_bytes(numpy_array: np.ndarray):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Input image data must be of type numpy ndarray.")

        try:

            # Load the image array into buffer
            mem_buffer = BytesIO()
            np.save(mem_buffer, numpy_array)

            return mem_buffer.getvalue()
        
        except OSError as _e:

            raise OSError("An error occurred while converting image numpy array image to bytes.") from _e
    
    @staticmethod
    def get_new_img_arrays(env_var: EnvVar):
        new_filenames = os.listdir(env_var.new_data_dir)
        for filename in new_filenames:

            image = Image.open(filename)
            image_array = np.asarray(image)

            yield filename, image_array


    @staticmethod
    def change_filepath_file_type(filename: str,
                                new_file_type :str)-> str:
        """
        This changes the file type of a file

        Args:
            filename (str): file name
            new_file_type (str): file type endings string

        Return: 
            destination file path
        Raises:
            TypeError: variables are not string
        """

        if not isinstance(new_file_type, str) or not isinstance(filename, str):
            raise TypeError("foldername and filename have to be string ")

        new_filename, _ = os.path.splitext(filename)
        new_filename += "." + new_file_type
        return new_filename

class ImageProcessor:
    """
    place holder
    """
    def __call__(self, image):
        return image