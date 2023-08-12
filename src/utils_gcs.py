"""
GCS Utils
"""
import os
import io
import pandas as pd
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden, GoogleAPIError, Conflict

class GCSBucket:
    """
    A class for interacting with Google Cloud Storage buckets.

    Attributes:
        bucket_name (str): The name of the Google Cloud Storage bucket.

    """
    def __init__(self, bucket_name):
        try:
            self.bucket_name = bucket_name
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)

        except NotFound as _e:
            raise NotFound(f"Bucket not found: {self.bucket_name}") from _e

    def read_data(self, file_path):
        """
        Reads the contents of a file from the Google Cloud Storage bucket.

        Args:
            file_path (str): The path to the file in the Google Cloud Storage bucket.

        Returns:
            bytes: The contents of the file.

        Raises:
            NotFound: If the specified file does not exist in the bucket.
            Forbidden: If the user does not have access to the specified file.
            GoogleAPIError: If there is an error with the Google Cloud Storage API.

        """
        try:
            blob = self.bucket.blob(file_path)
            content = blob.download_as_bytes()
            return content

        except NotFound as _e:
            print(f"File not found: {file_path}")
            raise _e
        except Forbidden as _e:
            print(f"Access denied: {file_path}")
            raise _e
        except GoogleAPIError as _e:
            print(f"API error: {_e}")
            raise _e

    def save_data(self, file_path, content):
        """
        Saves the contents of a file to the Google Cloud Storage bucket.

        Args:
            file_path (str): The path to the file in the Google Cloud Storage bucket.
            content (bytes): The contents of the file to be saved.

        Raises:
            Forbidden: If the user does not have permission to save the file.
            GoogleAPIError: If there is an error with the Google Cloud Storage API.

        """
        try:
            blob = self.bucket.blob(file_path)
            blob.upload_from_string(content, content_type='application/octet-stream')
        except Forbidden as _e:
            print(f"Access denied: {file_path}")
            raise _e
        except GoogleAPIError as _e:
            print(f"API error: {_e}")
            raise _e

    def download_file(self, file_path, local_path):
        """
        Downloads a file from the Google Cloud Storage bucket to a local file.

        Args:
            file_path (str): The path to the file in the Google Cloud Storage bucket.
            local_path (str): The local path where the file should be downloaded.

        Raises:
            NotFound: If the specified file does not exist in the bucket.
            Forbidden: If the user does not have access to the specified file.
            GoogleAPIError: If there is an error with the Google Cloud Storage API.

        """
        try:
            directory = os.path.dirname(local_path)

            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)

            blob = self.bucket.blob(file_path)
            blob.download_to_filename(local_path)

        except NotFound as _e:
            print(f"File not found: {file_path}")
            raise _e
        except Forbidden as _e:
            print(f"Access denied: {file_path}")
            raise _e
        except GoogleAPIError as _e:
            print(f"API error: {_e}")
            raise _e

    def upload_file(self, source_file_path, destination_blob_name):
        """Upload a file to the bucket.

        Args:
            source_file_path (str): Path to the file to be uploaded.
            destination_blob_name (str): Destination blob name for the file.

        Raises:
            ValueError: If source_file_path or destination_blob_name are not strings, or if
                the file does not exist.
            google.cloud.exceptions.GoogleCloudError: If the upload fails due to network issues.
        """
        if not isinstance(source_file_path, str):
            raise ValueError("source_file_path should be a string")
        if not isinstance(destination_blob_name, str):
            raise ValueError("destination_blob_name should be a string")

        if not os.path.isfile(source_file_path):
            raise ValueError(f"File not found: {source_file_path}")

        blob = self.bucket.blob(destination_blob_name)

        try:
            blob.upload_from_filename(source_file_path)
        except Exception as _e:
            print(_e)
            raise ValueError(f"Failed to upload file: {destination_blob_name}. Error: {_e}") from _e

        return f"gs://{self.bucket.name}/{destination_blob_name}"
    
    def read_metadata_csv(self):
        """
        Reads the 'meta_data.csv' file from the GCS bucket and returns it as a Pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the 'meta_data.csv' file.

        Raises:
            google.cloud.exceptions.NotFound: If the 'meta_data.csv' file does not exist in the GCS bucket.
            pandas.errors.ParserError: If the 'meta_data.csv' file cannot be parsed as a CSV file.
        """
        try:
            blob = self.bucket.blob('meta_data.csv')
            csv_data = blob.download_as_string()
            meta_data_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
            meta_data_df.sort_values(by="label", inplace=True)
            return meta_data_df
        except NotFound as _e:
            print(f"'meta_data.csv' file not found in GCS bucket '{self.bucket_name}'.")
            raise NotFound(_e) from _e
        except pd.errors.ParserError as _e:
            print("Could not parse 'meta_data.csv' file as a CSV file.")
            raise pd.errors.ParserError(_e) from _e
        except KeyError as _e:
            print(f"KeyError: {_e}")
            raise KeyError(_e) from _e
        
    def update_metadata_csv(self):

        """
        Updates the meta_data file or creates a new one if one doesnt exist.

        Returns:
            None
        """

        meta_data_df = pd.DataFrame(columns=["filenames","label"])

        for blob in self.bucket.list_blobs():
            filename = blob.name

            #ignore the meta_data file
            if "meta" in filename:
                continue
            
            type_subfolder = os.path.split(filename)[0]
            temp = pd.DataFrame([[filename,type_subfolder]], columns = ["filenames","label"])
            meta_data_df = pd.concat([meta_data_df, temp], ignore_index=True)

        meta_data_csv_string = meta_data_df.to_csv(index = False)
        self.save_data(file_path = "meta_data.csv", content = meta_data_csv_string)

def move_files_between_buckets(source_bucket,
                               destination_bucket,
                               source_blob_name,
                               destination_blob_name):
    """
    Move a file from one GCS bucket to another.

    Args:
        source_bucket (GCSBucket): Name of the source bucket.
        destination_bucket (GCSBucket): Name of the destination bucket.
        source_blob_name (str): Name of the source blob/file to move.
        destination_blob_name (str): Name of the destination blob/file to create.

    Raises:
        ValueError: If the source bucket or blob/file does not exist.
        ValueError: If the destination bucket already contains a blob/file with the same name.
        GoogleAPIError: If there is an error with the Google Cloud Storage API that is not related to the specific operation being performed.
        NotFound: If the requested bucket or blob is not found.
        Forbidden: If the user does not have permission to perform the requested operation.
        Conflict: If the destination blob already exists and the `overwrite` argument is set to `False`.

    Returns:
        None
    """
    # Check if both source and destination buckets are of class GCSBucket
    if not isinstance(source_bucket, GCSBucket) or not isinstance(destination_bucket, GCSBucket):
        raise ValueError("Variable source_bucket and destination_bucket need to be a GCSBucket class")
    # Check if both source and destination blobs are strings
    if not isinstance(source_blob_name, str) or not isinstance(destination_blob_name, str):
        raise TypeError("Variables source_blob_name and destination_blob_name have to be string ")

    # Check if the source blob/file exists
    source_blob = source_bucket.bucket.blob(source_blob_name)
    if not source_blob.exists():
        raise ValueError(f"Source blob {source_blob_name} does not exist in bucket {source_bucket.bucket_name}")

    # Check if the destination blob/file already exists
    destination_blob = destination_bucket.bucket.blob(destination_blob_name)
    if destination_blob.exists():
        raise ValueError(f"Destination blob {destination_blob_name} already exists in bucket {destination_bucket.bucket_name}")

    # Copy the file to the destination bucket
    try:
        source_bucket.bucket.copy_blob(source_blob, destination_bucket.bucket, destination_blob_name)
    except (GoogleAPIError, NotFound, Forbidden, Conflict) as _e:
        print(_e)
        raise _e

    # Delete the file from the source bucket
    try:
        source_blob.delete()
    except (GoogleAPIError, NotFound, Forbidden, Conflict) as _e:
        print(_e)
        raise _e