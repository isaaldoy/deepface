# recognize.py

import os
from deepface import DeepFace
import logging
import pandas # Import pandas to handle potential attribute access issues if needed

# --- Configuration ---
DB_PATH = "database"  # Path to the folder containing known faces
TEST_IMAGE_PATH = "test_images/test_image1.jpg" # Path to the image you want to test
MODEL_NAME = "VGG-Face" # Or "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
DETECTOR_BACKEND = "opencv" # Or "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8"
DISTANCE_METRIC = "cosine" # Or "euclidean", "euclidean_l2"

# --- Constants based on configuration ---
# It seems DeepFace might return the distance column simply as 'distance'
# Keep the configured metric name for reference or potential future use if API changes
EXPECTED_DISTANCE_COL = 'distance'

# Set logging level to suppress less important messages
# You can change this to logging.INFO or logging.DEBUG for more details
logging.basicConfig(level=logging.WARNING) # Set default level for root logger
logging.getLogger('tensorflow').setLevel(logging.ERROR) # Suppress verbose TF messages
# --- --- --- --- ---

def find_identity(image_path, db_path, model_name, detector_backend, distance_metric):
    """
    Finds the most likely identity of face(s) in an image against a database.

    Args:
        image_path (str): Path to the test image.
        db_path (str): Path to the database folder.
        model_name (str): Face recognition model to use.
        detector_backend (str): Face detector backend to use.
        distance_metric (str): Metric for comparing face embeddings.

    Returns:
        list: A list of pandas DataFrames, one for each detected face in the image_path.
              Each DataFrame contains potential matches from the db_path.
              Returns None if an error occurs or no faces are detected.
              Returns an empty list if detection is not enforced and no face is found.
    """
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return None
    # Check if db_path exists and is a directory
    if not os.path.isdir(db_path):
         print(f"Error: Database path {db_path} is not a valid directory.")
         return None
    # Check if db_path is empty (list content excluding hidden files)
    if not [f for f in os.listdir(db_path) if not f.startswith('.')]:
        print(f"Error: Database directory {db_path} is empty.")
        return None

    try:
        # DeepFace.find performs detection, embedding, and comparison.
        # Returns list of DataFrames. Important columns: 'identity', 'distance'
        dfs = DeepFace.find(
            img_path=image_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=True, # Raise error if no face detected in img_path
            align=True,             # Align faces before embedding (improves accuracy)
            silent=False            # Set to True to suppress DeepFace's download/progress bars
        )
        return dfs

    except ValueError as e:
        # Catches errors from enforce_detection=True if no face found, or other value errors.
        print(f"Could not process image {image_path}. Possible reason: No face detected or invalid input. Error: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during the find operation
        print(f"An unexpected error occurred during face recognition: {e}")
        # You might want to log the full traceback here for debugging
        # import traceback
        # traceback.print_exc()
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting face recognition for: {TEST_IMAGE_PATH}")
    print(f"Using database: {DB_PATH}")
    print(f"Model: {MODEL_NAME}, Detector: {DETECTOR_BACKEND}, Metric: {DISTANCE_METRIC}")
    print("-" * 30) # Separator

    results = find_identity(
        image_path=TEST_IMAGE_PATH,
        db_path=DB_PATH,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        distance_metric=DISTANCE_METRIC
    )

    if results is None:
        # This occurs if find_identity returned None due to errors (file not found, no face detected etc.)
        print("Face recognition process failed or could not proceed.")
    elif not isinstance(results, list):
         # Should not happen based on DeepFace docs, but good practice to check type
         print(f"Error: Unexpected result type from DeepFace.find: {type(results)}")
    elif not results:
         # This means DeepFace.find returned an empty list (e.g., if enforce_detection=False and no face found)
         # Note: With enforce_detection=True, this case shouldn't be reached; find_identity would return None instead.
         print("No faces were detected in the test image (or processing yielded no results).")
    else:
        # results is a non-empty list of DataFrames
        print("\n--- Recognition Results ---")
        # Iterate through each DataFrame (one per detected face in the test image)
        for i, df in enumerate(results):
            print(f"\nResults for Detected Face #{i+1}:")

            # Check if the DataFrame itself is valid and has expected columns
            if not isinstance(df, pandas.DataFrame):
                print(f"  Error: Result {i+1} is not a DataFrame ({type(df)}). Skipping.")
                continue

            if df.empty:
                print("  -> No matching face found in the database meeting the criteria.")
            elif 'identity' not in df.columns:
                 print(f"  Error: Result DataFrame {i+1} lacks 'identity' column. Columns: {df.columns.tolist()}")
            else:
                # Print the actual columns found for debugging if needed
                # print(f"  DEBUG: DataFrame Columns: {df.columns.tolist()}")

                print(f"  -> Most likely match(es) from database '{DB_PATH}':")
                # Iterate through the top few rows of the DataFrame
                # df is sorted by distance ascending (lower is better) by DeepFace.find
                for index, row in df.head(3).iterrows():
                    identity_path = row['identity']
                    identity_filename = os.path.basename(identity_path)

                    # Attempt to access the distance column using the expected standard name
                    try:
                        # *** Use the likely standard column name 'distance' ***
                        distance = row[EXPECTED_DISTANCE_COL]
                        print(f"    - Identity: {identity_filename} ({EXPECTED_DISTANCE_COL.capitalize()}: {distance:.4f})")
                    except KeyError:
                        # Handle case where the expected distance column is missing for this row
                        print(f"    - Identity: {identity_filename} (Distance column '{EXPECTED_DISTANCE_COL}' not found for this match)")
                        # Optional: Print available data for this row if needed for deeper debugging
                        # print(f"      Available row data: {row.to_dict()}")
                    except Exception as e_row:
                        # Catch other potential errors accessing row data
                         print(f"    - Error processing row for {identity_filename}: {e_row}")


        print("\n" + "-" * 30) # Separator

    print("Processing complete.")