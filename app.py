# app.py
import os
import base64
import io
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from deepface import DeepFace
from PIL import Image
import numpy as np
import secrets # For generating secret key

# --- Configuration ---
app = Flask(__name__)
# IMPORTANT: Change this to a long, random string in a real application!
# You can generate one using secrets.token_hex(16) in a Python console
app.secret_key = secrets.token_hex(16)

# Path where registered user faces are stored
USER_DB_PATH = "registered_users"
os.makedirs(USER_DB_PATH, exist_ok=True)

# DeepFace Model Configuration (keep consistent with your command-line version)
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"
# Threshold for face verification (tune this based on experiments)
# Lower value means stricter match. For Cosine distance, 0.4 is often used.
DISTANCE_THRESHOLD = 0.40

# Set logging level (adjust as needed)
logging.basicConfig(level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
app.logger.setLevel(logging.INFO) # Use Flask's logger

# --- Helper Functions ---
def decode_image(base64_string):
    """Decodes a base64 image string to a numpy array (OpenCV format)."""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        # Convert to RGB if necessary (DeepFace expects BGR often, but PIL opens RGB)
        # Let DeepFace handle color conversion internally if needed, pass numpy array
        return np.array(img)
    except Exception as e:
        app.logger.error(f"Error decoding image: {e}")
        return None

# --- Routes ---
@app.route('/')
def index():
    """Home page: Shows profile if logged in, otherwise links to login/register."""
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login')) # Or render a generic welcome page

@app.route('/login', methods=['GET'])
def login():
    """Serves the login page."""
    if 'username' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register_page():
    """Serves the registration page."""
    if 'username' in session:
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    """Handles user registration."""
    data = request.get_json()
    username = data.get('username')
    img_data_b64 = data.get('image')

    if not username or not img_data_b64:
        return jsonify({"success": False, "message": "Username and image data required."}), 400

    # Simple validation (add more robust checks!)
    if not username.isalnum() or ' ' in username:
         return jsonify({"success": False, "message": "Username must be alphanumeric."}), 400

    user_folder = os.path.join(USER_DB_PATH, username)
    if os.path.exists(user_folder):
        return jsonify({"success": False, "message": "Username already exists."}), 400

    img_np = decode_image(img_data_b64)
    if img_np is None:
        return jsonify({"success": False, "message": "Invalid image data."}), 400

    try:
        # Extract the face - ensure only one face is present for registration
        # `extract_faces` returns a list of dictionaries.
        # Set enforce_detection=True to error if no face found
        extracted_faces = DeepFace.extract_faces(
            img_path=img_np,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True
        )

        if len(extracted_faces) > 1:
             return jsonify({"success": False, "message": "Multiple faces detected. Please ensure only your face is clearly visible."}), 400
        if not extracted_faces: # Should be caught by enforce_detection=True, but double check
             return jsonify({"success": False, "message": "No face detected in the image."}), 400

        # Get the facial area from the result (it's already a numpy array)
        # The facial area is scaled by a factor, using it directly might be good.
        face_data = extracted_faces[0]['face']
        # DeepFace returns face data in BGR format if using OpenCV backend, but saved image needs RGB usually
        # Convert the extracted face (float 0-1 range) back to uint8 (0-255) BGR image
        face_img_bgr = (face_data * 255).astype(np.uint8)
        # Convert BGR to RGB for saving with PIL
        face_img_rgb = face_img_bgr[..., ::-1] # Efficient BGR -> RGB conversion

        # Save the extracted face image
        os.makedirs(user_folder, exist_ok=True)
        face_image_path = os.path.join(user_folder, "face.jpg")
        img_to_save = Image.fromarray(face_img_rgb)
        img_to_save.save(face_image_path, format="JPEG")

        app.logger.info(f"Registered user '{username}' successfully.")
        flash(f"User '{username}' registered successfully! Please log in.", "success")
        return jsonify({"success": True, "message": "Registration successful."})

    except ValueError as e:
        # Specific error from DeepFace if no face detected with enforce_detection=True
        app.logger.warning(f"Registration failed for {username}: No face detected. Error: {e}")
        return jsonify({"success": False, "message": f"Registration failed: No face detected."}), 400
    except Exception as e:
        app.logger.error(f"Error during registration for {username}: {e}")
        # Consider removing potentially incomplete user folder if error occurs mid-process
        if os.path.exists(user_folder):
            try:
                # Be cautious with recursive deletion!
                import shutil
                shutil.rmtree(user_folder)
            except Exception as rm_err:
                 app.logger.error(f"Failed to clean up user folder {user_folder}: {rm_err}")
        return jsonify({"success": False, "message": "An internal error occurred during registration."}), 500

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Handles face recognition login attempt."""
    data = request.get_json()
    img_data_b64 = data.get('image')

    if not img_data_b64:
        return jsonify({"success": False, "message": "Image data required."}), 400

    img_np = decode_image(img_data_b64)
    if img_np is None:
        return jsonify({"success": False, "message": "Invalid image data."}), 400

    try:
        # Use DeepFace.find to compare the captured face against the database
        # `find` returns a list of DataFrames, one per detected face in img_np
        # Since this is login, we expect only one face in img_np (the user)
        # Set enforce_detection=False initially, handle no face found more gracefully
        # Note: db_path needs to contain representations (images or vectors)
        found_dfs = DeepFace.find(
            img_path=img_np,
            db_path=USER_DB_PATH, # Path to folder containing subfolders like 'username/face.jpg'
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True, # Error if no face detected in login attempt image
            align=True,
            silent=True # Suppress find's own logging/progress
        )

        # find returns list of DFs. If login image has one face, list has one DF.
        if not found_dfs or found_dfs[0].empty:
             app.logger.warning("Authentication failed: No matching face found in database.")
             return jsonify({"success": False, "message": "Authentication failed: Face not recognized."})

        # Get the best match from the first (and likely only) DataFrame
        best_match_df = found_dfs[0]
        if 'identity' not in best_match_df.columns or 'distance' not in best_match_df.columns:
             app.logger.error(f"Authentication Error: Unexpected DataFrame columns: {best_match_df.columns.tolist()}")
             return jsonify({"success": False, "message": "Authentication error processing results."}), 500

        # Get the top match (lowest distance)
        top_match = best_match_df.iloc[0]
        matched_identity_path = top_match['identity']
        distance = top_match['distance']

        app.logger.info(f"Best match: Identity={matched_identity_path}, Distance={distance:.4f}")

        # Check if the distance is below our threshold
        if distance <= DISTANCE_THRESHOLD:
            # Extract username from the identity path
            # Path format is like: registered_users/some_username/face.jpg
            try:
                 # Get directory name (username) from the matched file path
                 username = os.path.basename(os.path.dirname(matched_identity_path))
                 # Store username in session
                 session['username'] = username
                 app.logger.info(f"Authentication successful for user: {username}")
                 return jsonify({"success": True, "message": "Login successful.", "username": username})
            except Exception as e:
                 app.logger.error(f"Error extracting username from path '{matched_identity_path}': {e}")
                 return jsonify({"success": False, "message": "Authentication error processing identity."}), 500
        else:
            app.logger.warning(f"Authentication failed: Best match distance ({distance:.4f}) exceeds threshold ({DISTANCE_THRESHOLD}).")
            return jsonify({"success": False, "message": "Authentication failed: Face not recognized."})

    except ValueError as e:
        # Usually means no face detected in the login image capture
        app.logger.warning(f"Authentication failed: No face detected in captured image. Error: {e}")
        return jsonify({"success": False, "message": "Authentication failed: No face detected."}), 400
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during authentication: {e}")
        # import traceback
        # traceback.print_exc() # Log full traceback for debugging
        return jsonify({"success": False, "message": "An internal server error occurred during authentication."}), 500


@app.route('/logout')
def logout():
    """Logs the user out."""
    session.pop('username', None) # Remove username from session
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# --- Run the App ---
if __name__ == '__main__':
    # Note: Default Flask server is for development only.
    # Use a production WSGI server (like Gunicorn or Waitress) for deployment.
    # Port 5000 is common, Codespaces usually forwards it automatically.
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True enables auto-reload and more error details