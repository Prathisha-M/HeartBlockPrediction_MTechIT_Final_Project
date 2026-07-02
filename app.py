from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# IMPORTANT: force the non-interactive Agg backend BEFORE importing pyplot.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import os
import json
import tempfile
import traceback

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app, origins=["*"], methods=["POST", "GET"])

# ---------------------------------------------------------------------------
# Gemini setup
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def validate_heart_image(image_bytes, mime_type="image/jpeg"):
    """
    Ask Gemini whether the uploaded image is actually a coronary angiogram /
    cardiac catheterization image. If it isn't, Gemini also tells us what the
    image actually shows so we can display that to the user.

    Returns a dict:
        {
            "is_heart_image": bool,
            "object_name": str,      # e.g. "a golden retriever dog"
            "confidence": float,     # 0-1
            "reasoning": str | None  # only set on internal errors
        }
    """
    if gemini_client is None:
        # No API key configured -> fail open so the tool still works,
        # but flag it clearly so it's obvious in the logs/response.
        return {
            "is_heart_image": True,
            "object_name": None,
            "confidence": 0.0,
            "reasoning": "GEMINI_API_KEY not configured; validation skipped",
        }

    prompt = (
        "You are a strict gatekeeper for a medical imaging tool that only "
        "accepts coronary angiogram / cardiac catheterization X-ray images "
        "(grayscale X-ray images of heart blood vessels, usually captured "
        "with contrast dye, sometimes showing a catheter).\n\n"
        "Look at the attached image and decide if it is genuinely this type "
        "of medical image.\n\n"
        "Respond with ONLY a JSON object, no markdown fences, matching this "
        "exact schema:\n"
        '{"is_heart_image": <true or false>, '
        '"object_name": "<short 2-6 word plain-English description of what '
        "the image actually shows, e.g. 'a golden retriever dog', 'a laptop "
        "keyboard', 'a chest X-ray (not coronary)', 'a coronary angiogram'>\", "
        '"confidence": <number between 0 and 1>}'
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        result = json.loads(response.text)
        return {
            "is_heart_image": bool(result.get("is_heart_image", False)),
            "object_name": result.get("object_name", "an unrecognized object"),
            "confidence": float(result.get("confidence", 0.0)),
            "reasoning": None,
        }
    except Exception as e:
        print(f"Gemini validation error: {e}")
        print(traceback.format_exc())
        # Fail open: if Gemini is down/misconfigured, don't block the whole
        # pipeline — just skip the extra check for this request.
        return {
            "is_heart_image": True,
            "object_name": None,
            "confidence": 0.0,
            "reasoning": f"validation error: {e}",
        }


def preprocess_image(image):
    """Enhanced preprocessing for medical images"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.medianBlur(gray, 5)
    
    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

def detect_initial_contour(image_shape):
    """Detect initial contour based on image dimensions"""
    height, width = image_shape
    
    # Create initial circle contour in the center of the image
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3
    
    theta = np.linspace(0, 2 * np.pi, 400)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    
    return np.array([x, y]).T

def analyze_stenosis(original_img, segmented_contour):
    """Analyze potential blockages in the coronary arteries"""
    height, width = original_img.shape
    
    # Calculate contour properties
    contour_length = len(segmented_contour)
    
    # Calculate diameter variations along the contour
    if contour_length > 10:
        center = np.mean(segmented_contour, axis=0)
        distances = np.linalg.norm(segmented_contour - center, axis=1)
        
        # Normalize distances
        if np.max(distances) - np.min(distances) > 0:
            normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        else:
            normalized_distances = np.ones_like(distances)
        
        # Detect significant constrictions (potential blockages)
        mean_dist = np.mean(normalized_distances)
        std_dist = np.std(normalized_distances)
        
        # Find regions with significant narrowing
        constriction_threshold = mean_dist - 1.5 * std_dist
        constricted_points = np.where(normalized_distances < constriction_threshold)[0]
        
        blockage_detected = len(constricted_points) > len(segmented_contour) * 0.1
        
        # Calculate regularity safely
        if mean_dist > 0:
            regularity = 1 - (std_dist / mean_dist)
        else:
            regularity = 0
            
        return {
            'blockage_detected': blockage_detected,
            'constriction_severity': len(constricted_points) / len(segmented_contour) if len(segmented_contour) > 0 else 0,
            'contour_regularity': max(0, min(1, regularity)),
            'message': 'Potential blockage detected' if blockage_detected else 'No significant blockage detected'
        }
    
    return {
        'blockage_detected': False,
        'constriction_severity': 0,
        'contour_regularity': 0,
        'message': 'Insufficient contour data for analysis'
    }


@app.route('/validate-image', methods=['POST'])
def validate_image():
    """
    Standalone endpoint the frontend calls BEFORE /analyze and /process.
    Uses Gemini to check the upload is really a coronary angiogram image.
    If not, returns the name of whatever object Gemini thinks it sees.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        file.seek(0)
        image_bytes = file.read()
        mime_type = file.mimetype or "image/jpeg"

        # Make sure OpenCV can actually decode it first
        image_data = np.frombuffer(image_bytes, np.uint8)
        decoded = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if decoded is None:
            return jsonify({'error': 'Invalid image file'}), 400

        validation = validate_heart_image(image_bytes, mime_type)
        return jsonify(validation)

    except Exception as e:
        print(f"Error in validate_image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        file = request.files['image']
        
        # Read and validate image
        file.seek(0)  # Reset file pointer
        raw_bytes = file.read()
        image_data = np.frombuffer(raw_bytes, np.uint8)
        original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if original_image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Gemini gate: refuse to run the pipeline on non-heart images
        validation = validate_heart_image(raw_bytes, file.mimetype or "image/jpeg")
        if not validation['is_heart_image']:
            return jsonify({
                'error': 'not_heart_image',
                'object_name': validation['object_name'],
                'confidence': validation['confidence'],
                'message': f"This looks like {validation['object_name']}, not a coronary angiogram image."
            }), 422

        # Preprocessing
        processed_image = preprocess_image(original_image)
        
        # Get initial contour
        initial_contour = detect_initial_contour(processed_image.shape)
        
        # Apply active contour segmentation
        smoothed_image = gaussian(processed_image, 3.0, preserve_range=False)
        
        # Active contour parameters optimized for medical images
        # CORRECTED: Changed max_iterations to max_num_iter
        segmented_contour = active_contour(
            smoothed_image,
            initial_contour,
            alpha=0.015,      # Elasticity parameter
            beta=0.1,         # Rigidity parameter  
            gamma=0.001,      # Time step
            max_num_iter=2000,  # CORRECTED PARAMETER NAME
            boundary_condition='periodic',
            w_line=0,         # Line attraction
            w_edge=1,         # Edge attraction
            convergence=0.1
        )
        
        analysis_results = analyze_stenosis(processed_image, segmented_contour)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(processed_image, cmap='gray')
        ax2.set_title('Preprocessed Image')
        ax2.axis('off')
        
        ax3.imshow(processed_image, cmap='gray')
        ax3.plot(initial_contour[:, 0], initial_contour[:, 1], '--r', lw=2, alpha=0.7, label='Initial Contour')
        ax3.plot(segmented_contour[:, 0], segmented_contour[:, 1], '-b', lw=3, label='Segmented Boundary')
        ax3.set_title(f'Segmentation Result\n{analysis_results["message"]}')
        ax3.legend()
        ax3.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_detailed():
    """Endpoint for detailed analysis with JSON response"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        file = request.files['image']
        file.seek(0)  # Reset file pointer
        raw_bytes = file.read()
        image_data = np.frombuffer(raw_bytes, np.uint8)
        original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if original_image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Gemini gate: refuse to run the pipeline on non-heart images
        validation = validate_heart_image(raw_bytes, file.mimetype or "image/jpeg")
        if not validation['is_heart_image']:
            return jsonify({
                'error': 'not_heart_image',
                'object_name': validation['object_name'],
                'confidence': validation['confidence'],
                'message': f"This looks like {validation['object_name']}, not a coronary angiogram image."
            }), 422
            
        processed_image = preprocess_image(original_image)
        initial_contour = detect_initial_contour(processed_image.shape)
        smoothed_image = gaussian(processed_image, 3.0, preserve_range=False)
        
        # CORRECTED: Changed max_iterations to max_num_iter
        segmented_contour = active_contour(
            smoothed_image,
            initial_contour,
            alpha=0.015,
            beta=0.1,
            gamma=0.001,
            max_num_iter=2000,  # CORRECTED PARAMETER NAME
            boundary_condition='periodic',
            w_line=0,
            w_edge=1,
            convergence=0.1
        )
        
        analysis = analyze_stenosis(processed_image, segmented_contour)
        
        # Calculate area metrics
        contour_area = cv2.contourArea(segmented_contour.astype(np.float32))
        image_area = processed_image.shape[0] * processed_image.shape[1]
        area_ratio = contour_area / image_area if image_area > 0 else 0
        
        analysis.update({
            'contour_area': float(contour_area),
            'image_area': float(image_area),
            'area_ratio': float(area_ratio),
            'contour_points': len(segmented_contour),
            'status': 'warning' if analysis['blockage_detected'] else 'normal'
        })
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Error in analyze_detailed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Heart Blockage Analysis API',
        'gemini_configured': gemini_client is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)