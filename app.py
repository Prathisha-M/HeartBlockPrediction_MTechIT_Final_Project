from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import os
import tempfile
import traceback

app = Flask(__name__)
# Fix CORS configuration - allow all origins for development
CORS(app, origins=["*"], methods=["POST", "GET"])

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

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        file = request.files['image']
        
        # Read and validate image
        file.seek(0)  # Reset file pointer
        image_data = np.frombuffer(file.read(), np.uint8)
        original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if original_image is None:
            return jsonify({'error': 'Invalid image file'}), 400

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
        image_data = np.frombuffer(file.read(), np.uint8)
        original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if original_image is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
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
    return jsonify({'status': 'healthy', 'service': 'Heart Blockage Analysis API'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)