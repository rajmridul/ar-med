from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import tempfile
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from posture_analysis import analyze_posture

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
PDF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload a .jpg, .jpeg, or .png image.'}), 400
    
    try:
        # Generate a unique filename
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        base_filename = original_filename.rsplit('.', 1)[0]
        extension = original_filename.rsplit('.', 1)[1] if '.' in original_filename else 'jpg'
        
        # Save the file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_{unique_id}.{extension}")
        file.save(image_path)
        
        # Run posture analysis
        result = analyze_posture(image_path, visualization=True, generate_pdf=True)
        
        # Extract the PDF path from the result message
        if "PDF report generated at:" in result:
            pdf_path = result.split("PDF report generated at: ")[1].strip()
            pdf_filename = os.path.basename(pdf_path)
            
            # Get visualization path based on naming convention from analyze_posture function
            vis_path = image_path.rsplit('.', 1)[0] + "_analysis.jpg"
            vis_filename = os.path.basename(vis_path) if os.path.exists(vis_path) else None
            
            return jsonify({
                'status': 'success',
                'message': 'Analysis completed successfully',
                'pdf_path': pdf_filename,
                'visualization_path': vis_filename
            })
        else:
            return jsonify({'error': 'Analysis failed: ' + result}), 500
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/reports/<filename>', methods=['GET'])
def get_report(filename):
    """Serve the generated PDF report"""
    try:
        # Find the report in the uploads directory (where analyze_posture saves it)
        report_path = None
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            if filename in files:
                report_path = os.path.join(root, filename)
                break
        
        if not report_path:
            return jsonify({'error': 'Report not found'}), 404
        
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/visualizations/<filename>', methods=['GET'])
def get_visualization(filename):
    """Serve the visualization image"""
    try:
        # Find the visualization in the uploads directory
        vis_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(vis_path):
            return jsonify({'error': 'Visualization not found'}), 404
        
        return send_file(vis_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 