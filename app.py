from flask import Flask, request, jsonify, send_file, Response, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
import cv2
import numpy as np
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from posture_analysis import analyze_posture, create_aggregated_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
PDF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames')
VISUALIZATIONS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'webm'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER
app.config['VISUALIZATIONS_FOLDER'] = VISUALIZATIONS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    video_extensions = {'mp4', 'avi', 'mov', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def extract_frames(video_path, output_dir, num_frames=10, max_duration=6.0):
    """
    Extract frames from a video at regular intervals
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        max_duration: Maximum duration of video to process in seconds
    
    Returns:
        List of paths to the extracted frames
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: FPS={fps}, Total frames={total_frames}, Duration={duration:.2f}s")
        
        # Limit to max_duration
        processed_duration = min(duration, max_duration)
        processed_frames = int(processed_duration * fps)
        
        # Calculate frame interval
        if num_frames <= 0 or processed_frames < num_frames:
            actual_frames = min(10, processed_frames)
        else:
            actual_frames = num_frames
            
        if processed_frames <= 0 or actual_frames <= 0:
            logger.error(f"Invalid processed frames: {processed_frames} or actual frames: {actual_frames}")
            return []
            
        interval = processed_frames / actual_frames
        
        logger.info(f"Extracting {actual_frames} frames with interval {interval:.2f} frames")
        
        # Extract frames
        frame_paths = []
        for i in range(actual_frames):
            # Calculate the frame number to extract
            frame_num = int(i * interval)
            
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {i} (position {frame_num})")
                break
            
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            logger.debug(f"Extracted frame {i} to {frame_path}")
        
        # Release the video capture
        cap.release()
        
        logger.info(f"Successfully extracted {len(frame_paths)} frames from {video_path}")
        return frame_paths
    
    except Exception as e:
        logger.exception(f"Error extracting frames from video: {str(e)}")
        return []

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['image']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload an image (.jpg, .jpeg, .png) or video (.mp4, .avi, .mov, .webm).'}), 400
    
    try:
        # Generate a unique identifier
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        base_filename = original_filename.rsplit('.', 1)[0]
        extension = original_filename.rsplit('.', 1)[1] if '.' in original_filename else 'jpg'
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_{unique_id}.{extension}")
        file.save(file_path)
        
        logger.info(f"File saved at {file_path}")
        
        # Check if it's a video or image
        if is_video_file(file.filename):
            logger.info(f"Processing video file: {file.filename}")
            
            # Extract frames from the video
            frames_dir = os.path.join(app.config['FRAMES_FOLDER'], unique_id)
            os.makedirs(frames_dir, exist_ok=True)
            
            frame_paths = extract_frames(file_path, frames_dir)
            
            if not frame_paths:
                logger.error("Failed to extract frames from video")
                return jsonify({'error': 'Failed to extract frames from video. Please ensure the video is valid.'}), 400
            
            logger.info(f"Successfully extracted {len(frame_paths)} frames")
            
            # Analyze each frame
            analysis_results = []
            for i, frame_path in enumerate(frame_paths):
                try:
                    logger.info(f"Analyzing frame {i+1}/{len(frame_paths)}: {frame_path}")
                    result = analyze_posture(frame_path, visualization=True, generate_pdf=False, return_data=True)
                    
                    if isinstance(result, dict) and not result.get("error"):
                        result['frame_number'] = i + 1
                        result['frame_path'] = frame_path
                        analysis_results.append(result)
                    else:
                        logger.warning(f"Frame {i+1} analysis failed: {result.get('error') if isinstance(result, dict) else 'Unknown error'}")
                except Exception as e:
                    logger.exception(f"Error analyzing frame {i+1}: {str(e)}")
            
            if not analysis_results:
                logger.error("Failed to analyze any frames from the video")
                return jsonify({'error': 'Failed to analyze any frames from the video.'}), 500
            
            logger.info(f"Successfully analyzed {len(analysis_results)} frames")
            
            # Generate aggregated report
            report_filename = f"video_analysis_{unique_id}.pdf"
            report_path = os.path.join(app.config['PDF_FOLDER'], report_filename)
            visualization_filename = f"video_visualization_{unique_id}.jpg"
            visualization_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], visualization_filename)
            
            # Create a simple visualization grid from analyzed frames
            try:
                # Get up to 4 frames to show in the grid
                selected_frame_paths = frame_paths[:min(4, len(frame_paths))]
                
                # Create the aggregated report with frame visualization
                result = create_aggregated_report(
                    analysis_results,
                    report_path,
                    visualization_path,
                    selected_frame_paths
                )
                
                if result.get("error"):
                    logger.error(f"Error creating aggregated report: {result.get('error')}")
                    return jsonify({"error": result["error"]}), 500
                
                logger.info(f"Successfully created aggregated report: {report_path}")
                logger.info(f"Visualization saved at: {visualization_path}")
                
                # Return success response
                return jsonify({
                    "is_video": True,
                    "frames_count": len(analysis_results),
                    "pdf_path": report_filename,
                    "visualization_path": os.path.basename(visualization_path),
                    "message": "Video analysis completed successfully"
                }), 200
                
            except Exception as e:
                logger.exception(f"Error creating aggregated report: {str(e)}")
                return jsonify({"error": f"Failed to create aggregated report: {str(e)}"}), 500
                
        else:
            # For single image analysis
            logger.info(f"Processing image file: {file.filename}")
            
            # For single image, use the original analyze_posture function
            result = analyze_posture(file_path, visualization=True, generate_pdf=True, return_data=True)
            
            if isinstance(result, dict) and result.get("error"):
                logger.error(f"Image analysis failed: {result.get('error')}")
                return jsonify({"error": result["error"]}), 400
            
            # Extract the filenames from the full paths
            pdf_filename = os.path.basename(result["pdf_path"])
            visualization_filename = os.path.basename(result["visualization_path"])
            
            logger.info(f"Image analysis successful. PDF: {pdf_filename}, Visualization: {visualization_filename}")
            
            # Return success response with PDF path and visualization
            return jsonify({
                "is_video": False,
                "pdf_path": pdf_filename,
                "visualization_path": visualization_filename,
                "message": "Analysis completed successfully"
            }), 200
            
    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/api/reports/<filename>', methods=['GET'])
def get_report(filename):
    """Serve the generated PDF report"""
    try:
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        # Check in PDF folder
        pdf_path = os.path.join(app.config['PDF_FOLDER'], filename)
        if os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True)
            
        # If not found, check other possible locations
        for folder in [app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER']]:
            for root, dirs, files in os.walk(folder):
                if filename in files:
                    return send_file(os.path.join(root, filename), as_attachment=True)
        
        logger.error(f"Report file not found: {filename}")
        return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        logger.exception(f"Error serving report {filename}: {str(e)}")
        return jsonify({'error': f'Error serving report: {str(e)}'}), 500

@app.route('/api/visualizations/<filename>', methods=['GET'])
def get_visualization(filename):
    """Serve the visualization image"""
    try:
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        # Check in visualizations folder
        vis_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], filename)
        if os.path.exists(vis_path):
            return send_file(vis_path, mimetype='image/jpeg')
            
        # If not found, check other possible locations
        for folder in [app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER']]:
            for root, dirs, files in os.walk(folder):
                if filename in files:
                    return send_file(os.path.join(root, filename), mimetype='image/jpeg')
        
        logger.error(f"Visualization file not found: {filename}")
        return jsonify({'error': 'Visualization not found'}), 404
    except Exception as e:
        logger.exception(f"Error serving visualization {filename}: {str(e)}")
        return jsonify({'error': f'Error serving visualization: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting AR-MED server...")
    app.run(debug=True, host='0.0.0.0', port=5000) 