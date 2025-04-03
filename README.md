# AR-MED: Advanced Posture Analysis System

AR-MED is an AI-powered posture analysis system that uses computer vision to detect and analyze body alignment issues from 2D images. The system generates detailed PDF reports with visualizations, measurements, and personalized recommendations.

## Features

- **AI-Powered Analysis**: Uses MediaPipe pose estimation to detect 33 body landmarks
- **Comprehensive Assessment**: Analyzes knee alignment, shoulder balance, hip alignment, and overall posture
- **Detailed Reports**: Generates professional PDF reports with visualizations and personalized recommendations
- **Modern Web Interface**: React.js frontend with Material UI for a clean, responsive user experience
- **Flask API Backend**: Robust Python backend that processes images and generates analysis

## Project Structure

```
AR-MED/
├── app.py                  # Flask API backend
├── posture_analysis.py     # Core posture analysis functionality
├── requirements.txt        # Python dependencies
├── uploads/                # Directory for uploaded images and reports
├── reports/                # Directory for generated PDF reports
└── frontend/
    └── ar-med/            # React.js frontend application
```

## Installation

### Backend Setup

1. Make sure you have Python 3.8+ installed
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Make sure you have Node.js (14+) and npm installed
2. Navigate to the frontend directory:

```bash
cd frontend/ar-med
```

3. Install the required npm packages:

```bash
npm install
```

## Running the Application

### Start the Backend Server

```bash
python app.py
```

The Flask API will start running on http://localhost:5000

### Start the Frontend Development Server

```bash
cd frontend/ar-med
npm start
```

The React application will start running on http://localhost:3000

## How to Use

1. Open your browser and navigate to http://localhost:3000
2. Click on "Analysis Tool" in the navigation menu or the "Try It Now" button
3. Upload a clear full-body image showing the person's posture
4. Click "Analyze Posture" to process the image
5. View the analysis results and download the detailed PDF report

## Backend API Endpoints

- `POST /api/analyze`: Upload an image for posture analysis
- `GET /api/reports/<filename>`: Download a generated PDF report
- `GET /api/visualizations/<filename>`: View a generated visualization image

## Technical Details

### Backend Technologies

- **Flask**: Web framework for the API endpoints
- **MediaPipe**: Google's framework for pose estimation
- **OpenCV**: Computer vision library for image processing
- **ReportLab**: PDF generation library

### Frontend Technologies

- **React.js**: Frontend library for building the user interface
- **Material UI**: React component library for responsive design
- **Axios**: HTTP client for API requests

## Disclaimer

AR-MED is a preliminary analysis tool only. The posture analysis provided is intended for informational and educational purposes only and is not meant to be a substitute for professional medical advice, diagnosis, or treatment.

For an accurate diagnosis of posture-related issues or any medical condition, always consult with a qualified healthcare provider such as a physician, physical therapist, or chiropractor.

## License

MIT 