import React, { useState, useRef } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  Divider,
  Grid,
  IconButton,
  Paper,
  Stack,
  Tab,
  Tabs,
  Tooltip,
  Typography,
  useTheme,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import WarningIcon from '@mui/icons-material/Warning';
import VideocamIcon from '@mui/icons-material/Videocam';
import ImageIcon from '@mui/icons-material/Image';
import InfoIcon from '@mui/icons-material/Info';
import axios from 'axios';

const AnalysisTool = () => {
  const theme = useTheme();
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [fileType, setFileType] = useState('image'); // 'image' or 'video'
  const [activeTab, setActiveTab] = useState(0); // 0 for image, 1 for video
  const videoRef = useRef(null);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    // Reset state when changing tabs
    setSelectedFile(null);
    setPreviewUrl(null);
    setError('');
    setResult(null);
    setFileType(newValue === 0 ? 'image' : 'video');
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Determine file type
    const currentFileType = file.type.startsWith('video/') ? 'video' : 'image';
    
    // Check if file type matches selected tab
    if (
      (activeTab === 0 && currentFileType === 'video') || 
      (activeTab === 1 && currentFileType === 'image')
    ) {
      const wrongTypeMessage = activeTab === 0 
        ? 'Please select an image file (jpg, jpeg, png)' 
        : 'Please select a video file (mp4, avi, mov, webm)';
      setError(wrongTypeMessage);
      return;
    }

    setFileType(currentFileType);

    // Check file size (max 100MB for videos, 10MB for images)
    const maxSize = currentFileType === 'video' ? 100 * 1024 * 1024 : 10 * 1024 * 1024;
    if (file.size > maxSize) {
      const sizeMessage = currentFileType === 'video' 
        ? 'File size should be less than 100MB' 
        : 'File size should be less than 10MB';
      setError(sizeMessage);
      return;
    }

    // For videos, check duration if possible
    if (currentFileType === 'video') {
      const videoElement = document.createElement('video');
      videoElement.preload = 'metadata';
      
      videoElement.onloadedmetadata = () => {
        URL.revokeObjectURL(videoElement.src);
        if (videoElement.duration > 10) {
          setError('Video should be less than 10 seconds long for optimal analysis.');
          return;
        }
        
        // If video is valid, proceed
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setError('');
        setResult(null);
      };
      
      videoElement.src = URL.createObjectURL(file);
    } else {
      // For images, proceed directly
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError('');
      setResult(null);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError('');
    setResult(null);
    if (videoRef.current) {
      videoRef.current.pause();
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError(`Please select a ${fileType} to analyze`);
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('image', selectedFile); // Backend endpoint uses 'image' for both image and video

    try {
      const response = await axios.post('/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      console.error('Error during analysis:', err);
      setError(
        err.response?.data?.error ||
          'An error occurred during analysis. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    if (result && result.pdf_path) {
      window.open(`/api/reports/${result.pdf_path}`, '_blank');
    }
  };

  const renderFileUploader = () => (
    <Box
      sx={{
        mt: 3,
        p: 4,
        border: '2px dashed',
        borderColor: 'primary.main',
        borderRadius: 2,
        bgcolor: 'rgba(25, 118, 210, 0.05)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        cursor: 'pointer',
      }}
      onClick={() => document.getElementById('upload-input').click()}
    >
      <input
        type="file"
        id="upload-input"
        accept={fileType === 'image' ? "image/png, image/jpeg, image/jpg" : "video/mp4, video/avi, video/mov, video/webm"}
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      {fileType === 'image' ? (
        <ImageIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
      ) : (
        <VideocamIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
      )}
      <Typography variant="h6" color="primary.main">
        Click to upload {fileType === 'image' ? 'an image' : 'a video'}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        {fileType === 'image' 
          ? 'Supported formats: JPG, JPEG, PNG' 
          : 'Supported formats: MP4, AVI, MOV, WEBM (max 6 seconds)'}
      </Typography>
    </Box>
  );

  const renderPreview = () => (
    <Box sx={{ mt: 3, position: 'relative' }}>
      {fileType === 'image' ? (
        <Box
          component="img"
          src={previewUrl}
          alt="Preview"
          sx={{
            width: '100%',
            maxHeight: 400,
            objectFit: 'contain',
            borderRadius: 2,
          }}
        />
      ) : (
        <Box
          component="video"
          ref={videoRef}
          src={previewUrl}
          controls
          sx={{
            width: '100%',
            maxHeight: 400,
            borderRadius: 2,
          }}
        />
      )}
      <IconButton
        sx={{
          position: 'absolute',
          top: 8,
          right: 8,
          bgcolor: 'rgba(255,255,255,0.8)',
          '&:hover': {
            bgcolor: 'rgba(255,255,255,0.9)',
          },
        }}
        onClick={handleRemoveFile}
      >
        <DeleteIcon />
      </IconButton>
    </Box>
  );

  const renderVideoAnalysisInfo = () => (
    <Box sx={{ mt: 3, mb: 3 }}>
      <Alert 
        icon={<InfoIcon />} 
        severity="info" 
        sx={{ 
          mb: 2, 
          '& .MuiAlert-message': { 
            width: '100%' 
          } 
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          How Video Analysis Works:
        </Typography>
        <Typography variant="body2">
          1. Upload a short video (max 6 seconds) of a person rotating to show all angles of standing position
        </Typography>
        <Typography variant="body2">
          2. Our system extracts 10 frames at 0.5-second intervals from the video
        </Typography>
        <Typography variant="body2">
          3. Each frame is analyzed individually for posture assessment
        </Typography>
        <Typography variant="body2">
          4. The analysis is aggregated into a comprehensive report showing averages and variations
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 'bold', mt: 1 }}>
          Video analysis provides a more robust assessment than a single image
        </Typography>
      </Alert>
    </Box>
  );

  const renderVideoResults = () => (
    <Box sx={{ mt: 2 }}>
      <Alert severity="success" sx={{ mb: 3 }}>
        Video analysis completed successfully! {result.frames_count} frames analyzed.
      </Alert>

      {result.visualization_path && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Frame Analysis Visualization
          </Typography>
          <Box
            component="img"
            src={`/api/visualizations/${result.visualization_path}`}
            alt="Analysis Visualization"
            sx={{
              width: '100%',
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'divider',
            }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Visual analysis of selected frames from your video
          </Typography>
        </Box>
      )}

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Your Multi-Frame Analysis Report is Ready
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            This comprehensive report includes:
          </Typography>
          <Box component="ul" sx={{ pl: 2, mb: 2 }}>
            <Box component="li">
              <Typography variant="body2" color="text.secondary">
                Averaged measurements across all analyzed frames
              </Typography>
            </Box>
            <Box component="li">
              <Typography variant="body2" color="text.secondary">
                Individual frame-by-frame analysis
              </Typography>
            </Box>
            <Box component="li">
              <Typography variant="body2" color="text.secondary">
                Statistical variance in measurements during rotation
              </Typography>
            </Box>
            <Box component="li">
              <Typography variant="body2" color="text.secondary">
                Personalized recommendations based on aggregated data
              </Typography>
            </Box>
          </Box>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleDownloadPDF}
            fullWidth
          >
            Download Multi-Frame PDF Report
          </Button>
        </CardContent>
      </Card>

      <Box sx={{ mt: 3 }}>
        <Alert severity="info" variant="outlined">
          <Typography variant="body2" fontWeight={500}>
            Important Reminder
          </Typography>
          <Typography variant="body2">
            This multi-frame analysis provides a more comprehensive assessment than a single image, but is still for informational purposes only. For an accurate diagnosis, please consult with a medical professional.
          </Typography>
        </Alert>
      </Box>
    </Box>
  );

  const renderImageResults = () => (
    <Box sx={{ mt: 2 }}>
      <Alert severity="success" sx={{ mb: 3 }}>
        Analysis completed successfully!
      </Alert>

      {result.visualization_path && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Analysis Visualization
          </Typography>
          <Box
            component="img"
            src={`/api/visualizations/${result.visualization_path}`}
            alt="Analysis Visualization"
            sx={{
              width: '100%',
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'divider',
            }}
          />
        </Box>
      )}

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Your Detailed PDF Report is Ready
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Your comprehensive posture analysis report includes detailed
            measurements, visualizations, and personalized recommendations.
          </Typography>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleDownloadPDF}
            fullWidth
          >
            Download PDF Report
          </Button>
        </CardContent>
      </Card>

      <Box sx={{ mt: 3 }}>
        <Alert severity="info" variant="outlined">
          <Typography variant="body2" fontWeight={500}>
            Important Reminder
          </Typography>
          <Typography variant="body2">
            This analysis is for informational purposes only. For an accurate
            diagnosis, please consult with a medical professional.
          </Typography>
        </Alert>
      </Box>
    </Box>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 8 }}>
      <Typography
        variant="h2"
        align="center"
        gutterBottom
        sx={{ mb: 2, color: 'primary.main' }}
      >
        Posture Analysis Tool
      </Typography>
      <Typography
        variant="h6"
        align="center"
        color="text.secondary"
        gutterBottom
        sx={{ mb: 6 }}
      >
        Upload an image or video to analyze posture and get detailed insights
      </Typography>

      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              p: 4,
              border: '1px dashed',
              borderColor: 'divider',
              bgcolor: 'background.paper',
              height: '100%',
            }}
          >
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
              <Tabs value={activeTab} onChange={handleTabChange} centered>
                <Tab 
                  label="Image Analysis" 
                  icon={<ImageIcon />} 
                  iconPosition="start"
                  disabled={loading}
                />
                <Tab 
                  label="Video Analysis" 
                  icon={<VideocamIcon />} 
                  iconPosition="start"
                  disabled={loading}
                />
              </Tabs>
            </Box>

            {activeTab === 1 && renderVideoAnalysisInfo()}

            <Typography variant="h5" gutterBottom>
              Upload Your {fileType === 'image' ? 'Image' : 'Video'}
            </Typography>
            
            <Typography variant="body2" color="text.secondary" paragraph>
              {fileType === 'image' 
                ? 'For best results, upload a clear full-body image. The person should be standing naturally, facing the camera directly or from the side.'
                : 'Upload a short video (max 6 seconds) where the person is rotating slowly to show all angles of their standing position.'}
            </Typography>

            {!previewUrl ? renderFileUploader() : renderPreview()}

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}

            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
              <Button
                variant="contained"
                size="large"
                disabled={!selectedFile || loading}
                onClick={handleSubmit}
                startIcon={loading && <CircularProgress size={24} color="inherit" />}
              >
                {loading ? 'Analyzing...' : `Analyze ${fileType === 'image' ? 'Posture' : 'Video'}`}
              </Button>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              p: 4,
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: 'background.paper',
              height: '100%',
            }}
          >
            <Typography variant="h5" gutterBottom>
              Results
            </Typography>

            {!result && !loading ? (
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: 300,
                  textAlign: 'center',
                }}
              >
                <Box
                  sx={{
                    width: 80,
                    height: 80,
                    borderRadius: '50%',
                    bgcolor: 'rgba(25, 118, 210, 0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mb: 2,
                  }}
                >
                  <WarningIcon sx={{ fontSize: 40, color: 'primary.main' }} />
                </Box>
                <Typography variant="h6" color="text.secondary">
                  No Analysis Yet
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Upload {fileType === 'image' ? 'an image' : 'a video'} and click "Analyze {fileType === 'image' ? 'Posture' : 'Video'}" to see results
                </Typography>
              </Box>
            ) : loading ? (
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: 300,
                  textAlign: 'center',
                }}
              >
                <CircularProgress size={60} sx={{ mb: 3 }} />
                <Typography variant="h6">
                  {fileType === 'image' 
                    ? 'Analyzing your posture...' 
                    : 'Analyzing your video...'}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {fileType === 'image'
                    ? 'This may take a few moments'
                    : 'This may take a minute as we analyze multiple frames'}
                </Typography>
              </Box>
            ) : (
              // Render the appropriate results section based on file type
              result.is_video ? renderVideoResults() : renderImageResults()
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default AnalysisTool; 