import React, { useState } from 'react';
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
  Typography,
  useTheme,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import WarningIcon from '@mui/icons-material/Warning';
import axios from 'axios';

const AnalysisTool = () => {
  const theme = useTheme();
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Check if file is an image
      if (!file.type.match('image.*')) {
        setError('Please select an image file (jpg, jpeg, png)');
        return;
      }

      // Check file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size should be less than 10MB');
        return;
      }

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
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError('Please select an image to analyze');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('image', selectedFile);

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
        Upload an image to analyze your posture and get detailed insights
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
            <Typography variant="h5" gutterBottom>
              Upload Your Image
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              For best results, upload a clear full-body image. The person should
              be standing naturally, facing the camera directly or from the side.
            </Typography>

            {!previewUrl ? (
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
                  accept="image/png, image/jpeg, image/jpg"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
                <CloudUploadIcon
                  sx={{ fontSize: 48, color: 'primary.main', mb: 2 }}
                />
                <Typography variant="h6" color="primary.main">
                  Click to upload an image
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Supported formats: JPG, JPEG, PNG
                </Typography>
              </Box>
            ) : (
              <Box sx={{ mt: 3, position: 'relative' }}>
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
            )}

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
                {loading ? 'Analyzing...' : 'Analyze Posture'}
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
                  Upload an image and click "Analyze Posture" to see results
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
                <Typography variant="h6">Analyzing your posture...</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  This may take a few moments
                </Typography>
              </Box>
            ) : (
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
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default AnalysisTool; 