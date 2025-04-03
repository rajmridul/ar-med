import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Grid,
  Typography,
  useTheme,
} from '@mui/material';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import AssessmentIcon from '@mui/icons-material/Assessment';
import ReceiptLongIcon from '@mui/icons-material/ReceiptLong';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import SecurityIcon from '@mui/icons-material/Security';
import SchoolIcon from '@mui/icons-material/School';

const Hero = () => {
  return (
    <Box
      sx={{
        bgcolor: 'rgba(25, 118, 210, 0.05)',
        pt: 12,
        pb: 8,
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography
              variant="h1"
              fontWeight={700}
              sx={{
                fontSize: { xs: '2.5rem', md: '3.5rem' },
                mb: 2,
                color: 'primary.main',
              }}
            >
              Advanced Posture Analysis
            </Typography>
            <Typography
              variant="h5"
              color="text.secondary"
              sx={{ mb: 4, fontWeight: 400 }}
            >
              AI-powered posture assessment for early detection of alignment issues
            </Typography>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                size="large"
                component={RouterLink}
                to="/analysis"
              >
                Try It Now
              </Button>
              <Button
                variant="outlined"
                size="large"
                component={RouterLink}
                to="/about"
              >
                Learn More
              </Button>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                '& img': {
                  maxWidth: '100%',
                  height: 'auto',
                  borderRadius: 2,
                  boxShadow: 3,
                },
              }}
            >
              <img
                src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
                alt="Posture Analysis"
              />
            </Box>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

const Benefits = () => {
  const benefits = [
    {
      icon: <AssessmentIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Comprehensive Analysis',
      description:
        'Detailed assessment of knee alignment, shoulder imbalance, hip alignment, and overall posture.',
    },
    {
      icon: <ReceiptLongIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Detailed Reports',
      description:
        'Generate professional PDF reports with visualizations, measurements, and personalized recommendations.',
    },
    {
      icon: <AutoFixHighIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'AI Technology',
      description:
        'Powered by advanced machine learning algorithms for accurate body landmark detection.',
    },
    {
      icon: <AccessTimeIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Quick Results',
      description:
        'Get comprehensive analysis in seconds rather than waiting for in-person appointments.',
    },
    {
      icon: <SecurityIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Privacy First',
      description:
        'Your images and data are processed securely and never shared with third parties.',
    },
    {
      icon: <SchoolIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Educational Tool',
      description:
        'Learn about your body alignment and understand how it affects your physical performance.',
    },
  ];

  return (
    <Box sx={{ py: 8 }}>
      <Container maxWidth="lg">
        <Typography
          variant="h2"
          align="center"
          gutterBottom
          sx={{ mb: 6, color: 'primary.main' }}
        >
          Benefits of AR-MED
        </Typography>
        <Grid container spacing={4}>
          {benefits.map((benefit, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Box sx={{ mb: 2 }}>{benefit.icon}</Box>
                  <Typography variant="h5" gutterBottom>
                    {benefit.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {benefit.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

const HowItWorks = () => {
  const steps = [
    {
      title: 'Upload Your Image',
      description:
        'Upload a full-body image showing your posture from front, side, or back view.',
    },
    {
      title: 'AI Analysis',
      description:
        'Our advanced algorithms will automatically detect body landmarks and analyze your posture.',
    },
    {
      title: 'Get Your Report',
      description:
        'View your results online and download a detailed PDF report with recommendations.',
    },
  ];

  return (
    <Box sx={{ py: 8, bgcolor: 'rgba(25, 118, 210, 0.05)' }}>
      <Container maxWidth="lg">
        <Typography
          variant="h2"
          align="center"
          gutterBottom
          sx={{ mb: 6, color: 'primary.main' }}
        >
          How It Works
        </Typography>
        <Grid container spacing={4}>
          {steps.map((step, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Box
                sx={{
                  textAlign: 'center',
                  p: 4,
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 80,
                    height: 80,
                    mx: 'auto',
                    mb: 2,
                    borderRadius: '50%',
                    bgcolor: 'primary.main',
                    color: 'white',
                  }}
                >
                  <Typography variant="h4" fontWeight={700}>
                    {index + 1}
                  </Typography>
                </Box>
                <Typography variant="h5" gutterBottom>
                  {step.title}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  {step.description}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
        <Box sx={{ textAlign: 'center', mt: 6 }}>
          <Button
            variant="contained"
            size="large"
            component={RouterLink}
            to="/analysis"
          >
            Try It Now
          </Button>
        </Box>
      </Container>
    </Box>
  );
};

const Disclaimer = () => {
  return (
    <Box sx={{ py: 8 }}>
      <Container maxWidth="md">
        <Typography
          variant="h2"
          align="center"
          gutterBottom
          sx={{ mb: 4, color: 'primary.main' }}
        >
          Important Disclaimer
        </Typography>
        <Card sx={{ p: 4 }}>
          <CardContent>
            <Typography variant="body1" paragraph>
              <strong>AR-MED is a preliminary analysis tool only:</strong> The
              posture analysis provided by this application is intended for
              informational and educational purposes only. It should not be
              considered as a substitute for professional medical advice,
              diagnosis, or treatment.
            </Typography>
            <Typography variant="body1" paragraph>
              <strong>Consult healthcare professionals:</strong> For an accurate
              diagnosis of posture-related issues or any medical condition,
              always consult with a qualified healthcare provider such as a
              physician, physical therapist, or chiropractor.
            </Typography>
            <Typography variant="body1">
              <strong>Limitations of 2D analysis:</strong> The analysis is based
              on 2D images and has inherent limitations compared to in-person
              clinical assessments. Results may vary based on image quality,
              positioning, and other factors.
            </Typography>
          </CardContent>
        </Card>
      </Container>
    </Box>
  );
};

const Home = () => {
  return (
    <>
      <Hero />
      <Benefits />
      <HowItWorks />
      <Disclaimer />
    </>
  );
};

export default Home; 