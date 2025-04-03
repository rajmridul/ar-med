import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Divider,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Typography,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ComputerIcon from '@mui/icons-material/Computer';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import BiotechIcon from '@mui/icons-material/Biotech';
import SchoolIcon from '@mui/icons-material/School';
import WarningIcon from '@mui/icons-material/Warning';

const About = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 8 }}>
      <Typography
        variant="h2"
        align="center"
        gutterBottom
        sx={{ mb: 2, color: 'primary.main' }}
      >
        About AR-MED
      </Typography>
      <Typography
        variant="h6"
        align="center"
        color="text.secondary"
        gutterBottom
        sx={{ mb: 6, maxWidth: 800, mx: 'auto' }}
      >
        Advancing preventive healthcare through AI-powered posture analysis
      </Typography>

      <Grid container spacing={6}>
        <Grid item xs={12} md={6}>
          <Typography variant="h4" gutterBottom color="primary">
            Our Mission
          </Typography>
          <Typography variant="body1" paragraph>
            AR-MED was developed with a clear mission: to make professional-grade
            posture analysis accessible to everyone. By leveraging cutting-edge
            artificial intelligence and computer vision technology, we've created
            a tool that can detect potential alignment issues early, before they
            develop into more serious conditions.
          </Typography>
          <Typography variant="body1" paragraph>
            Poor posture and body alignment are often overlooked but can lead to
            chronic pain, reduced mobility, and decreased quality of life. Our
            goal is to help users identify these issues early and provide
            guidance for improvement.
          </Typography>
          <Typography variant="body1">
            While AR-MED is not a substitute for professional medical care, it
            serves as a valuable preliminary assessment tool that can help users
            decide when to seek professional advice and track their progress over
            time.
          </Typography>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              p: 4,
              border: '1px solid',
              borderColor: 'divider',
              height: '100%',
            }}
          >
            <Typography variant="h5" gutterBottom color="primary">
              The Technology Behind AR-MED
            </Typography>
            <List>
              <ListItem sx={{ px: 0 }}>
                <ListItemIcon>
                  <ComputerIcon color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary="Computer Vision"
                  secondary="Uses advanced pose estimation models to accurately identify 33 body landmarks from simple 2D images."
                />
              </ListItem>
              <ListItem sx={{ px: 0 }}>
                <ListItemIcon>
                  <BiotechIcon color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary="Biomechanical Analysis"
                  secondary="Applies clinical biomechanical principles to analyze posture and alignment issues."
                />
              </ListItem>
              <ListItem sx={{ px: 0 }}>
                <ListItemIcon>
                  <HealthAndSafetyIcon color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary="Comprehensive Assessment"
                  secondary="Evaluates knee alignment, shoulder balance, hip alignment, and overall posture."
                />
              </ListItem>
              <ListItem sx={{ px: 0 }}>
                <ListItemIcon>
                  <SchoolIcon color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary="Evidence-Based Recommendations"
                  secondary="Provides personalized suggestions based on established clinical guidelines."
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Divider sx={{ my: 4 }} />
        </Grid>

        <Grid item xs={12} md={6}>
          <Typography variant="h4" gutterBottom color="primary">
            Benefits and Advantages
          </Typography>
          <List>
            {[
              {
                text: 'Early Detection of Alignment Issues',
                description:
                  'Identify potential posture and alignment problems before they progress to more serious conditions or injuries.',
              },
              {
                text: 'Comprehensive Analysis',
                description:
                  'Detailed assessment of multiple body regions including knees, shoulders, hips, and overall posture.',
              },
              {
                text: 'Accessibility',
                description:
                  'Professional-level posture analysis accessible anytime, anywhere, without expensive equipment or appointments.',
              },
              {
                text: 'Educational Value',
                description:
                  'Learn about your body mechanics and how alignment affects your health and physical performance.',
              },
              {
                text: 'Progress Tracking',
                description:
                  'Monitor improvements in your posture over time with consistent, objective measurements.',
              },
              {
                text: 'Preventative Health',
                description:
                  'Take proactive steps to prevent potential musculoskeletal issues before they develop.',
              },
            ].map((item, index) => (
              <ListItem key={index} sx={{ px: 0 }}>
                <ListItemIcon>
                  <CheckCircleOutlineIcon color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary={item.text}
                  secondary={item.description}
                  primaryTypographyProps={{ fontWeight: 500 }}
                />
              </ListItem>
            ))}
          </List>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%', bgcolor: 'rgba(25, 118, 210, 0.03)' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <WarningIcon color="warning" sx={{ mr: 1 }} />
                <Typography variant="h4" color="text.primary" gutterBottom>
                  Important Disclaimer
                </Typography>
              </Box>
              <Typography variant="body1" paragraph>
                <strong>AR-MED is a preliminary assessment tool only.</strong> The
                posture analysis provided is intended for informational and
                educational purposes only and is not meant to be a substitute for
                professional medical advice, diagnosis, or treatment.
              </Typography>
              <Typography variant="body1" paragraph>
                <strong>Limitations of our analysis include:</strong>
              </Typography>
              <List disablePadding>
                <ListItem sx={{ py: 0.5 }}>
                  <ListItemText primary="- 2D image analysis cannot capture all aspects of 3D posture" />
                </ListItem>
                <ListItem sx={{ py: 0.5 }}>
                  <ListItemText primary="- Analysis quality depends on image clarity and positioning" />
                </ListItem>
                <ListItem sx={{ py: 0.5 }}>
                  <ListItemText primary="- Cannot diagnose medical conditions or provide treatment" />
                </ListItem>
                <ListItem sx={{ py: 0.5 }}>
                  <ListItemText primary="- Not a replacement for in-person clinical assessment" />
                </ListItem>
              </List>
              <Typography variant="body1" paragraph sx={{ mt: 2 }}>
                <strong>Always consult with qualified healthcare providers</strong> such
                as physicians, physical therapists, or chiropractors for proper
                diagnosis and treatment of any posture-related issues or medical
                conditions.
              </Typography>
              <Box sx={{ mt: 3 }}>
                <Button
                  variant="contained"
                  component={RouterLink}
                  to="/analysis"
                  fullWidth
                >
                  Try the Analysis Tool
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default About; 