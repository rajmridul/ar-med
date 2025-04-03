import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Container, Divider, Grid, Link, Typography } from '@mui/material';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import { styled } from '@mui/material/styles';

const FooterLink = styled(Link)(({ theme }) => ({
  color: theme.palette.text.secondary,
  textDecoration: 'none',
  '&:hover': {
    color: theme.palette.primary.main,
    textDecoration: 'none',
  },
}));

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <Box component="footer" sx={{ bgcolor: 'white', py: 6, borderTop: 1, borderColor: 'divider' }}>
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <HealthAndSafetyIcon sx={{ mr: 1, color: 'primary.main', fontSize: 32 }} />
              <Typography variant="h5" color="primary" fontWeight={700}>
                AR-MED
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Advanced posture analysis powered by artificial intelligence for preliminary assessment of body alignment issues.
            </Typography>
            <Typography variant="caption" color="text.secondary">
              © {currentYear} AR-MED. All rights reserved.
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="h6" color="text.primary" gutterBottom>
              Quick Links
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <FooterLink component={RouterLink} to="/" sx={{ mb: 1 }}>
                Home
              </FooterLink>
              <FooterLink component={RouterLink} to="/analysis" sx={{ mb: 1 }}>
                Analysis Tool
              </FooterLink>
              <FooterLink component={RouterLink} to="/about" sx={{ mb: 1 }}>
                About
              </FooterLink>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="h6" color="text.primary" gutterBottom>
              Disclaimer
            </Typography>
            <Typography variant="body2" color="text.secondary">
              The analysis provided by AR-MED is for informational purposes only and is not intended to diagnose any medical condition. 
              Always consult with a qualified healthcare provider for professional medical advice.
            </Typography>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Footer; 