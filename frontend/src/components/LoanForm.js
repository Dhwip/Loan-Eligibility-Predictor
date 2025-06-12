import React, { useState } from 'react';
import axios from 'axios';
import {
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Typography,
  Container,
  Box,
  Alert,
  CircularProgress,
  useTheme,
  useMediaQuery,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
  Chip
} from '@mui/material';

const steps = ['Personal Information', 'Financial Details', 'Loan Information'];

const LoanForm = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const [formData, setFormData] = useState({
    gender: '',
    married: '',
    dependents: '',
    education: '',
    selfEmployed: '',
    applicantIncome: '',
    coapplicantIncome: '',
    loanAmount: '',
    loanAmountTerm: '',
    creditHistory: '',
    propertyArea: ''
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeStep, setActiveStep] = useState(0);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
    // Don't clear prediction result automatically - let user explicitly start new application
  };

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate that all required fields are filled
    const requiredFields = ['gender', 'married', 'dependents', 'education', 'selfEmployed', 
                          'applicantIncome', 'coapplicantIncome', 'loanAmount', 'loanAmountTerm', 
                          'creditHistory', 'propertyArea'];
    
    const missingFields = requiredFields.filter(field => !formData[field]);
    
    if (missingFields.length > 0) {
      setError(`Please fill in all required fields: ${missingFields.join(', ')}`);
      return;
    }
    
    setLoading(true);
    setError('');
    setPredictionResult(null);
    setHasSubmitted(false);

    try {
      console.log('Sending data to backend:', formData);
      const response = await axios.post('http://localhost:8000/api/predict', formData);
      console.log('Backend response:', response.data);
      
      // Spring Boot returns a boolean directly
      setPredictionResult(response.data);
      setHasSubmitted(true);
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      gender: '',
      married: '',
      dependents: '',
      education: '',
      selfEmployed: '',
      applicantIncome: '',
      coapplicantIncome: '',
      loanAmount: '',
      loanAmountTerm: '',
      creditHistory: '',
      propertyArea: ''
    });
    setPredictionResult(null);
    setHasSubmitted(false);
    setError('');
    setActiveStep(0);
  };

  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Gender</InputLabel>
                <Select
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  label="Gender"
                >
                  <MenuItem value="Male">Male</MenuItem>
                  <MenuItem value="Female">Female</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Married</InputLabel>
                <Select
                  name="married"
                  value={formData.married}
                  onChange={handleChange}
                  label="Married"
                >
                  <MenuItem value="Yes">Yes</MenuItem>
                  <MenuItem value="No">No</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Dependents"
                name="dependents"
                value={formData.dependents}
                onChange={handleChange}
                variant="outlined"
                placeholder="Number of dependents"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Education</InputLabel>
                <Select
                  name="education"
                  value={formData.education}
                  onChange={handleChange}
                  label="Education"
                >
                  <MenuItem value="Graduate">Graduate</MenuItem>
                  <MenuItem value="Not Graduate">Not Graduate</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Self Employed</InputLabel>
                <Select
                  name="selfEmployed"
                  value={formData.selfEmployed}
                  onChange={handleChange}
                  label="Self Employed"
                >
                  <MenuItem value="Yes">Yes</MenuItem>
                  <MenuItem value="No">No</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        );
      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Applicant Income"
                name="applicantIncome"
                type="number"
                value={formData.applicantIncome}
                onChange={handleChange}
                variant="outlined"
                placeholder="Monthly income"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Coapplicant Income"
                name="coapplicantIncome"
                type="number"
                value={formData.coapplicantIncome}
                onChange={handleChange}
                variant="outlined"
                placeholder="Monthly income"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Credit History</InputLabel>
                <Select
                  name="creditHistory"
                  value={formData.creditHistory}
                  onChange={handleChange}
                  label="Credit History"
                >
                  <MenuItem value={1}>Good Credit History</MenuItem>
                  <MenuItem value={0}>Poor Credit History</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        );
      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Loan Amount"
                name="loanAmount"
                type="number"
                value={formData.loanAmount}
                onChange={handleChange}
                variant="outlined"
                placeholder="Requested loan amount"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Loan Amount Term (months)"
                name="loanAmountTerm"
                type="number"
                value={formData.loanAmountTerm}
                onChange={handleChange}
                variant="outlined"
                placeholder="Loan term in months"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Property Area</InputLabel>
                <Select
                  name="propertyArea"
                  value={formData.propertyArea}
                  onChange={handleChange}
                  label="Property Area"
                >
                  <MenuItem value="Urban">Urban</MenuItem>
                  <MenuItem value="Semiurban">Semiurban</MenuItem>
                  <MenuItem value="Rural">Rural</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        );
      default:
        return <Typography>Unknown step</Typography>;
    }
  };

  const renderPredictionResult = () => {
    // Only show prediction result if user has actually submitted the form
    console.log('renderPredictionResult called:', { hasSubmitted, predictionResult });
    
    if (!hasSubmitted || predictionResult === null) {
      console.log('Not showing prediction result - hasSubmitted:', hasSubmitted, 'predictionResult:', predictionResult);
      return null;
    }

    const isApproved = predictionResult === true;
    console.log('Showing prediction result - isApproved:', isApproved);
    
    return (
      <Card sx={{ mt: 3, border: '2px solid', borderColor: isApproved ? 'success.main' : 'error.main' }}>
        <CardContent sx={{ textAlign: 'center', py: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
            <Box
              sx={{
                width: 48,
                height: 48,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mr: 2,
                bgcolor: isApproved ? 'success.main' : 'error.main',
                color: 'white'
              }}
            >
              {isApproved ? '✓' : '✗'}
            </Box>
            <Typography variant="h5" component="div">
              {isApproved ? 'Loan Approved!' : 'Loan Not Approved'}
            </Typography>
          </Box>
          <Chip 
            label={isApproved ? 'Eligible for Loan' : 'Not Eligible for Loan'} 
            color={isApproved ? 'success' : 'error'}
            variant="outlined"
            size="large"
            sx={{ mb: 2 }}
          />
          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              onClick={resetForm}
              size="large"
              sx={{
                background: 'linear-gradient(45deg, #1976d2, #42a5f5)',
                minWidth: 200,
                height: 48
              }}
            >
              Start New Application
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  };

  const isStepComplete = (step) => {
    switch (step) {
      case 0:
        return formData.gender && formData.married && formData.dependents && 
               formData.education && formData.selfEmployed;
      case 1:
        return formData.applicantIncome && formData.coapplicantIncome && 
               formData.creditHistory !== '';
      case 2:
        return formData.loanAmount && formData.loanAmountTerm && formData.propertyArea;
      default:
        return false;
    }
  };

  const isFormComplete = () => {
    const requiredFields = ['gender', 'married', 'dependents', 'education', 'selfEmployed', 
                          'applicantIncome', 'coapplicantIncome', 'loanAmount', 'loanAmountTerm', 
                          'creditHistory', 'propertyArea'];
    return requiredFields.every(field => formData[field]);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography 
          variant={isMobile ? "h4" : "h3"} 
          component="h1" 
          gutterBottom
          sx={{ 
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #1976d2, #42a5f5)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 2
          }}
        >
          Loan Eligibility Predictor
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
          Get instant loan eligibility prediction using AI
        </Typography>
      </Box>

      <Card elevation={3} sx={{ borderRadius: 3, overflow: 'hidden' }}>
        <Box sx={{ 
          background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
          color: 'white',
          p: 3,
          textAlign: 'center'
        }}>
          <Typography variant="h5" gutterBottom>
            Step {activeStep + 1} of {steps.length}
          </Typography>
          <Typography variant="body1">
            {steps[activeStep]}
          </Typography>
        </Box>

        <CardContent sx={{ p: 4 }}>
          {!isMobile && (
            <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
              {steps.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          )}

          <form onSubmit={handleSubmit}>
            {getStepContent(activeStep)}

            {error && (
              <Alert severity="error" sx={{ mt: 3 }}>
                {error}
              </Alert>
            )}

            {renderPredictionResult()}

            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
              <Button
                disabled={activeStep === 0 || hasSubmitted}
                onClick={handleBack}
                variant="outlined"
                size="large"
              >
                Back
              </Button>

              <Box>
                {activeStep === steps.length - 1 ? (
                  <Button
                    type="submit"
                    variant="contained"
                    size="large"
                    disabled={loading || !isFormComplete() || hasSubmitted}
                    sx={{
                      background: 'linear-gradient(45deg, #1976d2, #42a5f5)',
                      minWidth: 150,
                      height: 48
                    }}
                  >
                    {loading ? (
                      <CircularProgress size={24} color="inherit" />
                    ) : (
                      'Get Prediction'
                    )}
                  </Button>
                ) : (
                  <Button
                    variant="contained"
                    onClick={handleNext}
                    size="large"
                    disabled={!isStepComplete(activeStep) || hasSubmitted}
                    sx={{
                      background: 'linear-gradient(45deg, #1976d2, #42a5f5)',
                      minWidth: 150,
                      height: 48
                    }}
                  >
                    Next
                  </Button>
                )}
              </Box>
            </Box>
          </form>
        </CardContent>
      </Card>

      {/* Mobile Progress Indicator */}
      {isMobile && (
        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            Step {activeStep + 1} of {steps.length}: {steps[activeStep]}
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
            {steps.map((_, index) => (
              <Box
                key={index}
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  bgcolor: index <= activeStep ? 'primary.main' : 'grey.300',
                  mx: 0.5
                }}
              />
            ))}
          </Box>
        </Box>
      )}
    </Container>
  );
};

export default LoanForm;