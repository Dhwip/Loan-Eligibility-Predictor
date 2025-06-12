package com.loan.predictor.controller;

import com.loan.predictor.model.LoanApplication;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.beans.factory.annotation.Autowired;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:3000", allowedHeaders = "*", methods = {RequestMethod.GET, RequestMethod.POST, RequestMethod.OPTIONS})
public class LoanPredictionController {

    @Autowired
    private RestTemplate restTemplate;

    @GetMapping("/test")
    public ResponseEntity<String> test() {
        return ResponseEntity.ok("Spring Boot API is working! CORS should be enabled.");
    }

    @PostMapping("/predict")
    public ResponseEntity<Boolean> predictLoanEligibility(@RequestBody LoanApplication application) {
        try {
            System.out.println("Received prediction request: " + application);
            
            // Call the Python model API
            String modelApiUrl = "http://localhost:9000/predict";
            ResponseEntity<PredictionResponse> response = restTemplate.postForEntity(
                modelApiUrl, 
                application, 
                PredictionResponse.class
            );
            
            if (response.getBody() != null) {
                System.out.println("Model prediction: " + response.getBody().isPrediction());
                return ResponseEntity.ok(response.getBody().isPrediction());
            } else {
                System.out.println("No response from model API");
                return ResponseEntity.ok(false);
            }
        } catch (Exception e) {
            System.err.println("Error calling model API: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.ok(false);
        }
    }

    // Inner class to deserialize the response from Python API
    public static class PredictionResponse {
        private boolean prediction;
        
        public boolean isPrediction() {
            return prediction;
        }
        
        public void setPrediction(boolean prediction) {
            this.prediction = prediction;
        }
    }
} 