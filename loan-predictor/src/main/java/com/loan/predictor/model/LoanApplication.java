package com.loan.predictor.model;

public class LoanApplication {
    private String gender;
    private String married;
    private String dependents;
    private String education;
    private String selfEmployed;
    private int applicantIncome;
    private int coapplicantIncome;
    private int loanAmount;
    private int loanAmountTerm;
    private int creditHistory;
    private String propertyArea;

    // Default constructor
    public LoanApplication() {}

    // Getters and Setters
    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getMarried() {
        return married;
    }

    public void setMarried(String married) {
        this.married = married;
    }

    public String getDependents() {
        return dependents;
    }

    public void setDependents(String dependents) {
        this.dependents = dependents;
    }

    public String getEducation() {
        return education;
    }

    public void setEducation(String education) {
        this.education = education;
    }

    public String getSelfEmployed() {
        return selfEmployed;
    }

    public void setSelfEmployed(String selfEmployed) {
        this.selfEmployed = selfEmployed;
    }

    public int getApplicantIncome() {
        return applicantIncome;
    }

    public void setApplicantIncome(int applicantIncome) {
        this.applicantIncome = applicantIncome;
    }

    public int getCoapplicantIncome() {
        return coapplicantIncome;
    }

    public void setCoapplicantIncome(int coapplicantIncome) {
        this.coapplicantIncome = coapplicantIncome;
    }

    public int getLoanAmount() {
        return loanAmount;
    }

    public void setLoanAmount(int loanAmount) {
        this.loanAmount = loanAmount;
    }

    public int getLoanAmountTerm() {
        return loanAmountTerm;
    }

    public void setLoanAmountTerm(int loanAmountTerm) {
        this.loanAmountTerm = loanAmountTerm;
    }

    public int getCreditHistory() {
        return creditHistory;
    }

    public void setCreditHistory(int creditHistory) {
        this.creditHistory = creditHistory;
    }

    public String getPropertyArea() {
        return propertyArea;
    }

    public void setPropertyArea(String propertyArea) {
        this.propertyArea = propertyArea;
    }

    @Override
    public String toString() {
        return "LoanApplication{" +
                "gender='" + gender + '\'' +
                ", married='" + married + '\'' +
                ", dependents='" + dependents + '\'' +
                ", education='" + education + '\'' +
                ", selfEmployed='" + selfEmployed + '\'' +
                ", applicantIncome=" + applicantIncome +
                ", coapplicantIncome=" + coapplicantIncome +
                ", loanAmount=" + loanAmount +
                ", loanAmountTerm=" + loanAmountTerm +
                ", creditHistory=" + creditHistory +
                ", propertyArea='" + propertyArea + '\'' +
                '}';
    }
} 