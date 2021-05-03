# Grocery-Carbon-Footprint-Calculator

A serverless application (using AWS Lambda and AWS API Gateway) that provides the Carbon Footprint of the detected Grocery item from an image.

This repository consists of:

1. The Training code (using Stratified K-fold Cross-Validation) to train a ResNet-50 or EfficientNet-B1 on the Freiburg Grocery dataset to classify different types of Grocery items
2. The AWS Serverless Application Model (AWS SAM) code to deploy the serverless application which uses the trained model from Step 1.
