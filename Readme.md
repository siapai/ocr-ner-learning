# Document Scanning and Entity Extraction

This project focuses on using two primary data science technologies: Computer Vision and Natural Language Processing (NLP). 

## Project Overview

1. **Computer Vision**: 
    - Scanning documents.
    - Identifying text locations.
    - Extracting text from images.

2. **Natural Language Processing**:
    - Extracting entities from the text.
    - Performing text cleaning.
    - Parsing entities from the text.

## Python Libraries

### Computer Vision Module:
- **OpenCV**
- **Numpy**
- **Pytesseract**

### Natural Language Processing Module:
- **Spacy**
- **Pandas**
- **Regular Expression**
- **String**

## Development Stages

### Stage 1: Setup
- Install Python
- Install dependencies

### Stage 2: Data Preparation
- Gather images
- Overview of Pytesseract
- Extract text from images
- Clean and prepare text

### Stage 3: Labeling NER Data
- Manual labeling using BIO tagging:
  - **B**: Beginning
  - **I**: Inside
  - **O**: Outside

### Stage 4: Text Cleaning and Preprocessing
- Prepare training data for Spacy
- Convert data into Spacy format

### Stage 5: Model Training
- Configure the NER model
- Train the model

### Stage 6: Entity Prediction and Parsing
- Load the model
- Render and serve using Displacy
- Draw bounding boxes on images
- Parse entities from text


