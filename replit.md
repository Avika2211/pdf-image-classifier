# PDF Figure Extraction & Classification Tool

## Overview

This is a Streamlit-based web application that automatically extracts and classifies figures from PDF documents. The tool uses PyMuPDF for PDF processing, PIL for image manipulation, and implements a custom figure classification system. Users can upload PDF files, extract all embedded figures, and get them classified into different categories with an intuitive web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Architecture Pattern**: Single-page application with session state management
- **UI Components**: File uploader, sidebar navigation, main content area for results display
- **State Management**: Streamlit session state for maintaining extracted figures and classification results

### Backend Architecture
- **Processing Pipeline**: Modular design with separate components for extraction and classification
- **Core Components**:
  - `PDFFigureExtractor`: Handles PDF parsing and figure extraction using PyMuPDF
  - `FigureClassifier`: Implements rule-based and feature-based figure classification
  - `utils`: Provides utility functions for file handling and image processing

### Data Processing Flow
1. PDF upload through Streamlit file uploader
2. PDF processing using PyMuPDF (fitz) to extract embedded images
3. Image filtering (minimum size requirements, format validation)
4. Feature extraction from images for classification
5. Rule-based classification into predefined categories
6. Results display with download capabilities

## Key Components

### PDF Processing (`figure_extractor.py`)
- **Purpose**: Extract figures and images from PDF documents
- **Technology**: PyMuPDF (fitz library)
- **Key Features**:
  - Page-by-page image extraction
  - Image size filtering (minimum 50x50 pixels)
  - Format conversion (CMYK to RGB)
  - PIL Image integration

### Figure Classification (`figure_classifier.py`)
- **Purpose**: Classify extracted figures into categories
- **Approach**: Hybrid rule-based and feature-based classification
- **Features Extracted**:
  - Aspect ratio, brightness, contrast
  - Edge density, color diversity
  - Text ratio, line density
  - Shape analysis (circles, rectangles)
  - Symmetry analysis
  - Color statistics (saturation, hue variance)

### Web Interface (`app.py`)
- **Framework**: Streamlit
- **Layout**: Wide layout with sidebar for controls
- **Session Management**: Persistent state for extracted figures and results
- **User Experience**: Progress indication, file size display, batch processing

### Utilities (`utils.py`)
- **File Operations**: File size calculation, download link generation
- **Image Processing**: Resize for display, validation, metadata extraction
- **Helper Functions**: Support for main application workflow

## Data Flow

1. **Input**: User uploads PDF file through Streamlit interface
2. **Extraction**: PyMuPDF processes PDF, extracts embedded images
3. **Filtering**: Images filtered by size and format requirements
4. **Classification**: Each image processed through feature extraction and classification
5. **Storage**: Results stored in Streamlit session state
6. **Display**: Processed figures displayed with classification results
7. **Export**: Users can download individual figures or batch results

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **PyMuPDF (fitz)**: PDF processing and image extraction
- **PIL (Pillow)**: Image processing and manipulation
- **pandas**: Data structure management
- **opencv-python (cv2)**: Computer vision operations
- **scikit-learn**: Machine learning utilities (StandardScaler, RandomForestClassifier)
- **numpy**: Numerical computing

### System Dependencies
- **tempfile**: Temporary file handling
- **zipfile**: Archive creation for bulk downloads
- **io**: In-memory file operations
- **os**: Operating system interface
- **logging**: Application logging

## Deployment Strategy

### Development Environment
- **Platform**: Designed for Replit deployment
- **Configuration**: Streamlit configuration for web deployment
- **File Structure**: Modular Python application with clear separation of concerns

### Production Considerations
- **Scalability**: Session-based processing suitable for single-user instances
- **Memory Management**: Temporary file handling for large PDF processing
- **Error Handling**: Comprehensive logging and validation throughout pipeline
- **Performance**: Image size filtering and efficient PDF processing

### Deployment Requirements
- Python 3.7+ environment
- Sufficient memory for PDF and image processing
- Web server capability (provided by Streamlit)
- File system access for temporary file operations

## Technical Architecture Decisions

### PDF Processing Library Choice
- **Decision**: PyMuPDF (fitz) for PDF processing
- **Rationale**: Robust image extraction capabilities, good performance, PIL integration
- **Alternatives**: PDFPlumber, PyPDF2 (limited image extraction capabilities)

### Classification Approach
- **Decision**: Rule-based classification with feature extraction
- **Rationale**: Interpretable results, no need for large training datasets
- **Future Enhancement**: Machine learning models can be integrated using the existing feature extraction framework

### Web Framework Selection
- **Decision**: Streamlit for rapid prototyping and deployment
- **Rationale**: Quick development, built-in state management, suitable for data applications
- **Trade-offs**: Limited customization compared to Flask/Django but faster development

### Image Processing Strategy
- **Decision**: PIL for image manipulation with OpenCV for computer vision
- **Rationale**: PIL for basic operations, OpenCV for advanced feature extraction
- **Benefits**: Comprehensive image processing capabilities, good performance