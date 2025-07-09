import os
import json
import logging
import base64
import io
from PIL import Image
from google import genai
from google.genai import types

class AIFigureClassifier:
    """AI-powered figure classifier using Google Gemini."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.confidence_score = 0.0
        
        # Define comprehensive figure categories
        self.figure_categories = {
            "bar_chart": "Bar Chart - Shows data using rectangular bars",
            "pie_chart": "Pie Chart - Circular chart showing proportions",
            "line_graph": "Line Graph - Shows trends over time or continuous data",
            "scatter_plot": "Scatter Plot - Shows relationship between two variables",
            "histogram": "Histogram - Shows distribution of data",
            "box_plot": "Box Plot - Shows statistical distribution",
            "heatmap": "Heatmap - Shows data intensity with colors",
            "flowchart": "Flowchart - Shows process or workflow",
            "organizational_chart": "Organizational Chart - Shows hierarchy",
            "network_diagram": "Network Diagram - Shows connections between entities",
            "scientific_diagram": "Scientific Diagram - Technical/scientific illustration",
            "medical_diagram": "Medical Diagram - Anatomical or medical illustration",
            "engineering_diagram": "Engineering Diagram - Technical drawing or schematic",
            "map": "Map - Geographic or spatial representation",
            "floor_plan": "Floor Plan - Architectural layout",
            "timeline": "Timeline - Shows events over time",
            "table": "Table - Structured data in rows and columns",
            "infographic": "Infographic - Visual information presentation",
            "photograph": "Photograph - Real-world image",
            "screenshot": "Screenshot - Computer screen capture",
            "logo": "Logo - Brand or company symbol",
            "chart_other": "Other Chart Type - Specialized chart not in main categories",
            "diagram_other": "Other Diagram - General diagram or illustration",
            "unknown": "Unknown - Cannot determine figure type"
        }
    
    def classify_figure(self, image):
        """
        Classify a figure using Google Gemini AI.
        
        Args:
            image (PIL.Image): The image to classify
            
        Returns:
            dict: Classification results with type, confidence, and description
        """
        try:
            # Convert PIL image to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            image_bytes = img_buffer.read()
            
            # Create the classification prompt
            prompt = self._create_classification_prompt()
            
            # Call Gemini API
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            
            if response.text:
                result = json.loads(response.text)
                self.confidence_score = result.get('confidence', 0.5)
                
                return {
                    'classification': result.get('type', 'unknown'),
                    'confidence': self.confidence_score,
                    'description': result.get('description', 'No description available'),
                    'details': result.get('details', {}),
                    'reasoning': result.get('reasoning', '')
                }
            else:
                return self._fallback_classification()
                
        except Exception as e:
            self.logger.error(f"Error in AI classification: {str(e)}")
            return self._fallback_classification()
    
    def get_confidence(self):
        """Get the confidence score of the last classification."""
        return self.confidence_score
    
    def _create_classification_prompt(self):
        """Create a comprehensive classification prompt."""
        categories_text = "\n".join([f"- {key}: {desc}" for key, desc in self.figure_categories.items()])
        
        prompt = f"""
        Analyze this figure/image and classify it into one of the following categories. Be very precise and accurate.

        AVAILABLE CATEGORIES:
        {categories_text}

        CLASSIFICATION REQUIREMENTS:
        1. Look carefully at the visual elements, structure, and content
        2. Consider the purpose and typical use of the figure
        3. For charts/graphs, identify the specific type (bar, pie, line, scatter, etc.)
        4. For diagrams, determine the specific domain (scientific, medical, engineering, etc.)
        5. For images, distinguish between photographs, screenshots, logos, etc.

        SPECIAL CONSIDERATIONS:
        - Tables: Look for structured data in rows and columns
        - Charts: Identify data visualization patterns (bars, lines, circles, points)
        - Diagrams: Look for flowcharts, organizational structures, technical drawings
        - Scientific: Look for formulas, molecular structures, anatomical drawings
        - Maps: Geographic features, roads, boundaries, topographical elements

        OUTPUT FORMAT (JSON):
        {{
            "type": "category_key_from_list_above",
            "confidence": 0.95,
            "description": "Brief description of what you see",
            "details": {{
                "visual_elements": ["list", "of", "key", "elements"],
                "data_type": "type of data shown if applicable",
                "domain": "subject domain if applicable"
            }},
            "reasoning": "Why you chose this classification"
        }}

        Be extremely accurate. If you're not sure between two categories, pick the most specific one that fits best.
        """
        return prompt
    
    def _fallback_classification(self):
        """Fallback classification when AI fails."""
        self.confidence_score = 0.3
        return {
            'classification': 'unknown',
            'confidence': 0.3,
            'description': 'Could not classify figure',
            'details': {},
            'reasoning': 'AI classification failed, using fallback'
        }
    
    def get_supported_categories(self):
        """Get all supported figure categories."""
        return self.figure_categories
    
    def batch_classify(self, images, progress_callback=None):
        """
        Classify multiple images in batch.
        
        Args:
            images (list): List of PIL Images
            progress_callback (function): Optional callback for progress updates
            
        Returns:
            list: List of classification results
        """
        results = []
        total = len(images)
        
        for i, image in enumerate(images):
            result = self.classify_figure(image)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results