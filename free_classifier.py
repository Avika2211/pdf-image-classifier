import os
import json
import logging
import base64
import io
import requests
from PIL import Image
import numpy as np
import cv2

class FreeFigureClassifier:
    """Free figure classifier using Hugging Face models and local processing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.confidence_score = 0.0

        # Free image analysis endpoints (no API key required)
        self.hf_api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

        # Define comprehensive figure categories
        self.figure_categories = {
            "bar_chart": ["bar chart", "bar graph", "column chart", "histogram", "bars"],
            "pie_chart": ["pie chart", "pie graph", "circular chart", "donut chart"],
            "line_graph": ["line chart", "line graph", "trend", "curve", "time series"],
            "scatter_plot": ["scatter plot", "scatter chart", "dots", "points", "correlation"],
            "histogram": ["histogram", "distribution", "frequency", "bins"],
            "box_plot": ["box plot", "boxplot", "whisker", "quartile"],
            "heatmap": ["heatmap", "heat map", "intensity", "color map", "gradient"],
            "flowchart": ["flowchart", "flow chart", "process", "workflow", "diagram"],
            "organizational_chart": ["organizational chart", "org chart", "hierarchy", "structure"],
            "network_diagram": ["network", "graph", "nodes", "connections", "tree"],
            "scientific_diagram": ["molecule", "chemical", "formula", "scientific", "laboratory"],
            "medical_diagram": ["anatomy", "medical", "body", "organ", "health"],
            "engineering_diagram": ["circuit", "schematic", "technical", "blueprint", "engineering"],
            "map": ["map", "geographic", "location", "street", "geography", "satellite"],
            "floor_plan": ["floor plan", "blueprint", "layout", "room", "building"],
            "timeline": ["timeline", "chronology", "sequence", "history", "events"],
            "table": ["table", "grid", "rows", "columns", "data", "spreadsheet"],
            "infographic": ["infographic", "information", "visual", "statistics"],
            "photograph": ["photo", "picture", "image", "real", "camera", "scene"],
            "screenshot": ["screenshot", "screen", "interface", "software", "application"],
            "logo": ["logo", "brand", "symbol", "emblem", "company"],
            "chart_other": ["chart", "graph", "visualization", "data"],
            "diagram_other": ["diagram", "illustration", "drawing", "figure"],
            "unknown": ["unclear", "unknown", "indeterminate"]
        }

        self.figure_emojis = {
            "bar_chart": "📊",
            "pie_chart": "🥧",
            "line_graph": "📈",
            "scatter_plot": "📉",
            "histogram": "📊",
            "box_plot": "📦",
            "heatmap": "🌡️",
            "flowchart": "🔁",
            "organizational_chart": "🗂️",
            "network_diagram": "🔗",
            "scientific_diagram": "🧪",
            "medical_diagram": "🫀",
            "engineering_diagram": "📐",
            "map": "🗺️",
            "floor_plan": "🏠",
            "timeline": "🕒",
            "table": "📋",
            "infographic": "📌",
            "photograph": "📷",
            "screenshot": "🖥️",
            "logo": "🚩",
            "chart_other": "🔢",
            "diagram_other": "📝",
            "unknown": "❓"
        }

    def classify_figure(self, image):
        try:
            description = self._get_image_description(image)
            result = self._classify_from_description(description, image)
            # Inject 'type' and 'classification' for downstream compatibility
            classification = result.get("classification", "unknown")
            result["type"] = f"{self.figure_emojis.get(classification, '❓')} {classification.replace('_', ' ').title()}"
            result["classification"] = classification  # make sure it's present
            return result
        except Exception as e:
            self.logger.error(f"Error in classification: {str(e)}")
            return self._fallback_classification()

    def _get_image_description(self, image):
        try:
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            response = requests.post(
                self.hf_api_url,
                files={"inputs": img_buffer.getvalue()},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
            return self._analyze_image_locally(image)
        except Exception as e:
            self.logger.warning(f"BLIP failed, falling back: {str(e)}")
            return self._analyze_image_locally(image)

    def _analyze_image_locally(self, image):
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            aspect_ratio = width / height

            if len(img_array.shape) == 3:
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                color_diversity = unique_colors / (height * width)
                brightness = np.mean(img_array)
                gray = np.mean(img_array, axis=2)
                edges = self._simple_edge_detection(gray)
                edge_density = np.sum(edges > 0) / (height * width)
            else:
                color_diversity = 0
                brightness = np.mean(img_array)
                edge_density = 0

            desc = []
            if aspect_ratio > 1.5: desc.append("wide")
            elif aspect_ratio < 0.7: desc.append("tall")
            if color_diversity > 0.1: desc.append("colorful")
            elif color_diversity < 0.01: desc.append("simple")
            if edge_density > 0.2: desc.append("detailed diagram")
            elif edge_density > 0.1: desc.append("chart")
            else: desc.append("image")
            if brightness > 200: desc.append("bright")
            elif brightness < 100: desc.append("dark")

            return " ".join(desc) if desc else "visual content"
        except:
            return "visual content"

    def _simple_edge_detection(self, gray_image):
        try:
            return cv2.Canny(gray_image.astype(np.uint8), 50, 150)
        except:
            return np.zeros_like(gray_image)

    def _classify_from_description(self, description, image):
        desc_lower = description.lower()
        scores = {k: sum(len(w) for w in v if w in desc_lower) for k, v in self.figure_categories.items()}
        scores = {k: v for k, v in scores.items() if v > 0}
        enhanced = self._enhance_classification_with_analysis(image, scores)

        if enhanced:
            best = max(enhanced.items(), key=lambda x: x[1])
            classification = best[0]
            confidence = min(0.95, 0.5 + best[1] / 20)
            self.confidence_score = confidence
            return {
                "classification": classification,
                "confidence": confidence,
                "description": self._generate_description(classification, description),
                "details": {
                    "visual_elements": self._extract_visual_elements(description, classification),
                    "analysis_method": "Free local + HuggingFace analysis"
                },
                "reasoning": f"Classified as {classification} using description '{description}'"
            }
        else:
            return self._classify_by_visual_analysis(image, description)

    def _enhance_classification_with_analysis(self, image, scores):
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            ar = w / h
            enhanced = scores.copy()

            if 0.8 <= ar <= 1.5:
                for t in ['bar_chart', 'line_graph', 'scatter_plot']:
                    if t in enhanced: enhanced[t] += 5
            if ar > 2:
                for t in ['timeline', 'flowchart']:
                    if t in enhanced: enhanced[t] += 8
            if len(img_array.shape) == 3:
                uc = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                cd = uc / (h * w)
                if cd > 0.1:
                    for t in ['photograph', 'infographic', 'map']:
                        if t in enhanced: enhanced[t] += 6
                elif cd < 0.01:
                    for t in ['bar_chart', 'line_graph']:
                        if t in enhanced: enhanced[t] += 4
            return enhanced
        except:
            return scores

    def _classify_by_visual_analysis(self, image, description):
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            ar = w / h

            if len(img_array.shape) == 3:
                uc = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                cd = uc / (h * w)

                if cd > 0.1:
                    classification = "photograph"
                    confidence = 0.6
                elif ar > 2:
                    classification = "timeline"
                    confidence = 0.5
                elif 0.8 <= ar <= 1.2:
                    classification = "chart_other"
                    confidence = 0.5
                else:
                    classification = "diagram_other"
                    confidence = 0.4
            else:
                classification = "diagram_other"
                confidence = 0.4

            self.confidence_score = confidence
            return {
                'classification': classification,
                'confidence': confidence,
                'description': f"Visual analysis suggests a {classification.replace('_', ' ')}",
                'details': {
                    'visual_elements': ['visual content'],
                    'analysis_method': 'Fallback visual analysis'
                },
                'reasoning': f"No matching description, classified by aspect ratio {ar:.2f}"
            }
        except:
            return self._fallback_classification()

    def _generate_description(self, classification, original_description):
        readable = classification.replace("_", " ")
        if original_description and original_description not in ["visual content", "image with visual elements"]:
            return f"{readable.title()}. {original_description}"
        return readable.title()

    def _extract_visual_elements(self, description, classification):
        elements = []
        if 'chart' in classification or 'graph' in classification:
            elements.extend(['data visualization', 'axes', 'labels'])
        elif 'diagram' in classification:
            elements.extend(['shapes', 'connectors'])
        elif classification == 'photograph':
            elements.extend(['real objects', 'natural lighting'])
        if description:
            desc_lower = description.lower()
            if 'text' in desc_lower: elements.append('text content')
            if 'color' in desc_lower: elements.append('varied colors')
            if 'line' in desc_lower: elements.append('linear elements')
        return elements[:5] if elements else ['visual content']

    def _fallback_classification(self):
        self.confidence_score = 0.3
        return {
            'classification': 'unknown',
            'confidence': 0.3,
            'description': 'Could not classify figure reliably',
            'details': {
                'visual_elements': ['visual content'],
                'analysis_method': 'Fallback classification'
            },
            'reasoning': 'No meaningful description or features available'
        }

    def get_confidence(self):
        return self.confidence_score
