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

    def classify_figure(self, image):
        """Classify a figure using Hugging Face + local processing."""
        try:
            description = self._get_image_description(image)
            result = self._classify_from_description(description, image)

            # Final standard format
            return {
                "type": self._map_to_label(result.get("classification", "unknown")),
                "confidence": round(result.get("confidence", 0.3) * 100, 1),
                "description": result.get("description", "Could not classify"),
                "reasoning": result.get("reasoning", "No reasoning available"),
                "details": result.get("details", {})
            }

        except Exception as e:
            self.logger.error(f"Error classifying image: {str(e)}")
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
                if isinstance(result, list) and result:
                    return result[0].get('generated_text', '')
        except:
            pass

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
            if aspect_ratio < 0.7: desc.append("tall")
            if color_diversity > 0.1: desc.append("colorful")
            if color_diversity < 0.01: desc.append("simple")
            if edge_density > 0.2: desc.append("detailed diagram")
            elif edge_density > 0.1: desc.append("chart")
            else: desc.append("image")
            if brightness > 200: desc.append("bright")
            if brightness < 100: desc.append("dark")

            return " ".join(desc) if desc else "visual content"
        except:
            return "visual content"

    def _simple_edge_detection(self, gray_image):
        try:
            gray_uint8 = gray_image.astype(np.uint8)
            return cv2.Canny(gray_uint8, 50, 150)
        except:
            return np.zeros_like(gray_image)

    def _classify_from_description(self, description, image):
        try:
            desc_lower = description.lower()
            scores = {}

            for cat, keywords in self.figure_categories.items():
                scores[cat] = sum(len(k) for k in keywords if k in desc_lower)

            enhanced = self._enhance_classification_with_analysis(image, scores)

            if enhanced:
                best = max(enhanced.items(), key=lambda x: x[1])
                category, score = best
                confidence = min(0.95, 0.5 + score / 20)

                self.confidence_score = confidence
                return {
                    'classification': category,
                    'confidence': confidence,
                    'description': self._generate_description(category, description),
                    'details': {
                        'visual_elements': self._extract_visual_elements(description, category),
                        'analysis_method': 'Free local + HuggingFace analysis'
                    },
                    'reasoning': f"Classified as {category} based on visual + textual clues"
                }

            return self._classify_by_visual_analysis(image, description)
        except Exception as e:
            return self._fallback_classification()

    def _enhance_classification_with_analysis(self, image, base_scores):
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            aspect_ratio = w / h

            scores = base_scores.copy()
            if 0.8 <= aspect_ratio <= 1.5:
                for t in ['bar_chart', 'pie_chart', 'line_graph']: scores[t] += 5
            if aspect_ratio > 2:
                for t in ['timeline', 'flowchart']: scores[t] += 8

            if len(img_array.shape) == 3:
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                color_div = unique_colors / (h * w)
                if color_div > 0.1:
                    for t in ['photograph', 'map', 'infographic']: scores[t] += 6
                if color_div < 0.01:
                    for t in ['bar_chart', 'line_graph', 'flowchart']: scores[t] += 4

            return scores
        except:
            return base_scores

    def _classify_by_visual_analysis(self, image, description):
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            aspect_ratio = w / h

            if len(img_array.shape) == 3:
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                color_div = unique_colors / (h * w)

                if color_div > 0.1:
                    return self._format_result("photograph", 0.6, description)
                elif aspect_ratio > 2:
                    return self._format_result("timeline", 0.5, description)
                elif 0.8 <= aspect_ratio <= 1.2:
                    return self._format_result("chart_other", 0.5, description)
                else:
                    return self._format_result("diagram_other", 0.4, description)
            else:
                return self._format_result("diagram_other", 0.4, description)
        except:
            return self._fallback_classification()

    def _generate_description(self, classification, original_description):
        descriptions = {
            'bar_chart': 'A bar chart showing data with rectangular bars',
            'pie_chart': 'A pie chart displaying data as sectors of a circle',
            'line_graph': 'A line graph showing trends or changes over time',
            'scatter_plot': 'A scatter plot showing the relationship between variables',
            'timeline': 'A timeline showing events in chronological order',
            'table': 'A table organizing data in rows and columns',
            'photograph': 'A photographic image of real-world content'
        }
        base = descriptions.get(classification, f"A {classification.replace('_', ' ')}")
        return f"{base}. {original_description}" if original_description not in ["visual content", "image with visual elements"] else base

    def _extract_visual_elements(self, description, classification):
        elements = []
        if 'chart' in classification or 'graph' in classification:
            elements += ['data visualization', 'axes', 'labels']
        elif 'diagram' in classification:
            elements += ['shapes', 'connections', 'text']
        elif classification == 'photograph':
            elements += ['real objects', 'natural lighting']
        elif classification == 'table':
            elements += ['rows', 'columns', 'grid']

        if description:
            desc = description.lower()
            if any(w in desc for w in ['color', 'bright', 'dark']): elements.append('varied colors')
            if any(w in desc for w in ['text', 'label', 'title']): elements.append('text content')
            if any(w in desc for w in ['line', 'curve', 'edge']): elements.append('linear elements')

        return elements[:5] if elements else ['visual content']

    def _format_result(self, classification, confidence, desc):
        self.confidence_score = confidence
        return {
            'classification': classification,
            'confidence': confidence,
            'description': self._generate_description(classification, desc),
            'details': {
                'visual_elements': ['visual content'],
                'analysis_method': 'Visual characteristics analysis'
            },
            'reasoning': f"Classified based on visual properties"
        }

    def _map_to_label(self, category):
        mapping = {
            "bar_chart": "📊 Bar Chart",
            "line_graph": "📈 Line Graph",
            "pie_chart": "🟢 Pie Chart",
            "scatter_plot": "🔵 Scatter Plot",
            "timeline": "⏰ Timeline",
            "photograph": "📷 Photograph",
            "table": "📋 Table",
            "map": "🗺️ Map",
            "flowchart": "📐 Flowchart",
            "scientific_diagram": "📐 Scientific Diagram",
            "diagram_other": "📐 Other Diagram",
            "chart_other": "📊 Other Chart",
            "infographic": "🖼️ Infographic",
            "logo": "🏷️ Logo",
            "screenshot": "💻 Screenshot",
            "engineering_diagram": "📐 Engineering Diagram",
            "network_diagram": "🔗 Network Diagram",
            "organizational_chart": "📋 Organizational Chart",
            "floor_plan": "📐 Floor Plan",
            "box_plot": "📦 Box Plot",
            "heatmap": "🌡️ Heatmap",
            "histogram": "📊 Histogram",
            "medical_diagram": "🩺 Medical Diagram",
            "unknown": "❓ Unknown"
        }
        return mapping.get(category, "📐 Other Diagram")

    def get_confidence(self):
        return self.confidence_score

    def _fallback_classification(self):
        self.confidence_score = 0.3
        return {
            "type": "❓ Unknown",
            "confidence": 30.0,
            "description": "Could not classify figure reliably",
            "reasoning": "Classification failed, using fallback method",
            "details": {
                "visual_elements": ['visual content'],
                "analysis_method": "Fallback classification"
            }
        }
