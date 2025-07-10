import numpy as np
from PIL import Image, ImageStat
import cv2
from sklearn.cluster import KMeans
import logging

class FigureClassifier:
    """Classify extracted figures into different categories."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.confidence_score = 0.0
        self.feature_names = [
            'aspect_ratio', 'brightness', 'contrast', 'edge_density',
            'color_diversity', 'text_ratio', 'line_density', 'circle_ratio',
            'rectangle_ratio', 'symmetry_horizontal', 'symmetry_vertical',
            'dominant_color_count', 'saturation_mean', 'hue_variance'
        ]

    def classify_figure(self, image: Image.Image) -> dict:
        """
        Classify a figure and return a structured dictionary.
        """
        try:
            features = self._extract_features(image)
            category = self._rule_based_classification(features, image)

            return {
                "type": self._map_to_label(category),
                "confidence": round(self.confidence_score * 100, 1),
                "description": self._get_brief_description(category),
                "reasoning": f"Rule-based decision based on visual features: {category}"
            }

        except Exception as e:
            self.logger.error(f"Error classifying figure: {str(e)}")
            return {
                "type": "📐 Other Diagram",
                "confidence": 30.0,
                "description": "Grayscale content, likely diagram or text",
                "reasoning": f"Error during classification: {e}"
            }

    def _map_to_label(self, category):
        mapping = {
            "bar_chart": "📊 Bar Chart",
            "line_graph": "📈 Line Graph",
            "pie_chart": "🟢 Pie Chart",
            "timeline": "⏰ Timeline",
            "photograph": "📷 Photograph",
            "table": "📋 Table",
            "scatter_plot": "🔵 Scatter Plot",
            "flowchart": "📐 Flowchart",
            "scientific_diagram": "📐 Scientific Diagram",
            "map": "🗺️ Map",
            "diagram": "📐 Other Diagram",
        }
        return mapping.get(category, "📐 Other Diagram")

    def _get_brief_description(self, category):
        descs = {
            "bar_chart": "Vertical or horizontal bars used to compare values.",
            "line_graph": "Continuous line to show trends over time or data.",
            "pie_chart": "Circular chart divided into slices.",
            "timeline": "Horizontal layout showing chronological events.",
            "photograph": "High color and texture variation, likely image capture.",
            "table": "Grid of cells with rows and columns of text.",
            "scatter_plot": "Many small dots or shapes scattered across axes.",
            "flowchart": "Connected shapes representing steps or decisions.",
            "scientific_diagram": "Symmetric, structured layout with labeled parts.",
            "map": "Irregular shapes and colors suggest geographic layout.",
            "diagram": "Generic illustration, often grayscale or text-heavy."
        }
        return descs.get(category, "Grayscale content, likely diagram or text")

    def _extract_features(self, image):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        height, width = gray.shape
        aspect_ratio = width / height
        brightness = np.mean(gray)
        contrast = np.std(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)

        if len(img_array.shape) == 3:
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            color_diversity = unique_colors / (height * width)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation_mean = np.mean(hsv[:, :, 1])
            hue_variance = np.var(hsv[:, :, 0])
            dominant_color_count = self._count_dominant_colors(img_array)
        else:
            color_diversity = 0
            saturation_mean = 0
            hue_variance = 0
            dominant_color_count = 1

        text_ratio = self._estimate_text_ratio(gray)
        line_density = self._detect_lines(gray)
        circle_ratio = self._detect_circles(gray)
        rectangle_ratio = self._detect_rectangles(gray)
        symmetry_horizontal = self._calculate_symmetry(gray, axis=0)
        symmetry_vertical = self._calculate_symmetry(gray, axis=1)

        return {
            'aspect_ratio': aspect_ratio,
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'color_diversity': color_diversity,
            'text_ratio': text_ratio,
            'line_density': line_density,
            'circle_ratio': circle_ratio,
            'rectangle_ratio': rectangle_ratio,
            'symmetry_horizontal': symmetry_horizontal,
            'symmetry_vertical': symmetry_vertical,
            'dominant_color_count': dominant_color_count,
            'saturation_mean': saturation_mean,
            'hue_variance': hue_variance
        }

    def _rule_based_classification(self, features, image):
        self.confidence_score = 0.5

        if features['circle_ratio'] > 0.3:
            self.confidence_score = 0.8
            return "pie_chart"
        if features['rectangle_ratio'] > 0.4 and features['text_ratio'] < 0.3:
            self.confidence_score = 0.7
            return "bar_chart"
        if features['line_density'] > 0.3 and features['rectangle_ratio'] < 0.2:
            self.confidence_score = 0.7
            return "line_graph"
        if features['text_ratio'] > 0.4:
            self.confidence_score = 0.6
            return "table"
        if features['edge_density'] > 0.2 and features['rectangle_ratio'] > 0.2:
            self.confidence_score = 0.6
            return "flowchart"
        if features['edge_density'] < 0.1 and features['color_diversity'] > 0.1:
            self.confidence_score = 0.6
            return "photograph"
        if features['symmetry_horizontal'] > 0.7 or features['symmetry_vertical'] > 0.7:
            self.confidence_score = 0.6
            return "scientific_diagram"
        if self._is_scatter_plot(features, image):
            self.confidence_score = 0.7
            return "scatter_plot"
        if self._is_map_like(features, image):
            self.confidence_score = 0.6
            return "map"

        self.confidence_score = 0.4
        return "diagram"

    def _estimate_text_ratio(self, gray):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_area = 0
        total_area = gray.shape[0] * gray.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if h < 50 and w > 20 and area > 100:
                text_area += area
        return text_area / total_area

    def _detect_lines(self, gray):
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        return len(lines) / (gray.shape[0] * gray.shape[1] / 10000) if lines is not None else 0

    def _detect_circles(self, gray):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            total_circle_area = sum(np.pi * r * r for (x, y, r) in circles)
            total_area = gray.shape[0] * gray.shape[1]
            return total_circle_area / total_area
        return 0

    def _detect_rectangles(self, gray):
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangle_area = 0
        total_area = gray.shape[0] * gray.shape[1]
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                rectangle_area += cv2.contourArea(contour)
        return rectangle_area / total_area

    def _calculate_symmetry(self, gray, axis):
        if axis == 0:
            mid = gray.shape[0] // 2
            top, bottom = gray[:mid, :], np.flip(gray[mid:, :], axis=0)
            min_rows = min(top.shape[0], bottom.shape[0])
            top, bottom = top[:min_rows, :], bottom[:min_rows, :]
            corr = np.corrcoef(top.flatten(), bottom.flatten())[0, 1]
        else:
            mid = gray.shape[1] // 2
            left, right = gray[:, :mid], np.flip(gray[:, mid:], axis=1)
            min_cols = min(left.shape[1], right.shape[1])
            left, right = left[:, :min_cols], right[:, :min_cols]
            corr = np.corrcoef(left.flatten(), right.flatten())[0, 1]
        return max(0, corr) if not np.isnan(corr) else 0

    def _count_dominant_colors(self, img_array):
        pixels = img_array.reshape(-1, 3)
        try:
            kmeans = KMeans(n_clusters=min(8, len(np.unique(pixels, axis=0))), random_state=42)
            kmeans.fit(pixels)
            return len(kmeans.cluster_centers_)
        except:
            return 1

    def _is_scatter_plot(self, features, image):
        gray = np.array(image.convert('L'))
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=15, minRadius=1, maxRadius=10)
        return circles is not None and len(circles[0]) > 20

    def _is_map_like(self, features, image):
        return (features['color_diversity'] > 0.05 and 
                0.1 < features['edge_density'] < 0.3 and
                features['symmetry_horizontal'] < 0.3 and
                features['symmetry_vertical'] < 0.3)
