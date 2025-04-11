import torch
from torchvision import transforms
from PIL import Image
import lpips
from transformers import pipeline, CLIPProcessor, CLIPModel
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import time
import webbrowser
from jinja2 import Template
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.io as pio
from torchvision.models import vgg16
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the image folder path
COMIC_FOLDER = r"C:\Users\a5409.DESKTOP-474T628\Desktop\lora photo"

# Define supported image formats
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Load the configuration file
config = {
    "style_image_path": r"C:\Users\a5409.DESKTOP-474T628\Desktop\flux_dev_example.png",
    "reference_image_path": r"C:\Users\a5409.DESKTOP-474T628\Downloads\ComfyUI_temp_uygmr_00008_.png",
    "resize_shape": [256, 256],
    "layouts": {
        "ComfyUI_temp_vkrqr_00002_.png": "2x2",
        "ComfyUI_temp_vkrqr_00005_.png": "2x2",
        "eat_food_high_quality.png": "1x4",
        "school_joke_high_quality.png": "4x1",
    },
    "use_vgg": True # Whether to use the VGG model to extract features
}

class ComicEvaluator:
    def __init__(self, config, comic_folder):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.comic_folder = comic_folder
        self.style_path = config["style_image_path"]
        self.reference_path = config["reference_image_path"]
        self.resize_shape = tuple(config["resize_shape"])
        self.layouts = config.get("layouts", {})
        self.use_vgg = config.get("use_vgg", False)
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
        ])
        self.comic_paths = self._scan_comic_folder()
        if not self.comic_paths:
            logger.error(f"No valid comic images found in {self.comic_folder}")
            raise ValueError(f"No valid comic images found in {self.comic_folder}")
        logger.info(f"Found {len(self.comic_paths)} comic images: {self.comic_paths}")
        try:
            self.loss_fn = lpips.LPIPS(net='vgg').to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize LPIPS model: {e}")
            raise
        try:
            self.emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
        except Exception as e:
            logger.error(f"Failed to initialize emotion classifier: {e}")
            raise
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=False)
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise
        if self.use_vgg:
            try:
                self.vgg = vgg16(weights='IMAGENET1K_V1').features.to(self.device).eval()
            except Exception as e:
                logger.error(f"Failed to initialize VGG model: {e}")
                raise
        self.results = []
        self.emotion_plots = []
        self.summary_plots = {}
        self.emotion_anomalies = {}
        self.image_text_consistency = {}
        self.scene_prompts_dict = {} #Store the scene description of each comic

    def _scan_comic_folder(self):
        if not os.path.exists(self.comic_folder):
            logger.error(f"Comic folder {self.comic_folder} does not exist.")
            raise FileNotFoundError(f"Comic folder {self.comic_folder} does not exist.")
        comic_paths = [os.path.join(self.comic_folder, f) for f in os.listdir(self.comic_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        comic_paths = sorted(comic_paths)
        logger.info(f"Found {len(comic_paths)} images in {self.comic_folder}: {comic_paths}")
        return comic_paths

    def load_images(self, comic_path):
        try:
            self.comic_image = Image.open(comic_path).convert("RGB")
            self.style_image = Image.open(self.style_path).convert("RGB")
            self.reference_image = Image.open(self.reference_path).convert("RGB")
            logger.info(f"Loaded images: {comic_path}, {self.style_path}, {self.reference_path}")
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            raise
        self.comic_tensor = self.transform(self.comic_image).unsqueeze(0).to(self.device)
        self.style_tensor = self.transform(self.style_image).unsqueeze(0).to(self.device)
        self.reference_tensor = self.transform(self.reference_image).unsqueeze(0).to(self.device)

    def _determine_layout(self, width, height, comic_name):
        layout = self.layouts.get(comic_name, None)
        if layout:
            if layout == "2x2":
                rows, cols = 2, 2
            elif layout == "1x4":
                rows, cols = 1, 4
            elif layout == "4x1":
                rows, cols = 4, 1
            else:
                logger.error(f"Unsupported layout: {layout}")
                raise ValueError(f"Unsupported layout: {layout}")
            logger.info(f"Using specified layout for {comic_name}: {layout}")
        else:
            aspect_ratio = width / height
            if 0.5 < aspect_ratio < 2.0:
                layout = "2x2"
                rows, cols = 2, 2
            elif aspect_ratio >= 2.0:
                layout = "1x4"
                rows, cols = 1, 4
            else:
                layout = "4x1"
                rows, cols = 4, 1
            logger.info(f"Determined layout for image (width={width}, height={height}, aspect_ratio={aspect_ratio:.2f}): {layout}")
        return layout, rows, cols

    def split_panels(self, comic_path):
        width, height = self.comic_image.size
        comic_name = os.path.basename(comic_path)
        layout, rows, cols = self._determine_layout(width, height, comic_name)
        
        panel_width = width // cols
        panel_height = height // rows
        
        if width % cols != 0 or height % rows != 0:
            logger.warning(f"Image dimensions ({width}x{height}) not divisible by {cols}x{rows}. Resizing image to fit layout.")
            new_width = panel_width * cols
            new_height = panel_height * rows
            self.comic_image = self.comic_image.resize((new_width, new_height), Image.LANCZOS)
            width, height = new_width, new_height
            logger.info(f"Resized image to ({new_width}x{new_height})")
        
        panels = []
        for i in range(rows):
            for j in range(cols):
                left = j * panel_width
                upper = i * panel_height
                right = (j + 1) * panel_width
                lower = (i + 1) * panel_height
                logger.debug(f"Panel {len(panels) + 1}: left={left}, upper={upper}, right={right}, lower={lower}")
                panel = self.comic_image.crop((left, upper, right, lower))
                panels.append(panel)
        
        expected_panels = rows * cols
        actual_panels = len(panels)
        if actual_panels != expected_panels:
            logger.error(f"Expected {expected_panels} panels but got {actual_panels} panels. Check image dimensions and layout.")
            raise ValueError(f"Panel splitting error: Expected {expected_panels} panels, got {actual_panels}.")
        
        logger.info(f"Number of panels after splitting: {len(panels)}")
        panel_tensors = [self.transform(panel).unsqueeze(0).to(self.device) for panel in panels]
        return panels, panel_tensors

    def evaluate_style(self):
        style_loss = self.loss_fn(self.comic_tensor, self.style_tensor).item()
        logger.info(f"Overall Style Loss (lower is better): {style_loss:.4f}")
        return style_loss

    def evaluate_detail(self):
        detail_loss = self.loss_fn(self.comic_tensor, self.reference_tensor).item()
        logger.info(f"Overall Detail Loss (lower is better): {detail_loss:.4f}")
        return detail_loss

    def evaluate_emotions(self, panels):
        if not panels:
            logger.warning("No panels available for emotion evaluation. Returning empty emotion data.")
            return []
        
        emotion_data = []
        for i, panel in enumerate(panels[:4]):
            try:
                emotions = self.emotion_classifier(panel)
                logger.info(f"\nPanel {i+1} Emotions:")
                for emotion in emotions:
                    logger.info(f"  {emotion['label']}: {emotion['score']:.4f}")
                emotion_data.append({e['label']: e['score'] for e in emotions})
            except Exception as e:
                logger.error(f"Failed to evaluate emotions for panel {i+1}: {e}")
                emotion_data.append({})
        return emotion_data

    def analyze_emotion_trends(self, emotion_data):
        if not emotion_data or len(emotion_data) < 2:
            return []
        anomalies = []
        for i in range(len(emotion_data) - 1):
            for emotion in emotion_data[i]:
                score1 = emotion_data[i].get(emotion, 0.0)
                score2 = emotion_data[i+1].get(emotion, 0.0)
                if abs(score1 - score2) > 0.3:
                    anomalies.append(f"Emotion {emotion} has large change between panel {i+1} and {i+2}: {score1:.4f} -> {score2:.4f}")
        return anomalies

    def plot_emotions(self, comic_name, emotion_data):
        if not emotion_data or not any(emotion_data):
            logger.warning(f"No emotion data available for {comic_name}. Skipping plot.")
            fig = plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, "No Emotion Data Available", fontsize=16, ha='center', va='center')
            plt.xlabel("Panel Number", fontsize=14)
            plt.ylabel("Emotion Score", fontsize=14)
            plt.title(f"Emotion Trends Across Panels - {comic_name}", fontsize=16)
            plt.grid(True)
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close(fig)
            self.emotion_plots.append((comic_name, image_base64))
            return

        all_emotions = set()
        for panel_emotions in emotion_data:
            all_emotions.update(panel_emotions.keys())
        emotions = sorted(list(all_emotions))
        num_panels = min(len(emotion_data), 4)
        data = {emotion: [panel_emotions.get(emotion, 0.0) for panel_emotions in emotion_data[:4]] for emotion in emotions}

        logger.info(f"Plotting emotions for {comic_name}: {data}")

        has_variation = any(len(set(scores)) > 1 for scores in data.values())
        if not has_variation:
            logger.warning(f"Emotion scores for {comic_name} are constant across all panels. Plot will show flat lines.")

        fig = plt.figure(figsize=(12, 8))
        for emotion, scores in data.items():
            logger.info(f"Drawing line for {emotion}: {scores}")
            plt.plot(range(1, num_panels + 1), scores, marker='o', label=emotion, linewidth=2, markersize=8)
            for j, score in enumerate(scores):
                plt.text(j + 1, score, f"{score:.4f}", fontsize=10, ha='center', va='bottom')

        all_scores = [score for scores in data.values() for score in scores if score > 0]
        logger.info(f"All scores for {comic_name}: {all_scores}")
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            padding = (max_score - min_score) * 0.1 or 0.01
            plt.ylim(min_score - padding, max_score + padding)
        else:
            plt.ylim(0, 1)

        plt.xlabel("Panel Number", fontsize=14)
        plt.ylabel("Emotion Score", fontsize=14)
        plt.title(f"Emotion Trends Across Panels - {comic_name}", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(range(1, num_panels + 1), fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(f"debug_emotion_plot_{comic_name}.png", format='png', bbox_inches='tight')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close(fig)
        self.emotion_plots.append((comic_name, image_base64))

    def parse_prompt_scenes(self, prompt):
        scenes = []
        matches = re.findall(r'\[SCENE-\d\](.*?)(?=\[SCENE-\d\]|$)', prompt, re.DOTALL)
        for match in matches:
            scene_text = match.strip()
            if scene_text:
                scenes.append(scene_text)
        while len(scenes) < 4:
            scenes.append("")
        return scenes[:4]

    def evaluate_image_text_consistency(self, panel, prompt):
        if not prompt:
            logger.warning("No prompt provided for image-text consistency evaluation.")
            return 0.0, "No prompt provided"
        try:
            inputs = self.clip_processor(text=[prompt], images=panel, return_tensors="pt", padding=True).to(self.device)
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
            logger.info(f"Image-text consistency score with prompt '{prompt}': {score:.4f}")
            return score, "Success"
        except Exception as e:
            logger.error(f"Failed to evaluate image-text consistency: {e}")
            return 0.0, f"Failed: {str(e)}"

    def evaluate_continuity(self, panel_tensors):
        if len(panel_tensors) < 2:
            logger.warning("Not enough panels to compute continuity (need at least 2 panels).")
            return [], []
        
        panel_features = []
        if self.use_vgg:
            for img in panel_tensors:
                with torch.no_grad():
                    features = self.vgg(img)
                    features = torch.flatten(features)
                    logger.debug(f"VGG feature shape after flatten: {features.shape}")
                panel_features.append(features)
        else:
            for img in panel_tensors:
                img = torch.clamp(img, 0, 1)
                inputs = self.clip_processor(images=img, return_tensors="pt", do_rescale=False).to(self.device)
                features = self.clip_model.get_image_features(**inputs)
                logger.debug(f"CLIP feature shape: {features.shape}")
                panel_features.append(features)
        
        similarities = []
        for i in range(len(panel_tensors) - 1):
            sim = torch.cosine_similarity(panel_features[i], panel_features[i+1], dim=0).item()
            similarities.append(sim)
            logger.info(f"Similarity between panel {i+1} and {i+2}: {sim:.4f}")
        logger.info(f"\nAdjacent Panel Similarities: {similarities}")
        
        if len(set(similarities)) <= 1:
            logger.warning("Continuity scores are constant across all panels. Consider checking the feature extraction method or panel differences.")
        
        return similarities, panel_features

    def compare_images(self, panel_tensors1, panel_tensors2):
        if len(panel_tensors1) != len(panel_tensors2):
            logger.warning("The number of panels in the two images does not match.")
            return [], 0.0
        
        similarities = []
        panel_features1 = []
        panel_features2 = []
        
        if self.use_vgg:
            for img1, img2 in zip(panel_tensors1, panel_tensors2):
                with torch.no_grad():
                    features1 = self.vgg(img1)
                    features2 = self.vgg(img2)
                    features1 = torch.flatten(features1)
                    features2 = torch.flatten(features2)
                panel_features1.append(features1)
                panel_features2.append(features2)
        else:
            for img1, img2 in zip(panel_tensors1, panel_tensors2):
                img1 = torch.clamp(img1, 0, 1)
                img2 = torch.clamp(img2, 0, 1)
                inputs1 = self.clip_processor(images=img1, return_tensors="pt", do_rescale=False).to(self.device)
                inputs2 = self.clip_processor(images=img2, return_tensors="pt", do_rescale=False).to(self.device)
                features1 = self.clip_model.get_image_features(**inputs1)
                features2 = self.clip_model.get_image_features(**inputs2)
                panel_features1.append(features1)
                panel_features2.append(features2)
        
        for i in range(len(panel_features1)):
            sim = torch.cosine_similarity(panel_features1[i], panel_features2[i], dim=0).item()
            similarities.append(sim)
            logger.info(f"Similarity between panel {i+1} of image 1 and panel {i+1} of image 2: {sim:.4f}")
        logger.info(f"\nPanel-wise Similarities between images: {similarities}")
        
        avg_features1 = torch.mean(torch.stack(panel_features1), dim=0)
        avg_features2 = torch.mean(torch.stack(panel_features2), dim=0)
        overall_sim = torch.cosine_similarity(avg_features1, avg_features2, dim=0).item()
        logger.info(f"Overall Similarity between images: {overall_sim:.4f}")
        
        return similarities, overall_sim

    def save_results(self, comic_name, style_loss, detail_loss, emotions, similarities, panel_features, consistency_scores, prompt=None):
        # Store raw floating point numbers without pre-formatting
        result = {
            "Comic": comic_name,
            "Style Loss": style_loss,
            "Detail Loss": detail_loss,
            "Emotions": emotions,
            "Similarities": similarities,  
            "Panel Features": panel_features,
            "Consistency Scores": consistency_scores,  
            "Prompt": prompt
        }
        self.results.append(result)
        # Formatting data when saving to CSV
        df = pd.DataFrame(self.results)
        df['Style Loss'] = df['Style Loss'].map(lambda x: f"{x:.4f}")
        df['Detail Loss'] = df['Detail Loss'].map(lambda x: f"{x:.4f}")
        df['Similarities'] = df['Similarities'].map(lambda sims: [f"{sim:.4f}" for sim in sims])
        df['Consistency Scores'] = df['Consistency Scores'].map(lambda scores: [f"{score:.4f}" for score in scores])
        df.to_csv("evaluation_results.csv", index=False)
        logger.info("Results saved to evaluation_results.csv")

    def _plot_to_base64(self):
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        return image_base64

    def plot_summary(self):
        df = pd.DataFrame(self.results)
        if df.empty:
            logger.warning("No results to plot.")
            return

        if len(df) < 2:
            logger.warning("Only one comic processed. Some plots may not be meaningful.")
            return

        avg_style_loss = np.mean([float(result['Style Loss']) for result in self.results])
        avg_detail_loss = np.mean([float(result['Detail Loss']) for result in self.results])
        max_transitions = max(len(sims) for sims in df['Similarities']) if df['Similarities'].any() else 0
        avg_similarities = np.full(max_transitions, np.nan)
        for t in range(max_transitions):
            valid_sims = [sims[t] for sims in df['Similarities'] if t < len(sims)]
            if valid_sims:
                avg_similarities[t] = np.mean(valid_sims)

        all_emotions = set()
        for emotions in df['Emotions']:
            for panel_emotions in emotions:
                all_emotions.update(panel_emotions.keys())
        emotions = sorted(list(all_emotions))
        avg_emotions = {emotion: 0.0 for emotion in emotions}
        emotion_counts = {emotion: 0 for emotion in emotions}
        for comic_emotions in df['Emotions']:
            for panel_emotions in comic_emotions:
                for emotion, score in panel_emotions.items():
                    avg_emotions[emotion] += score
                    emotion_counts[emotion] += 1
        for emotion in avg_emotions:
            if emotion_counts[emotion] > 0:
                avg_emotions[emotion] /= emotion_counts[emotion]

        avg_consistency = np.mean([np.mean(scores) for scores in df['Consistency Scores'] if scores])

        logger.info(f"\nOverall Statistics:")
        logger.info(f"Average Style Loss: {avg_style_loss:.4f}")
        logger.info(f"Average Detail Loss: {avg_detail_loss:.4f}")
        logger.info(f"Average Continuity Scores: {[f'{sim:.4f}' for sim in avg_similarities]}")
        logger.info(f"Average Emotion Scores: { {k: f'{v:.4f}' for k, v in avg_emotions.items()} }")
        logger.info(f"Average Image-Prompt Consistency: {avg_consistency:.4f}")

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[[float(result['Style Loss']) for result in self.results], 
                         [float(result['Detail Loss']) for result in self.results]], width=0.5)
        plt.axhline(y=avg_style_loss, color='blue', linestyle='--', label=f'Avg Style Loss: {avg_style_loss:.4f}')
        plt.axhline(y=avg_detail_loss, color='orange', linestyle='--', label=f'Avg Detail Loss: {avg_detail_loss:.4f}')
        plt.title("Distribution of Style and Detail Losses Across Comics", fontsize=16)
        plt.ylabel("Loss Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        self.summary_plots['style_detail_loss_distribution'] = self._plot_to_base64()

        all_panel_emotions = []
        panel_labels = []
        for comic_name, comic_emotions in zip(df['Comic'], df['Emotions']):
            for j, panel_emotion in enumerate(comic_emotions[:4]):
                if panel_emotion:
                    emotion_scores = [panel_emotion.get(emotion, 0.0) for emotion in emotions]
                    all_panel_emotions.append(emotion_scores)
                    panel_labels.append(f"{comic_name} - Panel {j+1}")
        
        if all_panel_emotions:
            emotion_matrix = np.array(all_panel_emotions)
            plt.figure(figsize=(12, max(8, len(panel_labels) * 0.5)))
            sns.heatmap(emotion_matrix, xticklabels=emotions, yticklabels=panel_labels, cmap="YlGnBu", annot=True, fmt=".4f")
            plt.title("Emotion Scores Per Panel Across Comics", fontsize=16)
            plt.xlabel("Emotion", fontsize=14)
            plt.ylabel("Comic - Panel", fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.summary_plots['emotion_heatmap'] = self._plot_to_base64()
        else:
            logger.warning("No panel emotion data available for heatmap.")
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, "No Emotion Data Available", fontsize=16, ha='center', va='center')
            plt.title("Emotion Scores Per Panel Across Comics", fontsize=16)
            plt.xlabel("Emotion", fontsize=14)
            plt.ylabel("Comic - Panel", fontsize=14)
            self.summary_plots['emotion_heatmap'] = self._plot_to_base64()

        plt.figure(figsize=(10, 6))
        plt.bar(avg_emotions.keys(), avg_emotions.values(), color='skyblue')
        plt.title("Overall Average Emotion Scores", fontsize=16)
        plt.xlabel("Emotion", fontsize=14)
        plt.ylabel("Average Score", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        self.summary_plots['overall_emotion_distribution'] = self._plot_to_base64()

        all_similarities = [sim for sims in df['Similarities'] for sim in sims]
        has_variation = len(set(all_similarities)) > 1
        if not has_variation:
            logger.warning("Continuity scores are identical across all comics.")

        fig = go.Figure()
        line_styles = ['solid', 'dash', 'dot', 'dashdot']
        for i, similarities in enumerate(df['Similarities']):
            x = list(range(1, len(similarities) + 1))
            y = similarities  
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers+text',
                name=df['Comic'][i],
                text=[f"{sim:.4f}" for sim in similarities],
                textposition="top center",
                line=dict(dash=line_styles[i % len(line_styles)]),
                marker=dict(size=8)
            ))

        if has_variation:
            fig.add_trace(go.Scatter(
                x=list(range(1, max_transitions + 1)),
                y=avg_similarities,
                mode='lines+markers+text',
                name='Average',
                text=[f"{sim:.4f}" for sim in avg_similarities],
                textposition="top center",
                line=dict(dash='dash', color='black'),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title="Continuity Across Comics",
            xaxis_title="Panel Transition",
            yaxis_title="Similarity Score",
            legend=dict(font=dict(size=12)),
            showlegend=True,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            yaxis=dict(range=[min(all_similarities) - 0.05, max(all_similarities) + 0.05] if all_similarities else [0, 1])
        )
        self.summary_plots['continuity_trends'] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        if len(df) >= 2:
            panel_tensors1 = [self.transform(Image.open(os.path.join(self.comic_folder, df['Comic'][0]))).unsqueeze(0).to(self.device) for _ in range(4)]
            panel_tensors2 = [self.transform(Image.open(os.path.join(self.comic_folder, df['Comic'][1]))).unsqueeze(0).to(self.device) for _ in range(4)]
            panel_similarities, overall_sim = self.compare_images(panel_tensors1, panel_tensors2)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(panel_similarities) + 1)),
                y=panel_similarities,
                mode='lines+markers+text',
                name="Panel-wise Similarity",
                text=[f"{sim:.4f}" for sim in panel_similarities],
                textposition="top center",
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="Image Similarity Across Panels",
                xaxis_title="Panel Number",
                yaxis_title="Similarity Score",
                legend=dict(font=dict(size=12)),
                showlegend=True,
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[min(panel_similarities) - 0.05, max(panel_similarities) + 0.05])
            )
            self.summary_plots['image_similarity'] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        sim_data = [[] for _ in range(max_transitions)]
        for sim in df['Similarities']:
            for t in range(max_transitions):
                if t < len(sim):
                    sim_data[t].append(sim[t])
        sim_df = pd.DataFrame({f"Transition {i+1}": sim_data[i] for i in range(max_transitions)})
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=sim_df, width=0.5)
        sns.stripplot(data=sim_df, color='red', size=8, alpha=0.5)
        for i, avg in enumerate(avg_similarities):
            plt.axhline(y=avg, color='black', linestyle='--', alpha=0.5, label=f'Avg Transition {i+1}: {avg:.4f}' if i == 0 else "")
        plt.title("Distribution of Continuity Scores Across Transitions", fontsize=16)
        plt.ylabel("Similarity Score", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        self.summary_plots['continuity_distribution'] = self._plot_to_base64()

        plt.figure(figsize=(8, 6))
        avg_continuity = np.nanmean([np.mean(sims) for sims in df['Similarities']]) if df['Similarities'].any() else 0
        plt.bar(['Style Loss', 'Detail Loss', 'Continuity'], [avg_style_loss, avg_detail_loss, avg_continuity], color=['blue', 'orange', 'green'])
        plt.title("Overall Average Metrics", fontsize=16)
        plt.ylabel("Average Value", fontsize=14)
        plt.grid(True, axis='y')
        self.summary_plots['overall_metrics'] = self._plot_to_base64()

        # Adjust the Emotion vs Continuity Correlation chart to reduce the number of legends
        plt.figure(figsize=(12, 8))
        plotted_emotions = set()
        for i, (comic_emotions, similarities) in enumerate(zip(df['Emotions'], df['Similarities'])):
            for j, (panel_emotions, sim) in enumerate(zip(comic_emotions[1:], similarities)):
                for emotion, score in panel_emotions.items():
                    label = f"{emotion}" if emotion not in plotted_emotions else None
                    plt.scatter(sim, score, label=label, s=100, alpha=0.6)
                    plotted_emotions.add(emotion)
        plt.xlabel("Similarity Score", fontsize=14)
        plt.ylabel("Emotion Score", fontsize=14)
        plt.title("Emotion vs Continuity Correlation", fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        self.summary_plots['emotion_continuity_correlation'] = self._plot_to_base64()

    def generate_report(self):
        df = pd.DataFrame(self.results)
        if df.empty:
            logger.warning("No results to generate report. Skipping report generation.")
            return

        avg_style_loss = np.mean([float(result['Style Loss']) for result in self.results])
        avg_detail_loss = np.mean([float(result['Detail Loss']) for result in self.results])
        max_transitions = max(len(sims) for sims in df['Similarities']) if df['Similarities'].any() else 0
        avg_similarities = np.full(max_transitions, np.nan)
        for t in range(max_transitions):
            valid_sims = [sims[t] for sims in df['Similarities'] if t < len(sims)]
            if valid_sims:
                avg_similarities[t] = np.mean(valid_sims)
        avg_continuity = np.nanmean([np.mean(sims) for sims in df['Similarities']]) if df['Similarities'].any() else 0
        avg_consistency = np.mean([np.mean(scores) for scores in df['Consistency Scores'] if scores]) if df['Consistency Scores'].any() else 0

        all_emotions = set()
        for emotions in df['Emotions']:
            for panel_emotions in emotions:
                all_emotions.update(panel_emotions.keys())
        emotions = sorted(list(all_emotions))
        avg_emotions = {emotion: 0.0 for emotion in emotions}
        emotion_counts = {emotion: 0 for emotion in emotions}
        for comic_emotions in df['Emotions']:
            for panel_emotions in comic_emotions:
                for emotion, score in panel_emotions.items():
                    avg_emotions[emotion] += score
                    emotion_counts[emotion] += 1
        for emotion in avg_emotions:
            if emotion_counts[emotion] > 0:
                avg_emotions[emotion] /= emotion_counts[emotion]

        # Format fractions in image_text_consistency
        formatted_image_text_consistency = {}
        for comic_name in self.image_text_consistency:
            formatted_image_text_consistency[comic_name] = {}
            for i, score_status in self.image_text_consistency[comic_name].items():
                if "N/A" in score_status:
                    formatted_image_text_consistency[comic_name][i] = score_status
                else:
                    score = float(score_status.split(" ")[0])
                    status = score_status.split(" ")[1]
                    formatted_image_text_consistency[comic_name][i] = f"{score:.4f} {status}"

        # Formatting table data
        formatted_results = []
        for result in self.results:
            formatted_result = result.copy()
            formatted_result['Style Loss'] = f"{result['Style Loss']:.4f}"
            formatted_result['Detail Loss'] = f"{result['Detail Loss']:.4f}"
            formatted_result['Similarities'] = [f"{sim:.4f}" for sim in result['Similarities']]
            formatted_result['Consistency Scores'] = [f"{score:.4f}" for score in result['Consistency Scores']]
            formatted_results.append(formatted_result)

        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Comic Evaluation Report</title>
            <!-- Bootstrap CSS -->
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <!-- Font Awesome for icons -->
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
            <!-- Custom CSS -->
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f8f9fa;
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    text-align: center;
                    color: #007bff;
                    margin-bottom: 30px;
                }
                .summary-card {
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    margin-bottom: 30px;
                }
                .summary-card h2 {
                    color: #007bff;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }
                .summary-card p {
                    font-size: 1.1rem;
                    margin: 10px 0;
                }
                .summary-card ul {
                    list-style: none;
                    padding: 0;
                }
                .summary-card li {
                    font-size: 1rem;
                    margin: 5px 0;
                }
                .comic-section {
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    margin-bottom: 30px;
                }
                .comic-section h2 {
                    color: #343a40;
                    border-bottom: 2px solid #343a40;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }
                .anomalies {
                    color: #dc3545;
                }
                .consistency-score {
                    font-weight: bold;
                }
                .consistency-score.high {
                    color: #28a745;
                }
                .consistency-score.low {
                    color: #dc3545;
                }
                .scene-desc {
                    font-style: italic;
                    color: #555;
                    background-color: #f1f1f1;
                    padding: 5px 10px;
                    border-radius: 5px;
                    display: block;
                    margin-top: 5px;
                }
                .plot-section {
                    margin-bottom: 30px;
                }
                .plot-section h2 {
                    color: #343a40;
                    border-bottom: 2px solid #343a40;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }
                .table-responsive {
                    margin-top: 20px;
                }
                .table thead {
                    background-color: #007bff;
                    color: #fff;
                }
                .table th, .table td {
                    vertical-align: middle;
                }
                .table-striped tbody tr:nth-of-type(odd) {
                    background-color: #f8f9fa;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1><i class="fas fa-book-open me-2"></i> Comic Evaluation Report</h1>
                <div class="summary-card">
                    <h2><i class="fas fa-chart-bar me-2"></i> Summary Statistics</h2>
                    <p><i class="fas fa-paint-brush me-2"></i> Average Style Loss: {{ avg_style_loss }}</p>
                    <p><i class="fas fa-search me-2"></i> Average Detail Loss: {{ avg_detail_loss }}</p>
                    <p><i class="fas fa-link me-2"></i> Average Continuity Score: {{ avg_continuity }}</p>
                    <p><i class="fas fa-image me-2"></i> Average Image-Prompt Consistency: {{ avg_consistency }}</p>
                    <h3>Average Emotion Scores:</h3>
                    <ul>
                    {% for emotion, score in avg_emotions.items() %}
                        <li><i class="fas fa-heart me-2"></i> {{ emotion }}: {{ score }}</li>
                    {% endfor %}
                    </ul>
                </div>

                {% for comic_name, plot in emotion_plots %}
                <div class="comic-section">
                    <h2><i class="fas fa-comic me-2"></i> {{ comic_name }}</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h3>Emotion Trends</h3>
                            <img src="data:image/png;base64,{{ plot }}" alt="Emotion Trends - {{ comic_name }}" class="img-fluid">
                        </div>
                        <div class="col-md-6">
                            {% if emotion_anomalies[comic_name] %}
                            <div class="anomalies">
                                <h3>Emotion Anomalies:</h3>
                                <ul>
                                {% for anomaly in emotion_anomalies[comic_name] %}
                                    <li>{{ anomaly }}</li>
                                {% endfor %}
                                </ul>
                            </div>
                            {% endif %}
                            <h3>Image-Prompt Consistency Scores:</h3>
                            <ul>
                            {% for i, score in image_text_consistency[comic_name].items() %}
                                <li>
                                    <span class="consistency-score {% if 'N/A' not in score and score.split(' ')[0]|float > 22 %}high{% elif 'N/A' not in score %}low{% endif %}">
                                        Panel {{ i+1 }}: {{ score }}
                                    </span>
                                    <span class="scene-desc">[Scene {{ i+1 }}: {{ scene_prompts[comic_name][i] if scene_prompts[comic_name][i] else 'No scene description' }}]</span>
                                </li>
                            {% endfor %}
                            </ul>
                        </div>
                    </div>
                </div>
                {% endfor %}

                <div class="plot-section">
                    <h2><i class="fas fa-chart-pie me-2"></i> Overall Metrics</h2>
                    <img src="data:image/png;base64,{{ summary_plots['overall_metrics'] }}" alt="Overall Metrics" class="img-fluid">
                </div>

                <div class="plot-section">
                    <h2><i class="fas fa-boxes me-2"></i> Style and Detail Loss Distribution</h2>
                    <img src="data:image/png;base64,{{ summary_plots['style_detail_loss_distribution'] }}" alt="Style and Detail Loss Distribution" class="img-fluid">
                </div>

                <div class="plot-section">
                    <h2><i class="fas fa-smile me-2"></i> Overall Emotion Distribution</h2>
                    <img src="data:image/png;base64,{{ summary_plots['overall_emotion_distribution'] }}" alt="Overall Emotion Distribution" class="img-fluid">
                </div>

                <div class="plot-section">
                    <h2><i class="fas fa-fire me-2"></i> Emotion Heatmap</h2>
                    <img src="data:image/png;base64,{{ summary_plots['emotion_heatmap'] }}" alt="Emotion Heatmap" class="img-fluid">
                </div>

                <div class="plot-section">
                    <h2><i class="fas fa-link me-2"></i> Continuity Trends</h2>
                    {{ summary_plots['continuity_trends'] | safe }}
                </div>

                <div class="plot-section">
                    <h2><i class="fas fa-images me-2"></i> Image Similarity Across Panels</h2>
                    {{ summary_plots['image_similarity'] | safe }}
                </div>

                <div class="plot-section">
                    <h2><i class="fas fa-box-open me-2"></i> Distribution of Continuity Scores Across Transitions</h2>
                    <img src="data:image/png;base64,{{ summary_plots['continuity_distribution'] }}" alt="Continuity Distribution" class="img-fluid">
                </div>

                <div class="plot-section">
                    <h2><i class="fas fa-heartbeat me-2"></i> Emotion vs Continuity Correlation</h2>
                    <img src="data:image/png;base64,{{ summary_plots['emotion_continuity_correlation'] }}" alt="Emotion vs Continuity Correlation" class="img-fluid">
                </div>

                <div class="table-responsive">
                    <h2><i class="fas fa-table me-2"></i> Evaluation Results</h2>
                    <table class="table table-striped table-bordered">
                        <thead>
                            <tr>
                                <th>Comic</th>
                                <th>Style Loss</th>
                                <th>Detail Loss</th>
                                <th>Emotions</th>
                                <th>Similarities</th>
                                <th>Consistency Scores</th>
                                <th>Prompt</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for result in formatted_results %}
                        <tr>
                            <td>{{ result['Comic'] }}</td>
                            <td>{{ result['Style Loss'] }}</td>
                            <td>{{ result['Detail Loss'] }}</td>
                            <td>{{ result['Emotions'] }}</td>
                            <td>{{ result['Similarities'] | join(', ') }}</td>
                            <td>{{ result['Consistency Scores'] | join(', ') }}</td>
                            <td>{{ result['Prompt'] }}</td>
                        </tr>
                        {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Bootstrap JS and dependencies -->
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.min.js"></script>
        </body>
        </html>
        """
        template = Template(html_template)
        with open("report.html", "w", encoding="utf-8") as f:
            f.write(template.render(
                formatted_results=formatted_results,
                avg_style_loss=f"{avg_style_loss:.4f}",
                avg_detail_loss=f"{avg_detail_loss:.4f}",
                avg_continuity=f"{avg_continuity:.4f}",
                avg_consistency=f"{avg_consistency:.4f}",
                avg_emotions={emotion: f"{score:.4f}" for emotion, score in avg_emotions.items()},
                emotion_plots=self.emotion_plots,
                summary_plots=self.summary_plots,
                emotion_anomalies=self.emotion_anomalies,
                image_text_consistency=formatted_image_text_consistency,
                scene_prompts=self.scene_prompts_dict
            ))
        logger.info("Report generated and saved as report.html")
        webbrowser.open("report.html")

    def process_comic(self, comic_path, prompt=None):
        start_time = time.time()
        comic_name = os.path.basename(comic_path)
        logger.info(f"\nEvaluating comic: {comic_name}")
        try:
            self.load_images(comic_path)
            panels, panel_tensors = self.split_panels(comic_path)
            style_loss = self.evaluate_style()
            detail_loss = self.evaluate_detail()
            emotions = self.evaluate_emotions(panels)
            emotion_anomalies = self.analyze_emotion_trends(emotions)
            self.emotion_anomalies[comic_name] = emotion_anomalies
            logger.info(f"Emotion trend anomalies for {comic_name}: {emotion_anomalies}")
            self.plot_emotions(comic_name, emotions)
            similarities, panel_features = self.evaluate_continuity(panel_tensors)
            consistency_scores = []
            self.image_text_consistency[comic_name] = {}
            
            # Parse the prompt words into scene descriptions
            if prompt:
                scene_prompts = self.parse_prompt_scenes(prompt)
                self.scene_prompts_dict[comic_name] = scene_prompts
                for i, (panel, scene_prompt) in enumerate(zip(panels, scene_prompts)):
                    if scene_prompt:
                        consistency, status = self.evaluate_image_text_consistency(panel, scene_prompt)
                        consistency_scores.append(consistency)
                        self.image_text_consistency[comic_name][i] = f"{consistency:.4f} ({status})"
                        logger.info(f"Panel {i+1} image-text consistency with scene '{scene_prompt}': {consistency:.4f} ({status})")
                    else:
                        self.image_text_consistency[comic_name][i] = "N/A (No scene prompt for this panel)"
                        logger.warning(f"No scene prompt provided for panel {i+1}.")
            else:
                logger.warning(f"No prompt provided for {comic_name}. Skipping image-text consistency evaluation.")
                self.scene_prompts_dict[comic_name] = [""] * 4
                for i in range(len(panels)):
                    self.image_text_consistency[comic_name][i] = "N/A (No prompt provided)"
            
            self.save_results(comic_name, style_loss, detail_loss, emotions, similarities, panel_features, consistency_scores, prompt)
        except Exception as e:
            logger.error(f"Failed to process {comic_name}: {e}", exc_info=True)
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"Processing {comic_name} took {elapsed_time:.2f} seconds")

    def run(self, prompt=None):
        logger.info(f"Starting evaluation for {len(self.comic_paths)} comic images.")
        for comic_path in self.comic_paths:
            self.process_comic(comic_path, prompt)
        logger.info("Finished processing all comics.")
        self.plot_summary()
        self.generate_report()

    def run_with_prompts(self, prompt_list, generate_comics_func):
        all_results = []
        for prompt in prompt_list:
            logger.info(f"\nEvaluating comics for prompt: {prompt}")
            comic_paths = generate_comics_func(prompt, self.comic_folder)
            if not comic_paths:
                logger.warning(f"No comics generated for prompt: {prompt}")
                continue
            self.comic_paths = comic_paths
            self.results = []
            self.emotion_plots = []
            self.summary_plots = {}
            self.emotion_anomalies = {}
            self.image_text_consistency = {}
            self.scene_prompts_dict = {}
            self.run(prompt)
            all_results.extend([(prompt, result) for result in self.results])
        
        df_all = pd.DataFrame([
            {
                "Prompt": prompt,
                "Comic": result["Comic"],
                "Style Loss": result["Style Loss"],
                "Detail Loss": result["Detail Loss"],
                "Average Continuity": np.mean(result["Similarities"]) if result["Similarities"] else 0,
                "Average Consistency": np.mean(result["Consistency Scores"]) if result["Consistency Scores"] else 0
            }
            for prompt, result in all_results
        ])
        # Format the values ​​in prompt_comparison.csv
        df_all['Style Loss'] = df_all['Style Loss'].map(lambda x: f"{float(x):.4f}")
        df_all['Detail Loss'] = df_all['Detail Loss'].map(lambda x: f"{float(x):.4f}")
        df_all['Average Continuity'] = df_all['Average Continuity'].map(lambda x: f"{x:.4f}")
        df_all['Average Consistency'] = df_all['Average Consistency'].map(lambda x: f"{x:.4f}")
        df_all.to_csv("prompt_comparison.csv", index=False)
        logger.info("Prompt comparison results saved to prompt_comparison.csv")

if __name__ == "__main__":
    def dummy_generate_comics_func(prompt, comic_folder):
        logger.info(f"Generating comics for prompt: {prompt}")
        comic_paths = [os.path.join(comic_folder, f) for f in os.listdir(comic_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        return comic_paths

    evaluator = ComicEvaluator(config, COMIC_FOLDER)
    prompt = prompt = prompt = """[SCENE-1] <Taro>, a laid-back man with brown hair, sits on the couch, laughing at his cat <Mew> chasing yarn; he notices an odd glint in her eye, hinting at something otherworldly. [SCENE-2] As the scene unfolds, we see <Taro> pouring over an ancient-looking book, bewildered by cryptic symbols and diagrams that seem to be a mix of human and alien languages; Mew is now sitting beside him, watching intently as if trying to help decipher the codes. [SCENE-3] With the lights dimmed and only a faint moonlight illuminating the room, <Taro> suddenly feels an electric presence in the air, and upon closer inspection, he discovers a strange energy signature emanating from Mew's paw; it is then that Taro realizes - his beloved "cat" might not be of this world. [SCENE-4] As Taro's eyes widen with astonishment, <Mew> begins to transform before his very eyes, her body elongating and morphing into an extraterrestrial being with iridescent scales and large, glowing orbs for eyes; the air is filled with an unearthly hum as Mew reveals her true nature - a galactic explorer in disguise."""
    prompt_list = [prompt]
    evaluator.run_with_prompts(prompt_list, dummy_generate_comics_func)