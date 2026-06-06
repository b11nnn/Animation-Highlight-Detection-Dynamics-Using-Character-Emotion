# Animation-Highlight-Detection-Dynamics-Using-Character-Emotion
programming for artificial intelligence

This project detects highlight moments in animation videos by analyzing character-level emotional dynamics.

Instead of relying only on general visual or audio signals, we focus on how each character’s emotion changes over time.  
The main idea is that important scenes often occur when a character’s actual emotion becomes different from their expected emotional role.

## Project Overview

We define highlight moments as scenes where emotions become unusual, intense, unstable, or different from the recent emotional flow.

For example, if a character who is usually associated with happiness suddenly shows sadness or anger, this emotional mismatch may indicate an important narrative moment.

## Pipeline

The overall pipeline consists of:

1. Frame sampling from animation videos
2. Quality filtering for blurry, dark, or duplicate frames
3. Character detection and cropping using OWL-ViT
4. Character emotion classification using CLIP with LoRA fine-tuning
5. Segment-level feature extraction
6. Highlight prediction using regression, classification, and hybrid ensemble models

## Target Construction

We used YouTube Most Replayed heatmap data to construct the highlight score.

The original `youtube_highlight_score` was transformed into a final regression target:

```python
highlight_temporal_smooth_rank_score
