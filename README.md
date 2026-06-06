# Animation-Highlight-Detection-Dynamics-Using-Character-Emotion
programming for artificial intelligence

# Animation Highlight Detection Using Character Emotion Dynamics

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

The original `youtube_highlight_score` was transformed into the final regression target:

```python
highlight_temporal_smooth_rank_score
```

This target was created using:

- rank transformation
- temporal label smoothing
- final rank conversion

This allows the model to focus on identifying top highlight segments rather than predicting the exact raw replay score.

## Feature Design

We extracted five groups of features.

| Feature Group | Number of Features |
|---|---:|
| Scene information | 4 |
| Emotion mismatch | 5 |
| CLIP probability | 3 |
| Within-segment change | 3 |
| Temporal context | 7 |

These features help the model capture moments where emotions become unexpected, intense, mixed, or different from the recent context.

## Modeling

We compared several regression models:

- Ridge Regression
- ElasticNet
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

We also trained binary classifiers to predict whether each segment belongs to the top highlight group.

The final model combines regression predictions and classifier probabilities using a hybrid score:

```python
hybrid_score = alpha * regression_prediction + (1 - alpha) * classifier_probability
```

We tested alpha values of 0.3, 0.5, and 0.7.

To make highlight segments more important during training, we applied a higher sample weight to the top highlight segments.

```python
HIGHLIGHT_WEIGHT = 5.0
```

The final ensemble averages the top 3 hybrid models with equal weights.

```text
Ensemble weights = 1 : 1 : 1
```

## Results

The final model showed strong performance in detecting top-k highlight segments.

| Model | NDCG@K | Precision@K |
|---|---:|---:|
| 10-sec basic feature-only | 0.4009 | 0.1667 |
| 5-sec enhanced feature-only | 0.3636 | 0.1429 |
| Final top-3 hybrid ensemble | 0.9897 | 0.8571 |

In the test set, the model correctly identified 6 of the true top 7 highlight segments.

However, since the highlight extraction process can involve LLM-generated outputs, the final regression performance may vary depending on the quality and consistency of the generated results.

## Key Contribution

This project demonstrates that character-level emotional mismatch and temporal emotion dynamics can be useful signals for animation highlight detection.

Main contributions include:

- constructing highlight labels from YouTube replay heatmap data
- fine-tuning CLIP for animation character emotion classification
- modeling emotional mismatch between expected and predicted character emotions
- designing temporal emotion-based features
- combining regression and classification models through hybrid ensemble scoring

## Future Work

This approach can be extended to other animation datasets and more diverse story-based videos.
