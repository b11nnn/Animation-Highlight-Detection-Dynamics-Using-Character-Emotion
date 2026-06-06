# Animation-Highlight-Detection-Dynamics-Using-Character-Emotion

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

## Character Emotion Classification

We used CLIP as the base model for character emotion classification.

The base model was:

```python
openai/clip-vit-base-patch32
```

However, zero-shot CLIP can struggle with stylized animation characters because their facial expressions are often exaggerated or visually different from real human faces.

To improve emotion classification performance for animation images, we applied LoRA fine-tuning to CLIP.

## LoRA Fine-Tuning

We built an animation emotion dataset using cropped character face images.

Each image was organized into one of the target emotion classes, such as:

- happy
- sad
- angry
- fearful
- disgusted
- surprised
- neutral

Then, we defined a custom dataset class to load the cropped character images and their emotion labels.

```python
class AEAAAnimationDataset(Dataset):
    def __init__(self, base_dir, processor, emotion_list):
        self.base_dir = base_dir
        self.processor = processor
        self.emotion_list = emotion_list
        self.emotion_to_idx = {
            emotion: idx for idx, emotion in enumerate(self.emotion_list)
        }
```

We loaded the original CLIP model and inserted LoRA adapters into the visual encoder.

```python
base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

The LoRA configuration was:

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["visual_projection"]
)
```

This means that instead of updating the entire CLIP model, we only trained a small number of additional LoRA parameters.

This makes the model more efficient while still adapting it to the animation domain.

## Training Setup

For fine-tuning, we used text prompts for each emotion class.

```python
emotion_prompts = [f"a photo of a character feeling {e}" for e in EMOTIONS]
```

The model was trained using image-text similarity logits from CLIP.

```python
loss = criterion(outputs.logits_per_image, labels)
```

Training was conducted for 3 epochs with the following setup:

```python
batch_size = 16
learning_rate = 5e-5
optimizer = AdamW
loss_function = CrossEntropyLoss
```

After training, the LoRA weights were saved and later loaded back into the original CLIP model.

```python
lora_model.save_pretrained(OUTPUT_DIR)
```

If the trained LoRA weights were available, we combined them with the original CLIP model.

```python
lora_clip_model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH).to(DEVICE)
```

If the weights were not found, the pipeline used the original zero-shot CLIP model as a fallback.

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
- adapting CLIP to animation character emotion classification using LoRA fine-tuning
- modeling emotional mismatch between expected and predicted character emotions
- designing temporal emotion-based features
- combining regression and classification models through hybrid ensemble scoring

## Future Work

This approach can be extended to other animation datasets and more diverse story-based videos.

Future improvements may include:

- using a larger animation emotion dataset
- improving character detection accuracy
- comparing more vision-language models
- reducing dependency on LLM-generated intermediate outputs
- applying the pipeline to longer and more diverse animation videos
