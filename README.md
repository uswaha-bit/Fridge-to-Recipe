# Fridge to Recipe

Upload a photo of your fridge or pantry and get recipe suggestions based on what's inside. Combines a pretrained computer vision model for ingredient detection with an LLM for recipe generation.

## Pipeline

1. **Object detection (YOLOv8, PyTorch, fine-tuned)** — the uploaded photo is passed through a YOLOv8n model, fine-tuned via transfer learning on the [AICook dataset](https://universe.roboflow.com/aicook/aicook-lcv4d) (516 images covering 30 common fridge ingredients), to detect food items and draw bounding boxes around them. See `finetune_yolov8_fridge.ipynb` for the full training process.
2. **Human-in-the-loop review** — the detected ingredient list is shown in an editable text box, since any 30-class model trained on ~500 images won't catch every ingredient in every fridge — the user can add or correct items before generating recipes.
3. **Recipe generation (Gemini API)** — the confirmed ingredient list is sent to Gemini, which returns 2-3 recipe suggestions with ingredients and steps.
4. **UI (Gradio)** — ties the two steps together into a single interactive app.

## Dataset

Fine-tuning used the **AICook dataset** from Roboflow Universe: 516 images covering 30 common fridge ingredients, originally created by a postgraduate applied AI team at Erasmus Brussels for a similar fridge-to-recipe project.

- Source: https://universe.roboflow.com/aicook/aicook-lcv4d
- License: MIT

I did not collect or annotate this data myself — using an existing, purpose built public dataset was the right call given the project's scope, rather than spending time building a labeled dataset from scratch.

## Model performance

Fine-tuned via transfer learning from COCO-pretrained `yolov8n.pt`, 50 epochs (early-stopped on validation performance):

- **mAP50: 0.961**
- **mAP50-95: 0.639**

Per-class performance varies with training sample size — well-represented classes like `banana`, `cheese`, `eggs`, and `butter` scored mAP50 ≥ 0.99, while classes with fewer examples (e.g. `ham`, with only 7 validation instances) or visually similar categories (`chicken` vs. `chicken_breast`) scored lower. This is an expected effect of class imbalance in a small dataset, not a flaw in the training approach.

## Why this design

A pure LLM-only approach (just asking an LLM to "look" at the photo) skips the actual computer vision step entirely. Fine-tuning YOLOv8 means there's a real, trained, evaluated deep learning model in the pipeline with real, measurable limitations.

## Tech stack

- Python
- PyTorch (via Ultralytics YOLOv8)
- Gemini API (google-genai SDK)
- Gradio (UI)

## How to run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
python app.py
```

## what I'd improve with more time

- The training dataset is small (516 base images) and was shot in a single fridge with a fixed camera setup, so the model likely won't generalize as well to fridges that look very different (lighting, angle, packaging brands).
- Some visually similar classes (e.g. `chicken` vs. `chicken_breast`, `beef` vs. `ground_beef`) are harder for the model to distinguish and would benefit from more training examples.
- No handling for partially-visible or stacked items in cluttered photos.
- Could add confidence scores next to each detected label so the user knows which ones to double-check.
- Could cache/rate-limit the Gemini calls.

---

Built as a hands-on project combining a fine-tuned deep learning model with an LLM.
