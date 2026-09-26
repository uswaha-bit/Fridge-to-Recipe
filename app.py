from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from PIL import Image
from ultralytics import YOLO
import gradio as gr

GEMINI_MODEL_NAME = "gemini-3.8-flash"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
detector = YOLO("fridge_yolov8n_best.pt")


def detect_ingredients(image: Image.Image):
    """Step 1: Run the fine-tuned YOLOv8 detector and return the labeled/
    annotated image plus the list of detected ingredient labels. Every class
    in this model is a food ingredient, so no filtering is needed."""
    results = detector(image, conf=0.35)
    result = results[0]

    detected_labels = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = detector.names[class_id]
        detected_labels.append(label)

    # result.plot() returns a numpy array (BGR) with boxes drawn on it
    annotated_array = result.plot()
    annotated_image = Image.fromarray(annotated_array[:, :, ::-1])  # BGR -> RGB

    unique_labels = sorted(set(detected_labels))
    return annotated_image, ", ".join(unique_labels)


def generate_recipes(ingredients_text: str) -> str:
    """Step 2: Ask Gemini to suggest recipes using the (possibly edited)
    ingredient list the user confirms."""
    if not ingredients_text.strip():
        return "Please detect or enter at least one ingredient first."

    prompt = (
        f"I have these ingredients available: {ingredients_text}. "
        "Suggest 3 simple recipes I could make using some or all of these ingredients "
        "(they don't need to use every single one). For each recipe, give: "
        "a title, the ingredients used from my list (plus any common pantry staples "
        "like salt, oil, etc. if needed), and short numbered cooking steps. "
        "Format the response in clean Markdown with headers for each recipe."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )
    return response.text


with gr.Blocks(title="Fridge to Recipe") as demo:
    gr.Markdown("# 🥗 Fridge to Recipe")
    gr.Markdown(
        "**Pipeline:** a YOLOv8 model (PyTorch), fine-tuned on 30 common "
        "fridge ingredients, detects food items in your photo -> you review/edit "
        "the list -> Gemini generates recipe suggestions.\n\n"
        "Upload a fridge or pantry photo to get started."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a fridge/pantry photo")
            detect_btn = gr.Button("1. Detect ingredients", variant="secondary")
            annotated_output = gr.Image(label="Detected objects")

        with gr.Column():
            ingredients_box = gr.Textbox(
                label="Detected ingredients (edit or add to this list before generating recipes)",
                lines=2,
                placeholder="e.g. eggs, spinach, cheddar cheese"
            )
            recipe_btn = gr.Button("2. Generate recipes", variant="primary")
            recipes_output = gr.Markdown()

    detect_btn.click(
        fn=detect_ingredients,
        inputs=image_input,
        outputs=[annotated_output, ingredients_box],
    )

    recipe_btn.click(
        fn=generate_recipes,
        inputs=ingredients_box,
        outputs=recipes_output,
    )

    gr.Markdown(
        "---\n"
        "*Note: this model recognizes 30 common ingredients (trained on a small, "
        "516-image dataset), so it won't catch everything in every fridge -- "
        "edit the ingredients box to add anything the model missed.*"
    )


if __name__ == "__main__":
    demo.launch()
