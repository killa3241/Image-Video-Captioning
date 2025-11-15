import argparse
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Local Model Setup ---
# Load BLIP model from your local folder "blip"
MODEL_PATH = "./blip"

processor = BlipProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)

def caption_image(image_path: str):
    """
    Generates a caption for a given image file path.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    print(f"--- Processing: {os.path.basename(image_path)} ---")

    try:
        # Load image
        raw_image = Image.open(image_path).convert("RGB")
        print("Image loaded successfully.")

        # Preprocess + generate caption
        inputs = processor(raw_image, return_tensors="pt")
        output_ids = model.generate(**inputs, max_length=50)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)

        # Print caption
        print("\nGENERATED CAPTION:")
        print(f"\"{caption}\"")

        # Save caption
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = (
            os.path.splitext(os.path.basename(image_path))[0] + "_caption.txt"
        )
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w") as f:
            f.write(caption)

        print(f"\nCaption saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during captioning: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Image Captioning using BLIP")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    caption_image(args.image_path)
