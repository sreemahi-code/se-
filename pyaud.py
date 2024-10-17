import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import speech_recognition as sr
from PIL import Image

# Load the CLIP model and processor for text-to-image similarity
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the Stable Diffusion pipeline for image generation
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def get_text_embeddings(text_descriptions):
    inputs = processor(text=text_descriptions, return_tensors="pt", padding=True)
    text_embeddings = model.get_text_features(**inputs)
    return text_embeddings

def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    image_embeddings = model.get_image_features(**inputs)
    return image_embeddings

def find_best_match(text_descriptions, image_path):
    text_embeddings = get_text_embeddings(text_descriptions)
    image_embedding = get_image_embedding(image_path)

    # Compute cosine similarity between text embeddings and image embedding
    similarities = torch.cosine_similarity(text_embeddings, image_embedding)
    best_match_idx = similarities.argmax().item()

    return text_descriptions[best_match_idx]

def generate_shape_from_text(description):
    # Generate image from the text description
    image = pipe(description).images[0]
    return image

def generate_best_fit_shape(text_descriptions, reference_image_path):
    # Step 1: Find the best matching description for the image
    best_description = find_best_match(text_descriptions, reference_image_path)
    print(f"Best matching description: {best_description}")

    # Step 2: Generate the image from the best matching description
    generated_image = generate_shape_from_text(best_description)

    # Step 3: Display the generated image
    generated_image.show()

# Voice Bot Setup

def recognize_speech_from_mic(recognizer, microphone):
    """Capture speech and return recognized text"""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for a shape description...")
        audio = recognizer.listen(source)

    try:
        # Recognize the speech using Google's speech recognition engine
        response = recognizer.recognize_google(audio)
        print(f"Recognized: {response}")
        return response
    except sr.RequestError:
        print("API unavailable.")
    except sr.UnknownValueError:
        print("Unable to recognize speech.")
    return ""

def keyword_detection_and_generation():
    # Keywords or shape descriptions to listen for
    shape_keywords = [
        "circle", "square", "triangle", "star", "rectangle", 
        "a square with rounded corners", "hexagon"
    ]
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Loop to continuously listen for shape descriptions
    while True:
        # Capture spoken input
        spoken_text = recognize_speech_from_mic(recognizer, microphone)

        if spoken_text:
            # Check if any keywords are present in the spoken text
            for keyword in shape_keywords:
                if keyword in spoken_text.lower():
                    print(f"Keyword '{keyword}' detected, generating shape...")
                    
                    # Generate and show the shape
                    generated_image = generate_shape_from_text(keyword)
                    generated_image.show()

                    # Break loop after generating a shape
                    return

if __name__ == "__main__":
    print("Voice bot activated. Please describe a shape.")
    keyword_detection_and_generation()
