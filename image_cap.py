from PIL import Image
import torch
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and preprocess image
img_path = "dog.jpg"
image = Image.open(img_path).convert('RGB')

# Resize image to match model's requirements (adjust as needed)
resize = transforms.Resize((224, 224))
image = resize(image)

# Prepare text input
text = "The image of"

# Tokenize text input
inputs = processor(text=text, return_tensors="pt")

# Process image
transform = transforms.ToTensor()
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

# Generate caption
outputs = model.generate(pixel_values=image_tensor, **inputs, max_length=50)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)
