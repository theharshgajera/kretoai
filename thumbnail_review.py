import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

# Define properties and their corresponding parameters
PROPERTIES = {
    'face': ['anger', 'blurred', 'headwear', 'joy', 'sorrow', 'surprise'],
    'color': ['vibrancy', 'contrast', 'saturation', 'balance', 'harmony', 'clarity'],
    'composition': ['framing', 'focus', 'symmetry', 'depth', 'clutter', 'alignment'],
    'text': ['readability', 'size', 'contrast', 'placement', 'clarity', 'relevance'],
    'objects': ['prominence', 'variety', 'clarity', 'placement', 'context', 'quantity'],
    'background': ['distraction', 'contrast', 'depth', 'simplicity', 'relevance', 'focus']
}

def review_thumbnail(image_url):
    """Reviews a thumbnail image and provides ratings based on predefined properties."""
    try:
        # Fetch and validate image
        response = requests.get(image_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-pro')

        # Construct prompt with strict lowercase enforcement
        prompt = (
            "You are an expert in YouTube thumbnail analysis. Review the provided thumbnail image and rate it based on the following properties "
            "and their parameters on a scale of 0 to 5 (0 = poor, 5 = excellent). Also, provide an overall virality score out of 10 "
            "based on its potential to attract clicks. Properties and parameters are:\n"
        )
        for prop, params in PROPERTIES.items():
            prompt += f"- {prop}: {', '.join(params)}\n"  # Use lowercase property names in prompt
        prompt += (
            "Return the response EXACTLY as a valid JSON object with the structure: "
            "{'properties': {'property_name': {'parameter_name': rating, ...}, ...}, 'virality_score': overall_score}. "
            "Use integer values for all ratings (0-5 for properties, 0-10 for virality_score). Ensure all property names and parameter names "
            "in the JSON are in lowercase (e.g., 'face', 'anger'). Do not include any text outside the JSON, including code block markers."
        )

        # Call Gemini API with image and prompt
        response = model.generate_content([prompt, img])
        review_text = response.text.strip()

        # Log raw response for debugging
        print(f"Raw Gemini response: {review_text}")

        # Remove ```json``` markers and extract pure JSON
        review_text = re.sub(r'```json\s*|\s*```', '', review_text).strip()

        # Parse the response into a dictionary
        try:
            review_data = json.loads(review_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON response: {review_text}")
            return None, f"Invalid response from AI model: {str(e)}"

        # Normalize property names to lowercase
        if 'properties' in review_data:
            normalized_properties = {}
            for prop in review_data['properties']:
                normalized_prop = prop.lower()
                normalized_properties[normalized_prop] = {
                    param.lower(): rating for param, rating in review_data['properties'][prop].items()
                }
            review_data['properties'] = normalized_properties

        # Validate and ensure all ratings are integers
        for prop in review_data.get('properties', {}):
            for param in review_data['properties'].get(prop, {}):
                rating = review_data['properties'][prop].get(param)
                if rating is not None:
                    review_data['properties'][prop][param] = max(0, min(5, int(float(rating))))  # Clamp to 0-5
        virality = review_data.get('virality_score')
        if virality is not None:
            review_data['virality_score'] = max(0, min(10, int(float(virality))))  # Clamp to 0-10

        return review_data, None
    except Exception as e:
        print(f"❌ Error reviewing thumbnail {image_url}: {str(e)}")
        return None, f"Failed to review thumbnail: {str(e)}"