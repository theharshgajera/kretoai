from flask import Flask, request, jsonify
import google.generativeai as genai
from googleapiclient.discovery import build
import logging
import json
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)  # Enable debug logging

# Load API keys from environment variables
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'your_youtube_api_key')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key')

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

@app.route('/generate_titles', methods=['POST'])
def generate_titles():
    try:
        # Get JSON input
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        idea = data.get('idea')
        script = data.get('script')  # Optional, not used in this implementation

        # Validate input
        if not idea:
            return jsonify({'error': 'Idea is required'}), 400

        # Search YouTube for top 10 most viewed videos related to the idea
        try:
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            search_response = youtube.search().list(
                q=idea,
                part='snippet',
                type='video',
                order='viewCount',
                maxResults=10
            ).execute()
            titles = [item['snippet']['title'] for item in search_response.get('items', [])]
            app.logger.debug(f"YouTube titles: {titles}")
            if not titles:
                return jsonify({'error': 'No videos found for the given idea'}), 404
        except Exception as e:
            app.logger.error(f"YouTube API error: {str(e)}")
            return jsonify({'error': f'YouTube API failed: {str(e)}'}), 500

        # Prepare the prompt for Gemini
        prompt = (
            f"Given the idea: '{idea}' and the following popular video titles related to it: "
            f"{', '.join(titles)}, generate 5 viral video title options that are similar and likely to attract viewers. "
        )
        if script:
            prompt += (
                f"Also consider the following video script for context: '{script[:500]}' (truncated for brevity). "
                f"Use the script's themes to inform the titles, tags, and description. "
            )
        prompt += (
            f"For each title, provide a virality score out of 100. Also, provide 10 tags and one description that are common to all 5 titles. "
            f"Return the response in valid JSON format wrapped in ```json\n...\n``` with the following structure: "
            f"{{ 'titles': [{{'title': 'string', 'virality_score': int}}, ...], 'tags': ['string', ...], 'description': 'string' }}"
        )

        # Call Gemini API using google-generativeai
        try:
            client = genai.GenerativeModel('gemini-2.0-flash')
            response = client.generate_content(prompt)
            app.logger.debug(f"Gemini raw response: {response.text}")
            gemini_response = response.text
            # Extract JSON from ```json\n...\n``` if present
            if gemini_response.startswith('```json\n') and gemini_response.endswith('\n```'):
                gemini_response = gemini_response[7:-4]
            gemini_response = json.loads(gemini_response)  # Parse JSON string
            app.logger.debug(f"Parsed Gemini response: {gemini_response}")
        except Exception as e:
            app.logger.error(f"Gemini API error: {str(e)}")
            return jsonify({'error': f'Gemini API failed: {str(e)}'}), 500

        # Return the response
        return jsonify(gemini_response)

    except Exception as e:
        app.logger.error(f"General error in generate_titles: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
