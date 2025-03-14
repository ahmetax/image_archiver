# Image Archiver with LLaVa and Semantic Search

This is a Flask-based web application that scans directories for images, uses the LLaVa model through Ollama to generate detailed descriptions, and allows for both SQL and semantic search of the images.

## Features

- **Image Scanning**: Recursively scans directories for image files
- **Perceptual Hashing**: Identifies duplicate images regardless of filename
- **AI Image Description**: Uses LLaVa model via Ollama to generate detailed descriptions
- **SQL Filtering**: Query images using standard SQL
- **Semantic Search**: Find images based on natural language descriptions
- **Responsive Web Interface**: Browse and search your image collection

## Prerequisites

- Python 3.8+
- Ollama server running locally with LLaVa model installed
- Sufficient disk space for the image database

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-archiver.git
   cd image-archiver
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up Ollama with LLaVa:
   ```
   # Install Ollama from https://ollama.ai/
   # Then pull the LLaVa model
   ollama pull llava:latest
   ```

4. Start the Ollama server:
   ```
   ollama serve
   ```

5. Run the Flask application:
   ```
   python app.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Scanning Images

1. Enter the full path to the directory containing your images
2. Click "Start Scan"
3. Wait for the process to complete (can take some time depending on the number of images)

### SQL Search Examples

Search for images by description:
```sql
SELECT * FROM images WHERE description LIKE '%beach%' LIMIT 10
```

Find large images:
```sql
SELECT * FROM images WHERE width > 1920 AND height > 1080
```

Find recently scanned images:
```sql
SELECT * FROM images WHERE scan_date > datetime('now', '-1 day')
```

### Semantic Search

Enter natural language queries like:
- "sunset over water"
- "people smiling outdoors"
- "food on a table"
- "buildings in a city"

## How It Works

### Image Identification

The application uses perceptual hashing to identify images regardless of filename. This means if you rename an image or have duplicates, it will still recognize them as the same image and maintain a consistent description.

### Description Generation

When a new image is found, the application sends it to the LLaVa model via Ollama's API with a prompt requesting a detailed description. The description includes information about objects, colors, actions, environment, and any visible text.

### Embedding Generation

For semantic search capabilities, the application generates embeddings for each image description using the SentenceTransformer model. These embeddings capture the semantic meaning of the descriptions.

### Search Process

- SQL search: Directly queries the SQLite database
- Semantic search: Converts your search text to an embedding and compares it with stored embeddings to find the most similar matches

## Project Structure

- `app.py`: Main Flask application
- `templates/index.html`: Frontend interface
- `image_archive.db`: SQLite database (created on first run)

## Performance Considerations

- Processing large image collections can take significant time
- The application processes images in batches to prevent overwhelming the Ollama server
- Consider running initial scans overnight for large collections
- Semantic search performance depends on the quality of the generated descriptions

## Troubleshooting

### Ollama Connection Issues

If you encounter problems connecting to Ollama:
1. Ensure Ollama is running (`ollama serve`)
2. Verify the LLaVa model is installed (`ollama list`)
3. Check that the API URL in `app.py` matches your Ollama configuration

### Image Loading Issues

If images don't display in the browser:
1. This is expected for security reasons - browsers restrict loading local file URLs
2. The application will show placeholders when images can't be loaded
3. Consider implementing a Flask route to serve images if needed

## Future Improvements

- Add image serving functionality to view images directly in the browser
- Implement image clustering based on visual similarity
- Add support for tagging and categorizing images
- Enable multi-user support with authentication

## License
This project is licensed under the MIT license. See the LICENSE file for details.

## Contact
Ahmet Aksoy - @ahmetax

Project Link: https://github.com/ahmetax/image_archiver

This project was developed with the contribution of Grok3 - (https://grok.com)  and Claude 3.7 Sonnet - (https://claude.ai)


