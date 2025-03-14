# Add these imports to app.py
from flask import send_file
from werkzeug.utils import secure_filename

# Add this function to app.py

def validate_directory(directory_path):
    """
    Validate that the directory exists and is accessible
    Returns (is_valid, message)
    """
    if not directory_path:
        return False, "No directory specified"
    
    if not os.path.exists(directory_path):
        return False, f"Directory does not exist: {directory_path}"
    
    if not os.path.isdir(directory_path):
        return False, f"Not a directory: {directory_path}"
    
    # Check if we have read access
    try:
        os.listdir(directory_path)
        return True, "Directory is valid"
    except PermissionError:
        return False, f"Permission denied: {directory_path}"
    except Exception as e:
        return False, f"Error accessing directory: {str(e)}"

# Update the scan route in app.py
@app.route('/scan', methods=['POST'])
def scan():
    directory = request.json.get('directory')
    
    # Validate the directory
    is_valid, message = validate_directory(directory)
    if not is_valid:
        return jsonify({"error": message}), 400
    
    try:
        results = scan_directory(directory)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Scan error: {str(e)}"}), 500

# Add this route to serve images safely
@app.route('/view-image/<int:image_id>')
def view_image(image_id):
    """
    Safely serve an image from the database
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT file_path FROM images WHERE id = ?", (image_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return "Image not found", 404
        
    file_path = result[0]
    
    # Verify that the file exists
    if not os.path.exists(file_path):
        return "Image file not found", 404
        
    # Get file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Map file extensions to MIME types
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff'
    }
    
    mime_type = mime_types.get(file_ext, 'application/octet-stream')
    
    # Return the image file
    return send_file(file_path, mimetype=mime_type)

# Update JavaScript in templates/index.html to use the new image serving route
# Replace the part where images are displayed in displaySearchResults function:
"""
// Create image path for browser
const imgSrc = `/view-image/${image.id}`;  // Use our image serving route

// Add to card.innerHTML:
<img src="${imgSrc}" alt="${image.file_name}" class="img-thumbnail">
"""
