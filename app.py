# app.py
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import os
import sqlite3
import hashlib
import threading
from contextlib import contextmanager
from PIL import Image
import io
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from werkzeug.utils import secure_filename
from threading import Semaphore
from functools import wraps
import logging

logging.basicConfig(
    filename='image_archiver.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Kritik işlemleri try-except blokları içine alın ve logları kaydedin
# try:
#     # İşlem
# except Exception as e:
#     logging.error(f"Hata: {str(e)}", exc_info=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
DATABASE_PATH = 'image_archive.db'
SCAN_BATCH_SIZE = 10  # Process images in batches to avoid overloading
OLLAMA_API_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and lightweight model for embeddings
# Eşzamanlı istek sınırlaması
MAX_CONCURRENT_REQUESTS = 1  # Tek seferde yalnızca 1 istek işlenecek
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

# Thread-local storage for database connections
db_local = threading.local()

# Global variable to track scan progress
scan_progress = {
    "total_images": 0,
    "processed_images": 0,
    "current_image": "",
    "new_images": 0,
    "updated_images": 0,
    "status": "idle"  # idle, scanning, completed, error
}

@contextmanager
def get_db_connection():
    """
    Get a thread-safe database connection.
    Each thread gets its own connection to avoid locking issues.
    """
    # Check if this thread already has a connection
    if not hasattr(db_local, 'connection'):
        # Create a new connection for this thread
        db_local.connection = sqlite3.connect(DATABASE_PATH, timeout=30.0)
        # Enable foreign keys
        db_local.connection.execute('PRAGMA foreign_keys = ON')
        # Set row factory to return dictionaries
        db_local.connection.row_factory = sqlite3.Row
    
    try:
        yield db_local.connection
    except Exception as e:
        db_local.connection.rollback()
        raise e

def init_db():
    """Initialize the SQLite database with necessary tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        file_name TEXT,
        hash TEXT,  
        file_size INTEGER,
        width INTEGER,
        height INTEGER,
        format TEXT,
        description TEXT,
        scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        image_id INTEGER PRIMARY KEY,
        embedding BLOB,
        FOREIGN KEY (image_id) REFERENCES images (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def reset_database():
    """
    Reset the database completely to start fresh.
    This deletes the existing database and creates a new one.
    """
    import os
    
    # First check if the database file exists
    if os.path.exists(DATABASE_PATH):
        print(f"Removing existing database: {DATABASE_PATH}")
        # Rename the old database as backup
        backup_path = f"{DATABASE_PATH}.backup"
        try:
            # Remove existing backup if it exists
            if os.path.exists(backup_path):
                os.remove(backup_path)
            # Rename current database to backup
            os.rename(DATABASE_PATH, backup_path)
            print(f"Old database backed up to: {backup_path}")
        except Exception as e:
            print(f"Error backing up database: {e}")
    
    # Create new database with proper schema
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create images table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        file_name TEXT,
        hash TEXT,  
        file_size INTEGER,
        width INTEGER,
        height INTEGER,
        format TEXT,
        description TEXT,
        scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create embeddings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        image_id INTEGER PRIMARY KEY,
        embedding BLOB,
        FOREIGN KEY (image_id) REFERENCES images (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print("New database created successfully")
    return True

def get_image_hash(image_path):
    """
    Generate a perceptual hash of the image to identify it regardless of filename
    """
    try:
        with Image.open(image_path) as img:
            # Resize to small size for reliable hashing
            img = img.resize((8, 8), Image.LANCZOS).convert('L')
            # Calculate average pixel value
            pixels = list(img.getdata())
            avg_pixel = sum(pixels) / len(pixels)
            # Create binary hash
            bits = ''.join('1' if pixel >= avg_pixel else '0' for pixel in pixels)
            # Convert binary to hexadecimal
            hexadecimal = hex(int(bits, 2))[2:]
            return hexadecimal
    except Exception as e:
        print(f"Error hashing image {image_path}: {e}")
        return None


def rate_limited(max_per_minute):
    """
    Fonksiyon çağrılarını dakika başına belirli bir sayıyla sınırlandıran dekoratör
    """
    interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
                
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

def get_image_description(image_path):
    """
    Get image description using Ollama's LLaVa model with optimizations
    for reduced resource usage:
    1. Resize image to smaller dimensions
    2. Use quantized model if available
    """
    try:
        with Image.open(image_path) as img:
            # Resize image to reduce memory usage and processing time
            # The 512x512 size still provides enough detail for good descriptions
            # while significantly reducing the memory footprint
            max_size = (512, 512)
            img.thumbnail(max_size, Image.LANCZOS)
            
            # Convert image to base64
            img_byte_arr = io.BytesIO()
            # Use JPEG format with reduced quality to further decrease size
            img.save(img_byte_arr, format='JPEG', quality=85)
            
            # Base64 encode the image for Ollama API
            import base64
            image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Prepare prompt for LLaVa - simplified to reduce token count
            prompt = "Describe what's in this image concisely."
            
            # Send request to Ollama API with base64-encoded image
            # Try to use quantized model if available (llava:7b-q4 or similar)
            # Fall back to llava:latest if quantized model isn't available
            # model_name = "llava:7b-q4" # Quantized 7B model uses less resources
            model_name = "llava:latest" # Quantized 7B model uses less resources
            
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    # Add parameters to limit resource usage
                    "options": {
                        "num_ctx": 1024,       # Reduce context window
                        "num_thread": 2,        # Limit CPU threads
                        "num_batch": 1
                    }
                }
            )
            
            # If the quantized model isn't available, fall back to llava:latest
            if response.status_code == 404 and "model not found" in response.text.lower():
                print(f"Quantized model not found, falling back to llava:latest")
                response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": "llava:latest",
                        "prompt": prompt,
                        "images": [image_base64],
                        "stream": False,
                        "options": {
                            "num_ctx": 1024,
                            "num_thread": 2,
                            "num_batch": 1
                        }
                    }
                )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No description generated')
            else:
                print(f"Error from Ollama API: {response.status_code}, {response.text}")
                return "Error generating description"
                
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return "Error processing image"

@rate_limited(max_per_minute=10)  # Dakikada sadece 10 istek
def get_image_description_rate_limited(image_path):
    """
    Rate-limited olarak image description alma fonksiyonu
    """
    # Semaphore ile eşzamanlı istek sayısını sınırla
    with request_semaphore:
        return get_image_description(image_path)

def generate_embedding(text):
    """Generate embedding for text using SentenceTransformer"""
    if not text:
        return None
    embedding = EMBEDDING_MODEL.encode(text)
    return embedding

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

def scan_directory(directory_path):
    """
    Scan a directory and its subdirectories for images,
    process them, and store in the database using a simpler and more reliable approach
    """
    global scan_progress
    
    # Reset progress tracker
    scan_progress["total_images"] = 0
    scan_progress["processed_images"] = 0
    scan_progress["current_image"] = ""
    scan_progress["new_images"] = 0
    scan_progress["updated_images"] = 0
    scan_progress["status"] = "scanning"
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    # Update total count
    scan_progress["total_images"] = len(image_files)
    socketio.emit('scan_progress', scan_progress)
    
    print(f"Found {len(image_files)} images in {directory_path}")
    
    # Process images in batches
    total_processed = 0
    new_images = 0
    updated_images = 0
    
    try:
        for i in range(0, len(image_files), SCAN_BATCH_SIZE):
            batch = image_files[i:i+SCAN_BATCH_SIZE]
            
            for image_path in batch:
                try:
                    # Update current image being processed
                    scan_progress["current_image"] = os.path.basename(image_path)
                    socketio.emit('scan_progress', scan_progress)
                    
                    # First check if this exact path already exists
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM images WHERE file_path = ?", (image_path,))
                        existing_image = cursor.fetchone()
                        
                        if existing_image:
                            # This image is already in database, update scan date
                            cursor.execute(
                                "UPDATE images SET scan_date = CURRENT_TIMESTAMP WHERE id = ?",
                                (existing_image['id'],)
                            )
                            conn.commit()
                            updated_images += 1
                            scan_progress["updated_images"] = updated_images
                            
                            total_processed += 1
                            scan_progress["processed_images"] = total_processed
                            socketio.emit('scan_progress', scan_progress)
                            continue
                    
                    # Generate hash to identify duplicate images
                    try:
                        image_hash = get_image_hash(image_path)
                        if not image_hash:
                            print(f"Could not generate hash for {image_path}, skipping")
                            continue
                    except Exception as e:
                        print(f"Error generating hash for {image_path}: {e}")
                        continue
                    
                    # Get image metadata
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            img_format = img.format
                        
                        file_size = os.path.getsize(image_path)
                        file_name = os.path.basename(image_path)
                    except Exception as e:
                        print(f"Error reading image metadata for {image_path}: {e}")
                        continue
                    
                    # Get image description using LLaVa
                    try:
                        description = get_image_description_rate_limited(image_path)
                    except Exception as e:
                        print(f"Error getting description for {image_path}: {e}")
                        description = "Error generating description"
                    
                    # Insert the image into the database
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        
                        try:
                            # Use INSERT OR IGNORE to skip if file_path already exists
                            # This handles race conditions between threads
                            cursor.execute("""
                                INSERT OR IGNORE INTO images 
                                (file_path, file_name, hash, file_size, width, height, format, description) 
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (image_path, file_name, image_hash, file_size, width, height, img_format, description))
                            
                            if cursor.rowcount > 0:
                                # New image was inserted
                                image_id = cursor.lastrowid
                                new_images += 1
                                scan_progress["new_images"] = new_images
                                
                                # Generate and store embedding
                                try:
                                    embedding = generate_embedding(description)
                                    if embedding is not None:
                                        embedding_blob = embedding.tobytes()
                                        cursor.execute(
                                            "INSERT OR REPLACE INTO embeddings (image_id, embedding) VALUES (?, ?)",
                                            (image_id, embedding_blob)
                                        )
                                except Exception as e:
                                    print(f"Error generating embedding for {image_path}: {e}")
                            else:
                                # Image was skipped due to constraint
                                # This means another thread already processed it or it was already in DB
                                updated_images += 1
                                scan_progress["updated_images"] = updated_images
                            
                            conn.commit()
                            
                        except sqlite3.Error as e:
                            print(f"Database error processing {image_path}: {e}")
                            conn.rollback()
                            continue
                    
                    total_processed += 1
                    scan_progress["processed_images"] = total_processed
                    socketio.emit('scan_progress', scan_progress)
                    
                    # Optional: small delay to prevent overloading
                    time.sleep(0.1)
                
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    # Continue to next image
            
            # Log progress after each batch
            print(f"Processed {min(i + SCAN_BATCH_SIZE, len(image_files))}/{len(image_files)} images")
        
        # Scanning completed successfully
        scan_progress["status"] = "completed"
        socketio.emit('scan_progress', scan_progress)
        
    except Exception as e:
        # Major error occurred
        scan_progress["status"] = "error"
        scan_progress["error_message"] = str(e)
        socketio.emit('scan_progress', scan_progress)
        print(f"Scan error: {e}")
    
    return {
        "total_processed": total_processed,
        "new_images": new_images,
        "updated_images": updated_images
    }

def search_images_by_sql(query):
    """
    Search images using SQL query
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            results = [dict(row) for row in cursor.fetchall()]
            return results
        except sqlite3.Error as e:
            print(f"SQL error: {e}")
            return {"error": str(e)}

def search_images_by_semantic(search_text, limit=20):
    """
    Search images using semantic search with embeddings
    """
    if not search_text:
        return []
    
    # Generate embedding for search text
    search_embedding = generate_embedding(search_text)
    if search_embedding is None:
        return []
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get all embeddings
        cursor.execute("SELECT image_id, embedding FROM embeddings")
        results = cursor.fetchall()
        
        similarities = []
        for row in results:
            image_id = row['image_id']
            embedding_blob = row['embedding']
            
            # Convert blob back to numpy array
            stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = np.dot(search_embedding, stored_embedding) / (
                np.linalg.norm(search_embedding) * np.linalg.norm(stored_embedding)
            )
            
            similarities.append((image_id, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top matches
        top_matches = similarities[:limit]
        
        # Get image details for top matches
        image_details = []
        for image_id, similarity in top_matches:
            cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
            image = dict(cursor.fetchone())
            image['similarity'] = similarity
            image_details.append(image)
        
        return image_details

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_directory_route():
    directory = request.json.get('directory')
    
    # Validate the directory
    is_valid, message = validate_directory(directory)
    if not is_valid:
        return jsonify({"error": message}), 400
    
    # Start scanning in a background thread
    def scan_worker():
        try:
            scan_directory(directory)
        except Exception as e:
            print(f"Scan error: {str(e)}")
    
    thread = threading.Thread(target=scan_worker)
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "message": "Scan started"})

@app.route('/search/sql', methods=['POST'])
def sql_search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    results = search_images_by_sql(query)
    return jsonify(results)

@app.route('/search/semantic', methods=['POST'])
def semantic_search():
    search_text = request.json.get('text')
    if not search_text:
        return jsonify({"error": "Search text is required"}), 400
    
    limit = request.json.get('limit', 20)
    results = search_images_by_semantic(search_text, limit)
    return jsonify(results)

@app.route('/stats')
def stats():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        stats = {}
        cursor.execute("SELECT COUNT(*) as count FROM images")
        stats['total_images'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(DISTINCT hash) as count FROM images")
        stats['unique_images'] = cursor.fetchone()['count']
        
        return jsonify(stats)

@app.route('/scan/progress')
def get_scan_progress():
    return jsonify(scan_progress)

@app.route('/reset-database', methods=['POST'])
def reset_database_route():
    """API endpoint to reset the database"""
    try:
        success = reset_database()
        return jsonify({"success": success, "message": "Database reset successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/view-image/<int:image_id>')
def view_image(image_id):
    """
    Safely serve an image from the database
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT file_path FROM images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        
        if not result:
            return "Image not found", 404
            
        file_path = result['file_path']
        
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

# SocketIO event for requesting progress
@socketio.on('request_progress')
def handle_progress_request():
    emit('scan_progress', scan_progress)

if __name__ == '__main__':
    init_db()
    socketio.run(app, debug=True, port=5555, allow_unsafe_werkzeug=True)
