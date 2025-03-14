# Enhanced features that can be added to the image archiver

# 1. Image serving functionality
# Add this to app.py to serve images from local filesystem

@app.route('/images/<path:image_id>')
def serve_image(image_id):
    """
    Serve image from the database by ID
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

# 2. Image classification and tagging
# Add this class and function to app.py

def classify_image(image_path):
    """
    Classify image using Ollama's LLaVa model to generate tags
    """
    try:
        with Image.open(image_path) as img:
            # Convert image to base64
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
            
            # Base64 encode the image for Ollama API
            import base64
            image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Prepare prompt for LLaVa
            prompt = "Generate 5-10 tags for this image. Format as a comma-separated list."
            
            # Send request to Ollama API
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": "llava:latest",
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                tags_text = result.get('response', '')
                
                # Clean and parse tags
                tags = [tag.strip() for tag in tags_text.split(',')]
                return tags
            else:
                print(f"Error from Ollama API: {response.status_code}, {response.text}")
                return []
                
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

# Add this table to the init_db function
def init_db():
    # ... existing code ...
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER,
        tag TEXT,
        FOREIGN KEY (image_id) REFERENCES images (id),
        UNIQUE(image_id, tag)
    )
    ''')
    
    # ... rest of the function ...

# 3. Image clustering based on embeddings
# Add this function to app.py

def cluster_images(num_clusters=10):
    """
    Cluster images based on their embeddings
    Returns a dictionary mapping image IDs to cluster IDs
    """
    from sklearn.cluster import KMeans
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get all embeddings
    cursor.execute("SELECT image_id, embedding FROM embeddings")
    results = cursor.fetchall()
    
    if len(results) < num_clusters:
        return {}
    
    image_ids = []
    embeddings = []
    
    for row in results:
        image_id = row[0]
        embedding_blob = row[1]
        
        # Convert blob back to numpy array
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        
        image_ids.append(image_id)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_array)
    
    # Create dictionary mapping image IDs to cluster IDs
    image_clusters = {}
    for i, image_id in enumerate(image_ids):
        image_clusters[image_id] = int(clusters[i])
    
    # Store clusters in database
    cursor.execute("CREATE TABLE IF NOT EXISTS clusters (image_id INTEGER PRIMARY KEY, cluster_id INTEGER)")
    
    for image_id, cluster_id in image_clusters.items():
        cursor.execute("INSERT OR REPLACE INTO clusters (image_id, cluster_id) VALUES (?, ?)",
                      (image_id, cluster_id))
    
    conn.commit()
    conn.close()
    
    return image_clusters

@app.route('/cluster', methods=['POST'])
def cluster_endpoint():
    """
    Endpoint to trigger image clustering
    """
    num_clusters = request.json.get('num_clusters', 10)
    
    try:
        clusters = cluster_images(num_clusters)
        return jsonify({"success": True, "clusters": len(set(clusters.values())), "images": len(clusters)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clusters', methods=['GET'])
def get_clusters():
    """
    Get images grouped by cluster
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check if clusters table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'")
    if not cursor.fetchone():
        conn.close()
        return jsonify({"error": "No clusters available. Run clustering first."}), 400
    
    # Get all clusters
    cursor.execute("""
        SELECT c.cluster_id, COUNT(c.image_id) as count
        FROM clusters c
        GROUP BY c.cluster_id
        ORDER BY count DESC
    """)
    
    clusters = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({"clusters": clusters})

@app.route('/clusters/<int:cluster_id>', methods=['GET'])
def get_cluster_images(cluster_id):
    """
    Get images in a specific cluster
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get images in cluster
    cursor.execute("""
        SELECT i.* 
        FROM images i
        JOIN clusters c ON i.id = c.image_id
        WHERE c.cluster_id = ?
    """, (cluster_id,))
    
    images = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({"cluster_id": cluster_id, "images": images})

# 4. Image deduplication based on perceptual hash
# Add this function to app.py

@app.route('/duplicates')
def find_duplicates():
    """
    Find duplicate images based on hash
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Find hashes that appear more than once
    cursor.execute("""
        SELECT hash, COUNT(*) as count
        FROM images
        GROUP BY hash
        HAVING count > 1
        ORDER BY count DESC
    """)
    
    duplicate_hashes = cursor.fetchall()
    
    # Get details for each duplicate set
    duplicates = []
    for row in duplicate_hashes:
        hash_value = row['hash']
        
        cursor.execute("SELECT * FROM images WHERE hash = ?", (hash_value,))
        images = [dict(img) for img in cursor.fetchall()]
        
        duplicates.append({
            "hash": hash_value,
            "count": row['count'],
            "images": images
        })
    
    conn.close()
    return jsonify({"duplicate_sets": len(duplicates), "duplicates": duplicates})

# 5. Image statistics and reporting
# Add this function to app.py

@app.route('/statistics')
def get_statistics():
    """
    Get various statistics about the image collection
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    stats = {}
    
    # Total and unique images
    cursor.execute("SELECT COUNT(*) FROM images")
    stats['total_images'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT hash) FROM images")
    stats['unique_images'] = cursor.fetchone()[0]
    
    # Images by format
    cursor.execute("""
        SELECT format, COUNT(*) as count
        FROM images
        GROUP BY format
        ORDER BY count DESC
    """)
    stats['formats'] = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Images by resolution category
    cursor.execute("""
        SELECT 
            CASE
                WHEN width >= 3840 AND height >= 2160 THEN '4K+'
                WHEN width >= 1920 AND height >= 1080 THEN 'HD'
                WHEN width >= 1280 AND height >= 720 THEN 'SD'
                ELSE 'Low Res'
            END as resolution,
            COUNT(*) as count
        FROM images
        GROUP BY resolution
        ORDER BY count DESC
    """)
    stats['resolutions'] = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Recent scans
    cursor.execute("""
        SELECT DATE(scan_date) as date, COUNT(*) as count
        FROM images
        GROUP BY date
        ORDER BY date DESC
        LIMIT 10
    """)
    stats['recent_scans'] = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    return jsonify(stats)

# 6. Multi-user support with authentication
# Add these imports at the top of app.py
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# from werkzeug.security import generate_password_hash, check_password_hash

# Add this code to app.py for basic authentication

"""
# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# In-memory user store for simplicity - use database in production
users = {
    1: User(1, 'admin', generate_password_hash('admin123'))
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Find user by username
        for user in users.values():
            if user.username == username:
                if user.check_password(password):
                    login_user(user)
                    return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Protect routes with login_required
@app.route('/')
@login_required
def index():
    return render_template('index.html')
"""
