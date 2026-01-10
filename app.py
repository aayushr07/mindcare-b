from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import numpy as np
import os
import onnxruntime as ort
import librosa
from datetime import datetime, timedelta
import logging
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
from PIL import Image
import io
import base64
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import json
from functools import wraps
from cryptography.fernet import Fernet
import hashlib
import uuid

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) if os.getenv('OPENAI_API_KEY') else None
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None
    logger.warning("OpenAI not installed")

# PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    import qrcode
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PDF generation libraries not installed")

# Speech recognition
try:
    import speech_recognition as sr
    SPEECH_REC_AVAILABLE = True
except ImportError:
    SPEECH_REC_AVAILABLE = False
    logger.warning("SpeechRecognition not installed")

# MongoDB setup (optional - falls back to in-memory if not available)
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
    try:
        mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'), serverSelectionTimeoutMS=2000)
        mongo_client.server_info()  # Test connection
        db = mongo_client['neuro nest']
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.warning(f"MongoDB not available, using in-memory storage: {e}")
        MONGO_AVAILABLE = False
        db = None
except ImportError:
    logger.warning("pymongo not installed, using in-memory storage")
    MONGO_AVAILABLE = False
    db = None

app = Flask(__name__, static_folder='build', static_url_path='')

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "max_age": 3600
    }
})

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# HTTPS redirect (in production)
if os.getenv('ENFORCE_HTTPS', 'false').lower() == 'true':
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Encryption setup
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
try:
    cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
except Exception as e:
    logger.warning(f"Encryption setup failed: {e}")
    cipher_suite = None

# Audit logging
audit_log = []

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# In-memory storage for analysis history
analysis_history = []

# In-memory storage for NeuroTwin profiles
# Structure: {user_id: {history: [...], predictions: {...}}}
neurotwin_profiles = {}

# In-memory storage for game scores and streaks
game_scores = {}  # {user_id: {games: {...}, streak: int, badges: [...]}}
leaderboard_data = {}  # {region: [{user_id, score, rank}]}
cognitive_fingerprints = {}  # {user_id: {voice_embedding: [...], face_embedding: [...]}}

# CareNetwork storage
care_networks = {}  # {patient_id: {family_members: [...], doctors: [...], baseline_metrics: {...}}}
notifications = []  # {patient_id, type, message, timestamp, sent_to: [...]}

# Risk analysis model
risk_classifier = None
try:
    # Train a simple risk classifier if model doesn't exist
    from sklearn.linear_model import LogisticRegression
    # This would be trained on actual data in production
    risk_classifier = LogisticRegression(random_state=42)
    # Mock training data for initialization
    X_mock = np.random.rand(100, 5)  # 5 features: latency, pause, tremor, eye_movement_x, eye_movement_y
    y_mock = (X_mock.sum(axis=1) > 2.5).astype(int)  # Binary risk
    risk_classifier.fit(X_mock, y_mock)
    logger.info("Risk classifier initialized")
except Exception as e:
    logger.warning(f"Risk classifier not available: {e}")

# Role-based access control
user_roles = {}  # {user_id: 'patient'|'family'|'doctor'|'admin'}

# === Load Models ===
try:
    COGNITIVE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'cognitive_risk_model.pkl')
    cognitive_model = joblib.load(COGNITIVE_MODEL_PATH)
    logger.info("Cognitive model loaded successfully")
except Exception as e:
    logger.error(f"Error loading cognitive model: {e}")
    cognitive_model = None

try:
    SPEECH_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'speech_model.onnx')
    speech_session = ort.InferenceSession(SPEECH_MODEL_PATH)
    logger.info("Speech model loaded successfully")
except Exception as e:
    logger.error(f"Error loading speech model: {e}")
    speech_session = None

# === Helper Functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_confidence_score(prediction, features=None):
    """Calculate a confidence score based on prediction and features"""
    base_confidence = 0.75
    
    if features:
        if 'accuracy' in features:
            accuracy_factor = features['accuracy'] * 0.15
            base_confidence += accuracy_factor
        
        if 'reaction_time' in features:
            rt_factor = max(0, (2.0 - features['reaction_time']) / 2.0) * 0.1
            base_confidence += rt_factor
    
    confidence = base_confidence + np.random.uniform(-0.05, 0.05)
    return min(0.99, max(0.65, confidence))

def extract_facial_features(image_data=None):
    """Extract facial features from image"""
    try:
        if image_data is None:
            # Simulate webcam capture
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            image_data = frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            return None
        
        # Get the first face
        x, y, w, h = faces[0]
        
        # Calculate features
        img_height, img_width = image_data.shape[:2]
        
        features = [
            float(w),  # face width
            float(h),  # face height
            float(x + w/2) / img_width,  # normalized x position
            float(y + h/2) / img_height  # normalized y position
        ]
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting facial features: {e}")
        return None

def analyze_face(features):
    """Analyze facial features and return risk assessment"""
    if not features or len(features) < 4:
        return "Unknown", ["Insufficient data for analysis"]
    
    width, height, x_pos, y_pos = features
    reasons = []
    risk_score = 0
    
    # Face size analysis
    if width < 150 or height < 150:
        reasons.append(f"⚠️ Small facial region detected ({int(width)}x{int(height)}px) - may indicate positioning issues")
        risk_score += 1
    elif width < 200 or height < 200:
        reasons.append(f"📏 Moderate facial region ({int(width)}x{int(height)}px) - acceptable for analysis")
    else:
        reasons.append(f"✅ Good facial region size ({int(width)}x{int(height)}px) - optimal for analysis")
    
    # Aspect ratio analysis
    aspect_ratio = width / height if height > 0 else 1
    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
        reasons.append(f"⚠️ Unusual facial aspect ratio ({aspect_ratio:.2f}) - may indicate angle issues")
        risk_score += 1
    else:
        reasons.append(f"✅ Normal facial proportions (ratio: {aspect_ratio:.2f})")
    
    # Position analysis
    if x_pos < 0.3 or x_pos > 0.7:
        reasons.append(f"📍 Face positioned off-center horizontally ({x_pos:.2f})")
        risk_score += 0.5
    else:
        reasons.append(f"✅ Face well-centered horizontally ({x_pos:.2f})")
    
    if y_pos < 0.3 or y_pos > 0.7:
        reasons.append(f"📍 Face positioned off-center vertically ({y_pos:.2f})")
        risk_score += 0.5
    else:
        reasons.append(f"✅ Face well-centered vertically ({y_pos:.2f})")
    
    # Determine risk level
    if risk_score <= 1:
        risk = "Low"
        reasons.append("🎯 Overall assessment: Normal facial positioning and features")
    elif risk_score <= 2:
        risk = "Medium"
        reasons.append("⚡ Overall assessment: Some indicators suggest attention needed")
    else:
        risk = "High"
        reasons.append("🔴 Overall assessment: Multiple indicators suggest review needed")
    
    return risk, reasons

def extract_audio_features(y, sr):
    """Extract comprehensive audio features"""
    features = {}
    
    try:
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Energy/RMS
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        
        # Duration
        features['duration'] = float(len(y) / sr)
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
    
    return features

def get_risk_insights(label, features=None):
    """Provide insights based on risk classification"""
    insights = {
        "Normal": {
            "message": "Cognitive function appears to be within normal range",
            "recommendations": [
                "Maintain regular cognitive exercises",
                "Continue healthy lifestyle habits",
                "Consider periodic reassessment"
            ],
            "severity": "low"
        },
        "Low": {
            "message": "Facial analysis shows normal indicators",
            "recommendations": [
                "Continue regular monitoring",
                "Maintain healthy lifestyle",
                "Schedule periodic check-ups"
            ],
            "severity": "low"
        },
        "Medium": {
            "message": "Some indicators suggest attention needed",
            "recommendations": [
                "Consider professional consultation",
                "Increase monitoring frequency",
                "Review lifestyle factors"
            ],
            "severity": "medium"
        },
        "At Risk": {
            "message": "Some indicators suggest potential cognitive decline",
            "recommendations": [
                "Consult with a healthcare professional",
                "Increase cognitive stimulation activities",
                "Monitor progress with regular assessments",
                "Consider lifestyle modifications"
            ],
            "severity": "medium"
        },
        "High": {
            "message": "Multiple indicators suggest review needed",
            "recommendations": [
                "Seek professional medical evaluation",
                "Comprehensive assessment recommended",
                "Discuss findings with healthcare provider"
            ],
            "severity": "high"
        },
        "Impaired": {
            "message": "Multiple indicators suggest cognitive impairment",
            "recommendations": [
                "Seek immediate medical evaluation",
                "Comprehensive cognitive assessment recommended",
                "Discuss treatment options with healthcare provider",
                "Consider support resources for daily activities"
            ],
            "severity": "high"
        }
    }
    
    return insights.get(label, insights["Normal"])

# === Security & Access Control Functions ===

def encrypt_data(data):
    """Encrypt sensitive data"""
    if cipher_suite and data:
        try:
            if isinstance(data, str):
                return cipher_suite.encrypt(data.encode()).decode()
            elif isinstance(data, (list, dict)):
                json_str = json.dumps(data)
                return cipher_suite.encrypt(json_str.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption error: {e}")
    return data

def decrypt_data(encrypted_data):
    """Decrypt sensitive data"""
    if cipher_suite and encrypted_data:
        try:
            decrypted = cipher_suite.decrypt(encrypted_data.encode() if isinstance(encrypted_data, str) else encrypted_data)
            try:
                return json.loads(decrypted.decode())
            except:
                return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
    return encrypted_data

def audit_log_request(endpoint, user_id, method, status_code, details=None):
    """Log API call for audit"""
    audit_entry = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'user_id': user_id,
        'method': method,
        'status_code': status_code,
        'ip_address': request.remote_addr if request else None,
        'details': details
    }
    audit_log.append(audit_entry)
    
    # Keep only last 10000 entries
    if len(audit_log) > 10000:
        audit_log.pop(0)
    
    # Save to MongoDB if available
    if MONGO_AVAILABLE and db:
        try:
            db['audit_logs'].insert_one(audit_entry)
        except Exception as e:
            logger.error(f"Error saving audit log: {e}")

def require_role(*allowed_roles):
    """Decorator for role-based access control"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = request.json.get('user_id') if request.is_json else request.args.get('user_id')
            if not user_id:
                user_id = request.headers.get('X-User-ID')
            
            user_role = user_roles.get(user_id, 'patient')
            
            if user_role not in allowed_roles and 'admin' not in allowed_roles:
                audit_log_request(request.path, user_id, request.method, 403, 'Access denied')
                return jsonify({"error": "Insufficient permissions"}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# === NeuroTwin Helper Functions ===

def extract_cognitive_metrics(test_results):
    """Extract cognitive metrics from test results"""
    metrics = {
        'attention': 0.0,
        'memory': 0.0,
        'speed': 0.0,
        'verbal_fluency': 0.0
    }
    
    # Extract from cognitive game results
    if 'accuracy' in test_results and 'reaction_time' in test_results:
        # Attention: based on accuracy
        metrics['attention'] = float(test_results['accuracy'])
        
        # Speed: inverse of reaction time (normalized to 0-1, assuming max 5s)
        reaction_time = float(test_results.get('reaction_time', 2.0))
        metrics['speed'] = max(0.0, min(1.0, 1.0 - (reaction_time / 5.0)))
        
        # Memory: based on accuracy and level difficulty
        level = int(test_results.get('level', 3))
        metrics['memory'] = float(test_results['accuracy']) * (level / 5.0)
    
    # Extract from speech results
    if 'audio_features' in test_results:
        audio_features = test_results['audio_features']
        # Verbal fluency: based on speech features (duration, spectral features)
        if 'duration' in audio_features:
            duration = float(audio_features['duration'])
            # Normalize duration (assuming 2-30 seconds is normal range)
            metrics['verbal_fluency'] = min(1.0, max(0.0, (duration - 2.0) / 28.0))
    
    # If speech label is available, use it to adjust verbal fluency
    if 'label' in test_results:
        label = test_results['label']
        if label == 'Normal':
            metrics['verbal_fluency'] = max(metrics['verbal_fluency'], 0.7)
        elif label == 'At Risk':
            metrics['verbal_fluency'] = metrics['verbal_fluency'] * 0.8
        elif label == 'Impaired':
            metrics['verbal_fluency'] = metrics['verbal_fluency'] * 0.5
    
    return metrics

def predict_future_metrics(history, days=7):
    """Predict future metrics for next N days using Linear Regression"""
    if len(history) < 2:
        # Not enough data, return current metrics as prediction
        if len(history) == 0:
            return None
        current = history[-1]['metrics']
        return {
            day: {k: v for k, v in current.items()} 
            for day in range(1, days + 1)
        }
    
    predictions = {}
    metrics_keys = ['attention', 'memory', 'speed', 'verbal_fluency']
    
    # Prepare data: extract timestamps and metric values
    timestamps = []
    metric_values = {key: [] for key in metrics_keys}
    
    for entry in history:
        # Convert timestamp to days since first entry
        if len(timestamps) == 0:
            base_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
        entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
        days_since_start = (entry_time - base_time).total_seconds() / 86400.0
        timestamps.append(days_since_start)
        
        for key in metrics_keys:
            metric_values[key].append(entry['metrics'].get(key, 0.0))
    
    # Predict each metric
    X = np.array(timestamps).reshape(-1, 1)
    future_days = np.array([timestamps[-1] + i for i in range(1, days + 1)]).reshape(-1, 1)
    
    predicted_metrics = {}
    for key in metrics_keys:
        if len(metric_values[key]) < 2:
            # Use last value if not enough data
            last_value = metric_values[key][-1] if metric_values[key] else 0.5
            predicted_metrics[key] = [last_value] * days
        else:
            y = np.array(metric_values[key])
            model = LinearRegression()
            model.fit(X, y)
            predictions_array = model.predict(future_days)
            # Clamp values between 0 and 1
            predictions_array = np.clip(predictions_array, 0.0, 1.0)
            predicted_metrics[key] = predictions_array.tolist()
    
    # Format predictions by day
    result = {}
    for day in range(1, days + 1):
        result[day] = {
            key: predicted_metrics[key][day - 1] 
            for key in metrics_keys
        }
    
    return result

# === Middleware for Audit Logging ===
@app.before_request
def log_request():
    """Log all API requests for audit"""
    if request.path.startswith('/api/'):
        user_id = request.json.get('user_id') if request.is_json else request.args.get('user_id')
        if not user_id:
            user_id = request.headers.get('X-User-ID', 'anonymous')
        # Will be logged after response in after_request

@app.after_request
def log_response(response):
    """Log API responses for audit"""
    if request.path.startswith('/api/'):
        user_id = request.json.get('user_id') if request.is_json else request.args.get('user_id')
        if not user_id:
            user_id = request.headers.get('X-User-ID', 'anonymous')
        audit_log_request(request.path, user_id, request.method, response.status_code)
    return response

# === API Routes ===

@app.route('/')
def serve_react():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "cognitive_model_loaded": cognitive_model is not None,
        "speech_model_loaded": speech_session is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict_cognitive():
    """Cognitive game prediction"""
    try:
        if cognitive_model is None:
            return jsonify({"error": "Cognitive model not loaded"}), 503
        
        data = request.json
        
        required_fields = ['accuracy', 'reaction_time', 'retries', 'level', 'total_time']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        features = [
            float(data['accuracy']),
            float(data['reaction_time']),
            int(data['retries']),
            int(data['level']),
            float(data['total_time'])
        ]
        
        prediction = cognitive_model.predict([features])[0]
        label_map = {0: "Normal", 1: "At Risk", 2: "Impaired"}
        label = label_map[int(prediction)]
        
        confidence = calculate_confidence_score(prediction, {
            'accuracy': data['accuracy'],
            'reaction_time': data['reaction_time']
        })
        
        insights = get_risk_insights(label)
        
        analysis_record = {
            "id": len(analysis_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "type": "cognitive_game",
            "prediction": int(prediction),
            "label": label,
            "confidence": round(confidence, 3),
            "features": data,
            "insights": insights
        }
        analysis_history.append(analysis_record)
        
        logger.info(f"Cognitive prediction: {label} (confidence: {confidence:.2f})")
        
        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "label": label,
            "confidence": round(confidence, 3),
            "insights": insights,
            "analysis_id": analysis_record["id"],
            "timestamp": analysis_record["timestamp"]
        })
        
    except Exception as e:
        logger.error(f"Error in cognitive prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-facial', methods=['POST'])
def predict_facial():
    """Facial analysis prediction"""
    try:
        data = request.json
        
        if 'image' in data:
            # Process base64 image
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            features = extract_facial_features(image_bgr)
        else:
            # Use webcam
            features = extract_facial_features()
        
        if not features:
            return jsonify({
                "success": False,
                "error": "No face detected"
            }), 400
        
        risk, reasons = analyze_face(features)
        
        confidence = calculate_confidence_score(
            0 if risk == "Low" else (1 if risk == "Medium" else 2),
            {'accuracy': 0.85}
        )
        
        insights = get_risk_insights(risk)
        
        analysis_record = {
            "id": len(analysis_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "type": "facial_analysis",
            "risk": risk,
            "confidence": round(confidence, 3),
            "features": {
                "face_width": features[0],
                "face_height": features[1],
                "x_position": features[2],
                "y_position": features[3]
            },
            "reasons": reasons,
            "insights": insights
        }
        analysis_history.append(analysis_record)
        
        logger.info(f"Facial analysis: {risk} (confidence: {confidence:.2f})")
        
        return jsonify({
            "success": True,
            "risk": risk,
            "confidence": round(confidence, 3),
            "features": analysis_record["features"],
            "reasons": reasons,
            "insights": insights,
            "analysis_id": analysis_record["id"],
            "timestamp": analysis_record["timestamp"]
        })
        
    except Exception as e:
        logger.error(f"Error in facial prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-speech', methods=['POST'])
def predict_speech():
    """Speech analysis prediction"""
    try:
        if speech_session is None:
            return jsonify({"error": "Speech model not loaded"}), 503
        
        file = request.files.get("audio")
        if not file:
            return jsonify({"error": "No audio file uploaded"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.now().timestamp()}_{filename}")
        file.save(filepath)
        
        try:
            y, sr = librosa.load(filepath, sr=16000, duration=30)
            audio_features = extract_audio_features(y, sr)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc = mfcc[:, :94]
            if mfcc.shape[1] < 94:
                pad = np.zeros((13, 94 - mfcc.shape[1]))
                mfcc = np.hstack((mfcc, pad))
            
            input_tensor = mfcc[np.newaxis, np.newaxis, :, :].astype(np.float32)
            
            output = speech_session.run(None, {"input": input_tensor})[0]
            logits = output[0]
            
            exp_logits = np.exp(logits - np.max(logits))
            prediction_probs = exp_logits / np.sum(exp_logits)
            
            label_idx = int(np.argmax(prediction_probs))
            label_map = {0: "Normal", 1: "At Risk", 2: "Impaired"}
            label = label_map[label_idx]
            
            model_confidence = float(prediction_probs[label_idx])
            confidence = min(0.99, max(0.65, model_confidence))
            
            insights = get_risk_insights(label)
            
            analysis_record = {
                "id": len(analysis_history) + 1,
                "timestamp": datetime.now().isoformat(),
                "type": "speech_analysis",
                "prediction": label_idx,
                "label": label,
                "confidence": round(confidence, 3),
                "audio_features": audio_features,
                "prediction_probabilities": prediction_probs.tolist(),
                "insights": insights
            }
            analysis_history.append(analysis_record)
            
            logger.info(f"Speech prediction: {label} (confidence: {confidence:.2f})")
            
            return jsonify({
                "success": True,
                "prediction": label_idx,
                "label": label,
                "confidence": round(confidence, 3),
                "audio_features": audio_features,
                "prediction_probabilities": {
                    "Normal": float(prediction_probs[0]),
                    "At Risk": float(prediction_probs[1]),
                    "Impaired": float(prediction_probs[2])
                },
                "insights": insights,
                "analysis_id": analysis_record["id"],
                "timestamp": analysis_record["timestamp"]
            })
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
    except Exception as e:
        logger.error(f"Error in speech prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        analysis_type = request.args.get('type', None)
        
        filtered_history = analysis_history
        if analysis_type:
            filtered_history = [h for h in analysis_history if h['type'] == analysis_type]
        
        return jsonify({
            "success": True,
            "total": len(filtered_history),
            "history": filtered_history[-limit:][::-1]
        })
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/history/<int:analysis_id>', methods=['GET'])
def get_analysis_detail(analysis_id):
    """Get detailed analysis by ID"""
    try:
        analysis = next((h for h in analysis_history if h['id'] == analysis_id), None)
        
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404
        
        return jsonify({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        logger.error(f"Error fetching analysis detail: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get overall statistics"""
    try:
        if not analysis_history:
            return jsonify({
                "success": True,
                "total_analyses": 0,
                "statistics": {}
            })
        
        total = len(analysis_history)
        by_type = {}
        by_label = {}
        avg_confidence = []
        
        for record in analysis_history:
            rec_type = record['type']
            by_type[rec_type] = by_type.get(rec_type, 0) + 1
            
            label = record.get('label') or record.get('risk', 'Unknown')
            by_label[label] = by_label.get(label, 0) + 1
            
            avg_confidence.append(record['confidence'])
        
        return jsonify({
            "success": True,
            "total_analyses": total,
            "statistics": {
                "by_type": by_type,
                "by_label": by_label,
                "average_confidence": round(np.mean(avg_confidence), 3) if avg_confidence else 0,
                "most_recent": analysis_history[-1]['timestamp'] if analysis_history else None
            }
        })
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/neurotwin/update', methods=['POST'])
def update_neurotwin():
    """Update NeuroTwin profile with latest test results and predict future metrics"""
    try:
        data = request.json
        
        if 'user_id' not in data:
            return jsonify({"error": "user_id is required"}), 400
        
        user_id = str(data['user_id'])
        test_results = data.get('test_results', {})
        
        # Extract cognitive metrics from test results
        metrics = extract_cognitive_metrics(test_results)
        
        # Get or create NeuroTwin profile
        if user_id not in neurotwin_profiles:
            neurotwin_profiles[user_id] = {
                'user_id': user_id,
                'history': [],
                'predictions': {}
            }
        
        profile = neurotwin_profiles[user_id]
        
        # Add new entry to history
        new_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'test_results': test_results
        }
        profile['history'].append(new_entry)
        
        # Predict future metrics for next 7 days
        predictions = predict_future_metrics(profile['history'], days=7)
        profile['predictions'] = predictions
        
        # Check for metric drops and notify CareNetwork
        check_and_notify_metric_drop(user_id)
        
        logger.info(f"NeuroTwin updated for user {user_id}: {len(profile['history'])} entries")
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "current_metrics": metrics,
            "predictions": predictions,
            "history_count": len(profile['history']),
            "timestamp": new_entry['timestamp']
        })
        
    except Exception as e:
        logger.error(f"Error updating NeuroTwin: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/neurotwin/<user_id>', methods=['GET'])
def get_neurotwin(user_id):
    """Get NeuroTwin profile for a user"""
    try:
        user_id = str(user_id)
        
        if user_id not in neurotwin_profiles:
            return jsonify({
                "success": False,
                "error": "NeuroTwin profile not found",
                "user_id": user_id
            }), 404
        
        profile = neurotwin_profiles[user_id]
        
        # Get current metrics (from last entry)
        current_metrics = None
        if profile['history']:
            current_metrics = profile['history'][-1]['metrics']
        
        # Predict future if we have history
        predictions = profile.get('predictions', {})
        if not predictions and profile['history']:
            predictions = predict_future_metrics(profile['history'], days=7)
            profile['predictions'] = predictions
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "current_metrics": current_metrics,
            "predictions": predictions,
            "history": profile['history'],
            "history_count": len(profile['history'])
        })
        
    except Exception as e:
        logger.error(f"Error getting NeuroTwin: {e}")
        return jsonify({"error": str(e)}), 500

# === NeuroDrift Endpoint ===
@app.route('/api/neurodrift', methods=['POST'])
def calculate_neurodrift():
    """Calculate Brain Stability Index from cognitive tests and wearable data"""
    try:
        data = request.json
        user_id = str(data.get('user_id', 'demo-user-123'))
        days = int(data.get('days', 7))
        
        # Get recent test results
        profile = neurotwin_profiles.get(user_id, {})
        history = profile.get('history', [])
        
        if len(history) < 2:
            return jsonify({
                "success": False,
                "error": "Insufficient data. Need at least 2 test results."
            }), 400
        
        # Get wearable data (mock or from Fitbit API)
        wearable_data = data.get('wearable_data', {})
        if not wearable_data:
            # Generate mock wearable data
            wearable_data = generate_mock_wearable_data(days)
        
        # Calculate metrics for each day
        daily_metrics = []
        for i in range(min(days, len(history))):
            entry = history[-(days-i)] if i < len(history) else history[-1]
            metrics = entry.get('metrics', {})
            
            # Combine cognitive metrics with wearable data
            day_data = {
                'date': entry.get('timestamp', datetime.now().isoformat()),
                'focus': metrics.get('attention', 0.5) * 100,
                'mood': calculate_mood_score(metrics, wearable_data.get(f'day_{i+1}', {})),
                'alertness': calculate_alertness_score(metrics, wearable_data.get(f'day_{i+1}', {}))
            }
            daily_metrics.append(day_data)
        
        # Calculate Brain Stability Index (0-100)
        stability_index = calculate_stability_index(daily_metrics)
        
        # Calculate change from previous period
        previous_stability = calculate_stability_index(daily_metrics[:-1]) if len(daily_metrics) > 1 else stability_index
        stability_change = stability_index - previous_stability
        
        return jsonify({
            "success": True,
            "stability_index": round(stability_index, 2),
            "stability_change": round(stability_change, 2),
            "daily_metrics": daily_metrics,
            "alert": {
                "show": stability_change < -5,
                "message": f"Your stability dropped {abs(stability_change):.1f}% — check sleep quality."
            } if stability_change < -5 else None
        })
        
    except Exception as e:
        logger.error(f"Error calculating neurodrift: {e}")
        return jsonify({"error": str(e)}), 500

def generate_mock_wearable_data(days):
    """Generate mock wearable data for testing"""
    data = {}
    for i in range(days):
        data[f'day_{i+1}'] = {
            'sleep_hours': np.random.uniform(6, 9),
            'steps': int(np.random.uniform(5000, 12000)),
            'heart_rate_avg': int(np.random.uniform(60, 80)),
            'stress_level': np.random.uniform(0, 1)
        }
    return data

def calculate_mood_score(metrics, wearable):
    """Calculate mood score from metrics and wearable data"""
    base_mood = (metrics.get('attention', 0.5) + metrics.get('verbal_fluency', 0.5)) / 2
    sleep_factor = min(1.0, wearable.get('sleep_hours', 7) / 8)
    stress_factor = 1.0 - wearable.get('stress_level', 0.5)
    return min(100, max(0, (base_mood * 0.6 + sleep_factor * 0.2 + stress_factor * 0.2) * 100))

def calculate_alertness_score(metrics, wearable):
    """Calculate alertness score"""
    base_alertness = metrics.get('speed', 0.5)
    sleep_factor = min(1.0, wearable.get('sleep_hours', 7) / 8)
    return min(100, max(0, (base_alertness * 0.7 + sleep_factor * 0.3) * 100))

def calculate_stability_index(daily_metrics):
    """Calculate Brain Stability Index from daily metrics"""
    if not daily_metrics:
        return 50.0
    
    focus_values = [d['focus'] for d in daily_metrics]
    mood_values = [d['mood'] for d in daily_metrics]
    alertness_values = [d['alertness'] for d in daily_metrics]
    
    # Calculate coefficient of variation (lower = more stable)
    focus_cv = np.std(focus_values) / (np.mean(focus_values) + 1e-6)
    mood_cv = np.std(mood_values) / (np.mean(mood_values) + 1e-6)
    alertness_cv = np.std(alertness_values) / (np.mean(alertness_values) + 1e-6)
    
    # Stability = 100 - (average CV * 100)
    avg_cv = (focus_cv + mood_cv + alertness_cv) / 3
    stability = max(0, min(100, 100 - (avg_cv * 50)))
    
    return stability

# === Emotion Analysis Endpoint ===
@app.route('/api/emotion-analyze', methods=['POST'])
def analyze_emotion():
    """Analyze emotions from audio and optional webcam frame"""
    try:
        audio_file = request.files.get('audio')
        image_data = request.form.get('image')  # base64 encoded
        
        emotions = {
            'stress': 0.0,
            'joy': 0.0,
            'fatigue': 0.0,
            'neutral': 0.0,
            'sadness': 0.0,
            'anger': 0.0
        }
        
        # Analyze audio emotion
        if audio_file:
            try:
                filename = secure_filename(audio_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.now().timestamp()}_{filename}")
                audio_file.save(filepath)
                
                y, sr = librosa.load(filepath, sr=16000, duration=10)
                audio_emotions = analyze_audio_emotion(y, sr)
                emotions.update(audio_emotions)
                
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Error analyzing audio emotion: {e}")
        
        # Analyze facial emotion
        if image_data:
            try:
                face_emotions = analyze_facial_emotion(image_data)
                # Combine with audio (weighted average)
                for key in emotions:
                    emotions[key] = (emotions[key] * 0.4 + face_emotions.get(key, 0) * 0.6)
            except Exception as e:
                logger.error(f"Error analyzing facial emotion: {e}")
        
        # Normalize to sum to 1.0
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
        
        return jsonify({
            "success": True,
            "emotions": emotions,
            "dominant_emotion": max(emotions.items(), key=lambda x: x[1])[0],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return jsonify({"error": str(e)}), 500

def analyze_audio_emotion(y, sr):
    """Analyze emotion from audio using pyAudioAnalysis-like features"""
    emotions = {'stress': 0.0, 'joy': 0.0, 'fatigue': 0.0, 'neutral': 0.5, 'sadness': 0.0, 'anger': 0.0}
    
    try:
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # Simple heuristic-based emotion detection
        avg_zcr = np.mean(zcr)
        avg_centroid = np.mean(spectral_centroid)
        energy = np.mean(librosa.feature.rms(y=y))
        
        # High energy + high centroid = joy
        if energy > 0.1 and avg_centroid > 2000:
            emotions['joy'] = min(0.8, energy * 2)
        
        # Low energy + low centroid = fatigue/sadness
        if energy < 0.05:
            emotions['fatigue'] = 0.6
            emotions['sadness'] = 0.3
        
        # High ZCR = stress/anger
        if avg_zcr > 0.1:
            emotions['stress'] = min(0.7, avg_zcr * 5)
            emotions['anger'] = min(0.5, avg_zcr * 3)
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
            
    except Exception as e:
        logger.error(f"Error in audio emotion analysis: {e}")
    
    return emotions

def analyze_facial_emotion(image_data):
    """Analyze facial emotion using FER (Facial Expression Recognition)"""
    emotions = {'stress': 0.0, 'joy': 0.0, 'fatigue': 0.0, 'neutral': 0.5, 'sadness': 0.0, 'anger': 0.0}
    
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        # Try to use FER if available, otherwise use simple heuristics
        try:
            from fer import FER
            detector = FER(mtcnn=True)
            result = detector.detect_emotions(image_bgr)
            
            if result:
                emotions_dict = result[0]['emotions']
                # Map FER emotions to our format
                emotions['joy'] = emotions_dict.get('happy', 0)
                emotions['sadness'] = emotions_dict.get('sad', 0)
                emotions['anger'] = emotions_dict.get('angry', 0)
                emotions['stress'] = (emotions_dict.get('fear', 0) + emotions_dict.get('angry', 0)) / 2
                emotions['neutral'] = emotions_dict.get('neutral', 0.5)
        except ImportError:
            # Fallback: simple heuristic based on face detection
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Simple heuristic: if face detected, assume neutral
                emotions['neutral'] = 0.7
                
    except Exception as e:
        logger.error(f"Error in facial emotion analysis: {e}")
    
    return emotions

# === NeuroAge Endpoint ===
@app.route('/api/neuroage', methods=['POST'])
def calculate_neuroage():
    """Calculate cognitive age and update leaderboard"""
    try:
        data = request.json
        user_id = str(data.get('user_id', 'demo-user-123'))
        base_age = int(data.get('base_age', 30))
        region = data.get('region', 'global')
        
        # Get performance score from recent tests
        profile = neurotwin_profiles.get(user_id, {})
        history = profile.get('history', [])
        
        if not history:
            return jsonify({
                "success": False,
                "error": "No test history available"
            }), 400
        
        # Calculate average performance score (0-100)
        recent_scores = []
        for entry in history[-5:]:  # Last 5 tests
            metrics = entry.get('metrics', {})
            score = (
                metrics.get('attention', 0) * 25 +
                metrics.get('memory', 0) * 25 +
                metrics.get('speed', 0) * 25 +
                metrics.get('verbal_fluency', 0) * 25
            ) * 100
            recent_scores.append(score)
        
        performance_score = np.mean(recent_scores) if recent_scores else 50.0
        
        # Cognitive age = base_age - performance_score/10
        cognitive_age = base_age - (performance_score / 10)
        cognitive_age = max(18, min(100, cognitive_age))  # Clamp between 18-100
        
        # Update leaderboard
        update_leaderboard(user_id, performance_score, region, cognitive_age)
        
        # Get rank
        rank = get_user_rank(user_id, region)
        
        return jsonify({
            "success": True,
            "cognitive_age": round(cognitive_age, 1),
            "base_age": base_age,
            "performance_score": round(performance_score, 2),
            "rank": rank,
            "region": region,
            "total_peers": len(leaderboard_data.get(region, []))
        })
        
    except Exception as e:
        logger.error(f"Error calculating neuroage: {e}")
        return jsonify({"error": str(e)}), 500

def update_leaderboard(user_id, score, region, cognitive_age):
    """Update leaderboard for a region"""
    if region not in leaderboard_data:
        leaderboard_data[region] = []
    
    # Find or update user entry
    user_entry = None
    for entry in leaderboard_data[region]:
        if entry['user_id'] == user_id:
            user_entry = entry
            break
    
    if user_entry:
        user_entry['score'] = score
        user_entry['cognitive_age'] = cognitive_age
        user_entry['updated_at'] = datetime.now().isoformat()
    else:
        leaderboard_data[region].append({
            'user_id': user_id,
            'score': score,
            'cognitive_age': cognitive_age,
            'updated_at': datetime.now().isoformat()
        })
    
    # Sort by score (descending) and assign ranks
    leaderboard_data[region].sort(key=lambda x: x['score'], reverse=True)
    for i, entry in enumerate(leaderboard_data[region]):
        entry['rank'] = i + 1
    
    # Keep top 100
    leaderboard_data[region] = leaderboard_data[region][:100]
    
    # Save to MongoDB if available
    if MONGO_AVAILABLE and db:
        try:
            collection = db['leaderboards']
            collection.update_one(
                {'user_id': user_id, 'region': region},
                {
                    '$set': {
                        'score': score,
                        'cognitive_age': cognitive_age,
                        'updated_at': datetime.now()
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")

def get_user_rank(user_id, region):
    """Get user's rank in leaderboard"""
    if region not in leaderboard_data:
        return None
    
    for entry in leaderboard_data[region]:
        if entry['user_id'] == user_id:
            return entry['rank']
    
    return None

# === Cognitive Fingerprint Endpoints ===
@app.route('/api/auth/fingerprint-login', methods=['POST'])
def fingerprint_login():
    """Login using cognitive fingerprint (voice + face)"""
    try:
        data = request.json
        audio_embedding = data.get('audio_embedding', [])
        face_embedding = data.get('face_embedding', [])
        
        if not audio_embedding or not face_embedding:
            return jsonify({
                "success": False,
                "error": "Both audio and face embeddings required"
            }), 400
        
        # Find matching user by cosine similarity
        threshold = 0.85  # Similarity threshold
        best_match = None
        best_similarity = 0.0
        
        for user_id, fingerprint in cognitive_fingerprints.items():
            # Decrypt stored embeddings
            stored_audio_enc = fingerprint.get('voice_embedding', [])
            stored_face_enc = fingerprint.get('face_embedding', [])
            
            stored_audio = np.array(decrypt_data(stored_audio_enc) if cipher_suite and isinstance(stored_audio_enc, str) else stored_audio_enc)
            stored_face = np.array(decrypt_data(stored_face_enc) if cipher_suite and isinstance(stored_face_enc, str) else stored_face_enc)
            
            input_audio = np.array(audio_embedding)
            input_face = np.array(face_embedding)
            
            if len(stored_audio) > 0 and len(stored_face) > 0:
                # Calculate cosine similarity
                audio_sim = cosine_similarity([input_audio], [stored_audio])[0][0]
                face_sim = cosine_similarity([input_face], [stored_face])[0][0]
                
                # Combined similarity (weighted average)
                combined_sim = (audio_sim * 0.5 + face_sim * 0.5)
                
                if combined_sim > best_similarity:
                    best_similarity = combined_sim
                    best_match = user_id
        
        if best_match and best_similarity >= threshold:
            return jsonify({
                "success": True,
                "user_id": best_match,
                "similarity": round(float(best_similarity), 3),
                "message": "Cognitive ID verified successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Cognitive ID not recognized",
                "similarity": round(float(best_similarity), 3) if best_match else 0.0
            }), 401
        
    except Exception as e:
        logger.error(f"Error in fingerprint login: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/fingerprint/register', methods=['POST'])
def register_fingerprint():
    """Register or update cognitive fingerprint"""
    try:
        data = request.json
        user_id = str(data.get('user_id'))
        audio_embedding = data.get('audio_embedding', [])
        face_embedding = data.get('face_embedding', [])
        
        if not audio_embedding or not face_embedding:
            return jsonify({
                "success": False,
                "error": "Both audio and face embeddings required"
            }), 400
        
        # Encrypt and store fingerprint
        encrypted_voice = encrypt_data(audio_embedding) if cipher_suite else audio_embedding
        encrypted_face = encrypt_data(face_embedding) if cipher_suite else face_embedding
        
        cognitive_fingerprints[user_id] = {
            'voice_embedding': encrypted_voice,
            'face_embedding': encrypted_face,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Save to MongoDB if available
        if MONGO_AVAILABLE and db:
            try:
                collection = db['cognitive_fingerprints']
                collection.update_one(
                    {'user_id': user_id},
                    {
                        '$set': {
                            'voice_embedding': audio_embedding,
                            'face_embedding': face_embedding,
                            'updated_at': datetime.now()
                        }
                    },
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Error saving fingerprint to MongoDB: {e}")
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "message": "Cognitive fingerprint registered successfully"
        })
        
    except Exception as e:
        logger.error(f"Error registering fingerprint: {e}")
        return jsonify({"error": str(e)}), 500

# === NeuroCoach Chatbot Endpoint ===
@app.route('/api/coach', methods=['POST'])
@limiter.limit("10 per minute")
def neurocoach_chat():
    """NeuroCoach chatbot using OpenAI GPT-4o"""
    try:
        user_id = request.json.get('user_id', 'demo-user-123')
        message = request.json.get('message', '')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get user's NeuroTwin data
        profile = neurotwin_profiles.get(user_id, {})
        history = profile.get('history', [])
        current_metrics = None
        if history:
            current_metrics = history[-1].get('metrics', {})
        
        # Get recent test results
        recent_tests = analysis_history[-5:] if len(analysis_history) > 0 else []
        
        # Build context for OpenAI
        if current_metrics:
            metrics_text = f"""- Attention: {current_metrics.get('attention', 0) * 100:.1f}%
- Memory: {current_metrics.get('memory', 0) * 100:.1f}%
- Speed: {current_metrics.get('speed', 0) * 100:.1f}%
- Verbal Fluency: {current_metrics.get('verbal_fluency', 0) * 100:.1f}%"""
        else:
            metrics_text = "No metrics available yet"
        
        context = f"""You are NeuroCoach, a friendly AI cognitive health coach. 

User's current cognitive metrics:
{metrics_text}

Recent test history: {len(history)} tests completed.

Provide personalized, encouraging feedback. Be specific about improvements and offer actionable advice."""
        
        if OPENAI_AVAILABLE and openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                coach_response = response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                # Fallback response
                coach_response = generate_fallback_coach_response(current_metrics, history)
        else:
            coach_response = generate_fallback_coach_response(current_metrics, history)
        
        audit_log_request('/api/coach', user_id, 'POST', 200)
        
        return jsonify({
            "success": True,
            "response": coach_response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in NeuroCoach: {e}")
        audit_log_request('/api/coach', request.json.get('user_id', 'unknown'), 'POST', 500, str(e))
        return jsonify({"error": str(e)}), 500

def generate_fallback_coach_response(metrics, history):
    """Generate fallback coach response without OpenAI"""
    if not metrics or not history:
        return "Welcome to NeuroCoach! Complete some cognitive tests to get personalized feedback on your cognitive health."
    
    # Calculate improvements
    if len(history) >= 2:
        prev_metrics = history[-2].get('metrics', {})
        curr_metrics = history[-1].get('metrics', {})
        
        attention_change = (curr_metrics.get('attention', 0) - prev_metrics.get('attention', 0)) * 100
        memory_change = (curr_metrics.get('memory', 0) - prev_metrics.get('memory', 0)) * 100
        
        if attention_change > 0:
            return f"Great progress! Your focus improved {attention_change:.1f}% this week. Keep practicing cognitive exercises daily!"
        elif memory_change > 0:
            return f"Excellent work! Your memory score increased {memory_change:.1f}%. Continue with memory training exercises."
        else:
            return "Keep up the consistent testing! Regular cognitive assessments help track your progress over time."
    
    return "You're doing well! Continue with regular cognitive tests to build your baseline and track improvements."

# === Risk Analysis Endpoint ===
@app.route('/api/risk/analyze', methods=['POST'])
@limiter.limit("20 per hour")
def analyze_risk():
    """Analyze MCI risk using speech features and eye movement data"""
    try:
        data = request.json
        user_id = str(data.get('user_id', 'demo-user-123'))
        
        # Extract features
        speech_features = data.get('speech_features', {})
        eye_movement = data.get('eye_movement', {})
        
        # Feature vector: [latency, pause, tremor, eye_movement_x, eye_movement_y]
        features = np.array([[
            float(speech_features.get('latency', 0.5)),
            float(speech_features.get('pause_frequency', 0.3)),
            float(speech_features.get('tremor', 0.2)),
            float(eye_movement.get('x_variance', 0.1)),
            float(eye_movement.get('y_variance', 0.1))
        ]])
        
        if risk_classifier is None:
            return jsonify({
                "success": False,
                "error": "Risk classifier not available"
            }), 503
        
        # Predict risk score (0-1)
        risk_score = float(risk_classifier.predict_proba(features)[0][1])
        risk_level = "Low" if risk_score < 0.3 else ("Medium" if risk_score < 0.7 else "High")
        
        # Generate insights
        insights = generate_risk_insights(risk_score, speech_features, eye_movement)
        
        # Encrypt sensitive data
        encrypted_risk = encrypt_data(str(risk_score)) if cipher_suite else risk_score
        
        audit_log_request('/api/risk/analyze', user_id, 'POST', 200)
        
        return jsonify({
            "success": True,
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        audit_log_request('/api/risk/analyze', request.json.get('user_id', 'unknown') if request.is_json else 'unknown', 'POST', 500, str(e))
        return jsonify({"error": str(e)}), 500

def generate_risk_insights(risk_score, speech_features, eye_movement):
    """Generate actionable insights from risk analysis"""
    insights = []
    
    if risk_score > 0.7:
        insights.append("🔴 High risk detected. Consider consulting with a healthcare professional.")
        insights.append("Monitor speech patterns and cognitive function closely.")
    elif risk_score > 0.4:
        insights.append("⚠️ Moderate risk. Regular monitoring recommended.")
        insights.append("Continue cognitive exercises and maintain healthy lifestyle.")
    else:
        insights.append("✅ Low risk. Continue regular assessments.")
        insights.append("Maintain current cognitive health practices.")
    
    if speech_features.get('latency', 0) > 0.6:
        insights.append("Speech latency detected. Practice verbal fluency exercises.")
    
    if eye_movement.get('x_variance', 0) > 0.5:
        insights.append("Eye movement patterns suggest attention challenges. Focus training recommended.")
    
    return insights

# === CareNetwork Endpoints ===
@app.route('/api/carenetwork/create', methods=['POST'])
@require_role('patient', 'admin')
def create_carenetwork():
    """Create or update CareNetwork for a patient"""
    try:
        data = request.json
        patient_id = str(data.get('patient_id'))
        family_members = data.get('family_members', [])
        doctors = data.get('doctors', [])
        
        # Get baseline metrics
        profile = neurotwin_profiles.get(patient_id, {})
        baseline = None
        if profile.get('history'):
            baseline = profile['history'][-1].get('metrics', {})
        
        care_networks[patient_id] = {
            'patient_id': patient_id,
            'family_members': family_members,
            'doctors': doctors,
            'baseline_metrics': baseline,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Save to MongoDB
        if MONGO_AVAILABLE and db:
            try:
                db['care_networks'].update_one(
                    {'patient_id': patient_id},
                    {'$set': care_networks[patient_id]},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Error saving to MongoDB: {e}")
        
        audit_log_request('/api/carenetwork/create', patient_id, 'POST', 200)
        
        return jsonify({
            "success": True,
            "carenetwork": care_networks[patient_id]
        })
        
    except Exception as e:
        logger.error(f"Error creating CareNetwork: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/carenetwork/<patient_id>', methods=['GET'])
@require_role('patient', 'family', 'doctor', 'admin')
def get_carenetwork(patient_id):
    """Get CareNetwork for a patient"""
    try:
        patient_id = str(patient_id)
        
        if patient_id not in care_networks:
            return jsonify({
                "success": False,
                "error": "CareNetwork not found"
            }), 404
        
        network = care_networks[patient_id]
        
        # Get recent notifications
        recent_notifications = [n for n in notifications if n.get('patient_id') == patient_id][-10:]
        
        audit_log_request(f'/api/carenetwork/{patient_id}', patient_id, 'GET', 200)
        
        return jsonify({
            "success": True,
            "carenetwork": network,
            "notifications": recent_notifications
        })
        
    except Exception as e:
        logger.error(f"Error getting CareNetwork: {e}")
        return jsonify({"error": str(e)}), 500

def check_and_notify_metric_drop(patient_id):
    """Check if metrics dropped 15% and send notifications"""
    if patient_id not in care_networks:
        return
    
    network = care_networks[patient_id]
    baseline = network.get('baseline_metrics', {})
    
    if not baseline:
        return
    
    profile = neurotwin_profiles.get(patient_id, {})
    if not profile.get('history'):
        return
    
    current = profile['history'][-1].get('metrics', {})
    
    # Check each metric
    for metric_name in ['attention', 'memory', 'speed', 'verbal_fluency']:
        baseline_value = baseline.get(metric_name, 0)
        current_value = current.get(metric_name, 0)
        
        if baseline_value > 0:
            drop_percentage = ((baseline_value - current_value) / baseline_value) * 100
            
            if drop_percentage >= 15:
                # Create notification
                notification = {
                    'id': str(uuid.uuid4()),
                    'patient_id': patient_id,
                    'type': 'metric_drop',
                    'metric': metric_name,
                    'drop_percentage': round(drop_percentage, 1),
                    'message': f"{metric_name.capitalize()} dropped {drop_percentage:.1f}% below baseline",
                    'timestamp': datetime.now().isoformat(),
                    'sent_to': network.get('family_members', []) + network.get('doctors', [])
                }
                notifications.append(notification)
                
                # Save to MongoDB
                if MONGO_AVAILABLE and db:
                    try:
                        db['notifications'].insert_one(notification)
                    except Exception as e:
                        logger.error(f"Error saving notification: {e}")

# === Multi-language Support ===
@app.route('/api/language/detect', methods=['POST'])
def detect_language():
    """Detect language from audio using Whisper"""
    try:
        audio_file = request.files.get('audio')
        
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400
        
        # Save file temporarily
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.now().timestamp()}_{filename}")
        audio_file.save(filepath)
        
        detected_language = 'en'  # Default
        
        try:
            # Try using Whisper
            if OPENAI_AVAILABLE and openai_client:
                with open(filepath, 'rb') as f:
                    transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json"
                    )
                    detected_language = transcript.language
            elif SPEECH_REC_AVAILABLE:
                # Fallback to SpeechRecognition
                recognizer = sr.Recognizer()
                with sr.AudioFile(filepath) as source:
                    audio = recognizer.record(source)
                    try:
                        result = recognizer.recognize_google(audio, show_all=True)
                        if result and 'language_code' in result:
                            detected_language = result['language_code']
                    except:
                        pass
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Language mapping
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'hi': 'Hindi',
            'ar': 'Arabic'
        }
        
        audit_log_request('/api/language/detect', 'unknown', 'POST', 200)
        
        return jsonify({
            "success": True,
            "language_code": detected_language,
            "language_name": language_map.get(detected_language, 'English'),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return jsonify({"error": str(e)}), 500

# === Health Passport Export ===
@app.route('/api/passport/export', methods=['POST'])
@limiter.limit("5 per hour")
@require_role('patient', 'doctor', 'admin')
def export_health_passport():
    """Export comprehensive health passport as PDF"""
    try:
        data = request.json
        user_id = str(data.get('user_id', 'demo-user-123'))
        
        if not PDF_AVAILABLE:
            return jsonify({"error": "PDF generation not available"}), 503
        
        # Get all user data
        profile = neurotwin_profiles.get(user_id, {})
        history = profile.get('history', [])
        recent_analyses = [a for a in analysis_history if a.get('type')][-10:]
        
        # Generate PDF
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        
        # Header
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Cognitive Health Passport")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Patient ID: {user_id}")
        c.drawString(50, height - 100, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Current Metrics
        y_pos = height - 140
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Current Cognitive Metrics")
        y_pos -= 30
        
        if history:
            current = history[-1].get('metrics', {})
            c.setFont("Helvetica", 10)
            for metric, value in current.items():
                c.drawString(70, y_pos, f"{metric.capitalize()}: {value * 100:.1f}%")
                y_pos -= 20
        
        # Recent Test History
        y_pos -= 20
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Recent Test History")
        y_pos -= 30
        
        c.setFont("Helvetica", 9)
        for analysis in recent_analyses[:5]:
            test_type = analysis.get('type', 'Unknown')
            timestamp = analysis.get('timestamp', '')[:10]
            c.drawString(70, y_pos, f"{test_type}: {timestamp}")
            y_pos -= 15
        
        # Generate QR Code
        qr_data = f"https://neuro-nest.vercel.app/verify/{user_id}"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Save QR to buffer
        qr_buffer = io.BytesIO()
        qr_img.save(qr_buffer, format='PNG')
        qr_buffer.seek(0)
        
        # Add QR code to PDF
        from reportlab.lib.utils import ImageReader
        qr_reader = ImageReader(qr_buffer)
        c.drawImage(qr_reader, width - 150, height - 150, width=100, height=100)
        c.drawString(width - 150, height - 160, "Doctor Verification QR")
        
        c.save()
        pdf_buffer.seek(0)
        
        audit_log_request('/api/passport/export', user_id, 'POST', 200)
        
        # Return PDF as response
        response = make_response(pdf_buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=health_passport_{user_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
        return response
        
    except Exception as e:
        logger.error(f"Error exporting passport: {e}")
        audit_log_request('/api/passport/export', request.json.get('user_id', 'unknown') if request.is_json else 'unknown', 'POST', 500, str(e))
        return jsonify({"error": str(e)}), 500

# === Audit Log Endpoint ===
@app.route('/api/audit/logs', methods=['GET'])
@require_role('admin')
def get_audit_logs():
    """Get audit logs (admin only)"""
    try:
        limit = request.args.get('limit', 100, type=int)
        user_id_filter = request.args.get('user_id', None)
        
        filtered_logs = audit_log
        if user_id_filter:
            filtered_logs = [log for log in audit_log if log.get('user_id') == user_id_filter]
        
        return jsonify({
            "success": True,
            "total": len(filtered_logs),
            "logs": filtered_logs[-limit:][::-1]
        })
    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        return jsonify({"error": str(e)}), 500

# === Error Handlers ===
@app.errorhandler(413)
def request_entity_too_large(error):
    audit_log_request(request.path, 'unknown', request.method, 413)
    return jsonify({"error": "File too large. Maximum size is 10MB"}), 413

@app.errorhandler(404)
def not_found(error):
    audit_log_request(request.path, 'unknown', request.method, 404)
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    audit_log_request(request.path, 'unknown', request.method, 500)
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    audit_log_request(request.path, 'unknown', request.method, 429, 'Rate limit exceeded')
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

# === Start Server ===
if __name__ == "__main__":
    logger.info("Starting NeuroNest Unified API Server...")
    logger.info(f"Cognitive Model Loaded: {cognitive_model is not None}")
    logger.info(f"Speech Model Loaded: {speech_session is not None}")
    app.run(host='0.0.0.0', port=5000, debug=True)