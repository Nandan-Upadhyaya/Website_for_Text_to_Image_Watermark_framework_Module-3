import sys
import os
from pathlib import Path
import importlib.util

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

# Explicitly load models.py from the server directory
server_dir = Path(__file__).parent
models_path = server_dir / 'models.py'
firebase_config_path = server_dir / 'firebase_config.py'

# Load models module explicitly
spec = importlib.util.spec_from_file_location("server_models", models_path)
server_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_models)

# Load firebase_config module explicitly
spec_firebase = importlib.util.spec_from_file_location("server_firebase_config", firebase_config_path)
firebase_config_module = importlib.util.module_from_spec(spec_firebase)
spec_firebase.loader.exec_module(firebase_config_module)

# Extract what we need
User = server_models.User
Session = server_models.Session
verify_firebase_token = firebase_config_module.verify_firebase_token

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['name', 'email', 'password']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'message': f'{field} is required'}), 400
    
    # Create a session
    session = Session()
    
    try:
        # Check if user already exists
        existing_user = session.query(User).filter_by(email=data['email']).first()
        if existing_user:
            return jsonify({'message': 'Email already registered'}), 400
        
        # Create new user
        password_hash = generate_password_hash(data['password'])
        new_user = User(
            name=data['name'],
            email=data['email'],
            password_hash=password_hash
        )
        
        session.add(new_user)
        session.commit()
        
        # Generate JWT token
        token = new_user.generate_token()
        
        # Return user info and token
        return jsonify({
            'message': 'User created successfully',
            'user': new_user.to_dict(),
            'token': token
        }), 201
        
    except Exception as e:
        session.rollback()
        return jsonify({'message': f'Error creating user: {str(e)}'}), 500
    finally:
        session.close()

@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Validate required fields
    if not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Email and password are required'}), 400
    
    # Create a session
    session = Session()
    
    try:
        # Find user by email
        user = session.query(User).filter_by(email=data['email']).first()
        
        # Check if user exists and password is correct
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'message': 'Invalid email or password'}), 401
        
        # Generate JWT token
        token = user.generate_token()
        
        # Return user info and token
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'token': token
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error during login: {str(e)}'}), 500
    finally:
        session.close()

@auth_bp.route('/api/auth/firebase-login', methods=['POST'])
def firebase_login():
    data = request.get_json()
    
    # Validate required fields
    if not data.get('firebaseToken'):
        return jsonify({'message': 'Firebase token is required'}), 400
    
    # Verify Firebase token
    decoded_token = verify_firebase_token(data['firebaseToken'])
    if not decoded_token:
        return jsonify({'message': 'Invalid Firebase token'}), 401
    
    # Create a session
    session = Session()
    
    try:
        # Firebase tokens use 'sub' (subject) for user ID, not 'uid'
        firebase_uid = decoded_token.get('sub') or decoded_token.get('user_id')
        email = decoded_token.get('email')
        name = decoded_token.get('name') or data.get('user', {}).get('displayName', '')
        
        if not email:
            return jsonify({'message': 'Email is required from Firebase'}), 400
        
        if not firebase_uid:
            return jsonify({'message': 'User ID is required from Firebase'}), 400
        
        # Check if user exists with this email
        existing_user = session.query(User).filter_by(email=email).first()
        
        if existing_user:
            # Update existing user's Firebase UID if needed
            if not existing_user.firebase_uid:
                existing_user.firebase_uid = firebase_uid
                session.commit()
            user = existing_user
        else:
            # Create new user from Firebase data
            new_user = User(
                name=name or email.split('@')[0],
                email=email,
                firebase_uid=firebase_uid,
                password_hash=None  # No password needed for Firebase users
            )
            
            session.add(new_user)
            session.commit()
            user = new_user
        
        # Generate JWT token for backend authentication
        token = user.generate_token()
        
        # Return user info and token
        return jsonify({
            'message': 'Firebase login successful',
            'user': user.to_dict(),
            'token': token
        }), 200
        
    except Exception as e:
        session.rollback()
        return jsonify({'message': f'Error during Firebase login: {str(e)}'}), 500
    finally:
        session.close()

@auth_bp.route('/api/auth/verify', methods=['GET'])
def verify_token():
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'message': 'Authorization header missing or invalid'}), 401
    
    token = auth_header.split(' ')[1]
    
    # Verify token
    user_id = User.verify_token(token)
    if not user_id:
        return jsonify({'message': 'Invalid or expired token'}), 401
    
    # Create a session
    session = Session()
    
    try:
        # Find user by id
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Return user info
        return jsonify({
            'message': 'Token is valid',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error verifying token: {str(e)}'}), 500
    finally:
        session.close()

@auth_bp.route('/api/auth/user', methods=['GET'])
def get_user():
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'message': 'Authorization header missing or invalid'}), 401
    
    token = auth_header.split(' ')[1]
    
    # Verify token
    user_id = User.verify_token(token)
    if not user_id:
        return jsonify({'message': 'Invalid or expired token'}), 401
    
    # Create a session
    session = Session()
    
    try:
        # Find user by id
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Return user info
        return jsonify({
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error retrieving user: {str(e)}'}), 500
    finally:
        session.close()
