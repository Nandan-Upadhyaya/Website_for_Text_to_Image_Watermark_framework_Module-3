import os
import importlib.util
from flask import Blueprint, request, jsonify, send_file

# Load models using importlib to avoid conflicts with DF-GAN
models_path = os.path.join(os.path.dirname(__file__), 'models.py')
spec = importlib.util.spec_from_file_location("server.models", models_path)
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)

User = models_module.User
GeneratedImage = models_module.GeneratedImage
EvaluatedImage = models_module.EvaluatedImage
WatermarkedImage = models_module.WatermarkedImage
Session = models_module.Session

gallery_bp = Blueprint('gallery', __name__)

def get_user_from_token():
    """Extract user from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    user_id = User.verify_token(token)
    
    if not user_id:
        return None
    
    session = Session()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        return user
    finally:
        session.close()

@gallery_bp.route('/api/gallery/generated', methods=['GET'])
def get_generated_images():
    """Get user's generated images history"""
    user = get_user_from_token()
    
    if not user:
        return jsonify({'message': 'Unauthorized'}), 401
    
    session = Session()
    try:
        # Get query parameters for pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Query generated images
        query = session.query(GeneratedImage).filter_by(user_id=user.id).order_by(GeneratedImage.created_at.desc())
        
        # Calculate total
        total = query.count()
        
        # Paginate
        offset = (page - 1) * per_page
        images = query.limit(per_page).offset(offset).all()
        
        return jsonify({
            'images': [img.to_dict() for img in images],
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching generated images: {str(e)}'}), 500
    finally:
        session.close()

@gallery_bp.route('/api/gallery/evaluated', methods=['GET'])
def get_evaluated_images():
    """Get user's evaluated images history"""
    user = get_user_from_token()
    
    if not user:
        return jsonify({'message': 'Unauthorized'}), 401
    
    session = Session()
    try:
        # Get query parameters for pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Query evaluated images
        query = session.query(EvaluatedImage).filter_by(user_id=user.id).order_by(EvaluatedImage.created_at.desc())
        
        # Calculate total
        total = query.count()
        
        # Paginate
        offset = (page - 1) * per_page
        images = query.limit(per_page).offset(offset).all()
        
        return jsonify({
            'images': [img.to_dict() for img in images],
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching evaluated images: {str(e)}'}), 500
    finally:
        session.close()

@gallery_bp.route('/api/gallery/watermarked', methods=['GET'])
def get_watermarked_images():
    """Get user's watermarked images history"""
    user = get_user_from_token()
    
    if not user:
        return jsonify({'message': 'Unauthorized'}), 401
    
    session = Session()
    try:
        # Get query parameters for pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Query watermarked images
        query = session.query(WatermarkedImage).filter_by(user_id=user.id).order_by(WatermarkedImage.created_at.desc())
        
        # Calculate total
        total = query.count()
        
        # Paginate
        offset = (page - 1) * per_page
        images = query.limit(per_page).offset(offset).all()
        
        return jsonify({
            'images': [img.to_dict() for img in images],
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching watermarked images: {str(e)}'}), 500
    finally:
        session.close()

@gallery_bp.route('/api/gallery/image/<path:filename>', methods=['GET'])
def serve_gallery_image(filename):
    """Serve images from the gallery"""
    try:
        # Security: Prevent directory traversal
        filename = os.path.basename(filename)
        
        # Try different possible directories
        possible_dirs = [
            os.path.join(os.path.dirname(__file__), '..', 'generated_images'),
            os.path.join(os.path.dirname(__file__), '..', 'watermarked_images'),
            os.path.join(os.path.dirname(__file__), '..', 'evaluated_images'),
            os.path.join(os.path.dirname(__file__), '..', 'uploads')
        ]
        
        for directory in possible_dirs:
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                return send_file(file_path, mimetype='image/png')
        
        return jsonify({'message': 'Image not found'}), 404
        
    except Exception as e:
        return jsonify({'message': f'Error serving image: {str(e)}'}), 500

@gallery_bp.route('/api/gallery/stats', methods=['GET'])
def get_gallery_stats():
    """Get user's gallery statistics"""
    user = get_user_from_token()
    
    if not user:
        return jsonify({'message': 'Unauthorized'}), 401
    
    session = Session()
    try:
        generated_count = session.query(GeneratedImage).filter_by(user_id=user.id).count()
        evaluated_count = session.query(EvaluatedImage).filter_by(user_id=user.id).count()
        watermarked_count = session.query(WatermarkedImage).filter_by(user_id=user.id).count()
        
        return jsonify({
            'generated': generated_count,
            'evaluated': evaluated_count,
            'watermarked': watermarked_count,
            'total': generated_count + evaluated_count + watermarked_count
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching stats: {str(e)}'}), 500
    finally:
        session.close()

@gallery_bp.route('/api/gallery/save-generated', methods=['POST'])
def save_generated_image():
    """Save a generated image to user's gallery"""
    user = get_user_from_token()
    
    if not user:
        return jsonify({'message': 'Unauthorized. Please sign in to save images.'}), 401
    
    data = request.get_json()
    image_data = data.get('imageData')  # base64 string
    prompt = data.get('prompt')
    dataset = data.get('dataset', 'Unknown')
    
    if not image_data or not prompt:
        return jsonify({'message': 'Image data and prompt are required'}), 400
    
    try:
        import base64
        import uuid
        from pathlib import Path
        
        # Create directory for permanent storage
        generated_dir = Path(__file__).parent.parent / 'generated_images'
        generated_dir.mkdir(exist_ok=True)
        
        # Generate permanent filename
        permanent_filename = f"{uuid.uuid4().hex}.png"
        permanent_path = generated_dir / permanent_filename
        
        # Decode base64 and save image
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        with open(permanent_path, 'wb') as f:
            f.write(image_bytes)
        
        # Save to database
        session = Session()
        try:
            generated_img = GeneratedImage(
                user_id=user.id,
                prompt=prompt,
                file_path=str(permanent_path),
                dataset=dataset
            )
            session.add(generated_img)
            session.commit()
            
            return jsonify({
                'message': 'Image saved to gallery successfully!',
                'image_id': generated_img.id
            }), 200
        finally:
            session.close()
            
    except Exception as e:
        print(f"Error saving generated image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Error saving image: {str(e)}'}), 500

@gallery_bp.route('/api/gallery/save-watermarked', methods=['POST'])
def save_watermarked_image():
    """Save a watermarked image to user's gallery"""
    user = get_user_from_token()
    
    if not user:
        return jsonify({'message': 'Unauthorized. Please sign in to save images.'}), 401
    
    data = request.get_json()
    image_data = data.get('imageData')  # base64 string
    original_name = data.get('originalName', 'original.png')
    watermark_text = data.get('watermarkText', '')
    watermark_position = data.get('watermarkPosition', 'SE')
    watermark_opacity = data.get('watermarkOpacity', 50)
    
    if not image_data:
        return jsonify({'message': 'Image data is required'}), 400
    
    try:
        import base64
        import uuid
        from pathlib import Path
        
        # Create directory for permanent storage
        watermarked_dir = Path(__file__).parent.parent / 'watermarked_images'
        watermarked_dir.mkdir(exist_ok=True)
        
        # Generate permanent filename
        permanent_filename = f"{uuid.uuid4().hex}.png"
        permanent_path = watermarked_dir / permanent_filename
        
        # Decode base64 and save image
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        with open(permanent_path, 'wb') as f:
            f.write(image_bytes)
        
        # Save to database
        session = Session()
        try:
            watermarked_img = WatermarkedImage(
                user_id=user.id,
                original_image_path=original_name,
                watermarked_image_path=str(permanent_path),
                watermark_text=watermark_text,
                watermark_position=watermark_position,
                watermark_opacity=int(watermark_opacity)
            )
            session.add(watermarked_img)
            session.commit()
            
            return jsonify({
                'message': 'Watermarked image saved to gallery successfully!',
                'image_id': watermarked_img.id
            }), 200
        finally:
            session.close()
            
    except Exception as e:
        print(f"Error saving watermarked image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Error saving image: {str(e)}'}), 500

@gallery_bp.route('/api/gallery/save-evaluated', methods=['POST'])
def save_evaluated_image():
    """Save an evaluated image to user's gallery"""
    user = get_user_from_token()
    
    if not user:
        return jsonify({'message': 'Unauthorized. Please sign in to save images.'}), 401
    
    data = request.get_json()
    image_data = data.get('imageData')  # base64 string
    original_name = data.get('originalName', 'original.png')
    prompt = data.get('prompt', '')
    score = data.get('score')
    
    if not image_data:
        return jsonify({'message': 'Image data is required'}), 400
    
    try:
        import base64
        import uuid
        from pathlib import Path
        
        # Create directory for permanent storage
        evaluated_dir = Path(__file__).parent.parent / 'evaluated_images'
        evaluated_dir.mkdir(exist_ok=True)
        
        # Generate permanent filename
        permanent_filename = f"{uuid.uuid4().hex}.png"
        permanent_path = evaluated_dir / permanent_filename
        
        # Decode base64 and save image
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        with open(permanent_path, 'wb') as f:
            f.write(image_bytes)
        
        # Save to database
        session = Session()
        try:
            evaluated_img = EvaluatedImage(
                user_id=user.id,
                original_image_path=original_name,
                evaluated_image_path=str(permanent_path),
                prompt=prompt,
                score=score
            )
            session.add(evaluated_img)
            session.commit()
            
            return jsonify({
                'message': 'Evaluated image saved to gallery successfully!',
                'image_id': evaluated_img.id
            }), 200
        finally:
            session.close()
            
    except Exception as e:
        print(f"Error saving evaluated image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Error saving image: {str(e)}'}), 500

@gallery_bp.route('/api/gallery/generated/<image_id>', methods=['DELETE'])
def delete_generated_image(image_id):
    """Delete a generated image from user's gallery"""
    print(f"🗑️ DELETE request received for generated image ID: {image_id}")
    
    user = get_user_from_token()
    print(f"🔐 User from token: {user.email if user else 'None'}")
    
    if not user:
        print("❌ Unauthorized: No valid user token")
        return jsonify({'error': 'Unauthorized'}), 401
    
    session = Session()
    try:
        # Find the image
        image = session.query(GeneratedImage).filter_by(id=image_id, user_id=user.id).first()
        print(f"🔍 Found image: {image is not None}, Image ID: {image.id if image else 'None'}, User ID: {user.id}")
        
        if not image:
            return jsonify({'error': 'Image not found'}), 404
        
        # Delete the file from filesystem
        try:
            if image.file_path and os.path.exists(image.file_path):
                os.remove(image.file_path)
                print(f"Deleted file: {image.file_path}")
        except Exception as e:
            print(f"Error deleting file: {e}")
            import traceback
            traceback.print_exc()
        
        # Delete from database
        session.delete(image)
        session.commit()
        print(f"Deleted generated image {image_id} from database")
        
        return jsonify({'message': 'Image deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting generated image: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        return jsonify({'error': f'Error deleting image: {str(e)}'}), 500
    finally:
        session.close()

@gallery_bp.route('/api/gallery/watermarked/<image_id>', methods=['DELETE'])
def delete_watermarked_image(image_id):
    """Delete a watermarked image from user's gallery"""
    print(f"🗑️ DELETE request received for watermarked image ID: {image_id}")
    
    user = get_user_from_token()
    print(f"🔐 User from token: {user.email if user else 'None'}")
    
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    session = Session()
    try:
        # Find the image
        image = session.query(WatermarkedImage).filter_by(id=image_id, user_id=user.id).first()
        
        if not image:
            return jsonify({'error': 'Image not found'}), 404
        
        # Delete the file from filesystem
        try:
            if image.watermarked_image_path and os.path.exists(image.watermarked_image_path):
                os.remove(image.watermarked_image_path)
                print(f"Deleted file: {image.watermarked_image_path}")
        except Exception as e:
            print(f"Error deleting file: {e}")
            import traceback
            traceback.print_exc()
        
        # Delete from database
        session.delete(image)
        session.commit()
        print(f"Deleted watermarked image {image_id} from database")
        
        return jsonify({'message': 'Image deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting watermarked image: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        return jsonify({'error': f'Error deleting image: {str(e)}'}), 500
    finally:
        session.close()

@gallery_bp.route('/api/gallery/evaluated/<image_id>', methods=['DELETE'])
def delete_evaluated_image(image_id):
    """Delete an evaluated image from user's gallery"""
    print(f"🗑️ DELETE request received for evaluated image ID: {image_id}")
    
    user = get_user_from_token()
    print(f"🔐 User from token: {user.email if user else 'None'}")
    
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    session = Session()
    try:
        # Find the image
        image = session.query(EvaluatedImage).filter_by(id=image_id, user_id=user.id).first()
        
        if not image:
            return jsonify({'error': 'Image not found'}), 404
        
        # Delete the file from filesystem
        try:
            if image.evaluated_image_path and os.path.exists(image.evaluated_image_path):
                os.remove(image.evaluated_image_path)
                print(f"Deleted file: {image.evaluated_image_path}")
        except Exception as e:
            print(f"Error deleting file: {e}")
            import traceback
            traceback.print_exc()
        
        # Delete from database
        session.delete(image)
        session.commit()
        print(f"Deleted evaluated image {image_id} from database")
        
        return jsonify({'message': 'Image deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting evaluated image: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        return jsonify({'error': f'Error deleting image: {str(e)}'}), 500
    finally:
        session.close()
