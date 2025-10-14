import os
import datetime
import uuid
import jwt
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from werkzeug.security import generate_password_hash, check_password_hash

# Create database directory if it doesn't exist
DB_DIR = os.path.join(os.path.dirname(__file__), 'database')
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# Database configuration
DB_PATH = os.path.join(DB_DIR, 'ai_image_suite.db')
SQLALCHEMY_DATABASE_URI = f'sqlite:///{DB_PATH}'

# Create engine and session
engine = create_engine(SQLALCHEMY_DATABASE_URI)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# JWT Secret Key
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'development-secret-key')
JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(minutes=30)  # 30 minutes session timeout

class User(Base):
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=True)  # Nullable for Firebase users
    firebase_uid = Column(String(128), unique=True, nullable=True)  # For Firebase authentication
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    generated_images = relationship('GeneratedImage', back_populates='user', cascade='all, delete-orphan')
    evaluated_images = relationship('EvaluatedImage', back_populates='user', cascade='all, delete-orphan')
    watermarked_images = relationship('WatermarkedImage', back_populates='user', cascade='all, delete-orphan')
    
    def generate_token(self):
        payload = {
            'user_id': self.id,
            'exp': datetime.datetime.utcnow() + JWT_ACCESS_TOKEN_EXPIRES
        }
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')
    
    @staticmethod
    def verify_token(token):
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            # Token has expired
            return None
        except jwt.InvalidTokenError:
            # Invalid token
            return None
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class GeneratedImage(Base):
    __tablename__ = 'generated_images'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    prompt = Column(Text, nullable=False)
    file_path = Column(String(255), nullable=False)
    thumbnail_path = Column(String(255))
    dataset = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='generated_images')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'prompt': self.prompt,
            'file_path': self.file_path,
            'thumbnail_path': self.thumbnail_path,
            'dataset': self.dataset,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class EvaluatedImage(Base):
    __tablename__ = 'evaluated_images'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    original_image_path = Column(String(255), nullable=False)
    evaluated_image_path = Column(String(255), nullable=False)
    thumbnail_path = Column(String(255))
    prompt = Column(Text)
    score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='evaluated_images')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'original_image_path': self.original_image_path,
            'evaluated_image_path': self.evaluated_image_path,
            'thumbnail_path': self.thumbnail_path,
            'prompt': self.prompt,
            'score': self.score,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class WatermarkedImage(Base):
    __tablename__ = 'watermarked_images'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    original_image_path = Column(String(255), nullable=False)
    watermarked_image_path = Column(String(255), nullable=False)
    thumbnail_path = Column(String(255))
    watermark_text = Column(String(255))
    watermark_position = Column(String(50))
    watermark_opacity = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='watermarked_images')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'original_image_path': self.original_image_path,
            'watermarked_image_path': self.watermarked_image_path,
            'thumbnail_path': self.thumbnail_path,
            'watermark_text': self.watermark_text,
            'watermark_position': self.watermark_position,
            'watermark_opacity': self.watermark_opacity,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

def init_db():
    Base.metadata.create_all(engine)
    print("Database initialized.")

if __name__ == '__main__':
    init_db()