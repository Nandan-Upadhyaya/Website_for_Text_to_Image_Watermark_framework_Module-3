import os
import requests
from dotenv import load_dotenv
import jwt
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

load_dotenv()

def verify_firebase_token(id_token):
    """
    Verify Firebase ID token using Google's public keys
    This doesn't require Firebase Admin SDK or service account
    """
    try:
        project_id = os.getenv('FIREBASE_PROJECT_ID', 'project-561719770763')
        
        # Get Firebase public keys (these are X.509 certificates)
        jwks_url = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
        response = requests.get(jwks_url)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch Firebase public keys: {response.status_code}")
            return None
        
        certificates = response.json()
        
        # Decode the token header to get the key ID
        unverified_header = jwt.get_unverified_header(id_token)
        kid = unverified_header.get('kid')
        
        if not kid or kid not in certificates:
            print(f"❌ Invalid key ID in token header: {kid}")
            return None
        
        # Get the certificate for this token (it's an X.509 certificate, not a public key)
        certificate_str = certificates[kid]
        
        # Load the X.509 certificate
        certificate_bytes = certificate_str.encode('utf-8')
        cert = x509.load_pem_x509_certificate(certificate_bytes, default_backend())
        
        # Extract the public key from the certificate
        public_key = cert.public_key()
        
        # Convert to PEM format that PyJWT can use
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Decode and verify the token with leeway for clock skew
        # Leeway allows for time differences between client and server (up to 60 seconds)
        decoded_token = jwt.decode(
            id_token,
            key=public_key_pem,
            algorithms=["RS256"],
            audience=project_id,
            issuer=f"https://securetoken.google.com/{project_id}",
            leeway=60  # Allow 60 seconds of clock skew tolerance
        )
        
        print(f"✅ Firebase token verified for user: {decoded_token.get('email')}")
        print(f"🔍 Token fields: {list(decoded_token.keys())}")
        return decoded_token
        
    except jwt.ExpiredSignatureError:
        print("❌ Firebase token has expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid Firebase token: {e}")
        return None
    except Exception as e:
        print(f"❌ Firebase token verification error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_firebase_user(uid):
    """
    Get Firebase user by UID
    Note: This function is not used in the current implementation
    """
    # Not implemented without Firebase Admin SDK
    return None
