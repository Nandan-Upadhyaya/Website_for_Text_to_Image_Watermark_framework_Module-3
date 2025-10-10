# 🔐 Authentication & Data Storage

## Where Are Credentials Stored?

### **Email/Password Authentication**

#### Backend (Server-side):
- **Location**: `server/database/ai_image_suite.db`
- **Type**: SQLite database file
- **What's stored**:
  - User ID (UUID)
  - Name
  - Email (unique)
  - Password Hash (bcrypt - **NOT** plain text!)
  - Firebase UID (if linked to Google)
  - Created/Updated timestamps

**Password Security:**
- Passwords are **never** stored in plain text
- Uses bcrypt hashing with salt
- Hash example: `$2b$12$KXGfR.../abc123...` (irreversible)

#### Frontend (Browser):
- **Location**: `localStorage` in your browser
- **Keys**:
  - `ai_image_suite_auth_token`: JWT token for session
  - `ai_image_suite_user_info`: User profile (name, email, ID)
- **Access**: Developer Tools → Application → Local Storage → `http://localhost:3000`

**Token Example:**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMTIzNDU2Nzg5MCIsImV4cCI6MTYxNjI4OTYwMH0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

---

### **Google Authentication (Firebase)**

#### Backend:
- **Location**: Same database (`server/database/ai_image_suite.db`)
- **What's stored**:
  - Firebase UID (from Google)
  - Name (from Google profile)
  - Email (from Google)
  - Password Hash: `NULL` (Google manages authentication)

#### Frontend:
- **Firebase SDK**: Stores session tokens in `IndexedDB`
- **Our App**: Same JWT token in `localStorage`

---

## How to View Stored Data

### **View Database (Backend):**

1. **Using DB Browser for SQLite** (Recommended):
   - Download: https://sqlitebrowser.org/
   - Open: `server/database/ai_image_suite.db`
   - Browse Data → `users` table

2. **Using Python:**
   ```python
   import sqlite3
   
   conn = sqlite3.connect('server/database/ai_image_suite.db')
   cursor = conn.cursor()
   
   # View all users
   cursor.execute('SELECT id, name, email, firebase_uid, created_at FROM users')
   for row in cursor.fetchall():
       print(row)
   
   conn.close()
   ```

3. **Using Command Line:**
   ```powershell
   cd server/database
   sqlite3 ai_image_suite.db
   SELECT * FROM users;
   .quit
   ```

### **View Browser Storage (Frontend):**

1. **Chrome/Edge DevTools:**
   - Press `F12`
   - Go to **Application** tab
   - Left sidebar → **Local Storage** → `http://localhost:3000`
   - See: `ai_image_suite_auth_token` and `ai_image_suite_user_info`

2. **Firefox DevTools:**
   - Press `F12`
   - Go to **Storage** tab
   - **Local Storage** → `http://localhost:3000`

---

## Session Management

### **How Long Do Sessions Last?**
- **Default**: 24 hours (1 day)
- **Configurable in**: `server/models.py` → `JWT_ACCESS_TOKEN_EXPIRES`

### **What Happens on Logout?**
1. Clears `localStorage` in browser
2. Signs out from Firebase (if Google auth)
3. Redirects to login page
4. Token becomes invalid

### **Auto-Logout:**
- If token expires (after 24 hours)
- If user clicks "Logout"
- If user clears browser data

---

## Security Notes

### ✅ **What's Secure:**
- Passwords are hashed (bcrypt)
- JWT tokens have expiration
- Firebase handles Google OAuth securely
- HTTPS in production (you should add)

### ⚠️ **Current Limitations:**
- Running on `http://localhost` (not HTTPS)
- No rate limiting on login attempts
- No password reset functionality yet
- No 2FA (two-factor authentication)

---

## User Flow Diagram

```
┌─────────────────────────────────────────────┐
│         User Registration/Login             │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
    Email/Pass            Google OAuth
        │                       │
        ▼                       ▼
┌──────────────┐       ┌──────────────┐
│   Backend    │       │   Firebase   │
│  (SQLite)    │       │   (Google)   │
└──────┬───────┘       └──────┬───────┘
       │                      │
       └──────────┬───────────┘
                  │
                  ▼
         ┌────────────────┐
         │   JWT Token    │
         │   Generated    │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  localStorage  │
         │   (Browser)    │
         └────────────────┘
```

---

## Common Issues & Solutions

### **Problem**: "Invalid Firebase token"
**Solution**: Token verification is being fixed. Use email/password login for now.

### **Problem**: "User already exists"
**Solution**: Email is already registered. Try logging in or use a different email.

### **Problem**: "Session expired"
**Solution**: Your JWT token expired. Just log in again.

### **Problem**: Lost access to database
**Solution**: Delete `server/database/ai_image_suite.db` and run `python server/models.py` to recreate.

---

## For Developers

### **Reset Database:**
```powershell
Remove-Item server\database\ai_image_suite.db
myenv\Scripts\python.exe server\models.py
```

### **Change Token Expiry:**
Edit `server/models.py`:
```python
JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(days=7)  # Change to 7 days
```

### **Add New User Fields:**
1. Edit `server/models.py` → `User` class
2. Delete database
3. Recreate with `python server/models.py`

---

## Questions?

- **Where's the database?** → `server/database/ai_image_suite.db`
- **Where's the token?** → Browser localStorage
- **Is it secure?** → Password hashing yes, but add HTTPS for production
- **Can I see passwords?** → No! They're hashed (one-way encryption)
