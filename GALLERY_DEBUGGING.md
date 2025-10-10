# 🔍 Gallery Debugging Guide

## Common Issues & Solutions

### **Issue 1: Images Not Loading - CORS Error**

**Error in Browser Console:**
```
Access to fetch at 'http://localhost:5001/api/gallery/generated' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**Solution:**
The CORS headers are already configured in the backend, but make sure the Flask server is running.

---

### **Issue 2: 401 Unauthorized Error**

**Error in Browser Console:**
```
GET http://localhost:5001/api/gallery/generated 401 (Unauthorized)
```

**Cause:** JWT token is expired or invalid

**Solution:**
1. Sign out and sign in again
2. Check if token exists:
   ```javascript
   // Open browser console (F12) and type:
   localStorage.getItem('token')
   ```
3. If token is `null`, you need to sign in

---

### **Issue 3: Backend Not Running**

**Error in Browser Console:**
```
GET http://localhost:5001/api/gallery/generated net::ERR_CONNECTION_REFUSED
```

**Solution:**
Start the backend server:
```bash
cd AI-Image-Suite
.\myenv\Scripts\python.exe server\app.py
```

You should see:
```
✅ [SERVER] Database initialized successfully
* Running on http://0.0.0.0:5001/
```

---

### **Issue 4: Frontend Not Running**

**Solution:**
Start the frontend:
```bash
cd AI-Image-Suite
npm start
```

Browser should open to `http://localhost:3000`

---

### **Issue 5: No Images in Database**

**Error:** Gallery loads but shows "No images yet"

**Cause:** No images have been generated while logged in

**Solution:**
1. Make sure you're **signed in** (check navbar for your name)
2. Generate a new image:
   - Go to "Generate Images"
   - Enter prompt: "A beautiful landscape"
   - Click "Generate Images"
   - Wait for generation to complete
3. Check backend console for:
   ```
   🔍 [GENERATE] Authenticated user detected: your@email.com
   ✅ Saved 1 images to gallery for user your@email.com
   ```
4. If you see "No authenticated user", the token isn't being sent

---

### **Issue 6: Token Not Being Sent**

**Check Request Headers:**
1. Open browser DevTools (F12)
2. Go to **Network** tab
3. Click "Generate Images"
4. Click on the `/api/generate` request
5. Check **Request Headers** section
6. Should include: `Authorization: Bearer eyJhbGc...`

**If Authorization header is missing:**
- Clear browser cache (Ctrl+Shift+Delete)
- Hard refresh (Ctrl+Shift+R)
- Make sure you restarted the frontend after code changes

---

### **Issue 7: Database Not Initialized**

**Error in Backend Console:**
```
(sqlite3.OperationalError) no such table: generated_images
```

**Solution:**
```bash
cd AI-Image-Suite
.\myenv\Scripts\python.exe server\models.py
```

This will recreate the database with all tables.

---

### **Issue 8: Images Saved But Not Displayed**

**Check Image Paths:**
1. Open backend console
2. Look for save confirmation:
   ```
   ✅ Saved 1 images to gallery for user your@email.com
   ```
3. Check if directories exist:
   ```bash
   ls generated_images/
   ls watermarked_images/
   ```
4. Check database directly:
   ```bash
   cd server/database
   sqlite3 ai_image_suite.db
   SELECT * FROM generated_images;
   .exit
   ```

---

## 🧪 Step-by-Step Testing Procedure

### **Step 1: Verify Both Servers Are Running**

**Backend Terminal:**
```bash
cd C:\Users\Tejas\Desktop\stl\github\AI-Image-Suite
.\myenv\Scripts\python.exe server\app.py
```

Expected output:
```
✅ [SERVER] Database initialized successfully
 * Running on http://0.0.0.0:5001/
```

**Frontend Terminal:**
```bash
cd C:\Users\Tejas\Desktop\stl\github\AI-Image-Suite
npm start
```

Expected output:
```
Compiled successfully!
webpack compiled with 0 warnings
```

### **Step 2: Clear Browser Cache**

1. Press `Ctrl+Shift+Delete`
2. Check "Cached images and files"
3. Click "Clear data"
4. Or use Incognito/Private window

### **Step 3: Sign In**

1. Go to http://localhost:3000
2. Click "Sign In" button
3. Sign in with your account
4. Check navbar - should show your name with green checkmark

### **Step 4: Generate Test Image**

1. Click "Generate Images"
2. Enter prompt: "A test image"
3. Click "Generate Images"
4. **Watch Backend Console:**
   ```
   🔍 [GENERATE] Authenticated user detected: your@email.com
   Using COCO model for prompt: 'A test image'
   ✅ Saved 1 images to gallery for user your@email.com
   ```

### **Step 5: Check Gallery**

1. Click "Gallery" in navbar
2. You should see your image
3. If not, check browser console (F12) for errors

### **Step 6: Check Network Tab**

1. Open DevTools (F12)
2. Go to **Network** tab
3. Click "Gallery" again
4. Look for request to `/api/gallery/generated`
5. Check:
   - Status should be **200 OK**
   - Response should contain images array
   - Request should have `Authorization` header

---

## 🔍 Debugging Commands

### **Check if token exists:**
```javascript
// Browser console
localStorage.getItem('token')
// Should return: "eyJhbGciOiJIUzI1NiIsInR5c..."
```

### **Check if user is logged in:**
```javascript
// Browser console
JSON.parse(localStorage.getItem('user'))
// Should return: { id: "...", name: "...", email: "..." }
```

### **Test gallery API directly:**
```bash
# Get your token from browser console first
# Then test API:
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" http://localhost:5001/api/gallery/stats
```

### **Check database:**
```bash
cd server/database
sqlite3 ai_image_suite.db

-- Check users
SELECT * FROM users;

-- Check generated images
SELECT id, user_id, prompt, created_at FROM generated_images;

-- Check watermarked images  
SELECT id, user_id, watermark_text, created_at FROM watermarked_images;

.exit
```

---

## 📝 What to Share for Help

If still not working, please provide:

1. **Backend Console Output:**
   - Copy everything from when you start the server
   - Include the lines with 🔍 and ✅ emojis

2. **Browser Console Errors:**
   - Press F12 → Console tab
   - Screenshot any red errors

3. **Network Tab:**
   - Press F12 → Network tab
   - Click on `/api/gallery/generated` request
   - Share the Status code and Response

4. **Screenshots:**
   - Gallery page showing the issue
   - Navbar showing you're logged in

---

## ✅ Expected Behavior

**When Everything Works:**

1. **Backend Console:**
   ```
   🔍 [GENERATE] Authenticated user detected: user@email.com
   Using COCO model for prompt: 'test'
   ✅ Saved 1 images to gallery for user user@email.com
   ```

2. **Browser Console:**
   - No errors
   - Network tab shows 200 OK for all gallery requests

3. **Gallery Page:**
   - Shows stats (Generated: 1, Total: 1)
   - Shows image grid with your images
   - Each image shows prompt and timestamp
   - Download button works

---

## 🚨 Still Not Working?

Try this **complete reset:**

```bash
# 1. Stop both servers (Ctrl+C)

# 2. Delete database
rm server/database/ai_image_suite.db

# 3. Recreate database
.\myenv\Scripts\python.exe server\models.py

# 4. Clear browser data
# Browser: Ctrl+Shift+Delete → Clear all

# 5. Restart backend
.\myenv\Scripts\python.exe server\app.py

# 6. Restart frontend (new terminal)
npm start

# 7. Sign up with NEW account (or use existing)

# 8. Generate test image

# 9. Check gallery
```

This will give you a fresh start!
