# Testised v1.0 — Fresh Setup

Complete files for deploying the testis cell detector on Streamlit Cloud.

## Files Included

- **streamlit_app.py** — Main application
- **requirements.txt** — Python dependencies
- **packages.txt** — System dependencies (OpenCV, etc.)
- **README.md** — This file

## Quick Setup

### Step 1: Prepare Your GitHub Repository

1. Go to https://github.com/new
2. Create repository: `testis`
3. **Public** ✓
4. Click "Create repository"

### Step 2: Upload Files to GitHub

1. Go to your repo: https://github.com/YOUR_USERNAME/testis
2. Click **"Add file"** → **"Upload files"**
3. Upload these 3 files:
   - `streamlit_app.py`
   - `requirements.txt`
   - `packages.txt`
4. Click **"Commit changes"**

### Step 3: Deploy to Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click **"New app"**
4. Select: `YOUR_USERNAME/testis`
5. Branch: `main`
6. Main file: `streamlit_app.py`
7. Click **"Deploy"**

### Step 4: Configure Google Drive Model

1. **Upload `best.pt` to Google Drive**
2. Right-click → **Share** → Copy link
3. Extract file ID from link (between `/d/` and `/view`)
   - Example: `https://drive.google.com/file/d/1ABCxyz123.../view`
   - File ID: `1ABCxyz123`
4. Go to your Streamlit Cloud app
5. Click **⋮** (three dots) → **Settings**
6. Click **"Secrets"**
7. Add this line:
   ```
   GDRIVE_FILE_ID = "1ABCxyz123"
   ```
   (Replace with YOUR file ID)
8. Click **"Save"**
9. App reboots automatically

### Step 5: Test & Share

1. Your app is live at: `https://testis-XXXXX.streamlit.app`
2. Upload a test H&E image
3. Click "Detect cells"
4. Share the URL with colleagues! 🎉

---

## Features

✅ Upload H&E images (JPG, PNG, TIFF)
✅ Real-time cell detection
✅ Confidence slider
✅ Select which cell types to detect
✅ Tissue quality assessment
✅ Export to Excel (coordinates + confidence)
✅ Export to GeoJSON (QuPath-compatible)
✅ 100% online, works from any browser

---

## Troubleshooting

### "libGL.so.1 not found"
- This is fixed by `packages.txt`
- Make sure you uploaded it to GitHub
- Reboot the app

### "GDRIVE_FILE_ID not configured"
1. Check you added the secret correctly (no typos)
2. Make sure the file ID is between `/d/` and `/view` in the share link
3. Click Save in Secrets
4. Reboot app

### "Model downloading..."
- First load downloads the model (~1 min)
- Subsequent loads are instant
- This is normal!

---

## File Details

### streamlit_app.py
- Loads model from Google Drive (cached after first load)
- Preserves image aspect ratio during detection
- Proper coordinate scaling
- Clean UI with confidence slider
- Excel + GeoJSON export

### requirements.txt
- Streamlit 1.57.0
- YOLOv8 (ultralytics 8.4.52)
- PyTorch + torchvision
- OpenCV (headless version)
- Pandas + Pillow for processing

### packages.txt
- System libraries needed for OpenCV and graphics
- Installed by Streamlit Cloud automatically

---

## Local Testing (Optional)

Before deploying, test on your computer:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501

(Note: Local testing needs `best.pt` in the folder OR the GDRIVE_FILE_ID secret to work)

---

## Support

If something breaks:
1. Check the logs in Streamlit Cloud (Manage app → Logs)
2. Try rebooting the app
3. Delete and redeploy if needed

Good luck! 🚀
