# 📦 Complete File Upload Guide

## 🎯 Files for GitHub Repository (REQUIRED)

These files MUST be uploaded to your GitHub repository root:

### 1. Core Application Files
```
linkedin_social_listening.py   ← Main app (REQUIRED)
requirements.txt               ← Python dependencies (REQUIRED)
packages.txt                   ← System dependencies (REQUIRED)
README.md                      ← Documentation (Recommended)
```

### 2. Configuration Folder
```
.streamlit/
  └── config.toml              ← Streamlit settings (Recommended)
```

---

## 📂 Your GitHub Repository Structure

After uploading, your repo should look like this:

```
your-repo-name/
├── linkedin_social_listening.py
├── requirements.txt
├── packages.txt
├── README.md
└── .streamlit/
    └── config.toml
```

**That's it!** Just 5 files/items total.

---

## 🔧 Files for Local Use (NOT on GitHub)

These HTML tools are for YOU to use locally to collect data. Download them to your computer:

### Data Collection Tools
```
linkedin_copy_paste_extractor.html   ← Copy/paste method (Easiest!)
linkedin_data_entry.html             ← Manual entry form
```

**Where to save these:**
- Your Desktop
- A "LinkedIn Tools" folder on your computer
- Anywhere you can easily access them

**How to use:**
1. Double-click the HTML file to open in browser
2. Use it to collect LinkedIn data
3. Download the CSV
4. Upload CSV to your Streamlit app

---

## 📚 Documentation Files (Optional)

These are reference guides. You don't need to upload them to GitHub, but keep them for reference:

```
BOOKMARKLET_FIX.md        ← How to fix bookmarklet issues
BOOKMARKLET_GUIDE.md      ← Complete bookmarklet guide
DATA_COLLECTION.md        ← All data collection methods
DEPLOYMENT.md             ← Original deployment guide
DEPLOYMENT_FIX.md         ← Troubleshooting deployments
EXTRACTION_GUIDE.md       ← Overview of extraction methods
TROUBLESHOOTING.md        ← General troubleshooting
```

---

## 🚀 Step-by-Step Upload to GitHub

### Option A: Via GitHub Website (Easiest)

1. **Go to your GitHub repository**
   - If you don't have one, create it at github.com/new
   - Make it **Public** (required for free Streamlit Cloud)

2. **Upload main files:**
   - Click "Add file" → "Upload files"
   - Drag and drop these 4 files:
     - `linkedin_social_listening.py`
     - `requirements.txt`
     - `packages.txt`
     - `README.md`
   - Click "Commit changes"

3. **Create .streamlit folder:**
   - Click "Add file" → "Create new file"
   - Name it: `.streamlit/config.toml`
   - Paste the config.toml content
   - Click "Commit new file"

### Option B: Via Git Command Line

```bash
# Clone your repo
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Copy files (adjust paths as needed)
cp /path/to/linkedin_social_listening.py .
cp /path/to/requirements.txt .
cp /path/to/packages.txt .
cp /path/to/README.md .
mkdir -p .streamlit
cp /path/to/config.toml .streamlit/

# Commit and push
git add .
git commit -m "Initial commit - LinkedIn Social Listening App"
git push origin main
```

---

## ✅ Verification Checklist

Before deploying to Streamlit Cloud, verify:

- [ ] `linkedin_social_listening.py` is in repository root
- [ ] `requirements.txt` is in repository root
- [ ] `packages.txt` is in repository root
- [ ] `.streamlit/config.toml` exists (note the dot!)
- [ ] Repository is **Public**
- [ ] No syntax errors (check files in GitHub's viewer)

---

## 🎬 Deploy to Streamlit Cloud

Once files are on GitHub:

1. **Go to** https://streamlit.io/cloud
2. **Sign in** with GitHub
3. **Click** "New app"
4. **Select:**
   - Repository: `your-repo-name`
   - Branch: `main` (or `master`)
   - Main file: `linkedin_social_listening.py`
5. **Click** "Deploy!"
6. **Wait** 2-3 minutes

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 🛠️ Using the Data Collection Tools

### After Your App is Deployed:

1. **Download to your computer:**
   - `linkedin_copy_paste_extractor.html`
   - `linkedin_data_entry.html`

2. **When you want to collect data:**
   - Double-click `linkedin_copy_paste_extractor.html`
   - Go to LinkedIn, search for your topic
   - Copy the page (Ctrl+A, Ctrl+C)
   - Paste into the tool
   - Download CSV

3. **Upload to your app:**
   - Visit your Streamlit app URL
   - Click "Upload CSV" in sidebar
   - Select the CSV you just downloaded
   - Analyze!

---

## 📝 Quick Reference

### What Goes Where?

| File | Upload to GitHub? | Save Locally? | Purpose |
|------|-------------------|---------------|---------|
| linkedin_social_listening.py | ✅ YES | ❌ No | Main app |
| requirements.txt | ✅ YES | ❌ No | Dependencies |
| packages.txt | ✅ YES | ❌ No | System packages |
| .streamlit/config.toml | ✅ YES | ❌ No | Config |
| README.md | ⭐ Optional | ❌ No | Documentation |
| linkedin_copy_paste_extractor.html | ❌ No | ✅ YES | Data collection |
| linkedin_data_entry.html | ❌ No | ✅ YES | Data collection |
| All .md guides | ❌ No | ⭐ Optional | Reference |

---

## 🆘 Common Mistakes

### ❌ Wrong:
```
your-repo/
└── src/
    └── linkedin_social_listening.py  ← In subfolder!
```

### ✅ Correct:
```
your-repo/
└── linkedin_social_listening.py  ← In root!
```

### ❌ Wrong:
```
Repository set to "Private"  ← Won't work with free tier!
```

### ✅ Correct:
```
Repository set to "Public"  ← Required for Streamlit free tier
```

---

## 💡 Pro Tips

1. **Keep HTML tools handy:** Bookmark their location or keep them on your Desktop
2. **Regular monitoring:** Use the copy-paste tool weekly to track trends
3. **Save CSVs:** Keep a folder of historical data to see changes over time
4. **Multiple topics:** You can monitor different topics - just collect separate CSVs

---

## 🎉 You're All Set!

**Summary:**
1. ✅ Upload 5 items to GitHub (4 files + 1 folder with config)
2. ✅ Deploy to Streamlit Cloud
3. ✅ Save HTML tools locally
4. ✅ Use tools to collect data → Upload to app → Analyze!

Need help? Check the documentation files or visit https://discuss.streamlit.io/
