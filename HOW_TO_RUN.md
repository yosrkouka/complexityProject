# 🚀 How to Run the Dashboard

## Step-by-Step Instructions

### 1. Open Terminal/PowerShell
- Press `Windows Key + X` and select "Windows PowerShell" or "Terminal"
- Or open Command Prompt

### 2. Navigate to Your Project
```bash
cd c:\Users\abdel\Desktop\complexiteeProject
```

### 3. Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

### 5. View the Dashboard
- The dashboard will **automatically open** in your default web browser
- The URL will be: `http://localhost:8501`
- If it doesn't open automatically, copy the URL from the terminal and paste it in your browser

### 6. Stop the Dashboard
- Press `Ctrl + C` in the terminal to stop the server

---

## ⚠️ Important Notes

**Before running, make sure you have:**
1. ✅ All dependencies installed (`streamlit`, `pandas`, `numpy`, `matplotlib`, `plotly`)
2. ✅ The required folder structure:
   - `algorithms/` folder with `sa.py`, `ga.py`, `tabu.py`
   - `utils/` folder with `data.py`, `metrics.py`, `plots.py`
3. ✅ A CSV dataset file (or you can upload one in the dashboard)

---

## 🎯 Quick Test

To verify everything is ready:

```bash
# Check Streamlit is installed
streamlit --version

# Check Python can find the modules
python -c "import streamlit; print('OK')"
```

---

## 📊 Using the Dashboard

Once the dashboard opens:

1. **Load Dataset**: Use the sidebar to upload a CSV or specify a file path
2. **Set Parameters**: Configure algorithm parameters in the sidebar
3. **Run Algorithms**: Click "▶️ Run All" button
4. **Explore Results**: Navigate through the 4 tabs:
   - 📊 Overview
   - 📈 Convergence
   - 🔍 Constraints
   - ⚙️ Complexity

---

## 🐛 Troubleshooting

**Error: "No module named 'utils'"**
- The `utils/` folder is missing. You need to create it with the required Python files.

**Error: "No module named 'algorithms'"**
- The `algorithms/` folder is missing. You need to create it with the required Python files.

**Dashboard won't start**
- Make sure you're in the correct directory
- Check that `app.py` exists in the current folder
- Verify Streamlit is installed: `pip install streamlit`

**Browser doesn't open automatically**
- Look for the URL in the terminal output (usually `http://localhost:8501`)
- Copy and paste it into your browser manually

