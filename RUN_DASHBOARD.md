# 🚀 How to Run the Dashboard

## Quick Start

1. **Open PowerShell or Command Prompt**

2. **Navigate to your project folder:**
   ```bash
   cd c:\Users\abdel\Desktop\complexiteeProject
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **The dashboard will automatically open in your browser** at `http://localhost:8501`

---

## ✅ What's Fixed

- ✅ Created missing `utils/` folder with all required modules
- ✅ Created missing `algorithms/` folder with SA, GA, and Tabu Search wrappers
- ✅ Dataset path is correctly set to `cloud_task_scheduling_dataset.csv`
- ✅ All imports are working correctly

---

## 📊 Using the Dashboard

1. **Load Dataset**: 
   - The default path `cloud_task_scheduling_dataset.csv` is already set
   - Or upload your own CSV file using the sidebar

2. **Configure Parameters**: 
   - Set algorithm parameters in the sidebar
   - Adjust common constraints (capacity, thresholds)

3. **Run Algorithms**: 
   - Click "▶️ Run All" to execute all three algorithms
   - Wait for completion (may take a few minutes)

4. **Explore Results**: 
   - Navigate through 4 tabs:
     - 📊 **Overview**: KPI cards and metrics
     - 📈 **Convergence**: Fitness evolution charts
     - 🔍 **Constraints**: Gantt charts and diagnostics
     - ⚙️ **Complexity**: Theoretical and empirical analysis

5. **Export Results**: 
   - Use the sidebar to download results as CSV or JSON

---

## 🛑 To Stop the Dashboard

Press `Ctrl + C` in the terminal window where Streamlit is running.

---

## 📝 Notes

- The dashboard uses your dataset file: `cloud_task_scheduling_dataset.csv`
- All three algorithms (SA, GA, TS) are implemented and ready to run
- Results are cached to avoid recomputation
- You can adjust all parameters from the sidebar

---

**Ready to go! Just run `streamlit run app.py`** 🎉

