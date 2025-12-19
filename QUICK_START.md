# Quick Start Guide

## How to Run the Dashboard

### Step 1: Install Dependencies

Open PowerShell or Command Prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- streamlit
- pandas
- numpy
- matplotlib
- plotly

### Step 2: Ensure Required Files Exist

The dashboard requires these directories and files:

```
complexiteeProject/
├── app.py
├── algorithms/
│   ├── __init__.py
│   ├── sa.py
│   ├── ga.py
│   └── tabu.py
└── utils/
    ├── __init__.py
    ├── data.py
    ├── metrics.py
    └── plots.py
```

**If these files don't exist, the app will fail to start.**

### Step 3: Run the Dashboard

In PowerShell or Command Prompt, navigate to the project directory:

```bash
cd c:\Users\abdel\Desktop\complexiteeProject
```

Then run:

```bash
streamlit run app.py
```

### Step 4: Use the Dashboard

1. The dashboard will open automatically in your browser (usually at `http://localhost:8501`)
2. In the sidebar:
   - **Load your dataset**: Upload a CSV file or specify a local file path
   - **Configure parameters**: Set algorithm parameters
   - **Click "Run All"**: Execute all three algorithms
3. Navigate through the tabs to view results

## Troubleshooting

### Error: "No module named 'utils.data'"
- **Solution**: The `utils/` directory is missing. You need to create it with the required Python files.

### Error: "No module named 'algorithms.sa'"
- **Solution**: The `algorithms/` directory is missing. You need to create it with the required Python files.

### Error: "FileNotFoundError" when loading dataset
- **Solution**: Make sure your CSV file exists at the specified path, or upload it using the sidebar.

### Dashboard opens but shows errors
- Check that all dependencies are installed: `pip list | findstr streamlit`
- Verify your CSV has the required columns: Task_ID, Priority, Target (Optimal Scheduling), Execution_Time (s), CPU_Usage (%), Network_IO (MB/s)

## Quick Test

To verify Streamlit is installed correctly:

```bash
streamlit --version
```

If this works, you're ready to run the dashboard!

