@echo off
REM Add Streamlit to PATH temporarily
set PATH=%PATH%;C:\Users\Klimt\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts

REM Change to app directory
cd /d C:\Users\Klimt\CascadeProjects\streamlit-app

REM Start the Streamlit app
echo Starting Car Detection App...
streamlit run app.py

pause
