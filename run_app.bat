@echo off
echo ========================================
echo    House Price Prediction App
echo ========================================
echo.
echo Please ensure you have:
echo 1. Installed dependencies: pip install -r requirements.txt
echo 2. Run the Jupyter notebook first: house_price_prediction_ml.ipynb
echo 3. Generated model files in the models/ directory
echo.
echo The app will open in your default browser at:
echo http://localhost:8501
echo.
echo To stop the app, press Ctrl+C in this window.
echo.
echo ========================================
pause
echo Starting Streamlit app...
echo.
streamlit run house_price_app.py
