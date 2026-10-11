# SmartStudy AI — Final Submission Pack

## Overview
SmartStudy AI is a Streamlit dashboard for student performance analytics and final-grade estimation using a Random Forest regression model. Subject selection is displayed on the main screen; the sidebar is hidden for a cleaner dashboard layout.

## Dataset
UCI Machine Learning Repository — Student Performance:
https://archive.ics.uci.edu/dataset/320/student%2Bperformance

This is a public research dataset, not a government dataset. The app may download it on first run; if your version includes local CSV files, keep them beside `app.py`.

## Run locally (Windows)
1. Install Python 3.10 or newer.
2. Extract the project ZIP into a simple folder path.
3. Open CMD/PowerShell in the folder containing `app.py`.
4. Run:
   `python -m pip install -r requirements.txt`
5. Run:
   `python -m streamlit run app.py`

## Deployment
A public link is not created automatically by running locally. To deploy, push `app.py`, `requirements.txt`, and any needed local dataset files to a GitHub repository, then create an app at https://share.streamlit.io/ using that repository. Test the public URL before adding it to the PPT and report.

## Before submission
- Test Mathematics and Portuguese.
- Test all dashboard tabs and the prediction button.
- Capture genuine screenshots from your running app.
- Replace `ADD DEPLOYED URL HERE` in the PPT and documents after deployment.
- Copy actual MAE, RMSE and R² from the app; do not invent performance values.
- Do not use identifiable student records in a public demo.
