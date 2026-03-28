#  Cyclist Analysis App (NOC*NSF Project)

This project is a collaboration between **Ambient Intelligence** and the **Dutch Olympic Committee (NOC*NSF)**. 

The goal is to help track cyclists have a faster start. This app uses Computer Vision (AI) to watch videos of cyclists and measure their movements automatically.

##  What this app does:
* **Two AI Models:** You can choose between **MediaPipe** (fast) or **MMPose** (very accurate) to track the cyclist.
* **Knee Angles:** The app calculates the angle of the leg during the start.
* **Center of Mass (CoM):** It tracks how the body moves forward to see how much power the cyclist has.
* **Interactive Graphs:** You can see the data in live charts and zoom in on details.
* **Export to Excel:** Coaches can download the data as a CSV file to study it later.

##  Tech Tools:
* **Python** (The main language)
* **Streamlit** (To make the web interface)
* **MediaPipe & MMPose** (The AI for body tracking)
* **Plotly** (For the charts)

##  How to use it:
1. Upload a video of a cyclist start.
2. Choose the AI model in the menu.
3. Click "Run Analysis".
4. Look at the graphs and download the data file.
