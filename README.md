# Telecom-Churn-Prediction-App


This project is all about helping telecom companies figure out which customers are likely to leave—before they actually do. I built a simple, interactive web app using Streamlit that makes it easy to get churn predictions with just a few clicks.

#### What the App Does:
##### Single Prediction: 
You can enter details for one customer—like their call minutes, whether they have an international plan, how often they contact customer service—and the app will tell you if they’re likely to churn or stick around.

##### Batch Prediction: 
Got a whole list of customers? No problem. Upload a CSV file, and the app will run predictions on all of them at once. You’ll get a clean table with results, plus visual charts to quickly see the big picture.

##### How It Works:
Behind the scenes, there's a trained machine learning model doing the heavy lifting. It’s learned from past customer behavior and uses that knowledge to make predictions. The app also does some smart preprocessing, turning things like plan status (“yes”/“no”) into numbers the model understands.

##### Cool Extras:
* Instant prediction results

* Handy charts (like pie charts and count plots) so you can visualize churn vs. no churn

* Shows how fast predictions are made

* Option to download the results for sharing or future use
