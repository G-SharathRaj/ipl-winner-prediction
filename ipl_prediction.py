import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample IPL teams
teams = [
    "Mumbai Indians", "Chennai Super Kings", "Delhi Capitals", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals", "Punjab Kings",
    "Lucknow Super Giants", "Gujarat Titans"
]

# Sample IPL dataset (replace with actual dataset)
data_size = 500
df = pd.DataFrame({
    "team1": np.random.choice(teams, data_size),
    "team2": np.random.choice(teams, data_size),
    "venue": np.random.randint(1, 10, data_size),  # Simulating venue ID
    "toss_winner": np.random.choice(teams, data_size),
    "bat_or_bowl": np.random.randint(0, 2, data_size),  # 0 = Bat, 1 = Bowl
    "winner": np.random.choice(teams, data_size)  # Actual winner (random for now)
})

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=["team1", "team2", "toss_winner"])

# Define input features and target variable
X = df_encoded.drop(columns=["winner"])
y = df["winner"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 1000))

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=random.randint(1, 1000))
rf_model.fit(X_train, y_train)

# Train Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=random.randint(1, 1000))
gb_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Evaluate Models
rf_accuracy = accuracy_score(y_test, rf_predictions)
gb_accuracy = accuracy_score(y_test, gb_predictions)

# Print results
print("\nRandom Forest Model Performance:")
print(f"Accuracy: {rf_accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

print("\nGradient Boosting Model Performance:")
print(f"Accuracy: {gb_accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, gb_predictions))

# Inject randomness into final predictions
def get_final_winner(model_prediction):
    if random.random() < 0.2:  # 20% chance to introduce randomness
        return random.choice(teams)
    return model_prediction

predicted_winner_rf = get_final_winner(random.choice(rf_predictions))
predicted_winner_gb = get_final_winner(random.choice(gb_predictions))

print(f"\nPredicted Winner (Random Forest): {predicted_winner_rf}")
print(f"Predicted Winner (Gradient Boosting): {predicted_winner_gb}")
