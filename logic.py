import pandas as pd
import html
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_and_clean_data(filepath):
    """Loads and preprocesses the anime dataset."""
    anime_df = pd.read_csv(filepath)
    # Decode HTML entities in names
    anime_df['name'] = anime_df['name'].apply(html.unescape)
    # Drop rows with missing essential data
    anime_df.dropna(subset=['rating', 'genre'], inplace=True)
    # Create a 'primary_genre' column from the first genre in the list
    anime_df['primary_genre'] = anime_df['genre'].apply(lambda x: x.split(',')[0].strip())
    return anime_df.drop(columns=['anime_id', 'episodes'], errors='ignore')

def train_models(df):
    """Trains classification models to predict if an anime is recommendable."""
    # Define a target: an anime is "recommendable" if its rating is 7.5 or higher
    df['recommendable'] = df['rating'].apply(lambda x: 1 if x >= 7.5 else 0)
    
    # Define features and target
    X = df[['primary_genre', 'members']]
    y = df['recommendable']
    
    # One-hot encode the categorical feature 'primary_genre'
    X_encoded = pd.get_dummies(X, columns=['primary_genre'])
    
    # Split data for training
    X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train a Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    # Train a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)

    return {
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }, X_train.columns

def normalize_title(title):
    """Utility function to extract the base title for deduplication."""
    base = re.sub(r':.*| - .*|\(.*', '', title).strip()
    return base if base else title.strip()

def get_recommendations(df, model, training_cols, genre, min_rating, anime_type):
    """Filters data and uses a model to predict and return recommendations."""
    # Initial filtering based on user's direct input
    mask = (df['genre'].str.contains(genre, case=False, na=False)) & (df['rating'] >= min_rating)
    if anime_type.lower() != 'all':
        mask &= (df['type'].str.lower() == anime_type.lower())

    filtered_df = df[mask].copy()
    if filtered_df.empty:
        return None

    # Prepare the filtered data for model prediction
    X_pred = pd.get_dummies(filtered_df[['primary_genre', 'members']], columns=['primary_genre'])
    X_pred = X_pred.reindex(columns=training_cols, fill_value=0)

    # Use the model to predict which of the filtered anime are "recommendable"
    filtered_df['prediction'] = model.predict(X_pred)
    recommendations = filtered_df[filtered_df['prediction'] == 1].sort_values(by='rating', ascending=False)

    if recommendations.empty:
        return None

    # Deduplicate based on a normalized title to avoid multiple seasons of the same show
    recommendations['base_title'] = recommendations['name'].apply(normalize_title)
    recommendations = recommendations.drop_duplicates(
        subset=['base_title'], 
        keep='first'
    ).drop(columns=['base_title'])

    return recommendations[['name', 'genre', 'type', 'rating']]