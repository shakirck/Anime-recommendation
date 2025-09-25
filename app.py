import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle

# --- App Initialization ---
app = Flask(__name__)

# --- Configuration ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key')
basedir = os.path.abspath(os.path.dirname(__file__))
# Ensure the instance folder exists for the database
instance_path = os.path.join(basedir, 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Database and Login Manager Setup ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'home'

# --- Database Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Initialize Database ---
with app.app_context():
    db.create_all()

# --- ML Model & Data Loading ---
try:
    similarity = pickle.load(open('similarity.pkl','rb'))
    anime_df = pd.read_csv('anime.csv')
except Exception as e:
    similarity = None
    anime_df = None
    print(f"CRITICAL ERROR: Could not load model/data. {e}")

# --- Recommendation Logic ---
def recommend(anime_title):
    if anime_df is None or similarity is None:
        return []
    try:
        anime_index = anime_df[anime_df['name'] == anime_title].index[0]
    except IndexError:
        return []
    distances = similarity[anime_index]
    anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_anime = [anime_df.iloc[i[0]].name for i in anime_list]
    return recommended_anime

# --- Routes ---
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return redirect(url_for('index'))
    flash('Invalid username or password.', 'error')
    return redirect(url_for('home'))

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    password = request.form.get('password')
    if User.query.filter_by(username=username).first():
        flash('Username already exists.', 'error')
        return redirect(url_for('home'))
    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    flash('Account created successfully! Please log in.', 'success')
    return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/index')
@login_required
def index():
    anime_titles = anime_df['name'].values if anime_df is not None else []
    return render_template('index.html', username=current_user.username, anime_list=anime_titles)

@app.route('/recommend', methods=['POST'])
@login_required
def get_recommendations():
    selected_anime = request.form.get('anime')
    recommendations = recommend(selected_anime)
    anime_titles = anime_df['name'].values if anime_df is not None else []
    return render_template('index.html', 
                           username=current_user.username, 
                           anime_list=anime_titles, 
                           recommendations=recommendations,
                           selected_anime=selected_anime)

# --- Run App (for local development only) ---
if __name__ == "__main__":
    app.run(debug=True)
