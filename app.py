import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import logic # Imports your recommendation logic

# --- App Initialization ---
app = Flask(__name__)

# --- Configuration ---
app.config['SECRET_KEY'] = 'a_very_secret_key_that_should_be_changed'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'users.db')
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

# --- ML Dataset & Models ---
try:
    # CORRECTED PATH: Look for anime.csv in the root folder
    df_cleaned = logic.load_and_clean_data("anime.csv") 
    models, training_cols = logic.train_models(df_cleaned.copy())
    print("Dataset loaded & models trained")
except Exception as e:
    print(f" Error loading dataset/models: {e}")
    df_cleaned, models, training_cols = None, None, None

# --- Data for Forms ---
ANIME_GENRES = [
    'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror',
    'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural'
]
ANIME_TYPES = ['All', 'TV', 'Movie', 'OVA', 'Special']

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
    else:
        flash('Invalid username or password.', 'error')
        return redirect(url_for('home'))

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    password = request.form.get('password')
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('Username already exists.', 'error')
        return redirect(url_for('home'))
    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    flash('Account created! Please log in.', 'success')
    return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html', genres=ANIME_GENRES, types=ANIME_TYPES, username=current_user.username)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if df_cleaned is None or models is None:
        flash('Dataset or models not available. Please contact an administrator.', 'error')
        return redirect(url_for('index'))
    try:
        genre = request.form['genre']
        anime_type = request.form['type']
        min_rating = float(request.form['min_rating'])

        # Always use Random Forest as it's generally more robust
        selected_model = models["Random Forest"]

        recommendations = logic.get_recommendations(
            df_cleaned, selected_model, training_cols, genre, min_rating, anime_type
        )

        if recommendations is None or recommendations.empty:
            flash('No recommendations found for your criteria. Try adjusting the rating!', 'error')
            return redirect(url_for('index'))

        recs_list = recommendations.head(10).to_dict('records')
        return render_template('result.html', recommendations=recs_list, username=current_user.username)
    except Exception as e:
        flash(f'An error occurred: {e}', 'error')
        return redirect(url_for('index'))

# --- Run App ---
if __name__ == "__main__":
    with app.app_context():
        instance_path = os.path.join(basedir, 'instance')
        if not os.path.exists(instance_path):
            os.makedirs(instance_path)
        db.create_all()
    app.run(debug=True, port=5000)