"""
app.py  —  Streamlit UI for Hybrid AI Movie Recommendation System
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import heapq
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0d0d14;
    --surface: #16161f;
    --card: #1e1e2e;
    --accent: #e8c84a;
    --accent2: #c084fc;
    --text: #e2e8f0;
    --muted: #8b8fa8;
    --border: #2a2a3e;
    --green: #4ade80;
    --red: #f87171;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Header */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    background: linear-gradient(135deg, #0d0d14 0%, #1a0a2e 50%, #0d0d14 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.main-header p {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}

/* Movie Card */
.movie-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
    border-radius: 12px 0 0 12px;
}
.movie-card:hover {
    border-color: var(--accent);
}
.rank-badge {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--border);
    position: absolute;
    top: 0.8rem; right: 1.2rem;
    line-height: 1;
}
.movie-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.5rem;
    padding-right: 3rem;
}
.genre-tag {
    display: inline-block;
    background: rgba(232,200,74,0.12);
    color: var(--accent);
    border: 1px solid rgba(232,200,74,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.meta-row {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-top: 0.6rem;
}
.meta-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.meta-label {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.meta-value {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text);
}
.score-bar-wrap {
    margin-top: 0.9rem;
}
.score-bar-label {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
}
.score-bar-outer {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.score-bar-inner {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.why-tag {
    margin-top: 0.8rem;
    font-size: 0.78rem;
    color: var(--muted);
    font-style: italic;
}

/* Stats card */
.stat-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--accent);
}
.stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, var(--accent), #d4a017) !important;
    color: #0d0d14 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.5px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Inputs */
.stSelectbox label, .stSlider label, .stNumberInput label, .stMultiSelect label {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Section title */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
}

/* Alert boxes */
.info-box {
    background: rgba(192,132,252,0.08);
    border: 1px solid rgba(192,132,252,0.25);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    color: var(--accent2);
    font-size: 0.88rem;
    margin-bottom: 1rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# BACKEND CLASSES  (same logic as your original script)
# ─────────────────────────────────────────────────────────

class CSPModule:
    def __init__(self, df):
        self.df = df

    def filter(self, preferences):
        filtered = self.df.copy()
        if preferences.get('genres'):
            normalized = [g.strip().title() for g in preferences['genres']]
            preferences['genres'] = normalized
            filtered = filtered[filtered['genere'].isin(normalized)]
        if 'min_rating' in preferences:
            filtered = filtered[filtered['rating'] >= preferences['min_rating']]
        if 'year_min' in preferences:
            filtered = filtered[filtered['release_year'] >= preferences['year_min']]
        if 'year_max' in preferences:
            filtered = filtered[filtered['release_year'] <= preferences['year_max']]
        if 'runtime_min' in preferences:
            filtered = filtered[filtered['runtime'] >= preferences['runtime_min']]
        if 'runtime_max' in preferences:
            filtered = filtered[filtered['runtime'] <= preferences['runtime_max']]
        if len(filtered) == 0 and preferences.get('genres'):
            filtered = self.df[self.df['genere'].isin(preferences['genres'])].copy()
        return filtered


class HeuristicModule:
    def calculate_score(self, movie, preferences):
        score = 0
        if preferences.get('genres'):
            score += 40 if movie['genere'] in preferences['genres'] else -20
        rating = movie['rating']
        if rating >= 7.0:
            score += ((rating - 7.0) / 3.0) * 30
        else:
            score += (rating / 10.0) * 15
        pref_year = preferences.get('preferred_year', 2015)
        score += max(0, 15 - abs(movie['release_year'] - pref_year) * 1.5)
        pref_rt = preferences.get('preferred_runtime', 120)
        score += max(0, 10 - abs(movie['runtime'] - pref_rt) / 5)
        score += min(5, movie.get('vote_count', 1000) / 2000)
        return round(min(100, max(0, score)), 2)

    def rank(self, movies_df, preferences, top_n=50):
        movies_df = movies_df.copy()
        movies_df['heuristic_score'] = movies_df.apply(
            lambda row: self.calculate_score(row, preferences), axis=1)
        return movies_df.sort_values('heuristic_score', ascending=False).head(top_n)


class AStarSearch:
    def __init__(self, df, heuristic):
        self.df = df
        self.heuristic = heuristic

    def search(self, filtered_df, preferences, n=10):
        if filtered_df.empty:
            return []
        candidates = filtered_df.copy()
        candidates['a_star_score'] = candidates.apply(
            lambda row: self.heuristic.calculate_score(row, preferences), axis=1)
        if preferences.get('genres'):
            mask = candidates['genere'].isin(preferences['genres'])
            candidates.loc[mask, 'a_star_score'] += 10
        heap = []
        for idx, row in candidates.iterrows():
            heapq.heappush(heap, (-row['a_star_score'], idx, row['movie_id']))
        results, seen_ids = [], set()
        while heap and len(results) < n:
            neg_score, idx, movie_id = heapq.heappop(heap)
            if movie_id in seen_ids:
                continue
            seen_ids.add(movie_id)
            rows = self.df[self.df['movie_id'] == movie_id]
            if rows.empty:
                continue
            results.append({'movie': rows.iloc[0], 'score': -neg_score})
        return results


def predict_rating(movie, ann_model, scaler_model, reference_df):
    genre_columns = pd.get_dummies(reference_df['genere']).columns
    feature_vector = [1 if movie['genere'] == g else 0 for g in genre_columns]
    feature_vector += [movie['release_year'], movie['runtime']]
    X_single = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler_model.transform(X_single)
    return round(float(ann_model.predict(X_scaled)[0]), 1)


# ─────────────────────────────────────────────────────────
# DATA + MODEL LOADING  (cached so it only runs once)
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🎬 Loading dataset & training AI models…")
def load_everything():
    df = pd.read_csv('random_20000.csv')

    # ── Column normalization ──────────────────────────────
    # Lowercase all column names first to avoid case issues
    df.columns = df.columns.str.strip().str.lower()

    # Genre — try every possible name
    if 'genere' not in df.columns:
        for candidate in ['genres', 'genre', 'category', 'type']:
            if candidate in df.columns:
                df.rename(columns={candidate: 'genere'}, inplace=True)
                break
        else:
            # Last resort: use first string column
            str_cols = df.select_dtypes(include='object').columns.tolist()
            if str_cols:
                df.rename(columns={str_cols[0]: 'genere'}, inplace=True)
            else:
                df['genere'] = 'Unknown'

    # Release year
    if 'release_year' not in df.columns:
        if 'release_date' in df.columns:
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        elif 'year' in df.columns:
            df.rename(columns={'year': 'release_year'}, inplace=True)
        else:
            df['release_year'] = 2010

    # Runtime
    if 'runtime' not in df.columns:
        if 'run_time' in df.columns:
            df.rename(columns={'run_time': 'runtime'}, inplace=True)
        else:
            df['runtime'] = 120

    # Rating
    if 'rating' not in df.columns:
        if 'vote_average' in df.columns:
            df.rename(columns={'vote_average': 'rating'}, inplace=True)
        else:
            df['rating'] = 7.0

    # Title
    if 'title' not in df.columns:
        if 'name' in df.columns:
            df.rename(columns={'name': 'title'}, inplace=True)
        else:
            df['title'] = df.index.astype(str)

    # Movie ID
    if 'id' in df.columns and 'movie_id' not in df.columns:
        df.rename(columns={'id': 'movie_id'}, inplace=True)
    elif 'movie_id' not in df.columns:
        df['movie_id'] = range(len(df))

    # Vote count
    if 'vote_count' not in df.columns:
        df['vote_count'] = 1000

    # ── Cleaning ──────────────────────────────────────────
    # Fill missing genre instead of dropping — safer
    df['genere'] = df['genere'].fillna('Unknown')
    df = df.dropna(subset=['genere'])
    for col in ['rating', 'release_year', 'runtime']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['rating']       = df['rating'].fillna(df['rating'].median())
    df['release_year'] = df['release_year'].fillna(2010)
    df['runtime']      = df['runtime'].fillna(120)
    df = df[(df['rating'] > 0) & (df['release_year'] >= 1900) & (df['runtime'] > 0)]

    # ── Extract first clean genre from combined strings ───
    # Handles: "Action-Comedy-Crime", "Action,Comedy", "Action|Comedy"
    df['genere'] = df['genere'].astype(str).str.strip()
    if df['genere'].str.contains('-', na=False).mean() > 0.3:
        df['genere'] = df['genere'].str.split('-').str[0]
    if df['genere'].str.contains(',', na=False).mean() > 0.3:
        df['genere'] = df['genere'].str.split(',').str[0]
    if df['genere'].str.contains(r'\|', na=False).mean() > 0.3:
        df['genere'] = df['genere'].str.split('|').str[0]

    # Remove rows where genre looks like an ID or gibberish
    df = df[df['genere'].str.len().between(2, 30)]
    df = df[~df['genere'].str.match(r'^\d+$', na=False)]

    df['genere'] = df['genere'].str.strip().str.title()

    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['genere'])

    # ── K-Means ───────────────────────────────────────────
    genre_dummies = pd.get_dummies(df['genere'])
    cf = pd.concat([genre_dummies, df[['release_year', 'runtime']]], axis=1)
    scaler_km = StandardScaler()
    cf_scaled = scaler_km.fit_transform(cf)
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(cf_scaled)

    # ── ANN ───────────────────────────────────────────────
    genre_dummies_ann = pd.get_dummies(df['genere'])
    ann_feat = pd.concat([genre_dummies_ann, df[['release_year', 'runtime']]], axis=1)
    scaler_ann = StandardScaler()
    X = scaler_ann.fit_transform(ann_feat)
    y = df['rating'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ann = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu',
                       solver='adam', max_iter=500, random_state=42,
                       early_stopping=True, verbose=False)
    ann.fit(X_train, y_train)
    mse = mean_squared_error(y_test, ann.predict(X_test))
    r2  = r2_score(y_test, ann.predict(X_test))

    return df, km, ann, scaler_ann, mse, r2


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
try:
    df, kmeans_model, ann_model, ann_scaler, mse, r2 = load_everything()
    csp       = CSPModule(df)
    heuristic = HeuristicModule()
    search    = AStarSearch(df, heuristic)
    data_ok   = True
except FileNotFoundError:
    data_ok = False


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎬 AI Movie Recommender</h1>
    <p>Hybrid CSP · Heuristic · A* · Neural Network</p>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error("❌ **random_20000.csv** file nahi mila. Please CSV file same folder mein rakhein.")
    st.stop()


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Navigation")
    page = st.radio("", ["🔍 Search by Preferences", "🎥 Find Similar Movies",
                         "⭐ Top Rated", "🎲 Random Discovery", "📊 Statistics"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🤖 AI Model Info")
    st.markdown(f"""
    <div style='font-size:0.8rem; color:#8b8fa8; line-height:1.9'>
    <b style='color:#e2e8f0'>MSE</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{mse:.4f}<br>
    <b style='color:#e2e8f0'>R² Score</b> &nbsp;{r2:.4f}<br>
    <b style='color:#e2e8f0'>Movies</b> &nbsp;&nbsp;{len(df):,}<br>
    <b style='color:#e2e8f0'>Genres</b> &nbsp;&nbsp;{df['genere'].nunique()}<br>
    <b style='color:#e2e8f0'>Clusters</b> &nbsp;8 (K-Means)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HELPER: RENDER MOVIE CARD
# ─────────────────────────────────────────────────────────
def render_card(rank, rec, preferences=None):
    movie  = rec['movie']
    title  = str(movie.get('title', f"Movie #{movie.get('movie_id', rank)}"))
    genre  = movie['genere']
    rating = movie['rating']
    pred   = rec.get('predicted_rating', '—')
    year   = int(movie['release_year'])
    rt     = int(movie['runtime'])
    score  = rec['score']
    pct    = int(min(score, 100))

    genre_match = (preferences and preferences.get('genres') and genre in preferences['genres'])
    match_badge = (f'<span style="color:#4ade80;font-size:0.75rem">✔ Genre Match</span>'
                   if genre_match else '')

    reasons = []
    if genre_match:
        reasons.append(f"Matches '{genre}'")
    if rating >= 8.5:
        reasons.append("Highly rated 8.5+")
    if score >= 70:
        reasons.append("Close year & runtime fit")
    why_html = f'<div class="why-tag">💡 {" &nbsp;|&nbsp; ".join(reasons)}</div>' if reasons else ''

    st.markdown(f"""
    <div class="movie-card">
        <div class="rank-badge">#{rank}</div>
        <div class="movie-title">{title[:60]}</div>
        <span class="genre-tag">{genre}</span>
        &nbsp;&nbsp;{match_badge}
        <div class="meta-row">
            <div class="meta-item">
                <span class="meta-label">Actual Rating</span>
                <span class="meta-value">⭐ {rating}/10</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">AI Predicted</span>
                <span class="meta-value">🤖 {pred}/10</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Year</span>
                <span class="meta-value">📅 {year}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Runtime</span>
                <span class="meta-value">⏱ {rt} min</span>
            </div>
        </div>
        <div class="score-bar-wrap">
            <div class="score-bar-label">
                <span>Match Score</span>
                <span>{score:.1f} / 100</span>
            </div>
            <div class="score-bar-outer">
                <div class="score-bar-inner" style="width:{pct}%"></div>
            </div>
        </div>
        {why_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE 1: SEARCH BY PREFERENCES
# ─────────────────────────────────────────────────────────
if page == "🔍 Search by Preferences":
    st.markdown('<div class="section-title">Search by Preferences</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        available_genres = sorted(df['genere'].unique())
        genres = st.multiselect("Select Genre(s)", available_genres)
        min_rating = st.slider("Minimum Rating", 0.0, 10.0, 6.5, 0.1)
        year_min, year_max = st.select_slider(
            "Release Year Range",
            options=list(range(1950, 2025)),
            value=(2000, 2023)
        )

    with col2:
        runtime_min = st.number_input("Min Runtime (minutes)", 30, 300, 80, 5)
        runtime_max = st.number_input("Max Runtime (minutes)", 30, 300, 160, 5)
        preferred_year    = st.number_input("Ideal Release Year",    1950, 2024, 2015)
        preferred_runtime = st.number_input("Ideal Runtime (minutes)", 30, 300, 120, 5)
        n_results = st.slider("Number of Recommendations", 1, 20, 10)

    if st.button("🔍 Get Recommendations", use_container_width=True):
        preferences = {
            'min_rating'       : min_rating,
            'year_min'         : year_min,
            'year_max'         : year_max,
            'runtime_min'      : runtime_min,
            'runtime_max'      : runtime_max,
            'preferred_year'   : preferred_year,
            'preferred_runtime': preferred_runtime,
        }
        if genres:
            preferences['genres'] = genres

        with st.spinner("🤖 AI is finding your movies…"):
            filtered = csp.filter(preferences)
            if filtered.empty:
                st.error("No movies found. Try broadening your filters.")
            else:
                pool   = max(n_results * 5, 50)
                ranked = heuristic.rank(filtered, preferences, top_n=pool)
                recs   = search.search(ranked, preferences, n_results)
                for rec in recs:
                    rec['predicted_rating'] = predict_rating(rec['movie'], ann_model, ann_scaler, df)

                st.success(f"✅ Found {len(recs)} recommendations from {len(filtered):,} matching movies")
                st.markdown("---")
                for i, rec in enumerate(recs, 1):
                    render_card(i, rec, preferences)

                # CSV download
                if recs:
                    rows = []
                    for i, rec in enumerate(recs, 1):
                        m = rec['movie']
                        rows.append({
                            'Rank': i, 'Title': m.get('title',''), 'Genre': m['genere'],
                            'Rating': m['rating'], 'AI Predicted': rec['predicted_rating'],
                            'Year': int(m['release_year']), 'Runtime': int(m['runtime']),
                            'Score': round(rec['score'], 2)
                        })
                    csv = pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Results as CSV", csv,
                                       "recommendations.csv", "text/csv")


# ─────────────────────────────────────────────────────────
# PAGE 2: FIND SIMILAR MOVIES
# ─────────────────────────────────────────────────────────
elif page == "🎥 Find Similar Movies":
    st.markdown('<div class="section-title">Find Similar Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Ek movie ka naam type karein — AI aapko similar movies dhundh dega K-Means clustering ki madad se.</div>', unsafe_allow_html=True)

    movie_name = st.text_input("🎬 Movie Title", placeholder="e.g. Inception, The Dark Knight…")
    n_results  = st.slider("Number of Similar Movies", 1, 20, 10)

    if st.button("🔎 Find Similar", use_container_width=True) and movie_name:
        match = df[df['title'].str.lower() == movie_name.lower()]
        if match.empty:
            match = df[df['title'].str.lower().str.contains(movie_name.lower(), na=False)]

        if match.empty:
            st.error(f"Movie '{movie_name}' not found in dataset.")
        else:
            source = match.iloc[0]
            st.markdown(f"""
            <div class="movie-card" style="border-color:#e8c84a">
                <div class="movie-title">📽️ {source['title']}</div>
                <span class="genre-tag">{source['genere']}</span>
                <div class="meta-row" style="margin-top:0.6rem">
                    <div class="meta-item"><span class="meta-label">Rating</span><span class="meta-value">⭐ {source['rating']}/10</span></div>
                    <div class="meta-item"><span class="meta-label">Year</span><span class="meta-value">📅 {int(source['release_year'])}</span></div>
                    <div class="meta-item"><span class="meta-label">Runtime</span><span class="meta-value">⏱ {int(source['runtime'])} min</span></div>
                    <div class="meta-item"><span class="meta-label">Cluster</span><span class="meta-value">🔵 #{source['cluster']}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🤖 Finding similar movies…"):
                pool = df[(df['cluster'] == source['cluster']) & (df['movie_id'] != source['movie_id'])]
                if len(pool) < n_results * 2:
                    pool = df[(df['genere'] == source['genere']) & (df['movie_id'] != source['movie_id'])]

                prefs = {
                    'genres'           : [source['genere']],
                    'min_rating'       : max(5.0, source['rating'] - 1.5),
                    'preferred_year'   : source['release_year'],
                    'preferred_runtime': source['runtime'],
                }
                ranked = heuristic.rank(pool, prefs, top_n=max(n_results * 5, 50))
                recs   = search.search(ranked, prefs, n_results)
                for rec in recs:
                    rec['predicted_rating'] = predict_rating(rec['movie'], ann_model, ann_scaler, df)

            st.markdown(f"### 🎯 Top {len(recs)} Similar Movies")
            for i, rec in enumerate(recs, 1):
                render_card(i, rec, prefs)


# ─────────────────────────────────────────────────────────
# PAGE 3: TOP RATED
# ─────────────────────────────────────────────────────────
elif page == "⭐ Top Rated":
    st.markdown('<div class="section-title">Top Rated Movies</div>', unsafe_allow_html=True)
    n = st.slider("How many to show?", 5, 50, 20)
    genre_filter = st.selectbox("Filter by Genre (optional)", ["All"] + sorted(df['genere'].unique()))

    pool = df if genre_filter == "All" else df[df['genere'] == genre_filter]
    top  = pool.nlargest(n, 'rating')

    for rank, (_, movie) in enumerate(top.iterrows(), 1):
        rec = {'movie': movie, 'score': movie['rating'] * 10,
               'predicted_rating': predict_rating(movie, ann_model, ann_scaler, df)}
        render_card(rank, rec)


# ─────────────────────────────────────────────────────────
# PAGE 4: RANDOM DISCOVERY
# ─────────────────────────────────────────────────────────
elif page == "🎲 Random Discovery":
    st.markdown('<div class="section-title">Random Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">🎲 Har baar button dabaein — naye random movies milenge!</div>', unsafe_allow_html=True)

    n = st.slider("How many random movies?", 5, 20, 10)
    genre_filter = st.selectbox("Filter by Genre (optional)", ["All"] + sorted(df['genere'].unique()))

    if st.button("🎲 Discover!", use_container_width=True):
        pool   = df if genre_filter == "All" else df[df['genere'] == genre_filter]
        sample = pool.sample(n=min(n, len(pool)))
        for rank, (_, movie) in enumerate(sample.iterrows(), 1):
            rec = {'movie': movie, 'score': movie['rating'] * 10,
                   'predicted_rating': predict_rating(movie, ann_model, ann_scaler, df)}
            render_card(rank, rec)


# ─────────────────────────────────────────────────────────
# PAGE 5: STATISTICS
# ─────────────────────────────────────────────────────────
elif page == "📊 Statistics":
    st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, number, label in [
        (c1, f"{len(df):,}", "Total Movies"),
        (c2, str(df['genere'].nunique()), "Unique Genres"),
        (c3, f"{df['rating'].mean():.2f}", "Avg Rating"),
        (c4, f"{int(df['release_year'].max())}", "Latest Year"),
    ]:
        col.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{number}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎭 Genre Distribution")
    counts = df['genere'].value_counts().head(15).reset_index()
    counts.columns = ['Genre', 'Count']
    st.bar_chart(counts.set_index('Genre'), use_container_width=True, color="#e8c84a")

    st.markdown("#### ⭐ Rating Distribution")
    hist_data = pd.cut(df['rating'], bins=10).value_counts().sort_index()
    st.bar_chart(hist_data, use_container_width=True, color="#c084fc")

    st.markdown("#### 📅 Movies by Decade")
    df['decade'] = (df['release_year'] // 10 * 10).astype(int)
    decade_counts = df['decade'].value_counts().sort_index()
    st.bar_chart(decade_counts, use_container_width=True, color="#4ade80")
