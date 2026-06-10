import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load dataset (ganti nama file jika perlu)
# Jika kamu rename file menjadi dataset.csv, biarkan baris ini.
# Jika tetap pakai spotify-tracks.csv, ubah jadi "spotify-tracks.csv"
df = pd.read_csv("spotify-tracks.csv")

# --- Penyesuaian untuk dataset asli ---
# Kolom artist di file asli b ernama 'artists', bukan 'artist'
# Maka kita rename kolom 'artists' menjadi 'artist' agar kode tetap berjalan
if 'artists' in df.columns:
    df = df.rename(columns={'artists': 'artist'})

# Hapus baris yang memiliki nilai kosong pada fitur yang diperlukan
features = ['tempo', 'energy', 'danceability', 'valence']
df = df.dropna(subset=features + ['track_name', 'artist'])

# Normalisasi fitur
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

# Hitung similarity matrix
sim_matrix = cosine_similarity(scaled)

# Fungsi rekomendasi
def recommend(song_name, top_n=5):
    idx = df[df['track_name'] == song_name].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended_idx = [i[0] for i in sorted_scores[1:top_n+1]]
    return df.iloc[recommended_idx][['track_name', 'artist']]

# UI Streamlit
st.title("🎵 Sistem Rekomendasi Musik")
st.write("Pilih lagu favorit, dapatkan rekomendasi lagu mirip.")

selected = st.selectbox("Lagu favorit:", df['track_name'].values)

if st.button("Rekomendasikan"):
    recs = recommend(selected)
    st.write("### Rekomendasi untukmu:")
    for _, row in recs.iterrows():
        st.write(f"🎧 **{row['track_name']}** — {row['artist']}")