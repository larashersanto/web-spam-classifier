import streamlit as st
import joblib
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title='Classifier Demo', page_icon=':mag:')

# Load artefak dengan cache agar cepat
@st.cache_resource
def load_artefak():
    model = joblib.load('model.pkl')
    vec   = joblib.load('vectorizer.pkl')
    le    = joblib.load('label_encoder.pkl')
    with open('threshold.txt') as f:
        thr = float(f.read().strip())
    return model, vec, le, thr

# Jalankan fungsi loading
try:
    model, vec, le, threshold = load_artefak()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

st.title(':mag: Web Klasifikasi Teks')
st.write(f"Threshold: {threshold:.3f}")

teks_input = st.text_area('Masukkan teks yang ingin diklasifikasi:', height=150)

if st.button('Klasifikasikan'):
    if teks_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        try:
            # Prediksi
            X = vec.transform([teks_input])
            proba = model.predict_proba(X)[0, 1]
            pred = int(proba >= threshold)
            kelas_pred = le.classes_[pred]
            
            # Tampilkan hasil
            st.success(f'Prediksi: **{kelas_pred}**')
            st.write(f'Probabilitas: {proba:.4f}')
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")