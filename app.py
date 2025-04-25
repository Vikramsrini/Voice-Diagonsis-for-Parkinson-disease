from flask import Flask, request, jsonify
import numpy as np
import parselmouth
import librosa
import soundfile as sf
import tempfile
import os
import joblib
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('xgb_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

try:
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Scaler loading failed: {e}")
    scaler = None

@app.route('/audio_predict', methods=['POST'])
def audio_predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_file.stream.seek(0, os.SEEK_END)
        file_size = audio_file.stream.tell()
        audio_file.stream.seek(0)
        print(f"Received audio: {audio_file.filename}, size: {file_size} bytes")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            audio_path = tmp.name
        
        # Validate audio
        try:
            data, sr = sf.read(audio_path)
            print(f"Audio loaded: {len(data)} samples, sample rate: {sr}")
        except Exception as e:
            os.unlink(audio_path)
            return jsonify({'error': f"Invalid audio file: {str(e)}"}), 400
        
        # Extract features
        features = extract_voice_features(audio_path)
        print(f"Features extracted: {features}")
        
        # Clean up
        os.unlink(audio_path)
        
        # Prepare feature array
        feature_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]
        feature_array = np.array([[features[name] for name in feature_names]])
        print(f"Feature array shape: {feature_array.shape}")
        
        # Scale features if scaler is available
        if scaler:
            try:
                feature_array = scaler.transform(feature_array)
                print("Features scaled successfully")
            except Exception as e:
                print(f"Scaling error: {e}")
        
        # Make prediction
        prediction = "Sample Prediction"
        probability = None
        if model:
            try:
                # Assume binary classification (0=Negative, 1=Positive)
                prediction_prob = model.predict_proba(feature_array)[0]
                prediction = "Positive" if model.predict(feature_array)[0] == 1 else "Negative"
                probability = prediction_prob[1]  # Probability of Positive class
                print(f"Prediction: {prediction}, Probability: {probability}")
            except Exception as e:
                print(f"Prediction error: {e}")
        
        return jsonify({
            'status': 'success',
            'prediction': str(prediction),
            'probability': float(probability) if probability is not None else None,
            'features': features
        })
        
    except Exception as e:
        print("Error in audio_predict:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def extract_voice_features(audio_path):
    try:
        sound = parselmouth.Sound(audio_path)
        print("Sound loaded")
        pitch = sound.to_pitch()
        print("Pitch computed")
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        print("Pulses computed")
        
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values != 0]
        print(f"F0 values: {len(f0_values)}")
        if len(f0_values) == 0:
            f0_values = np.array([150.0])
            print("Warning: No voiced frames, using default pitch")
        
        jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) or 0
        jitter_abs = parselmouth.praat.call(pulses, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) or 0
        jitter_rap = parselmouth.praat.call(pulses, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) or 0
        jitter_ppq5 = parselmouth.praat.call(pulses, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) or 0
        jitter_ddp = parselmouth.praat.call(pulses, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) or 0
        
        shimmer_local = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        shimmer_local_db = parselmouth.praat.call([sound, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        shimmer_apq3 = parselmouth.praat.call([sound, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        shimmer_apq5 = parselmouth.praat.call([sound, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        shimmer_apq11 = parselmouth.praat.call([sound, pulses], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        shimmer_dda = parselmouth.praat.call([sound, pulses], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0) or 0
        nhr = 1/hnr if hnr > 0 else 0
        
        y, sr = librosa.load(audio_path, sr=None)
        print(f"Librosa loaded: {len(y)} samples, sr={sr}")
        
        # Downsample or truncate signal to avoid huge computations
        max_samples = 44100 * 5  # Max 5 seconds at 44.1 kHz
        if len(y) > max_samples:
            y = y[:max_samples]
            print(f"Truncated signal to {len(y)} samples")
        
        rpde = compute_rpde(y) if len(y) > 100 else 0.0
        dfa = compute_dfa(y) if len(y) > 100 else 0.0
        ppe = compute_ppe(f0_values) if len(f0_values) > 1 else 0.0
        spread1, spread2, d2 = compute_nonlinear_measures(y) if len(y) > 100 else (0.0, 0.0, 0.0)
        
        return {
            'MDVP:Fo(Hz)': float(np.mean(f0_values)),
            'MDVP:Fhi(Hz)': float(np.max(f0_values)),
            'MDVP:Flo(Hz)': float(np.min(f0_values)),
            'MDVP:Jitter(%)': float(jitter_local * 100),
            'MDVP:Jitter(Abs)': float(jitter_abs),
            'MDVP:RAP': float(jitter_rap),
            'MDVP:PPQ': float(jitter_ppq5),
            'Jitter:DDP': float(jitter_ddp),
            'MDVP:Shimmer': float(shimmer_local),
            'MDVP:Shimmer(dB)': float(shimmer_local_db),
            'Shimmer:APQ3': float(shimmer_apq3),
            'Shimmer:APQ5': float(shimmer_apq5),
            'MDVP:APQ': float(shimmer_apq11),
            'Shimmer:DDA': float(shimmer_dda),
            'NHR': float(nhr),
            'HNR': float(hnr),
            'RPDE': float(rpde),
            'DFA': float(dfa),
            'spread1': float(spread1),
            'spread2': float(spread2),
            'D2': float(d2),
            'PPE': float(ppe)
        }
    except Exception as e:
        print(f"Error in extract_voice_features: {e}")
        raise

def compute_rpde(signal, emb_dim=10, tau=3, eps=0.1):
    try:
        if len(signal) < emb_dim * tau + 1:
            print("Signal too short for RPDE")
            return 0.0
        
        # Limit embedding size
        max_M = 10000  # Cap number of vectors
        N = len(signal)
        M = min(N - (emb_dim - 1) * tau, max_M)
        print(f"RPDE: N={N}, M={M}")
        
        embedded = np.zeros((M, emb_dim))
        for i in range(emb_dim):
            embedded[:, i] = signal[i * tau : i * tau + M]
        
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import entropy
        
        # Compute distances efficiently
        dist_matrix = squareform(pdist(embedded, 'euclidean'))
        recurrence = (dist_matrix < eps).astype(int)
        
        diag_lines = []
        for i in range(1, M):
            diag = np.diag(recurrence, i)
            changes = np.where(np.diff(diag) != 0)[0] + 1
            segments = np.split(diag, changes)
            diag_lines.extend([len(s) for s in segments if s[0] == 1])
        
        if not diag_lines:
            return 0
        
        hist = np.histogram(diag_lines, bins=range(1, max(diag_lines)+2))[0]
        prob = hist / hist.sum()
        return entropy(prob)
    except Exception as e:
        print(f"RPDE error: {e}")
        return 0.0

def compute_dfa(signal):
    try:
        if len(signal) < 100:
            return 0.0
        N = len(signal)
        n_values = np.logspace(np.log10(4), np.log10(N//10), 10).astype(int)
        n_values = np.unique(n_values)
        
        F = []
        for n in n_values:
            blocks = np.array_split(signal, N // n)
            rms = []
            for block in blocks:
                y = np.cumsum(block - np.mean(block))
                x = np.arange(len(y))
                coef = np.polyfit(x, y, 1)
                trend = np.polyval(coef, x)
                y_detrended = y - trend
                rms.append(np.sqrt(np.mean(y_detrended**2)))
            F.append(np.mean(rms))
        
        if len(F) < 2:
            return 0
        
        coef = np.polyfit(np.log(n_values), np.log(F), 1)
        return coef[0]
    except Exception as e:
        print(f"DFA error: {e}")
        return 0.0

def compute_ppe(f0_values):
    try:
        if len(f0_values) < 2:
            return 0
        from scipy.stats import entropy
        periods = 1 / f0_values
        diffs = np.abs(np.diff(periods))
        diffs_norm = diffs / np.mean(diffs)
        hist = np.histogram(diffs_norm, bins=10)[0]
        prob = hist / hist.sum()
        return entropy(prob)
    except Exception as e:
        print(f"PPE error: {e}")
        return 0.0

def compute_nonlinear_measures(signal):
    try:
        if len(signal) < 100:
            return 0.0, 0.0, 0.0
        
        # Compute spread1 and spread2
        embedding = librosa.feature.stack_memory(signal, n_steps=10)
        U, s, V = np.linalg.svd(embedding, full_matrices=False)
        spread1 = s[0] / s[1] if s[1] > 0 else 0
        spread2 = s[1] / s[2] if len(s) > 2 and s[2] > 0 else 0
        
        # Compute D2 with limited embedding
        max_M = 10000  # Cap number of vectors
        emb_dim = 10
        tau = 3
        N = len(signal)
        M = min(N - (emb_dim - 1) * tau, max_M)
        print(f"Nonlinear: N={N}, M={M}")
        
        embedded = np.zeros((M, emb_dim))
        for i in range(emb_dim):
            embedded[:, i] = signal[i * tau : i * tau + M]
        
        from scipy.spatial.distance import pdist
        distances = pdist(embedded, 'euclidean')
        r = np.mean(distances)
        C_r = np.sum(distances < r) / len(distances)
        D2 = np.log(C_r) / np.log(r) if r > 0 and C_r > 0 else 0
        
        return spread1, spread2, D2
    except Exception as e:
        print(f"Nonlinear measures error: {e}")
        return 0.0, 0.0, 0.0

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)