import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
_SVM_PATH = os.path.join(_MODEL_DIR, 'face_svm.pkl')
_SCALER_PATH = os.path.join(_MODEL_DIR, 'scaler.pkl')
_CLASSES_PATH = os.path.join(_MODEL_DIR, 'classes.npy')
_CENTROIDS_PATH = os.path.join(_MODEL_DIR, 'centroids.npz')


def _get_pipeline():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
    )


def _compute_centroids_and_thresholds(X_scaled, y):
    unique = np.unique(y)
    centroids = {}
    thresholds = {}
    for c in unique:
        mask = y == c
        emb = X_scaled[mask]
        centroid = np.mean(emb, axis=0)
        centroids[c] = centroid
        dists = np.linalg.norm(emb - centroid, axis=1)
        thresholds[c] = float(np.mean(dists) + 2.5 * np.std(dists))
    return centroids, thresholds


def train(X, y, test_size=0.2):
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    pipeline = _get_pipeline()
    pipeline.fit(X_train, y_train)
    scaler = pipeline.named_steps['standardscaler']
    centroids, dist_thresholds = _compute_centroids_and_thresholds(
        scaler.transform(X_train), y_train
    )
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    return pipeline, (train_acc, test_acc), (X_test, y_test), centroids, dist_thresholds


def save_model(pipeline, classes, centroids, dist_thresholds):
    os.makedirs(_MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, _SVM_PATH)
    np.save(_CLASSES_PATH, np.array(classes))
    scaler = pipeline.named_steps['standardscaler']
    joblib.dump(scaler, _SCALER_PATH)
    c_keys = list(centroids.keys())
    c_vals = np.array([centroids[k] for k in c_keys])
    t_vals = np.array([dist_thresholds[k] for k in c_keys])
    np.savez(_CENTROIDS_PATH, keys=c_keys, centroids=c_vals, thresholds=t_vals)


def load_model():
    if not os.path.exists(_SVM_PATH):
        return None, None, None, None
    pipeline = joblib.load(_SVM_PATH)
    classes = np.load(_CLASSES_PATH, allow_pickle=True)
    if not os.path.exists(_CENTROIDS_PATH):
        return pipeline, classes, None, None
    data = np.load(_CENTROIDS_PATH, allow_pickle=True)
    centroids = {k: v for k, v in zip(data['keys'], data['centroids'])}
    thresholds = {k: v for k, v in zip(data['keys'], data['thresholds'])}
    return pipeline, classes, centroids, thresholds


def predict(pipeline, classes, embedding, centroids, dist_thresholds,
            prob_threshold=0.6, dist_factor=1.0):
    if centroids is not None and dist_thresholds is not None:
        scaler = pipeline.named_steps['standardscaler']
        emb_scaled = scaler.transform([embedding])[0]
        min_rel_dist = float('inf')
        nearest_class = None
        for c in classes:
            if c not in centroids or c not in dist_thresholds:
                continue
            centroid = centroids[c]
            thresh = dist_thresholds[c]
            if thresh <= 0:
                continue
            dist = float(np.linalg.norm(emb_scaled - centroid))
            rel_dist = dist / thresh
            if rel_dist < min_rel_dist:
                min_rel_dist = rel_dist
                nearest_class = c
        if min_rel_dist > dist_factor:
            adjusted_conf = max(0.0, 1.0 - (min_rel_dist - dist_factor))
            return 'Unknown', adjusted_conf
    svm_probs = pipeline.predict_proba([embedding])[0]
    max_prob = float(np.max(svm_probs))
    if max_prob < prob_threshold:
        return 'Unknown', max_prob
    idx = int(np.argmax(svm_probs))
    if centroids is not None and dist_thresholds is not None and nearest_class is not None:
        predicted_class = classes[idx]
        if min_rel_dist > 0.8:
            adjusted = max_prob * (1.0 - (min_rel_dist - 0.8) * 2.0)
            if adjusted < prob_threshold:
                return 'Unknown', max(0.0, adjusted)
            return predicted_class, adjusted
    return classes[idx], max_prob
