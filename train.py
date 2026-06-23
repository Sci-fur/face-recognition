from engine import FaceRecognitionEngine

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = FaceRecognitionEngine()
    return _engine


def _progress(step, pct):
    print(f'[{pct}%] {step}')


def train_model():
    engine = _get_engine()
    info = engine.get_dataset_info()
    if not info['persons']:
        print('Error: dataset/ directory not found. Collect data first.')
        return
    if len(info['persons']) < 2:
        print('Error: Need at least 2 persons in dataset/ to train.')
        return
    print('Dataset:')
    for p in info['persons']:
        print(f'  {p["name"]}: {p["count"]} images')
    print(f'  Total: {info["total_images"]} images')
    result = engine.train(progress_callback=_progress)
    if result['success']:
        print(f'\nTraining accuracy:   {result["train_acc"]:.2%}')
        print(f'Validation accuracy: {result["val_acc"]:.2%}')
        print(f'Model saved with {result["n_persons"]} persons ({result["n_images"]} embeddings)')
    else:
        print(f'Error: {result["error"]}')
