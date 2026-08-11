"""Quick test to verify ECAPA model loading."""
import sys
sys.path.insert(0, 'src')
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from ecapa import ECAPAModel

ecapa = ECAPAModel('./ecapa_model', 'cpu')
print('check_model_exists:', ecapa.check_model_exists())

ecapa.load()
print('Model loaded:', ecapa.loaded)
print('Model info:', ecapa.model_info)

# Test with a small audio
audio = np.random.randn(16000 * 3).astype(np.float32)
print('Testing encode_batch...')
embs = ecapa.encode_batch([audio, audio])
print('Embedding shape:', embs.shape)
sim = ecapa.compute_similarity(embs[0], embs[1])
print('Self-similarity:', sim)
print('ECAPA TEST PASSED!')
