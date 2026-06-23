import torch
from facenet_pytorch import InceptionResnetV1 as _InceptionResnetV1
import numpy as np

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

_resnet = _InceptionResnetV1(pretrained='vggface2').eval().to(_device)


def get_embedding(aligned_face_tensor):
    if aligned_face_tensor is None:
        return None
    if aligned_face_tensor.dim() == 3:
        aligned_face_tensor = aligned_face_tensor.unsqueeze(0)
    with torch.no_grad():
        aligned_face_tensor = aligned_face_tensor.to(_device)
        embeddings = _resnet(aligned_face_tensor)
    return embeddings[0].cpu().numpy()


def get_embeddings_batch(aligned_faces_tensor):
    if aligned_faces_tensor is None:
        return None
    with torch.no_grad():
        aligned_faces_tensor = aligned_faces_tensor.to(_device)
        embeddings = _resnet(aligned_faces_tensor)
    return embeddings.cpu().numpy()
