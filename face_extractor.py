import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.ops import nms
import numpy as np
from PIL import Image
import dlib
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1
import face_recognition
from collections import defaultdict
import threading
import time


class AdvancedFaceNet(nn.Module):
    """Gelişmiş yüz tanıma ağı - EfficientNet benzeri yapı"""

    def __init__(self, embedding_dim=512):
        super(AdvancedFaceNet, self).__init__()

        # Depthwise Separable Convolutions
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(3, 32, 3, 2),  # 224->112
            self._make_conv_block(32, 64, 3, 2),  # 112->56
            self._make_conv_block(64, 128, 3, 2),  # 56->28
            self._make_conv_block(128, 256, 3, 2),  # 28->14
            self._make_conv_block(256, 512, 3, 2),  # 14->7
        ])

        # Attention mechanism
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)

        # Global pooling ve fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                      kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        # Feature extraction
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Final embedding
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn(x)

        return F.normalize(x, p=2, dim=1)


class MultiFaceDetector:
    """Çoklu yüz algılama sistemi - MTCNN, MediaPipe, RetinaFace"""

    def __init__(self, device='cpu'):
        self.device = device
        self.detectors = {}

        # MTCNN (en hassas)
        try:
            self.detectors['mtcnn'] = MTCNN(
                image_size=160, margin=0, min_face_size=40,
                thresholds=[0.6, 0.7, 0.7], factor=0.709,
                post_process=False, device=device
            )
            print("✓ MTCNN yüklendi")
        except Exception as e:
            print(f"✗ MTCNN yüklenemedi: {e}")

        # MediaPipe (hızlı)
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.detectors['mediapipe'] = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            print("✓ MediaPipe yüklendi")
        except Exception as e:
            print(f"✗ MediaPipe yüklenemedi: {e}")

        # Dlib (geleneksel ama güvenilir)
        try:
            self.detectors['dlib'] = dlib.get_frontal_face_detector()
            print("✓ Dlib yüklendi")
        except Exception as e:
            print(f"✗ Dlib yüklenemedi: {e}")

        # OpenCV Haar Cascade (fallback)
        try:
            self.detectors['opencv'] = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("✓ OpenCV Haar Cascade yüklendi")
        except Exception as e:
            print(f"✗ OpenCV yüklenemedi: {e}")

    def detect_faces(self, image, min_confidence=0.5, min_size=60):
        """Tüm detektörlerden sonuçları birleştir"""
        all_detections = []
        h, w = image.shape[:2]

        # MTCNN
        if 'mtcnn' in self.detectors:
            try:
                pil_image = Image.fromarray(image)
                boxes, probs = self.detectors['mtcnn'].detect(pil_image)

                if boxes is not None:
                    for box, prob in zip(boxes, probs):
                        if prob > min_confidence:
                            x1, y1, x2, y2 = box.astype(int)
                            if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                                all_detections.append({
                                    'bbox': (max(0, y1), min(w, x2), min(h, y2), max(0, x1)),
                                    'confidence': prob,
                                    'detector': 'mtcnn'
                                })
            except Exception as e:
                print(f"MTCNN hata: {e}")

        # MediaPipe
        if 'mediapipe' in self.detectors:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
                results = self.detectors['mediapipe'].process(rgb_image)

                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box

                        x1 = int(bbox.xmin * w)
                        y1 = int(bbox.ymin * h)
                        x2 = int((bbox.xmin + bbox.width) * w)
                        y2 = int((bbox.ymin + bbox.height) * h)

                        if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                            all_detections.append({
                                'bbox': (y1, x2, y2, x1),
                                'confidence': detection.score[0],
                                'detector': 'mediapipe'
                            })
            except Exception as e:
                print(f"MediaPipe hata: {e}")

        # Dlib
        if 'dlib' in self.detectors:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                faces = self.detectors['dlib'](gray)

                for face in faces:
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                        all_detections.append({
                            'bbox': (y1, x2, y2, x1),
                            'confidence': 0.8,  # Dlib confidence vermez
                            'detector': 'dlib'
                        })
            except Exception as e:
                print(f"Dlib hata: {e}")

        # Non-Maximum Suppression uygula
        filtered_detections = self._apply_nms(all_detections, iou_threshold=0.6)

        return filtered_detections

    def _apply_nms(self, detections, iou_threshold=0.6):
        """Non-Maximum Suppression uygula"""
        if not detections:
            return []

        boxes = []
        scores = []

        for det in detections:
            top, right, bottom, left = det['bbox']
            boxes.append([left, top, right, bottom])
            scores.append(det['confidence'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)

        keep_indices = nms(boxes, scores, iou_threshold)

        return [detections[i] for i in keep_indices.tolist()]


class FaceQualityAssessment:
    """Gelişmiş yüz kalite değerlendirmesi"""

    @staticmethod
    def calculate_comprehensive_quality(face_region, landmarks=None):
        """Kapsamlı kalite değerlendirmesi"""
        if face_region.size == 0:
            return 0.0

        quality_metrics = {}

        # Gri tonlama dönüşümü
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray_face = face_region

        # 1. Keskinlik (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        quality_metrics['sharpness'] = min(laplacian_var / 1000.0, 1.0)

        # 2. Kontrast analizi
        contrast = gray_face.std()
        quality_metrics['contrast'] = min(contrast / 60.0, 1.0)

        # 3. Parlaklık uniformitesi
        mean_brightness = gray_face.mean()
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
        quality_metrics['brightness'] = max(0.0, brightness_score)

        # 4. Yüz boyutu
        face_area = face_region.shape[0] * face_region.shape[1]
        size_score = min(face_area / (160 * 160), 1.0)
        quality_metrics['size'] = size_score

        # 5. Simetri analizi
        if face_region.shape[1] > 20:  # Minimum genişlik kontrolü
            left_half = gray_face[:, :face_region.shape[1] // 2]
            right_half = gray_face[:, face_region.shape[1] // 2:]
            right_half_flipped = np.fliplr(right_half)

            # Boyut uyumlu hale getir
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]

            if left_half.shape == right_half_flipped.shape:
                symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float))) / 255.0
                quality_metrics['symmetry'] = max(0.0, symmetry)
            else:
                quality_metrics['symmetry'] = 0.5
        else:
            quality_metrics['symmetry'] = 0.5

        # 6. Gürültü analizi (Sobel gradient)
        sobel_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        noise_score = 1.0 - min(gradient_magnitude.std() / 50.0, 1.0)
        quality_metrics['noise'] = max(0.0, noise_score)

        # Ağırlıklı toplam
        weights = {
            'sharpness': 0.25,
            'contrast': 0.15,
            'brightness': 0.15,
            'size': 0.20,
            'symmetry': 0.15,
            'noise': 0.10
        }

        total_quality = sum(quality_metrics[metric] * weights[metric]
                            for metric in quality_metrics)

        return total_quality


class EnhancedFaceExtractor:
    """Gelişmiş yüz çıkarma sistemi"""

    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Kullanılan cihaz: {self.device}")

        # Model yükleme
        self.face_detector = MultiFaceDetector(self.device)
        self.quality_assessor = FaceQualityAssessment()

        # FaceNet modeli (pretrained)
        try:
            self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print("✓ FaceNet (InceptionResnet) yüklendi")
        except Exception as e:
            print(f"✗ FaceNet yüklenemedi, özel model kullanılacak: {e}")
            self.face_recognizer = AdvancedFaceNet().to(self.device)
            self.face_recognizer.eval()

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Clustering için DBSCAN
        self.clustering_threshold = 0.7

    def extract_face_encoding(self, face_region):
        """Yüz encoding'i çıkar"""
        try:
            face_pil = Image.fromarray(face_region)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                encoding = self.face_recognizer(face_tensor)
                return encoding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Encoding çıkarma hatası: {e}")
            return None

    def calculate_similarity_matrix(self, encodings):
        # Cosine similarity -1 ile 1 arasıdır, 0-1 aralığına normalize edelim
        sim_matrix = cosine_similarity(encodings)
        sim_matrix = (sim_matrix + 1) / 2
        return sim_matrix

    def cluster_faces_advanced(self, encodings, min_samples=1, eps=0.3):
        """Gelişmiş kümeleme algoritması"""
        if len(encodings) < 2:
            return list(range(len(encodings)))

        similarity_matrix = self.calculate_similarity_matrix(encodings)  # 0-1 arası
        distance_matrix = 1 - similarity_matrix
        distance_matrix = np.clip(distance_matrix, 0, 1)  # Negatif değerleri önle

        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clusterer.fit_predict(distance_matrix)

        return cluster_labels

    def process_video_optimized(self, video_path, output_folder,
                                max_frames=None, quality_threshold=0.4,
                                similarity_threshold=0.75, progress_callback=None):
        """Optimize edilmiş video işleme"""

        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Video açılamadı: {video_path}")

        # Video bilgileri
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if total_frames <= 0:
            total_frames = self._count_frames_manually(cap)

        print(f"Video bilgileri: {total_frames} frame, {fps:.2f} FPS")

        # Adaptive frame sampling
        if max_frames is None:
            max_frames = min(total_frames, 200)

        frame_step = max(1, total_frames // max_frames)

        # Veri yapıları
        all_encodings = []
        all_qualities = []
        all_face_data = []

        frame_count = 0
        processed_faces = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_step != 0:
                    frame_count += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Yüz algılama
                detections = self.face_detector.detect_faces(rgb_frame, min_confidence=0.6)

                for detection in detections:
                    top, right, bottom, left = detection['bbox']

                    # Güvenli sınır kontrolü
                    h, w = rgb_frame.shape[:2]
                    top, left = max(0, top), max(0, left)
                    bottom, right = min(h, bottom), min(w, right)

                    if top >= bottom or left >= right:
                        continue

                    face_region = rgb_frame[top:bottom, left:right]

                    if face_region.size == 0:
                        continue

                    # Kalite değerlendirmesi
                    quality = self.quality_assessor.calculate_comprehensive_quality(face_region)

                    if quality < quality_threshold:
                        continue

                    # Encoding çıkarma
                    encoding = self.extract_face_encoding(face_region)
                    if encoding is None:
                        continue

                    # Veri kaydetme
                    all_encodings.append(encoding)
                    all_qualities.append(quality)
                    all_face_data.append({
                        'face_region': face_region,
                        'frame_number': frame_count,
                        'bbox': (top, right, bottom, left),
                        'quality': quality,
                        'detector': detection['detector']
                    })

                    processed_faces += 1

                frame_count += 1

                # Progress callback
                if progress_callback:
                    progress = min(100, (frame_count / total_frames) * 100)
                    progress_callback(progress, processed_faces)

                if len(all_encodings) >= max_frames * 5:  # Çok fazla yüz bulunursa dur
                    break

        finally:
            cap.release()

        print(f"\n{processed_faces} yüz bulundu, kümeleme başlıyor...")

        # Kümeleme
        if not all_encodings:
            print("Hiç yüz bulunamadı!")
            return {}

        cluster_labels = self.cluster_faces_advanced(all_encodings, eps=1 - similarity_threshold)

        # En iyi yüzleri seç ve kaydet
        face_map = self._save_best_faces(cluster_labels, all_face_data, all_qualities, output_folder)

        return face_map

    def _count_frames_manually(self, cap):
        """Frame sayısını manuel say"""
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        count = 0
        while cap.read()[0]:
            count += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        return count

    def _save_best_faces(self, cluster_labels, face_data, qualities, output_folder):
        """Her küme için en iyi kaliteli yüzü kaydet"""
        face_map = {}
        unique_labels = set(cluster_labels)

        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in unique_labels:
            indices = [i for i, l in enumerate(cluster_labels) if l == label]
            if not indices:
                continue

            # Kalitesi en yüksek olanı seç
            best_idx = max(indices, key=lambda i: qualities[i])
            best_face = face_data[best_idx]

            filename = f"cluster_{label:03d}_frame{best_face['frame_number']}.jpg"
            filepath = os.path.join(output_folder, filename)

            face_bgr = cv2.cvtColor(best_face['face_region'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, face_bgr)

            face_map[label] = {
                'path': filepath,
                'quality': best_face['quality'],
                'frame_number': best_face['frame_number'],
                'bbox': best_face['bbox'],
                'detector': best_face['detector']
            }

        return face_map


# Ana fonksiyon - Geriye uyumluluk için
def extract_distinct_faces(video_path, output_folder, max_frames=100, tolerance=0.75,
                           min_face_size=60, quality_threshold=0.6, progress_status=None):
    """Gelişmiş yüz çıkarma - ana fonksiyon"""

    def progress_callback(progress, face_count):
        if progress_status:
            progress_status['progress'] = int(progress)
            progress_status['face_count'] = face_count

    try:
        extractor = EnhancedFaceExtractor()

        # Tolerance'ı similarity threshold'a çevir
        similarity_threshold = tolerance

        face_map = extractor.process_video_optimized(
            video_path=video_path,
            output_folder=output_folder,
            max_frames=max_frames,
            quality_threshold=quality_threshold,
            similarity_threshold=similarity_threshold,
            progress_callback=progress_callback
        )

        if progress_status:
            progress_status['status'] = 'done_extract'
            progress_status['progress'] = 100

        print(f"\n✅ İşlem tamamlandı! {len(face_map)} farklı yüz kaydedildi.")

        return face_map

    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        if progress_status:
            progress_status['status'] = 'error'
            progress_status['error'] = str(e)
        return {}


