import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.ops import batched_nms
import numpy as np
from PIL import Image
import dlib


class SimpleFaceNet(torch.nn.Module):
    """Basit yüz tanıma ağı - ResNet benzeri"""

    def __init__(self):
        super(SimpleFaceNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, padding=1)

        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = torch.nn.Linear(256 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)


def detect_faces_enhanced(image, device, min_face_size=60, use_dlib=True):
    """Geliştirilmiş yüz algılama - Haar Cascade + Dlib kombinasyonu"""
    face_locations = []

    # OpenCV Haar Cascade yüz algılayıcı
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Dlib detector (daha hassas)
    if use_dlib:
        try:
            detector = dlib.get_frontal_face_detector()
            dlib_faces = detector(gray)

            for face in dlib_faces:
                # dlib rectangle'ı (left, top, right, bottom) formatına çevir
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()

                # Minimum boyut kontrolü
                if (right - left) >= min_face_size and (bottom - top) >= min_face_size:
                    face_locations.append((top, right, bottom, left))

            print(f"Dlib ile {len(face_locations)} yüz bulundu")

        except Exception as e:
            print(f"Dlib kullanılamadı: {e}, OpenCV Haar Cascade kullanılıyor")
            use_dlib = False

    # Dlib bulunamazsa veya yüz bulunamazsa Haar Cascade kullan
    if not use_dlib or len(face_locations) == 0:
        # Çoklu ölçek algılama için farklı parametreler dene
        scale_factors = [1.05, 1.1, 1.2]
        min_neighbors_list = [3, 5, 7]

        all_faces = []

        for scale_factor in scale_factors:
            for min_neighbors in min_neighbors_list:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_face_size, min_face_size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                # (x, y, w, h) formatından (left, top, right, bottom) formatına çevir
                for (x, y, w, h) in faces:
                    all_faces.append((y, x + w, y + h, x))  # (top, right, bottom, left)

        # Çakışan yüzleri filtrele (Non-Maximum Suppression benzeri)
        if all_faces:
            face_locations = filter_overlapping_faces(all_faces)

        print(f"Haar Cascade ile {len(face_locations)} yüz bulundu")

    return face_locations


def filter_overlapping_faces(face_locations, overlap_threshold=0.3):
    """Çakışan yüzleri filtrele"""
    if len(face_locations) <= 1:
        return face_locations

    # IoU (Intersection over Union) hesapla
    def calculate_iou(box1, box2):
        top1, right1, bottom1, left1 = box1
        top2, right2, bottom2, left2 = box2

        # Kesişim alanı
        inter_left = max(left1, left2)
        inter_top = max(top1, top2)
        inter_right = min(right1, right2)
        inter_bottom = min(bottom1, bottom2)

        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0

        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        # Birleşim alanı
        area1 = (right1 - left1) * (bottom1 - top1)
        area2 = (right2 - left2) * (bottom2 - top2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    # Yüz boyutuna göre sırala (büyükten küçüğe)
    faces_with_area = []
    for face in face_locations:
        top, right, bottom, left = face
        area = (right - left) * (bottom - top)
        faces_with_area.append((face, area))

    faces_with_area.sort(key=lambda x: x[1], reverse=True)

    filtered_faces = []
    for i, (face1, area1) in enumerate(faces_with_area):
        is_overlapping = False

        for face2, area2 in filtered_faces:
            iou = calculate_iou(face1, face2)
            if iou > overlap_threshold:
                is_overlapping = True
                break

        if not is_overlapping:
            filtered_faces.append((face1, area1))

    return [face for face, area in filtered_faces]


def calculate_face_quality(face_region, landmarks=None):
    """Yüz kalitesini değerlendir"""
    if face_region.size == 0:
        return 0.0

    quality_score = 0.0

    # 1. Görüntü keskinliği (Laplacian variance)
    if len(face_region.shape) == 3:
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    else:
        gray_face = face_region

    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize
    quality_score += sharpness_score * 0.4

    # 2. Yüz boyutu (büyük yüzler daha iyi)
    face_area = face_region.shape[0] * face_region.shape[1]
    size_score = min(face_area / (200 * 200), 1.0)  # 200x200 referans
    quality_score += size_score * 0.3

    # 3. Parlaklık analizi
    mean_brightness = np.mean(gray_face)
    # İdeal parlaklık 100-180 arası
    if 100 <= mean_brightness <= 180:
        brightness_score = 1.0
    else:
        brightness_score = max(0.0, 1.0 - abs(mean_brightness - 140) / 140)
    quality_score += brightness_score * 0.3

    return quality_score


def safe_get_video_property(cap, prop):
    """Video özelliklerini güvenle al"""
    try:
        value = cap.get(prop)
        if value is None or np.isnan(value) or np.isinf(value):
            return None
        return int(value) if prop in [cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH,
                                      cv2.CAP_PROP_FRAME_HEIGHT] else value
    except Exception as e:
        print(f"Video özelliği alınırken hata: {e}")
        return None


def count_frames_manually(cap):
    """Frame sayısını manuel olarak say"""
    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    return frame_count


def extract_distinct_faces(video_path, output_folder, max_frames=50, tolerance=0.6,
                           min_face_size=60, quality_threshold=0.3, progress_status=None):
    """Optimize edilmiş yüz çıkarma fonksiyonu"""
    os.makedirs(output_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    face_net = SimpleFaceNet().to(device)
    face_net.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video açılamadı: {video_path}")

    # Video özelliklerini güvenli şekilde al
    total_frames = safe_get_video_property(cap, cv2.CAP_PROP_FRAME_COUNT)

    if total_frames is None or total_frames <= 0:
        print("Frame sayısı okunamadı, manuel sayım yapılıyor...")
        total_frames = count_frames_manually(cap)
        print(f"Toplam frame sayısı: {total_frames}")
        # Başa dön
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # max_frames parametresi varsa kullan, yoksa tüm videoyu işle
    if max_frames is None or max_frames <= 0:
        max_frames = total_frames
    else:
        max_frames = min(max_frames, total_frames)

    frame_skip = max(1, total_frames // 100)  # En fazla 100 frame işle
    if frame_skip < 30:
        frame_skip = 30  # En az 30 frame atlama

    print(f"Toplam {total_frames} frame, {frame_skip} aralıkla işlenecek, maksimum {max_frames} frame")

    known_encodings, face_qualities, face_map = [], [], {}
    face_id = frame_count = processed_frames = 0

    try:
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                face_locations = detect_faces_enhanced(rgb_frame, device, min_face_size)

                for loc in face_locations:
                    try:
                        top, right, bottom, left = map(int, loc)
                        top, left = max(0, top), max(0, left)
                        bottom, right = min(h, bottom), min(w, right)

                        # Geçerli koordinatları kontrol et
                        if top >= bottom or left >= right:
                            continue

                        face_region = rgb_frame[top:bottom, left:right]

                        if face_region.size == 0 or face_region.shape[0] < 10 or face_region.shape[1] < 10:
                            continue

                        quality = calculate_face_quality(face_region)
                        if quality < quality_threshold:
                            continue

                        face_pil = Image.fromarray(face_region)
                        face_tensor = transform(face_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            encoding = face_net(face_tensor).cpu().numpy().flatten()

                        similarity_scores = []
                        for ke in known_encodings:
                            try:
                                sim = np.dot(encoding, ke) / (np.linalg.norm(encoding) * np.linalg.norm(ke) + 1e-6)
                                similarity_scores.append(sim)
                            except Exception as e:
                                similarity_scores.append(0.0)

                        if similarity_scores:
                            max_sim = max(similarity_scores)
                            match_idx = np.argmax(similarity_scores)
                        else:
                            max_sim = -1
                            match_idx = -1

                        # Benzerlik eşiği - tolerance parametresini kullan
                        matched = max_sim > tolerance

                        def save_face(face_img, fid):
                            try:
                                path = os.path.join(output_folder, f"{fid}.jpg")
                                # BGR formatına çevir (OpenCV için)
                                face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                                if cv2.imwrite(path, face_bgr):
                                    face_map[str(fid)] = (frame_count, left, top, right - left, bottom - top)
                                    return True
                                return False
                            except Exception as e:
                                print(f"Yüz kaydetme hatası: {e}")
                                return False

                        print(
                            f"Frame {frame_count}: Max sim: {max_sim:.3f}, matched: {matched}, quality: {quality:.3f}")

                        if matched and match_idx < len(face_qualities):
                            # Daha iyi kalitedeyse güncelle
                            if quality > face_qualities[match_idx]:
                                known_encodings[match_idx] = encoding
                                face_qualities[match_idx] = quality
                                # Eski dosyayı güncelle
                                if save_face(face_region, match_idx):
                                    print(f"Yüz güncellendi: {match_idx}.jpg (kalite: {quality:.3f})")
                        else:
                            # Yeni yüz
                            if save_face(face_region, face_id):
                                print(f"Yeni yüz kaydedildi: {face_id}.jpg (kalite: {quality:.3f})")
                                known_encodings.append(encoding)
                                face_qualities.append(quality)
                                face_id += 1

                    except Exception as e:
                        print(f"Yüz işleme hatası: {e}")
                        continue

            except Exception as e:
                print(f"Frame {frame_count} işleme hatası: {e}")

            frame_count += 1
            processed_frames += 1

            # İlerleme güncelle
            if progress_status is not None:
                progress_status['progress'] = int((processed_frames / max_frames) * 100)

            # Her 10 frame'de bir rapor
            if processed_frames % 10 == 0:
                print(f"İşlenen frame: {processed_frames}/{max_frames} ({processed_frames / max_frames * 100:.1f}%)")

    except Exception as e:
        print(f"Genel hata: {e}")
    finally:
        cap.release()

    if progress_status is not None:
        progress_status['status'] = 'done_extract'
        progress_status['progress'] = 100

    print(f"\nİşlem tamamlandı. {face_id} farklı yüz bulundu.")
    if face_qualities:
        print(f"Ortalama kalite: {np.mean(face_qualities):.3f}")
        print(f"En yüksek kalite: {np.max(face_qualities):.3f}")
        print(f"En düşük kalite: {np.min(face_qualities):.3f}")
    else:
        print("Yüz bulunamadı.")

    return face_map