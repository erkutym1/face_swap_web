import os
import uuid

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from moviepy import VideoFileClip, AudioFileClip
from scipy.spatial import Delaunay
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


def to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor


def to_numpy(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def get_enhanced_face_points(landmarks_68):
    """68 noktalı landmark'ları genişletilmiş yüz noktalarına çevir"""
    points = np.array([[p.x, p.y] for p in landmarks_68.parts()], dtype=np.float32)

    # Orijinal 68 nokta
    enhanced_points = points.copy()

    # Alın bölgesi için ek noktalar ekle (saç çizgisi modelleme)
    forehead_points = []

    # Yüz sınırları
    left_face = points[0:9]  # Sol yüz kenarı
    right_face = points[8:17]  # Sağ yüz kenarı

    # Alın genişliği (kaş arası mesafe * 1.3)
    eyebrow_left = points[17:22]  # Sol kaş
    eyebrow_right = points[22:27]  # Sağ kaş

    # Kaş merkez noktaları
    left_brow_center = np.mean(eyebrow_left, axis=0)
    right_brow_center = np.mean(eyebrow_right, axis=0)
    center_brow = (left_brow_center + right_brow_center) / 2

    # Alın yüksekliği hesapla (kaş-burun mesafesi * 0.8)
    nose_bridge_top = points[27]  # Burun köprüsü üst
    forehead_height = np.linalg.norm(center_brow - nose_bridge_top) * 0.8

    # Alın eğrisi için noktalar (saç çizgisi)
    forehead_width = np.linalg.norm(right_brow_center - left_brow_center) * 1.4

    # Saç çizgisi noktaları (9 nokta - daha detaylı)
    for i in range(9):
        ratio = i / 8.0
        # X koordinatı (soldan sağa)
        x = left_brow_center[0] + ratio * (right_brow_center[0] - left_brow_center[0])
        x += (ratio - 0.5) * forehead_width * 0.3  # Hafif eğri

        # Y koordinatı (saç çizgisi eğrisi)
        base_y = center_brow[1] - forehead_height
        curve_offset = forehead_height * 0.2 * np.sin(ratio * np.pi)  # Eğri
        y = base_y - curve_offset

        forehead_points.append([x, y])

    # Yan alın noktaları (şakak bölgesi)
    temple_points = []

    # Sol şakak
    left_temple_x = points[0][0] - (points[16][0] - points[0][0]) * 0.3
    left_temple_y = left_brow_center[1] - forehead_height * 0.5
    temple_points.append([left_temple_x, left_temple_y])

    # Sağ şakak
    right_temple_x = points[16][0] + (points[16][0] - points[0][0]) * 0.3
    right_temple_y = right_brow_center[1] - forehead_height * 0.5
    temple_points.append([right_temple_x, right_temple_y])

    # Üst şakak noktaları
    left_upper_temple_x = left_temple_x + forehead_width * 0.1
    left_upper_temple_y = left_temple_y - forehead_height * 0.3
    temple_points.append([left_upper_temple_x, left_upper_temple_y])

    right_upper_temple_x = right_temple_x - forehead_width * 0.1
    right_upper_temple_y = right_temple_y - forehead_height * 0.3
    temple_points.append([right_upper_temple_x, right_upper_temple_y])

    # Çene altı noktaları (daha detaylı çene hattı)
    chin_points = []
    chin_center = points[8]  # Çene merkezi
    jaw_left = points[4]  # Sol çene
    jaw_right = points[12]  # Sağ çene

    # Çene altı eğrisi (5 nokta)
    for i in range(5):
        ratio = i / 4.0
        x = jaw_left[0] + ratio * (jaw_right[0] - jaw_left[0])
        y = chin_center[1] + np.abs(ratio - 0.5) * 15  # Çene altı eğrisi
        chin_points.append([x, y])

    # Kulak arkası noktaları
    ear_points = []

    # Sol kulak arkası
    left_ear_x = points[0][0] - 20
    left_ear_y = (points[0][1] + points[3][1]) / 2
    ear_points.append([left_ear_x, left_ear_y])

    # Sağ kulak arkası
    right_ear_x = points[16][0] + 20
    right_ear_y = (points[16][1] + points[13][1]) / 2
    ear_points.append([right_ear_x, right_ear_y])

    # Tüm ek noktaları birleştir
    all_additional = np.array(forehead_points + temple_points + chin_points + ear_points)
    enhanced_points = np.vstack([enhanced_points, all_additional])

    return enhanced_points


def get_detailed_triangulation(points, image_shape):
    """Detaylı Delaunay triangulation"""
    height, width = image_shape[:2]

    # Görüntü köşe noktalarını ekle (sınır noktaları)
    corner_points = np.array([
        [0, 0], [width // 4, 0], [width // 2, 0], [3 * width // 4, 0], [width - 1, 0],
        [0, height // 4], [width - 1, height // 4],
        [0, height // 2], [width - 1, height // 2],
        [0, 3 * height // 4], [width - 1, 3 * height // 4],
        [0, height - 1], [width // 4, height - 1], [width // 2, height - 1],
        [3 * width // 4, height - 1], [width - 1, height - 1]
    ])

    # Tüm noktaları birleştir
    all_points = np.vstack([points, corner_points])

    # Delaunay triangulation
    tri = Delaunay(all_points)

    return tri.simplices, all_points


def warp_affine_torch(src_tensor, M, out_size):
    device = src_tensor.device
    H, W = src_tensor.shape[1:]
    out_w, out_h = out_size

    # Affine transformation matrix'i GPU'ya taşı
    M_inv = cv2.invertAffineTransform(M)

    # Grid koordinatlarını GPU'da oluştur
    xs = torch.linspace(0, out_w - 1, out_w, device=device)
    ys = torch.linspace(0, out_h - 1, out_h, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    ones = torch.ones_like(grid_x)
    coords = torch.stack([grid_x, grid_y, ones], dim=2).view(-1, 3).t()

    coords = coords.to(device).float()
    M_inv_tensor = torch.from_numpy(M_inv).to(device).float()
    src_coords = torch.matmul(M_inv_tensor, coords)

    # Grid normalization
    src_x_norm = 2 * (src_coords[0] / (W - 1)) - 1
    src_y_norm = 2 * (src_coords[1] / (H - 1)) - 1

    grid = torch.stack([src_x_norm, src_y_norm], dim=1).view(out_h, out_w, 2)
    grid = grid.unsqueeze(0)

    src_tensor = src_tensor.unsqueeze(0)
    warped = F.grid_sample(src_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped[0]


def create_smooth_mask(points, image_shape, feather_amount=15):
    """Yumuşak geçişli maske oluştur"""
    height, width = image_shape[:2]

    # Ana maske
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    # Gaussian blur ile yumuşatma
    mask_smooth = cv2.GaussianBlur(mask, (feather_amount * 2 + 1, feather_amount * 2 + 1), feather_amount / 3)

    return mask_smooth


def safe_get_video_property(cap, prop):
    """Video özelliklerini güvenli şekilde al"""
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


def swap_faces(video_path, new_face_path, output_path, progress_status):
    """
    Bir videodaki yüzü, verilen bir resimdeki yüz ile değiştiren ana fonksiyon.
    Ses ve video ayrıştırma/birleştirme stratejisi ile güncellenmiştir.
    """
    # Cihazı ayarla (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Uygulama şu cihazda çalışıyor: {device}')

    # Geçici dosyalar için benzersiz ID
    unique_id = str(uuid.uuid4())
    temp_dir = "temp"  # Geçici dosyaların kaydedileceği klasör
    os.makedirs(temp_dir, exist_ok=True)  # temp klasörünü oluştur

    temp_video_no_audio_path = os.path.join(temp_dir, f"temp_video_only_{unique_id}.mp4")
    temp_audio_path = os.path.join(temp_dir, f"temp_audio_only_{unique_id}.mp3")
    temp_processed_video_path = os.path.join(temp_dir, f"temp_processed_video_no_audio_{unique_id}.mp4")

    # Yüz algılayıcı (MTCNN) ve landmark modeli (dlib) yükle
    detector = MTCNN(keep_all=True, device=device)
    try:
        predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        raise FileNotFoundError(
            f"dlib landmark predictor dosyası bulunamadı: shape_predictor_68_face_landmarks.dat. Lütfen doğru yola indirin. Hata: {e}")

    # Temizlik için tüm geçici dosyaları takip et
    temp_files_to_clean = []

    try:
        # ----- 1. Adım: Orijinal Videodan Ses ve Video Ayırma -----
        progress_status['status'] = "extracting_audio_video"
        print("Orijinal videodan ses ve video ayrıştırılıyor...")

        # `with` ifadesi, klibin otomatik olarak kapatılmasını sağlar
        with VideoFileClip(video_path) as original_clip:
            # Video kısmını kaydet
            original_clip.write_videofile(temp_video_no_audio_path,
                                          codec='libx264',
                                          audio_codec=None,  # Ses olmadan kaydet
                                          fps=original_clip.fps,
                                          verbose=False,  # Konsol çıktısını azalt
                                          logger=None)  # Logger'ı devre dışı bırak
            temp_files_to_clean.append(temp_video_no_audio_path)
            print(f"Video ayrıştırıldı: {temp_video_no_audio_path}")

            # Ses kısmını kaydet (varsa)
            if original_clip.audio:
                original_clip.audio.write_audiofile(temp_audio_path, codec='mp3', verbose=False, logger=None)
                temp_files_to_clean.append(temp_audio_path)
                print(f"Ses ayrıştırıldı: {temp_audio_path}")
            else:
                print("Orijinal videoda ses bulunamadı.")
                temp_audio_path = None  # Ses olmadığı bilgisini sakla

        # ----- 2. Adım: Yeni yüz resmini işle -----
        progress_status['status'] = "processing_new_face"
        img_new_face = cv2.imread(new_face_path)
        if img_new_face is None:
            raise FileNotFoundError(f"Yeni yüz resmi bulunamadı: {new_face_path}")

        # MTCNN, PIL görüntüsü bekler, bu yüzden OpenCV'den PIL'e dönüştürün
        img_new_face_pil = Image.fromarray(cv2.cvtColor(img_new_face, cv2.COLOR_BGR2RGB))
        boxes_new, _ = detector.detect(img_new_face_pil)

        if boxes_new is None or len(boxes_new) == 0:
            raise Exception("Sağlanan yeni yüz resminde herhangi bir yüz tespit edilemedi.")

        bbox_new_coords = boxes_new[0]
        shape_new = dlib.rectangle(
            int(bbox_new_coords[0]), int(bbox_new_coords[1]),
            int(bbox_new_coords[2]), int(bbox_new_coords[3])
        )
        # Yeni yüz resmini gri tonlamalıya çevirip dlib'e verin
        landmarks_new = predictor(cv2.cvtColor(img_new_face, cv2.COLOR_BGR2GRAY), shape_new)
        landmarks_new = [(p.x, p.y) for p in landmarks_new.parts()]

        # ----- 3. Adım: Ayrıştırılmış Videoyu İşle (Yüz Değiştirme) -----
        progress_status['status'] = "swapping_faces"
        print("Video kareleri üzerinde yüz değiştirme işlemi başlatılıyor...")

        cap = cv2.VideoCapture(temp_video_no_audio_path)
        if not cap.isOpened():
            raise IOError(f"Geçici video dosyası açılamadı: {temp_video_no_audio_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            # Manuel frame sayımı
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            frame_count = count
            if frame_count == 0:
                raise ValueError("Geçici video dosyasında okunacak kare (frame) bulunamadı.")

        out = cv2.VideoWriter(temp_processed_video_path, fourcc, fps, (width, height))
        temp_files_to_clean.append(temp_processed_video_path)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MTCNN, PIL görüntüsü bekler, bu yüzden OpenCV'den PIL'e dönüştürün
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = detector.detect(frame_pil)

            if boxes is None or len(boxes) == 0:
                out.write(frame)  # Yüz bulunamazsa orijinal kareyi yaz
            else:
                bbox_coords = boxes[0]  # İlk bulunan yüzü al

                # dlib.rectangle için koordinatları int'e çevir ve dörtlüye ayarla
                shape = dlib.rectangle(
                    int(bbox_coords[0]), int(bbox_coords[1]),
                    int(bbox_coords[2]), int(bbox_coords[3])
                )
                # Video karesini gri tonlamalıya çevirip dlib'e verin
                landmarks_video = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), shape)
                landmarks_video = [(p.x, p.y) for p in landmarks_video.parts()]

                # `swap_faces_single` için beklenen bbox formatı: [x, y, w, h]
                x1, y1, x2, y2 = map(int, bbox_coords)
                bbox_for_clone = [x1, y1, x2 - x1, y2 - y1]

                # `swap_faces_single` fonksiyonunu çağır
                swapped_frame = swap_faces_single(frame, img_new_face, landmarks_video, landmarks_new, bbox_for_clone)
                out.write(swapped_frame)

            frame_idx += 1
            if frame_count > 0:
                progress_status['progress'] = int(frame_idx / frame_count * 100)

        cap.release()
        out.release()
        print(f"Video kareleri işlendi ve geçici dosyaya kaydedildi: {temp_processed_video_path}")
        progress_status['progress'] = 100
        progress_status['status'] = "video_processed"

        # ----- 4. Adım: İşlenmiş Video ve Sesi Birleştirme -----
        progress_status['status'] = "combining_audio_video"
        print("İşlenmiş video ve sesi birleştiriliyor...")

        with VideoFileClip(temp_processed_video_path) as processed_video_clip:
            if temp_audio_path and os.path.exists(temp_audio_path):
                with AudioFileClip(temp_audio_path) as audio_clip:
                    # processed_video_clip'in sesini ayarla
                    final_clip = processed_video_clip.set_audio(audio_clip)

                    # Son çıktıyı kaydet
                    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=fps, verbose=False,
                                               logger=None)
                    final_clip.close()
            else:
                # Ses yoksa sadece işlenmiş videoyu çıktı olarak kaydet
                processed_video_clip.write_videofile(output_path, codec='libx264', audio_codec=None, fps=fps,
                                                     verbose=False, logger=None)
                print("Orijinal videoda ses olmadığı için sadece video kaydedildi.")

        print(f"İşlem tamamlandı. Son video kaydedildi: {output_path}")
        progress_status['status'] = "done"

    except Exception as e:
        print(f"Hata oluştu: {e}")
        progress_status['status'] = "error"
        progress_status['error'] = str(e)

    finally:
        # ----- 5. Adım: Geçici Dosyaları Temizleme -----
        print("Geçici dosyalar temizleniyor...")
        for temp_file in temp_files_to_clean:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Uyarı: Geçici dosya silinemedi {temp_file}: {e}")
        # Eğer temp klasörü boşsa, onu da sil (isteğe bağlı)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Uyarı: Geçici klasör silinemedi {temp_dir}: {e}")
        print("Geçici dosyalar temizlendi.")


def swap_faces_single(img_video_frame, img_new_face, landmarks_video_frame, landmarks_new_face, bbox_video_frame):
    """
    Tek bir karede yüz değiştirme işlemini gerçekleştirir.
    img_video_frame: Video karesi (Hedef)
    img_new_face: Yeni yüz resmi (Kaynak)
    landmarks_video_frame: Video karesindeki yüzün landmark'ları
    landmarks_new_face: Yeni yüz resmindeki yüzün landmark'ları
    bbox_video_frame: Video karesindeki yüzün sınırlayıcı kutusu [x, y, w, h]
    """
    # Landmark indeksleri (dlib'in 68 noktalı modeline göre)
    # Bu indeksler yüzün önemli kısımlarını (çene çizgisi, kaşlar, burun, gözler, ağız) kapsar.
    # Bu seçimin, yüzün ana hatlarını doğru yakaladığından emin olun.
    points_indexes = list(range(17)) + list(range(17, 27)) + list(range(27, 36)) + list(range(36, 48)) + \
                     list(range(48, 60)) + list(range(60, 68))  # Tüm 68 landmark'ı dahil edelim

    try:
        points1 = np.float32([landmarks_new_face[i] for i in points_indexes])  # Yeni yüzdeki noktalar
        points2 = np.float32([landmarks_video_frame[i] for i in points_indexes])  # Video karesindeki noktalar
    except IndexError as e:
        print(f"Uyarı: Landmark indeksleme hatası - {e}. Yüz değiştirme atlanıyor.")
        return img_video_frame

    # Konveks dışbükey bölgeleri bulma
    hull1 = cv2.convexHull(points1)  # Yeni yüzdeki dışbükey kabuk
    hull2 = cv2.convexHull(points2)  # Video karesindeki dışbükey kabuk

    # Delaunay üçgenlemesi için dikdörtgen alanı ve üçgenleri bulma
    rect = cv2.boundingRect(hull2)  # Video karesindeki yüzün dışbükey kabuğunu içeren dikdörtgen
    subdiv = cv2.Subdiv2D(rect)
    for p in points2:
        subdiv.insert(tuple(p))
    triangles = subdiv.getTriangleList()

    # Landmark noktalarının indeksini bulmak için yardımcı fonksiyon
    def find_index(pt, points_list):
        for i, p in enumerate(points_list):
            if np.linalg.norm(pt - p) < 1.0:  # Noktalar arasında çok küçük bir farka izin ver
                return i
        return -1

    triangle_indices = []
    for t in triangles:
        pts = [t[0:2], t[2:4], t[4:6]]  # Her üçgenin 3 köşesi
        idx = []
        for p in pts:
            # points2'deki orijinal indeksleri bul
            i = find_index(np.float32(p), points2)
            if i != -1:
                idx.append(i)
        if len(idx) == 3:
            # Orijinal points2 listesindeki indeksleri tut
            triangle_indices.append(tuple(idx))

    # Yeni yüzü video karesine göre warped (eğilmiş) hale getireceğimiz boş bir görüntü
    img1_warped = np.zeros_like(img_video_frame, dtype=np.float32)  # float32 yapıyoruz çünkü çarpma işlemleri olacak

    # Her üçgen için affine dönüşüm uygulama
    for tri in triangle_indices:
        x, y, z = tri
        # Yeni yüzdeki üçgenin köşeleri
        t1 = np.float32([points1[x], points1[y], points1[z]])
        # Video karesindeki üçgenin köşeleri
        t2 = np.float32([points2[x], points2[y], points2[z]])

        # Üçgenlerin sınırlayıcı dikdörtgenleri
        r1 = cv2.boundingRect(t1)  # Yeni yüzdeki üçgenin bbox'ı
        r2 = cv2.boundingRect(t2)  # Video karesindeki üçgenin bbox'ı

        # Dikdörtgenlere göre üçgen koordinatlarını ofsetleme
        t1_rect = np.array([[p[0] - r1[0], p[1] - r1[1]] for p in t1], dtype=np.float32)
        t2_rect = np.array([[p[0] - r2[0], p[1] - r2[1]] for p in t2], dtype=np.float32)

        # Dönüşüm için maske oluşturma
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)  # float32 maske
        cv2.fillConvexPoly(mask, np.int32(t2_rect), (1, 1, 1))  # Üçgeni beyaz (1) ile doldur

        # Yeni yüz resminden üçgen bölgesini kesme
        img1_rect = img_new_face[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]].copy()

        # Boş boyutlu kesitler için kontrol
        if img1_rect.shape[0] == 0 or img1_rect.shape[1] == 0:
            continue  # Bu üçgeni atla

        # Affine dönüşüm matrisini hesaplama
        warp_mat = cv2.getAffineTransform(t1_rect, t2_rect)

        # Yeni yüzdeki üçgeni video karesindeki üçgenin konumuna dönüştürme
        warped_triangle = cv2.warpAffine(img1_rect, warp_mat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)

        # Warped üçgeni maske ile çarpıp hedef görüntüye ekleme
        # Önce mevcut alanı temizle, sonra warped üçgeni ekle
        img1_warped[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = \
            img1_warped[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (1 - mask) + (
                        warped_triangle.astype(np.float32) * mask)

    # Son yüz maskesi oluşturma
    mask = np.zeros_like(img_video_frame[:, :, 0], dtype=np.uint8)
    convex_hull = cv2.convexHull(np.int32(points2))  # Video karesindeki yüzün dışbükey kabuğu
    cv2.fillConvexPoly(mask, convex_hull, 255)  # Bu bölgeyi maske olarak kullan

    # seamlessClone için yüzün merkezi
    # bbox_video_frame formatı: [x, y, w, h]
    center = (bbox_video_frame[0] + bbox_video_frame[2] // 2, bbox_video_frame[1] + bbox_video_frame[3] // 2)

    # seamlessClone kullanarak yüzü harmanlama
    # np.uint8'e dönüştürmeyi unutmayın
    output = cv2.seamlessClone(
        np.uint8(img1_warped),  # Kaynak görüntü (warped yeni yüz)
        np.uint8(img_video_frame),  # Hedef görüntü (video karesi)
        np.uint8(mask),  # Kaynak görüntünün maskesi
        center,  # Hedef görüntüdeki konum
        cv2.NORMAL_CLONE  # Harmanlama modu
    )

    return output
