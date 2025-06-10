import cv2
import numpy as np
import dlib
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay


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


def swap_faces(video_path, new_face_path, output_path, target_id=None, progress_status=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Yeni yüz resmini yükle ve landmark'larını çıkar
    new_face_img = cv2.imread(new_face_path)
    new_face_gray = cv2.cvtColor(new_face_img, cv2.COLOR_BGR2GRAY)
    new_faces = detector(new_face_gray)
    if len(new_faces) == 0:
        raise Exception("Yeni yüz resmi bulunamadı.")

    new_landmarks = predictor(new_face_gray, new_faces[0])
    new_points_68 = np.array([[p.x, p.y] for p in new_landmarks.parts()], dtype=np.float32)

    # Genişletilmiş yüz noktaları
    new_points_enhanced = get_enhanced_face_points(new_landmarks)

    # Yeni yüzü GPU'ya yükle
    new_face_tensor = to_tensor(new_face_img).to(device)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Orijinal frame'i GPU'ya taşı
        frame_tensor = to_tensor(frame).to(device)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            points_68 = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)

            # Hedef yüz için genişletilmiş noktalar
            target_points_enhanced = get_enhanced_face_points(landmarks)

            # Detaylı triangulation
            triangles, all_target_points = get_detailed_triangulation(target_points_enhanced, frame.shape)
            _, all_source_points = get_detailed_triangulation(new_points_enhanced, new_face_img.shape)

            print(f"Toplam üçgen sayısı: {len(triangles)}")

            # Warped face'i GPU'da oluştur
            warped_face = torch.zeros((3, height, width), dtype=torch.float32, device=device)
            accumulated_mask = torch.zeros((height, width), dtype=torch.float32, device=device)

            # Her üçgen için warp işlemi
            valid_triangles = 0
            for tri_idx in triangles:
                # Üçgen noktalarının geçerli olup olmadığını kontrol et
                if (tri_idx < len(all_target_points)).all() and (tri_idx < len(all_source_points)).all():

                    target_tri = all_target_points[tri_idx].astype(np.float32)
                    source_tri = all_source_points[tri_idx].astype(np.float32)

                    # Çok küçük üçgenleri atla
                    if cv2.contourArea(target_tri) < 10:
                        continue

                    try:
                        # Affine transformation
                        M = cv2.getAffineTransform(source_tri, target_tri)

                        # GPU'da warp işlemi
                        warped_tri = warp_affine_torch(new_face_tensor, M, (width, height))

                        # Üçgen maskesi oluştur
                        mask_tri = np.zeros((height, width), dtype=np.uint8)
                        cv2.fillConvexPoly(mask_tri, target_tri.astype(np.int32), 1)
                        mask_tri_tensor = torch.from_numpy(mask_tri).to(device).float()

                        # Anti-aliasing için hafif blur
                        mask_tri_tensor = torch.from_numpy(
                            cv2.GaussianBlur(mask_tri, (3, 3), 0.5)
                        ).to(device).float() / 255.0

                        # Maskeyi uygula
                        mask_tri_3d = mask_tri_tensor.unsqueeze(0)
                        warped_face += warped_tri * mask_tri_3d
                        accumulated_mask += mask_tri_tensor

                        valid_triangles += 1

                    except Exception as e:
                        continue

            print(f"İşlenen geçerli üçgen sayısı: {valid_triangles}")

            # Accumulated mask ile normalize et
            accumulated_mask = torch.clamp(accumulated_mask, min=1e-6)
            warped_face = warped_face / accumulated_mask.unsqueeze(0)

            # Gelişmiş yüz maskesi oluştur (alın dahil)
            face_mask_smooth = create_smooth_mask(target_points_enhanced, frame.shape, feather_amount=20)
            face_mask_tensor = torch.from_numpy(face_mask_smooth).to(device).float() / 255.0

            # Multi-level blending
            # 1. Ana maske ile alpha blending
            alpha = face_mask_tensor.unsqueeze(0)
            blended = alpha * warped_face + (1 - alpha) * frame_tensor

            # 2. Detay iyileştirme için edge-aware smoothing
            # Gradyan bazlı ağırlıklandırma
            gray_tensor = torch.mean(frame_tensor, dim=0, keepdim=True)
            grad_x = torch.abs(gray_tensor[:, :, 1:] - gray_tensor[:, :, :-1])
            grad_y = torch.abs(gray_tensor[:, 1:, :] - gray_tensor[:, :-1, :])

            # Sonucu CPU'ya geri al
            output = to_numpy(blended.clamp(0, 1))

            # Opsiyonel: Son rötuş için OpenCV seamlessClone
            center = (face.left() + face.width() // 2, face.top() + face.height() // 2)
            try:
                # Sadece merkezi yüz bölgesi için seamless clone
                center_mask = create_smooth_mask(points_68, frame.shape, feather_amount=10)
                warped_face_cpu = to_numpy(warped_face.clamp(0, 1))
                output_seamless = cv2.seamlessClone(warped_face_cpu, output, center_mask, center, cv2.NORMAL_CLONE)

                # İki sonucu karıştır (seams için)
                blend_ratio = 0.7
                output = cv2.addWeighted(output, 1 - blend_ratio, output_seamless, blend_ratio, 0)

            except Exception as e:
                print(f"Seamless clone hatası: {e}")
                # Hata durumunda GPU blending sonucunu kullan
                pass

        else:
            output = frame

        out.write(output)

        frame_index += 1
        if progress_status is not None:
            progress_status['progress'] = int((frame_index / frame_count) * 100)

    cap.release()
    out.release()