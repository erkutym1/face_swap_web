import os
import threading
import shutil
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
from face_extractor import extract_distinct_faces
from face_swap import swap_faces

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
FACE_FOLDER = 'static/faces'

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, FACE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global değişkenler
face_map = {}
progress_status = {
    'status': 'idle',  # idle, extracting, done_extract, swapping, done_swap
    'progress': 0
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', processed=False)

@app.route('/upload', methods=['POST'])
def upload():
    global face_map, progress_status

    video = request.files.get('video')
    if not video:
        return "Video yüklenmedi!", 400

    video_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    video.save(video_path)

    # Önce yüz klasörünü temizle
    for f in os.listdir(FACE_FOLDER):
        os.remove(os.path.join(FACE_FOLDER, f))

    progress_status['status'] = 'extracting'
    progress_status['progress'] = 0

    def extract_task():
        global face_map, progress_status
        face_map = extract_distinct_faces(video_path, FACE_FOLDER, max_frames=100, tolerance=0.6, progress_status=progress_status)
        progress_status['status'] = 'done_extract'
        progress_status['progress'] = 100

    threading.Thread(target=extract_task).start()

    return render_template('progress_extract.html')

@app.route('/progress_extract')
def progress_extract():
    return jsonify({
        'status': progress_status['status'],
        'progress': progress_status['progress']
    })

@app.route('/select_face', methods=['GET'])
def select_face():
    if progress_status['status'] != 'done_extract':
        return redirect(url_for('index'))

    face_files = sorted(os.listdir(FACE_FOLDER))
    return render_template('select_face.html', face_files=face_files)

@app.route('/process', methods=['POST'])
def process():
    global progress_status

    selected_face_id = request.form.get('selected_face')
    face_image = request.files.get('face_image')

    if not selected_face_id or not face_image:
        return "Yüz seçilmedi ya da yüz fotoğrafı yüklenmedi!", 400

    # Seçilen yüzün yolu (şu anda kullanılmıyor ama istenirse target_id eşleştirmesi yapılabilir)
    face_img_path = os.path.join(FACE_FOLDER, selected_face_id)
    if not os.path.exists(face_img_path):
        return "Seçilen yüz bulunamadı!", 404

    # Yüklenen yeni yüz resmi
    new_face_path = os.path.join(UPLOAD_FOLDER, "new_face.jpg")
    face_image.save(new_face_path)

    # Video yolu ve çıktı yolu
    input_video_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    output_video_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')

    progress_status['status'] = 'swapping'
    progress_status['progress'] = 0

    def swap_task():
        try:
            swap_faces(
                video_path=input_video_path,
                new_face_path=new_face_path,
                output_path=output_video_path,
                progress_status=progress_status
            )
            progress_status['status'] = 'done_swap'
            progress_status['progress'] = 100
        except Exception as e:
            progress_status['status'] = 'error'
            progress_status['progress'] = 0
            print(f"Yüz değiştirme hatası: {e}")

    threading.Thread(target=swap_task).start()

    return jsonify({"message": "Yüz değiştirme işlemi başladı."})

@app.route('/progress_swap')
def progress_swap():
    return jsonify({
        'status': progress_status.get('status', 'idle'),
        'progress': progress_status.get('progress', 0)
    })

@app.route('/swapping')
def swapping():
    return render_template('swapping.html')

@app.route('/progress')
def progress():
    return jsonify(progress_status)

@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/download')
def download():
    output_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
    if not os.path.exists(output_path):
        return "Çıktı dosyası bulunamadı!", 404
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
