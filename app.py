from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from backend import extract_embedding, recognize, get_all_names  # Thêm dòng này
import torch

app = Flask(__name__)

students = [{"name": name, "present": False} for name in get_all_names()]

@app.route('/')
def index():
    return render_template('main.html', students=students)

@app.route('/mark_present', methods=['POST'])
def mark_present():
    name = request.json.get('name')
    for s in students:
        if s['name'] == name:
            s['present'] = True
    return jsonify({"success": True})

@app.route('/students')
def get_students():
    return jsonify(students)


@app.route('/stream_recognize', methods=['POST'])
def stream_recognize():
    data = request.json.get('image')
    if not data:
        return jsonify({'success': False})
    img_data = data.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return jsonify({'success': True})
    # Lấy khuôn mặt lớn nhất
    x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
    face_img = img[y:y+h, x:x+w]
    emb = extract_embedding(face_img)
    name, dist = recognize(emb)
    print(name)
    # Đánh dấu đã điểm danh nếu nhận diện được
    if name != "Unknown":
        for s in students:
            if s['name'] == name:
                s['present'] = True
                break
    return jsonify({'success': True})


if __name__ == '__main__':   
    print(students)
    app.run(debug=True)