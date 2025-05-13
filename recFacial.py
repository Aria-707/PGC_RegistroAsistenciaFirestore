'''from flask import Flask, render_template, Response, request, redirect
import cv2
import os
import numpy as np
import time
import requests
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Cargar clave desde variable de entorno
cred = credentials.Certificate({
    "type": "service_account",
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.environ.get("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL"),
    "universe_domain": "googleapis.com",
})
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataPath = os.path.join(BASE_DIR, 'Data')
model_path = os.path.join(BASE_DIR, 'modeloLBPHReconocimientoOpencv.xml')

# Reconocimiento facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Carga previa del modelo si existe
if os.path.exists(model_path):
    face_recognizer.read(model_path)
    imagePaths = os.listdir(dataPath)
else:
    imagePaths = []

# Variables globales
cap = None
duracion_reconocimiento = 3
estudiantes_reconocidos = set()
tiempos_reconocimiento = {}

def entrenar_modelo():
    global face_recognizer, imagePaths

    peopleList = os.listdir(dataPath)
    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        for fileName in os.listdir(personPath):
            image = cv2.imread(os.path.join(personPath, fileName), 0)
            if image is not None:
                labels.append(label)
                facesData.append(image)
        label += 1

    if facesData:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        face_recognizer.write(model_path)
        imagePaths = peopleList
        print("Modelo entrenado con éxito.")

@app.route('/')
def index():
    return render_template('reconocimiento.html')

@app.route('/videoReC')
def videoRec():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global cap, imagePaths

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1] < 70 and result[0] < len(imagePaths):
                nombre = imagePaths[result[0]]
                if nombre not in tiempos_reconocimiento:
                    tiempos_reconocimiento[nombre] = time.time()
                elif time.time() - tiempos_reconocimiento[nombre] >= duracion_reconocimiento:
                    if nombre not in estudiantes_reconocidos:
                        estudiantes_reconocidos.add(nombre)
                        print(f"[✔] Reconocido: {nombre}")
                        registrar_asistencia(nombre)
                cv2.putText(frame, nombre, (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/registro')
def registro():
    return render_template('registro.html')

@app.route('/registrar', methods=['POST'])
def registrar():
    global cap
    estudiante = request.form['estudiante']
    personPath = os.path.join(dataPath, estudiante)

    if not os.path.exists(personPath):
        os.makedirs(personPath)

    count = 0
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), face)
            count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Entrena una vez que se han capturado los datos
    entrenar_modelo()
    return redirect('/')

def generate_frames_registro():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_registro')
def video_registro():
    return Response(generate_frames_registro(), mimetype='multipart/x-mixed-replace; boundary=frame')

def registrar_asistencia(nombre):
    try:
        doc_ref = db.collection('asistenciaReconocimiento').document()
        doc_ref.set({
            'estudiante': nombre,
            'estadoAsistencia': 'Presente',
            'fechaYhora': firestore.SERVER_TIMESTAMP,
            'asignatura': 'Física'
        })
        print(f"[✔] Asistencia registrada para {nombre} en Firestore.")
    except Exception as e:
        print(f"[✖] Error al registrar en Firestore: {e}")

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, firestore
import base64
from io import BytesIO
from PIL import Image

# Inicializar Firebase
cred = credentials.Certificate('C:/Users/arias/Downloads/asistenciaconreconocimiento-firebase-adminsdk-fbsvc-793e372c66.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataPath = os.path.join(BASE_DIR, 'Data')
model_path = os.path.join(BASE_DIR, 'modeloLBPHReconocimientoOpencv.xml')

# Clasificador y reconocedor
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cargar modelo si existe
if os.path.exists(model_path):
    face_recognizer.read(model_path)
    imagePaths = os.listdir(dataPath)
else:
    imagePaths = []

# Variables globales
duracion_reconocimiento = 3
estudiantes_reconocidos = set()
tiempos_reconocimiento = {}

def entrenar_modelo():
    global face_recognizer, imagePaths
    peopleList = os.listdir(dataPath)
    labels, facesData = [], []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        for fileName in os.listdir(personPath):
            image = cv2.imread(os.path.join(personPath, fileName), 0)
            if image is not None:
                labels.append(label)
                facesData.append(image)
        label += 1

    if facesData:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        face_recognizer.write(model_path)
        imagePaths = peopleList
        print("Modelo entrenado con éxito.")

@app.route('/')
def index():
    return render_template('reconocimiento.html')

@app.route('/registro')
def registro():
    return render_template('registro.html')

@app.route('/analizar_frame', methods=['POST'])
def analizar_frame():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    nombre_detectado = None

    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        if result[1] < 70 and result[0] < len(imagePaths):
            nombre = imagePaths[result[0]]
            nombre_detectado = nombre

            cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if nombre not in tiempos_reconocimiento:
                tiempos_reconocimiento[nombre] = time.time()
            elif time.time() - tiempos_reconocimiento[nombre] >= duracion_reconocimiento:
                if nombre not in estudiantes_reconocidos:
                    estudiantes_reconocidos.add(nombre)
                    registrar_asistencia(nombre)

        else:
            cv2.putText(frame, "Desconocido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'nombre': nombre_detectado,
        'imagen': f'data:image/jpeg;base64,{frame_base64}'
    }

@app.route('/registrar_frame', methods=['POST'])
def registrar_frame():
    data = request.get_json()
    nombre = data['nombre']
    image_data = data['imagen'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Convertir a OpenCV (color) y voltear horizontalmente (efecto espejo)
    frame_color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame_color = cv2.flip(frame_color, 1)

    # Convertir a escala de grises para detección de rostro
    image_cv = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(image_cv, 1.3, 5)
    if faces is None or len(faces) == 0:
        return {'message': 'No se detectó ningún rostro'}, 400

    personPath = os.path.join(dataPath, nombre)
    os.makedirs(personPath, exist_ok=True)

    existing_files = len(os.listdir(personPath))
    count = 0

    for (x, y, w, h) in faces:
        rostro = image_cv[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        filename = f'rostro_{existing_files + count}.jpg'
        cv2.imwrite(os.path.join(personPath, filename), rostro)
        count += 1

    return {'message': f'{count} rostro(s) capturado(s)'}



@app.route('/entrenar_modelo/<nombre>', methods=['POST'])
def entrenar_modelo_ruta(nombre):
    entrenar_modelo()
    return {'message': f'Modelo entrenado para {nombre}'}


def registrar_asistencia(nombre):
    try:
        db.collection('asistenciaReconocimiento').add({
            'estudiante': nombre,
            'estadoAsistencia': 'Presente',
            'fechaYhora': firestore.SERVER_TIMESTAMP,
            'asignatura': 'Física'
        })
        print(f"[✔] Asistencia registrada para {nombre}")
    except Exception as e:
        print(f"[✖] Error registrando asistencia: {e}")

if __name__ == '__main__':
    entrenar_modelo()
    app.run(debug=True)
