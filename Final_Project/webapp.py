from flask import Flask, render_template, Response, redirect, url_for, request, session, jsonify
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime, timedelta
import json
from flask_mail import Mail, Message
import pyotp
import bcrypt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import re
from typing import Tuple
from bleach import clean

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

import logging
from logging.handlers import RotatingFileHandler

from flask_talisman import Talisman

import qrcode
from io import BytesIO
import base64

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Setup logging configuration
logging.basicConfig(
    handlers=[
        RotatingFileHandler(
            'logs/app.log', 
            maxBytes=10000000,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ],
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

app = Flask(__name__)  # initializing
app.secret_key = 'your-secret-key-here'  # Required for sessions

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kalstaraiman01@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'otis ploz bemv ydha'     # Replace with your app password
mail = Mail(app)

# database credentials
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "",
        "storageBucket": "",
    },
)

bucket = storage.bucket()

# Add after app initialization
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Add a failed login attempts counter to track login failures
failed_login_attempts = {}
MAX_ATTEMPTS = 5
LOCKOUT_TIME = 900  # 15 minutes in seconds

# Add these constants at the top with other configurations
UPLOAD_FOLDER = './static/Files/Images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Add these to store OTP information
otp_storage = {}

# Security headers configuration
csp = {
    'default-src': "'self'",
    'img-src': ["'self'", '*', 'data:'],
    'script-src': ["'self'", "'unsafe-inline'", 'https://unpkg.com'],
    'style-src': ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com', 'https://unpkg.com'],
    'font-src': ["'self'", 'https://fonts.gstatic.com', 'https://unpkg.com'],
    'frame-ancestors': "'none'"
}

Talisman(app,
    content_security_policy=csp,
    force_https=True,
    strict_transport_security=True,
    session_cookie_secure=True,
    session_cookie_http_only=True,
    feature_policy={
        'geolocation': "'none'",
        'camera': "'self'",
        'microphone': "'none'"
    }
)

# Add these constants at the top
FACE_CONFIDENCE_THRESHOLD = 0.6
MAX_FACE_DISTANCE = 0.6
REQUIRED_FACE_SIZE = (160, 160)  # Minimum face size for quality check

# Add after other imports
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_file_upload(file):
    if not file:
        raise ValueError("No file provided")
        
    if not allowed_file(file.filename):
        raise ValueError("Invalid file type. Allowed types: PNG, JPG, JPEG")
        
    filename = secure_filename(file.filename)
    
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB")
    
    # Check if it's a valid image
    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
        file.seek(0)
    except Exception:
        raise ValueError("Invalid image file")
        
    return filename

def validate_image(file):
    if file is None:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No selected file"
        
    if not allowed_file(file.filename):
        return False, "Invalid file type. Only PNG, JPG, JPEG allowed"
        
    if file.content_length and file.content_length > MAX_FILE_SIZE:
        return False, "File size too large. Maximum size is 5MB"
        
    return True, "File is valid"

def is_account_locked(id):
    if id in failed_login_attempts:
        attempts = failed_login_attempts[id]
        if attempts['count'] >= MAX_ATTEMPTS:
            lockout_time = attempts['timestamp'] + LOCKOUT_TIME
            if datetime.now().timestamp() < lockout_time:
                return True
            else:
                # Reset attempts after lockout period
                failed_login_attempts.pop(id)
    return False

def record_failed_attempt(id):
    current_time = datetime.now().timestamp()
    if id in failed_login_attempts:
        failed_login_attempts[id]['count'] += 1
        failed_login_attempts[id]['timestamp'] = current_time
    else:
        failed_login_attempts[id] = {
            'count': 1,
            'timestamp': current_time
        }

def reset_failed_attempts(id):
    if id in failed_login_attempts:
        failed_login_attempts.pop(id)

def dataset(id):
    studentInfo = db.reference(f"Students/{id}").get()
    blob = bucket.get_blob(f"./static/Files/Images/{id}.png")
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
    datetimeObject = datetime.strptime(
        studentInfo["last_attendance_time"], "%Y-%m-%d %H:%M:%S"
    )
    secondElapsed = (datetime.now() - datetimeObject).total_seconds()
    return studentInfo, imgStudent, secondElapsed


already_marked_id_student = []
already_marked_id_admin = []


def generate_frame():
    # Background and Different Modes

    # video camera
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    imgBackground = cv2.imread("static/Files/Resources/background.png")

    folderModePath = "static/Files/Resources/Modes/"
    modePathList = os.listdir(folderModePath)
    imgModeList = []

    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

    modeType = 0
    id = -1
    imgStudent = []
    counter = 0

    # encoding loading ---> to identify if the person is in our database or not.... to detect faces that are known or not

    file = open("EncodeFile.p", "rb")
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodedFaceKnown, studentIDs = encodeListKnownWithIds

    while True:
        success, img = capture.read()

        if not success:
            break
        else:
            imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

            faceCurrentFrame = face_recognition.face_locations(imgSmall)
            encodeCurrentFrame = face_recognition.face_encodings(
                imgSmall, faceCurrentFrame
            )
            imgBackground[162 : 162 + 480, 55 : 55 + 640] = img
            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]

            if faceCurrentFrame:
                for encodeFace, faceLocation in zip(
                    encodeCurrentFrame, faceCurrentFrame
                ):
                    # Get face image for quality check
                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    face_image = img[y1:y2, x1:x2]
                    
                    # Check face quality
                    quality_ok, quality_msg = check_face_quality(face_image)
                    if not quality_ok:
                        logger.warning(f"Face quality check failed: {quality_msg}")
                        continue
                        
                    # Verify face
                    is_match, student_id, confidence = verify_face(
                        encodeFace, 
                        encodedFaceKnown, 
                        studentIDs
                    )
                    
                    if is_match:
                        logger.info(f"Face match found - ID: {student_id}, Confidence: {confidence}%")
                        id = student_id
                        counter = 1
                        modeType = 1
                    else:
                        logger.warning("No face match found or confidence too low")
                        modeType = 4
                        counter = 0
                        imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                            modeType
                        ]

                if counter != 0:
                    if counter == 1:
                        studentInfo, imgStudent, secondElapsed = dataset(id)
                        if secondElapsed > 60:
                            ref = db.reference(f"Students/{id}")
                            studentInfo["total_attendance"] += 1
                            ref.child("total_attendance").set(
                                studentInfo["total_attendance"]
                            )
                            ref.child("last_attendance_time").set(
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        else:
                            modeType = 3
                            counter = 0
                            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                modeType
                            ]

                            already_marked_id_student.append(id)
                            already_marked_id_admin.append(id)

                    if modeType != 3:
                        if 5 < counter <= 10:
                            modeType = 2

                        imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                            modeType
                        ]

                        if counter <= 5:
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["total_attendance"]),
                                (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["major"]),
                                (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(id),
                                (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["standing"]),
                                (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["year"]),
                                (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["starting_year"]),
                                (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )

                            (w, h), _ = cv2.getTextSize(
                                str(studentInfo["name"]), cv2.FONT_HERSHEY_COMPLEX, 1, 1
                            )

                            offset = (414 - w) // 2
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["name"]),
                                (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (50, 50, 50),
                                1,
                            )

                            imgStudentResize = cv2.resize(imgStudent, (216, 216))

                            imgBackground[
                                175 : 175 + 216, 909 : 909 + 216
                            ] = imgStudentResize

                        counter += 1

                        if counter >= 10:
                            counter = 0
                            modeType = 0
                            studentInfo = []
                            imgStudent = []
                            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                modeType
                            ]

            else:
                modeType = 0
                counter = 0

            ret, buffer = cv2.imencode(".jpeg", imgBackground)
            frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg \r\n\r\n" + frame + b"\r\n")


#########################################################################################################################


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


#########################################################################################################################


def generate_and_send_otp(email):
    # Generate a random OTP
    totp = pyotp.TOTP(pyotp.random_base32())
    otp = totp.now()
    
    # Send OTP via email
    msg = Message('Your 2FA Code',
                 sender='your-email@gmail.com',
                 recipients=[email])
    msg.body = f'Your verification code is: {otp}'
    mail.send(msg)
    
    return totp.secret


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def login():
    if request.method == "POST":
        id = sanitize_input(request.form.get("id_number"))
        password = request.form.get("password")
        user_type = request.form.get("user_type")  # 'admin' or 'student'
        
        if verify_password(id, password, is_admin=(user_type == 'admin')):
            user_info = dataset(id)[0]
            email = user_info['email']
            
            # Generate both 2FA methods
            qr_secret, qr_code, email_otp = generate_2fa_codes(id, email, is_admin=(user_type == 'admin'))
            
            # Store in session
            session['qr_secret'] = qr_secret
            session['email_otp'] = email_otp
            session['pending_user_id'] = id
            session['pending_user_type'] = user_type
            
            return render_template('verify_2fa.html', qr_code=qr_code)
            
        return render_template("login.html", error="Invalid credentials")
        
    return render_template("login.html")


@app.route("/student_login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def student_login():
    if request.method == "POST":
        id = sanitize_input(request.form.get("id_number", False))
        email = sanitize_input(request.form.get("email", False))
        password = request.form.get("password", False)  # Don't sanitize passwords
        
        logger.info(f"Login attempt for student ID: {id}, email: {email}")
        
        if is_account_locked(id):
            logger.warning(f"Attempted login to locked account - ID: {id}")
            return render_template("student_login.html", 
                data=" ❌ Account temporarily locked. Please try again later.")

        studentIDs, _ = add_image_database()

        if id:
            if id not in studentIDs:
                return render_template("student_login.html", 
                    data=" ❌ The id is not registered")
            
            student_info = dataset(id)[0]
            stored_password = student_info.get("password", "")
            
            try:
                password_matches = bcrypt.checkpw(password.encode('utf-8'), 
                    stored_password.encode('utf-8'))
            except ValueError:
                password_matches = (password == stored_password)
            
            if password_matches and student_info["email"] == email:
                logger.info(f"Successful login for student ID: {id}")
                reset_failed_attempts(id)
                secret = generate_and_send_otp(email)
                session['otp_secret'] = secret
                session['pending_student_id'] = id
                return redirect(url_for('student_otp'))
            else:
                logger.warning(f"Failed login attempt for student ID: {id}")
                record_failed_attempt(id)
                return render_template("student_login.html", 
                    data=" ❌ Email/Password Incorrect")
    
    return render_template("student_login.html")


@app.route("/student_otp", methods=["GET"])
def student_otp():
    if 'otp_secret' not in session:
        return redirect(url_for('student_login'))
    return render_template("student_otp.html")


@app.route("/verify_student_otp", methods=["POST"])
def verify_student_otp():
    if 'otp_secret' not in session:
        return redirect(url_for('student_login'))
    
    otp = request.form.get("otp")
    if not otp:
        return render_template("student_otp.html", data="Please enter OTP")
    
    totp = pyotp.TOTP(session['otp_secret'])
    if totp.verify(otp):
        # OTP is valid
        student_id = session['pending_student_id']
        secret_key = f"{student_id}anythingyoulike"
        hash_secret_key = str(hash(secret_key))
        session.pop('otp_secret', None)
        session.pop('pending_student_id', None)
        return redirect(url_for("student", data=student_id, title=hash_secret_key))
    else:
        return render_template("student_otp.html", data="❌ Invalid OTP")


@app.route("/student/<data>/<title>")
def student(data, title=None):
    studentInfo, imgStudent, secondElapsed = dataset(data)
    hoursElapsed = round((secondElapsed / 3600), 2)

    info = {
        "studentInfo": studentInfo,
        "lastlogin": hoursElapsed,
        "image": imgStudent,
    }
    return render_template("student.html", data=info)


@app.route("/student_attendance_list")
def student_attendance_list():
    unique_id_student = list(set(already_marked_id_student))
    student_info = []
    for i in unique_id_student:
        student_info.append(dataset(i))
    return render_template("student_attendance_list.html", data=student_info)

#########################################################################################################################


@app.route("/admin_login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def admin_login():
    if request.method == "POST":
        id = request.form.get("id_number", False)
        email = request.form.get("email", False)
        password = request.form.get("password", False)
        
        logger.info(f"Admin login attempt for ID: {id}, email: {email}")
        
        if is_account_locked(id):
            logger.warning(f"Attempted login to locked admin account - ID: {id}")
            return render_template("admin_login.html", 
                data=" ❌ Account temporarily locked. Please try again later.")

        studentIDs, _ = add_image_database()

        if id:
            if id not in studentIDs:
                return render_template("admin_login.html", 
                    data=" ❌ The id is not registered")
            
            admin_info = dataset(id)[0]
            stored_password = admin_info.get("password", "")
            
            try:
                password_matches = bcrypt.checkpw(password.encode('utf-8'), 
                    stored_password.encode('utf-8'))
            except ValueError:
                password_matches = (password == stored_password)
            
            if password_matches and admin_info["email"] == email:
                logger.info(f"Successful login for admin ID: {id}")
                reset_failed_attempts(id)
                secret = generate_and_send_otp(email)
                session['admin_otp_secret'] = secret
                session['pending_admin_id'] = id
                return redirect(url_for('admin_otp'))
            else:
                logger.warning(f"Failed login attempt for admin ID: {id}")
                record_failed_attempt(id)
                return render_template("admin_login.html", 
                    data=" ❌ Email/Password Incorrect")
    
    return render_template("admin_login.html")


@app.route("/admin_otp", methods=["GET"])
def admin_otp():
    if 'admin_otp_secret' not in session:
        return redirect(url_for('admin_login'))
    return render_template("admin_otp.html")


@app.route("/verify_admin_otp", methods=["POST"])
def verify_admin_otp():
    if 'admin_otp_secret' not in session:
        return redirect(url_for('admin_login'))
    
    otp = request.form.get("otp")
    if not otp:
        return render_template("admin_otp.html", data="Please enter OTP")
    
    totp = pyotp.TOTP(session['admin_otp_secret'])
    if totp.verify(otp):
        # OTP is valid
        session.pop('admin_otp_secret', None)
        session.pop('pending_admin_id', None)
        return redirect(url_for("admin"))
    else:
        return render_template("admin_otp.html", data="❌ Invalid OTP")


@app.route("/admin")
def admin():
    all_student_info = []
    studentIDs, _ = add_image_database()
    for i in studentIDs:
        all_student_info.append(dataset(i))
    return render_template("admin.html", data=all_student_info)


@app.route("/admin/admin_attendance_list", methods=["GET", "POST"])
def admin_attendance_list():
    if request.method == "POST":
        if request.form.get("button_student") == "VALUE1":
            already_marked_id_student.clear()
            return redirect(url_for("admin_attendance_list"))
        else:
            request.form.get("button_admin") == "VALUE2"
            already_marked_id_admin.clear()
            return redirect(url_for("admin_attendance_list"))
    else:
        unique_id_admin = list(set(already_marked_id_admin))
        student_info = []
        for i in unique_id_admin:
            student_info.append(dataset(i))
        return render_template("admin_attendance_list.html", data=student_info)



#########################################################################################################################

def add_image_database():
    folderPath = "./static/Files/Images"
    imgPathList = os.listdir(folderPath)
    imgList = []
    studentIDs = []

    for path in imgPathList:
        imgList.append(cv2.imread(os.path.join(folderPath, path)))
        studentIDs.append(os.path.splitext(path)[0])

        fileName = f"{folderPath}/{path}"
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)

    return studentIDs, imgList


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


@app.route("/admin/add_user", methods=["GET", "POST"])
def add_user():
    if request.method == "POST":
        try:
            # Get form data
            password = request.form.get("password", False)
            if not password:
                return render_template("add_user.html", error="Password is required")

            # Validate password
            is_valid, msg = PasswordValidator.validate_password(password)
            if not is_valid:
                return render_template("add_user.html", error=msg)

            # Hash password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            id = request.form.get("id", False)
            name = request.form.get("name", False)
            email = request.form.get("email", False)
            
            logger.info(f"Adding new user - ID: {id}, Name: {name}")
            
            if 'image' not in request.files:
                return render_template("add_user.html", error="No file uploaded")
                
            file = request.files['image']
            try:
                filename = secure_file_upload(file)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
                # Store user with hashed password
                ref = db.reference("Students")
                ref.child(id).set({
                    "name": name,
                    "email": email,
                    "password": hashed_password.decode('utf-8'),  # Store as string
                    "total_attendance": 0,
                    "last_attendance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                return redirect(url_for('admin'))
            except ValueError as e:
                return render_template("add_user.html", error=str(e))
                
        except Exception as e:
            logger.error(f"Error adding user - ID: {id}: {str(e)}")
            return render_template("add_user.html", error=str(e))

    return render_template("add_user.html")


#########################################################################################################################


@app.route("/admin/edit_user", methods=["POST", "GET"])
def edit_user():
    value = request.form.get("edit_student")

    studentInfo, imgStudent, secondElapsed = dataset(value)
    hoursElapsed = round((secondElapsed / 3600), 2)

    info = {
        "studentInfo": studentInfo,
        "lastlogin": hoursElapsed,
        "image": imgStudent,
    }

    return render_template("edit_user.html", data=info)


#########################################################################################################################


@app.route("/admin/save_changes", methods=["POST", "GET"])
def save_changes():
    content = request.get_data()

    dic_data = json.loads(content.decode("utf-8"))

    dic_data = {k: v.strip() for k, v in dic_data.items()}

    dic_data["year"] = int(dic_data["year"])
    dic_data["total_attendance"] = int(dic_data["total_attendance"])
    dic_data["starting_year"] = int(dic_data["starting_year"])

    update_student = db.reference(f"Students")

    update_student.child(dic_data["id"]).update(
        {
            "id": dic_data["id"],
            "name": dic_data["name"],
            "dob": dic_data["dob"],
            "address": dic_data["address"],
            "phone": dic_data["phone"],
            "email": dic_data["email"],
            "major": dic_data["major"],
            "starting_year": dic_data["starting_year"],
            "standing": dic_data["standing"],
            "total_attendance": dic_data["total_attendance"],
            "year": dic_data["year"],
            "last_attendance_time": dic_data["last_attendance_time"],
            "content": dic_data["content"],
        }
    )

    return "Data received successfully!"


#########################################################################################################################


def delete_image(student_id):
    filepath = f"./static/Files/Images/{student_id}.png"

    os.remove(filepath)

    bucket = storage.bucket()
    blob = bucket.blob(filepath)
    blob.delete()

    return "Successful"


@app.route("/admin/delete_user", methods=["POST", "GET"])
def delete_user():
    try:
        content = request.get_data()
        student_id = json.loads(content.decode("utf-8"))
        logger.info(f"Attempting to delete user - ID: {student_id}")
        
        delete_student = db.reference(f"Students")
        delete_student.child(student_id).delete()

        delete_image(student_id)

        studentIDs, imgList = add_image_database()

        encodeListKnown = findEncodings(imgList)

        encodeListKnownWithIds = [encodeListKnown, studentIDs]

        file = open("EncodeFile.p", "wb")
        pickle.dump(encodeListKnownWithIds, file)
        file.close()

        logger.info(f"Successfully deleted user - ID: {student_id}")
        return "Successful"
    except Exception as e:
        logger.error(f"Error deleting user - ID: {student_id}: {str(e)}")
        raise


#########################################################################################################################

@app.route("/resend_otp", methods=["POST"])
def resend_otp():
    try:
        if 'pending_student_id' in session:
            student_info = dataset(session['pending_student_id'])[0]
            email = student_info['email']
            logger.info(f"Resending OTP for student: {session['pending_student_id']}")
        elif 'pending_admin_id' in session:
            admin_info = dataset(session['pending_admin_id'])[0]
            email = admin_info['email']
            logger.info(f"Resending OTP for admin: {session['pending_admin_id']}")
        else:
            logger.warning("Resend OTP attempted without pending verification")
            return jsonify({'error': 'No pending verification'}), 400

        # Generate and send new OTP
        secret = generate_and_send_otp(email)
        if 'pending_student_id' in session:
            session['otp_secret'] = secret
        else:
            session['admin_otp_secret'] = secret
        
        logger.info(f"Successfully resent OTP to {email}")
        return jsonify({'message': 'New verification code sent'}), 200

    except Exception as e:
        logger.error(f"Error in resend_otp: {str(e)}")
        return jsonify({'error': 'Failed to send new code'}), 500

def verify_otp(otp, email):
    if email not in otp_storage:
        return False, "No OTP found for this email"
        
    otp_data = otp_storage[email]
    if datetime.now() > otp_data['expires_at']:
        del otp_storage[email]  # Clean up expired OTP
        return False, "Code has expired"
        
    totp = pyotp.TOTP(otp_data['secret'])
    if totp.verify(otp):
        del otp_storage[email]  # Clean up used OTP
        return True, "Success"
        
    return False, "Invalid code"

#########################################################################################################################

class PasswordValidator:
    MIN_LENGTH = 8
    MAX_LENGTH = 30

    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """
        Validate password against security requirements.
        Returns: Tuple(is_valid: bool, message: str)
        """
        # Check password length
        if len(password) < PasswordValidator.MIN_LENGTH:
            return False, f"Password must be at least {PasswordValidator.MIN_LENGTH} characters long"
        
        if len(password) > PasswordValidator.MAX_LENGTH:
            return False, f"Password must not exceed {PasswordValidator.MAX_LENGTH} characters"

        # Check for uppercase letters
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter (A-Z)"

        # Check for lowercase letters
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter (a-z)"

        # Check for numbers
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number (0-9)"

        # Check for special characters
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"

        # Check for common patterns
        common_patterns = [
            r"12345",
            r"qwerty",
            r"password",
            r"admin",
            r"abc123",
        ]
        for pattern in common_patterns:
            if pattern in password.lower():
                return False, f"Password contains a common pattern ({pattern})"

        # Check for repeating characters
        if re.search(r"(.)\1{2,}", password):
            return False, "Password must not contain repeating characters (e.g., 'aaa')"

        return True, "Password meets all requirements"

@app.route("/change_password", methods=["POST"])
def change_password():
    new_password = request.form.get("new_password")
    confirm_password = request.form.get("confirm_password")
    
    # Check if passwords match
    if new_password != confirm_password:
        return render_template("change_password.html", 
            error="Passwords do not match")
    
    # Validate new password
    is_valid, message = PasswordValidator.validate_password(new_password)
    if not is_valid:
        return render_template("change_password.html", error=message)
    
    # Continue with password change...

#########################################################################################################################

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return "An internal error occurred", 500

@app.after_request
def log_after_request(response):
    if response.status_code >= 400:
        logger.warning(
            f"Request to {request.path} failed with status {response.status_code}"
        )
    return response

def verify_face(face_encoding, stored_encodings, stored_ids):
    """
    Verify face with additional security checks
    Returns: (is_match, student_id, confidence)
    """
    try:
        if not face_encoding.any():
            return False, None, 0
            
        # Get face distances
        face_distances = face_recognition.face_distance(stored_encodings, face_encoding)
        
        if len(face_distances) == 0:
            return False, None, 0
            
        # Get best match
        best_match_index = np.argmin(face_distances)
        min_distance = face_distances[best_match_index]
        
        # Check if the match is good enough
        if min_distance > MAX_FACE_DISTANCE:
            logger.warning(f"Face match distance too high: {min_distance}")
            return False, None, 0
            
        # Calculate confidence score (0-100%)
        confidence = (1 - min_distance) * 100
        
        if confidence < FACE_CONFIDENCE_THRESHOLD * 100:
            logger.warning(f"Face match confidence too low: {confidence}%")
            return False, None, confidence
            
        return True, stored_ids[best_match_index], confidence
        
    except Exception as e:
        logger.error(f"Face verification error: {str(e)}")
        return False, None, 0

def check_face_quality(face_image):
    """
    Check if face image meets quality requirements
    """
    try:
        # Check image size
        height, width = face_image.shape[:2]
        if height < REQUIRED_FACE_SIZE[1] or width < REQUIRED_FACE_SIZE[0]:
            return False, "Face too small in image"
            
        # Check if face is too blurry
        laplacian = cv2.Laplacian(face_image, cv2.CV_64F).var()
        if laplacian < 100:  # Adjust threshold as needed
            return False, "Image too blurry"
            
        # Check face angle (assuming face landmarks are available)
        face_landmarks = face_recognition.face_landmarks(face_image)
        if not face_landmarks:
            return False, "Cannot detect facial features clearly"
            
        return True, "OK"
        
    except Exception as e:
        logger.error(f"Face quality check error: {str(e)}")
        return False, str(e)

def sanitize_input(text):
    """Sanitize user input to prevent XSS attacks"""
    if not text:
        return None
        
    # Define allowed HTML tags and attributes
    ALLOWED_TAGS = []  # No HTML tags allowed
    ALLOWED_ATTRIBUTES = {}  # No attributes allowed
    
    # Clean and strip the input
    cleaned_text = clean(
        str(text),
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True,
        strip_comments=True
    )
    
    # Remove any potential script injections
    cleaned_text = cleaned_text.replace('javascript:', '')
    cleaned_text = cleaned_text.replace('data:', '')
    
    return cleaned_text

def generate_2fa_codes(user_id, email, is_admin=False):
    """Generate both QR and Email OTP"""
    # Generate QR secret
    qr_secret = pyotp.random_base32()
    totp = pyotp.TOTP(qr_secret)
    issuer = "Admin Portal" if is_admin else "Student Portal"
    provisioning_uri = totp.provisioning_uri(email, issuer_name=issuer)
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert QR to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    qr_code = base64.b64encode(buffered.getvalue()).decode()
    
    # Generate email OTP
    email_otp = pyotp.TOTP(pyotp.random_base32()).now()
    
    # Send email OTP
    msg = Message('Your Login Verification Code',
                 sender=app.config['MAIL_USERNAME'],
                 recipients=[email])
    msg.body = f'Your email verification code is: {email_otp}\n\nOr scan the QR code in the login page with your authenticator app.'
    mail.send(msg)
    
    return qr_secret, qr_code, email_otp

@app.route("/verify_login", methods=["POST"])
def verify_login():
    user_id = session.get('pending_user_id')
    user_type = session.get('pending_user_type')
    
    if not user_id:
        return redirect(url_for('login'))
    
    qr_code = request.form.get('qr_code')
    email_otp = request.form.get('email_otp')
    
    # Verify either QR code or email OTP
    is_valid = False
    if qr_code:
        totp = pyotp.TOTP(session.get('qr_secret'))
        is_valid = totp.verify(qr_code)
    elif email_otp:
        is_valid = (email_otp == session.get('email_otp'))
    
    if is_valid:
        # Clear session data
        session.pop('qr_secret', None)
        session.pop('email_otp', None)
        session.pop('pending_user_id', None)
        session.pop('pending_user_type', None)
        
        if user_type == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    
    return render_template('verify_2fa.html', 
                         error="Invalid verification code",
                         qr_code=session.get('qr_code'))

def verify_password(id, password, is_admin=False):
    """
    Verify user password with security checks
    Returns: (is_valid, message)
    """
    try:
        # Input validation
        if not id or not password:
            return False, "ID and password required"
            
        # Get user info
        user_info = dataset(id)[0]
        if not user_info:
            logger.warning(f"Login attempt with invalid ID: {id}")
            return False, "Invalid credentials"
            
        stored_password = user_info.get("password", "")
        
        # Verify password
        try:
            is_valid = bcrypt.checkpw(
                password.encode('utf-8'), 
                stored_password.encode('utf-8')
            )
        except ValueError:
            # Fallback for legacy passwords
            is_valid = (password == stored_password)
            
        if not is_valid:
            logger.warning(f"Failed login attempt for ID: {id}")
            return False, "Invalid credentials"
            
        logger.info(f"Successful password verification for ID: {id}")
        return True, "Success"
        
    except Exception as e:
        logger.error(f"Password verification error: {str(e)}")
        return False, "An error occurred"

if __name__ == "__main__":
    app.run(debug=True)
