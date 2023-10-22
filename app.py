from flask import Flask, render_template, flash, redirect, url_for, request, current_app, jsonify
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from flask_mail import Mail, Message
from flask_cors import CORS
from itsdangerous import URLSafeTimedSerializer as Serializer
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from subprocess import call
import time
import json
import shutil
import os
import cv2
import argparse

from config import Config
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flask_wtf.file import FileField, FileRequired
import jwt

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = os.path.join('upload')

app = Flask(__name__)
CORS(app)

app.config.from_object(Config)
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail = Mail(app)
login = LoginManager(app)
login.login_view = 'login'
login.login_message = 'You must login to access this page'
login.login_message_category = 'info'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fps_trans(inputfile, fps_out):
    video_capture = cv2.VideoCapture("upload/" + str(inputfile[0]))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_in = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('upload/'+ str(int(fps_out)) + 'fps_' + str(inputfile[0]), fourcc, fps_out, (width, height))
    # Set the frame interpolation factor
    interp_factor = fps_in / fps_out
    if (interp_factor < 1):
        interp_factor = fps_out / fps_in
    
    # Process each frame in the original video
    while True:
        # Read the next frame from the original video
        ret, frame = video_capture.read()
    
        # If there are no more frames, break out of the loop
        if not ret:
            break
    
        # Perform frame interpolation to create additional frames
        for i in range(int(interp_factor)):
            out.write(frame)
    
    # Release the video capture and writer objects
    video_capture.release()
    out.release()

# def send_reset_password_mail(user, token):
#     msg = Message("[偏鄉醫療-心率血壓檢測追蹤網站] Reset Your Password",
#                   sender=current_app.config['MAIL_USERNAME'],
#                   recipients=[user.email],
#                   html=render_template('reset_password_mail.html', user=user, token=token))
#     mail.send(msg)

class User(db.Model, UserMixin):                                     # 可藉由cmd創建database
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False) # unique 表示該用戶名為唯一，只能註冊一次
    password = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)   # unique 表示該Email為唯一，只能註冊一次
    avatar_img = db.Column(db.String(120), default='static/images/profile-picture.jpg', nullable=False)
    
    # Define relationship with HeartPressureRecord
    records = db.relationship('HeartPressureRecord', back_populates='user')
    
    def __repr__(self):
        return '<User %r>' % self.username                           # 註冊成功回傳 <User 註冊用戶名>
    
    def generate_reset_password_token(self, expires_in=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expires_in=expires_in)
        return s.dumps({'reset_id': self.id})
        #return jwt.encode({"id": self.id}, current_app.config['SECRET_KEY'], algorithm="HS256")
    
    @staticmethod
    def check_reset_password_token(token):
        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            return User.query.filter_by(id=data['id']).first()
        except:
            return

class HeartPressureRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    bpm = db.Column(db.Float)
    systolic = db.Column(db.Float)
    diastolic = db.Column(db.Float)
    record_time = db.Column(db.DateTime, default=datetime.utcnow)

    # Define relationship with User
    user = db.relationship('User', back_populates='records')

''' ---------- 表單設定區 ---------- '''
# 註冊表單
class RegisterForm(FlaskForm):                                                                          # 創建表單
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])              # 使用者名最短3個字最多20個字
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=20)])            # 密碼最短8個字最多20個字
    confirm = PasswordField('Repeat Password',  validators=[DataRequired(), EqualTo('password')])       # 密碼確認
    #recaptch = RecaptchaField()
    submit = SubmitField('Register')
    
    def validate_username(self, username):                                     # 檢查該用戶名是否已存在database
        user = User.query.filter_by(username = username.data).first()
        if user:
            raise ValidationError('Username already token')
            
    def validate_email(self, email):                                           # 檢查該Email是否已存在database
        user = User.query.filter_by(email = email.data).first()
        if user:
            raise ValidationError('Email already token')

# 登入表單
class LoginForm(FlaskForm):                                                                          # 創建表單
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])           # 使用者名最短3個字最多20個字
    #email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=20)])         # 密碼最短8個字最多20個字
    remember = BooleanField('Remember')
    submit = SubmitField('Sign In')
    
# 忘記密碼要求重設的email信件表單
class PasswordResetRequestForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Length(5, 30), Email()])
    submit = SubmitField('Send')
    def validate_email(self, email): 
        """
        驗證是否有相關的EMAIL在資料庫內，若沒有就不寄信
        """
        user = User.query.filter_by(email = email.data).first()
        if not user:
            raise ValidationError('Email not exists, Please Check!')
            
# 重置密碼表單
class ResetPasswordForm(FlaskForm):                                                                          # 創建表單
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=20)])            # 密碼最短8個字最多20個字
    confirm = PasswordField('Repeat Password',  validators=[DataRequired(), EqualTo('password')])       # 密碼確認
    submit = SubmitField('Reset Password')
  
# 頭像上傳表單
class PhotoForm(FlaskForm):
    photo = FileField(validators=[FileRequired()])
    submit = SubmitField('Upload')

''' ---------- 表單設定區 ---------- '''

@login.user_loader
def load_user(id):
    return User.query.filter_by(id=id).first()

# 登入
@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    title = 'User Login'
    username = None
    password = None
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        remember = form.remember.data
        # 檢測密碼是否與使用者匹配
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            # User exists and password matched
            login_user(user, remember=remember)
            #flash('Login Success', category='info')
            if request.args.get('next'):
                next_page = request.args.get('next')
                return redirect(next_page)
            return redirect(url_for('index'))
        flash('User not exists or password not match', category='danger')
    return render_template('login.html', form=form, title=title)

# 登出
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# 註冊
@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    title = 'Create a Account Now'
    username = None
    email = None
    password = None
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        form.username.data=""
        email = form.email.data
        form.email.data=""
        password = bcrypt.generate_password_hash(form.password.data)
        form.password.data=""
        user = User(username=username, email=email, password=password)
        os.mkdir('./static/' + str(username))
        db.session.add(user)
        db.session.commit()
        flash('Congrats, registeration success', category='success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form, title=title)

# 若忘記密碼可到此頁面申請發送重置密碼Email(有bug，先擱著待修)
@app.route('/send_password_reset_request', methods=['GET','POST'])
def send_password_reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    title = 'Send Reset Password Email'
    form = PasswordResetRequestForm()
    if form.validate_on_submit():
        #  取得使用者資料
        user = User.query.filter_by(email=form.email.data).first()
        
        #token = user.generate_reset_password_token()
        print(user)
        #send_reset_password_mail(user, token)
        #flash('Password reset request mail is sent, please check your mailbox.', category='info')
    return render_template('send_password_reset_request.html', form=form, title=title)

# 重置密碼介面
@app.route('/reset_password', methods=['GET','POST'])
@login_required
def reset_password():
    title = 'Reset Your Password'
    form = ResetPasswordForm()
    if form.validate_on_submit():
        email = current_user.email
        user = User.query.filter_by(email=email).first()
        print(user)
        if user:
            user.password = bcrypt.generate_password_hash(form.password.data)
            db.session.commit()
            flash('Your password reset is done', category='info')
            return redirect(url_for('reset_password'))
    return render_template('reset_password.html', form=form, title=title)

# 帳戶資訊管理
@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    title = 'Your Profile'
    return render_template('profile.html', title=title)

# 上傳頭像
@app.route('/upload_photo', methods=['GET','POST'])
@login_required
def upload_photo():
    title = 'Upload Your Avatar Image'
    form = PhotoForm()
    if form.validate_on_submit():
        f = form.photo.data
        if f.filename == '':
            flash('No selected file', category='danger')
            return render_template('upload_photo.html', title=title, form=form)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join('./static/' + str(current_user.username), filename))
            current_user.avatar_img = './static/' + str(current_user.username) + '/' + filename
            db.session.commit()
            return redirect(url_for('profile'))
    return render_template('upload_photo.html', title=title, form=form)

# PPG與ECG介紹介面渲染  
@app.route('/ppg_and_ecg', methods=['GET','POST'])
def ppg_and_ecg():
    title = 'PPG與ECG介紹'
    return render_template('ppg_and_ecg.html', title=title)  

# HRV介紹介面渲染 
@app.route('/hrv', methods=['GET','POST'])
def hrv():
    title = 'HRV介紹'
    return render_template('hrv.html', title=title)

# ABP介紹介面渲染 
@app.route('/abp', methods=['GET','POST'])
def abp():
    title = 'ABP介紹'
    return render_template('abp.html', title=title)

# References介面渲染
@app.route('/references', methods=['GET','POST'])
def references():
    title = 'References'
    return render_template('references.html', title=title)
    
# 檢測介面渲染
@app.route('/detection', methods=['GET','POST'])
@login_required
def detection():
    title = '心律&血壓檢測'
    return render_template('detection.html', title=title)

# 開始檢測與紀錄每次數據
@app.route('/perform_calculation', methods=['GET','POST'])
@login_required
def perform_calculation():
    file = request.files['fileToUpload']
    # if not file:
    #     return jsonify({'error': '請選擇一個文件進行計算。'})
    if file.filename != '':
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    
    # ------------  模型計算區間  ------------
    # 初始所有資料檔案
    for f in os.listdir('./ABP_data/beats/00000'):
        os.remove(os.path.join('./ABP_data/beats/00000', f))
    for f in os.listdir('./ABP_data/extracted/00000'):
        os.remove(os.path.join('./ABP_data/extracted/00000', f))
    for f in os.listdir('./ABP_data/preprocessed/00000'):
        os.remove(os.path.join('./ABP_data/preprocessed/00000', f))
    for f in os.listdir('./ABP_data/videos/00000'):
        os.remove(os.path.join('./ABP_data/videos/00000', f))
    for f in os.listdir('./ABP_data'):
        if f == 'test_output.p' or f == 'test_output_approximate.p':
            os.remove(os.path.join('./ABP_data', f))
            
    # 初始化將數據歸零
    with open("config.json", mode='r') as file:
        data = json.load(file)
    data["bpm"] = 0.0
    data["ibi"] = 0.0
    data["sdnn"] = 0.0
    data["sdsd"] = 0.0
    data["rmssd"] = 0.0
    data["pnn20"] = 0.0
    data["pnn50"] = 0.0
    data["lf"] = 0.0
    data["hf"] = 0.0
    data["lf/hf"] = 0.0
    data["Systolic"] = 0.0
    data["Diastolic"] = 0.0
    with open("config.json", mode='w') as file:
        json.dump(data, file)    

    # 將upload裡的檔案轉換成各自需求並複製到模型下的data\videos\00000，並刪除upload中的檔案
    path_upload = 'upload/'
    inputfile = os.listdir(path_upload)
    print('transfer the video to 125 fps...')
    fps_trans(inputfile, 125.0)

    shutil.copyfile(path_upload + '125fps_' + str(inputfile[0]), './ABP_data/videos/00000/' + str(inputfile[0]))
    for f in os.listdir(path_upload):
        os.remove(os.path.join(path_upload, f))

    # 呼叫檔案執行
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--force_redo', action="store_true",)
    parser.add_argument('-f', '--features_onwards', action="store_true", )
    parser.add_argument('-o', '--offset', action="store", type=int, default=0)

    args = parser.parse_args()

    scripts_offset = args.offset

    if args.features_onwards:
        scripts_offset = 6

    scripts = [
        "./ABP_signal_extractor.py",  # video to signal
        "./ABP_signal_preprocessor.py",  # preprocess signal
        "./ABP_signal_beat_separation.py",  # separate beats
        "./signal_calculation.py",
        "./ABP_transform.py",          # prepare top.p file
        "./predict_test.py"
    ][scripts_offset:]

    for s in scripts:
        cmd = "python {} {}".format(
            s,
            "-r" if args.force_redo else ""
        )
        cmd = cmd.strip()
        print("###### %s ######" % cmd)
        print("cmd -> ", cmd)
        cmd = cmd.split(" ")
        call(cmd)
    
    # 初始所有資料檔案
    for f in os.listdir('./ABP_data/beats/00000'):
        os.remove(os.path.join('./ABP_data/beats/00000', f))
    for f in os.listdir('./ABP_data/extracted/00000'):
        os.remove(os.path.join('./ABP_data/extracted/00000', f))
    for f in os.listdir('./ABP_data/preprocessed/00000'):
        os.remove(os.path.join('./ABP_data/preprocessed/00000', f))
    for f in os.listdir('./ABP_data/videos/00000'):
        os.remove(os.path.join('./ABP_data/videos/00000', f))
    for f in os.listdir('./ABP_data'):
        if f == 'test_output.p' or f == 'test_output_approximate.p':
            os.remove(os.path.join('./ABP_data', f))
    
    # 將儲存資料的數據回傳網頁
    with open("config.json", mode='r') as file:
        calculated_result = json.load(file)
    # ------------  模型計算區間  ------------

    # calculated_result = {
    #     'bpm': 80.0,
    #     'Systolic': 120.0,
    #     'Diastolic': 70.0
    # }    

    # 將數值保留到database
    bpm = float(calculated_result['bpm'])
    systolic = float(calculated_result['Systolic'])
    diastolic = float(calculated_result['Diastolic'])
    if bpm != 0 and systolic != 0 and diastolic != 0 and systolic >= 200 and diastolic >= 200 and systolic <= 0 and diastolic <= 0:
        record_time = datetime.now()
        record = HeartPressureRecord(user=current_user, bpm=bpm, systolic=systolic, diastolic=diastolic, record_time=record_time)   
        db.session.add(record)
        db.session.commit()
    else:
        print("One or more values are 0, not saving to database")
    
    # 印出每次紀錄數值
    user_records = HeartPressureRecord.query.filter_by(user_id=current_user.id).all()
    print(current_user)
    for record in user_records:
        print(f"Time: {record.record_time}, Heart Rate: {record.bpm}, Systolic: {record.systolic}, Diastolic: {record.diastolic}")
    
    return jsonify(calculated_result)

# 一週紀錄介面渲染
@app.route('/week_rec', methods=['GET','POST'])
@login_required
def week_rec():
    title = '心律&血壓一週紀錄'
    return render_template('week_rec.html', title=title)

# 一週紀錄折線圖繪製
@app.route('/draw', methods=['GET','POST'])
def draw():
    if current_user.is_authenticated:
        user_id = current_user.id
        # 只顯示7天內的資料
        seven_days_ago = datetime.now() - timedelta(days=7)# minutes=5  days=7
        records = HeartPressureRecord.query.filter_by(user_id=user_id).filter(HeartPressureRecord.record_time >= seven_days_ago).all()
        export_data = []
        for record in records:
            record_dict = {
                'bpm': record.bpm,
                'systolic': record.systolic,
                'diastolic': record.diastolic,
                'record_time': record.record_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            export_data.append(record_dict)
        with open('record.json', 'w') as json_file:
            json.dump(export_data, json_file, indent=4)
        return jsonify({'export_data': export_data})
            
# HOME介面渲染
@app.route('/', methods=['GET','POST'])
def index():
    title = '偏鄉醫療-心率血壓檢測追蹤網站'
    return render_template('index.html', title=title)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4445)#, debug=True