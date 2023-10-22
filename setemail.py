from flask import current_app, render_template
from flask_mail import Message
from app import mail

def send_reset_password_mail(user, token):
    msg = Message("[偏鄉醫療-心率血壓檢測追蹤網站] Reset Your Password",
                  sender=current_app.config['MAIL_USERNAME'],
                  recipients=[user.email],
                  html=render_template('reset_password_mail.html', user=user, token=token))
    mail.send(msg)