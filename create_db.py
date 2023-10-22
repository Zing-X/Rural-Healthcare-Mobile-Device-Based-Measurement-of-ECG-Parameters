# 創建資料庫
from app import app, db

with app.app_context():
    db.create_all()