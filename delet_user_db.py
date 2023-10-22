# 刪除目前database中所有用戶
from app import app, db
from app import User

with app.app_context():
    db.session.query(User).delete()
    db.session.commit()
