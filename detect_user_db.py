# 顯示目前database中所有用戶
from app import app
from app import User

with app.app_context():
    users = User.query.all()
    for user in users:
        print(user.username)
