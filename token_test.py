from app import app
from app import User

#with app.app_context():
#    user = User.query.all()[0]
#    print(user.generate_reset_password_token())
    
with app.app_context():
    user = User.query.all()[0]
    print(user.check_reset_password_token(b'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MX0.o85_mpjLKHHBtX05k0UD1s3_9Af0Cz-0RdO3X6Q1600'))