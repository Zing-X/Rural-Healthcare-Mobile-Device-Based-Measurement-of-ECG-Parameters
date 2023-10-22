from app import db, HeartPressureRecord

# 刪除 HeartPressureRecord 表中的所有數據
db.session.query(HeartPressureRecord).delete()
db.session.commit()