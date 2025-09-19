import paho.mqtt.client as mqtt
import mysql.connector
import json

# ===== DB 연결 설정 =====
db = mysql.connector.connect(
    host="localhost",         # 관제센터 PC에서 MySQL 실행 중
    user="droneuser",
    password="0000",
    database="drone_parking"
)
cursor = db.cursor()

# ===== INSERT SQL 문 =====
insert_sql = """
INSERT INTO vehicle_detection_log (
    vehicle_number, detected_time, is_illegal,
    illegal_confidence, detect_confidence,
    latitude, longitude, image_path
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

# ===== MQTT 메시지 수신 처리 =====
def on_message(client, userdata, message):
    try:
        data = json.loads(message.payload.decode())
        values = (
            data.get("vehicle_number"),
            data.get("detected_time"),
            int(data.get("is_illegal", 0)),
            data.get("illegal_confidence"),
            data.get("detect_confidence"),
            data.get("latitude"),
            data.get("longitude"),
            data.get("image_path")
        )
        cursor.execute(insert_sql, values)
        db.commit()
        print(f"✅ Inserted into DB: {values}")
    except Exception as e:
        print(f"❌ DB insert failed: {e}")

# ===== MQTT 클라이언트 실행 =====
client = mqtt.Client("control-center-receiver")
client.on_message = on_message

client.connect("43.200.4.222", 1883)
client.subscribe("vehicle/detection")
print("📡 Waiting for detection data...")
client.loop_forever()
