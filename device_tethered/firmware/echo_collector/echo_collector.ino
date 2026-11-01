// ECHO Collector -- tethered sensor head firmware (ESP32-S3).
//
// The head is ONLY sensors: a BNO085 IMU and a VL53L5CX 8x8 ToF on
// one I2C bus, plus a record button and a status LED. It streams
// ECHO-T1 binary frames (see ../../README.md for the protocol spec;
// device_tethered/protocol.py is the reference implementation and
// this sketch must emit byte-identical frames) over USB serial to
// the PC, which does all recording, compute and display. The UVC
// camera plugs straight into the PC and is not this sketch's
// business.
//
// Arduino IDE setup:
//   Board:   "ESP32S3 Dev Module" (or your board's S3 profile),
//            with "USB CDC On Boot: Enabled" so Serial is the USB
//            port the PC recorder opens.
//   Libraries (Library Manager): "SparkFun BNO08x Arduino Library",
//            "SparkFun VL53L5CX Arduino Library".
//
// Wiring (breadboard, all pins changeable below):
//   I2C SDA = GPIO8, SCL = GPIO9, 3.3V + GND to both breakouts.
//   BNO085 at I2C 0x4B (its default), VL53L5CX at 0x29: no clash.
//   Button: GPIO11 to GND (internal pull-up). LED: GPIO10 + resistor.
//
// This sketch has NOT been compiled or run on hardware from the repo
// (no toolchain here). The bring-up checklist in ../../README.md
// covers what to verify in the first hardware session.

#include <Wire.h>
#include <SparkFun_BNO08x_Arduino_Library.h>
#include <SparkFun_VL53L5CX_Library.h>

// ---------------------------------------------------------------- pins
#define I2C_SDA_PIN   8
#define I2C_SCL_PIN   9
#define BUTTON_PIN    11
#define LED_PIN       10

// ------------------------------------------------------------- protocol
// Frame: A5 5A | type u8 | tick u32 LE (micros(), wraps at 2^32) |
// len u16 LE | payload | crc16 LE (CCITT-FALSE over type..payload).
#define FRAME_IMU     0x01
#define FRAME_TOF     0x02
#define FRAME_BUTTON  0x03
#define FRAME_HELLO   0x04
#define BUTTON_PRESS  0x01

#define IMU_REPORT_MS 10    // 100 Hz rotation vector (+ accel/gyro)
#define TOF_HZ        15    // max ranging frequency at 8x8

SFE_BNO08x imu;
SparkFun_VL53L5CX tof;
VL53L5CX_ResultsData tofData;

static uint16_t crc16Ccitt(const uint8_t *data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; b++)
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021)
                           : (uint16_t)(crc << 1);
  }
  return crc;
}

static void sendFrame(uint8_t type, const uint8_t *payload, uint16_t len) {
  uint8_t buf[9 + 128 + 2];
  uint32_t tick = micros();
  buf[0] = 0xA5; buf[1] = 0x5A; buf[2] = type;
  memcpy(buf + 3, &tick, 4);
  memcpy(buf + 7, &len, 2);
  memcpy(buf + 9, payload, len);
  uint16_t crc = crc16Ccitt(buf + 2, 7 + len);
  memcpy(buf + 9 + len, &crc, 2);
  Serial.write(buf, 9 + len + 2);
}

static void sendHello() {
  const char *text = "ECHO-T1 fw=0.1 proto=1 imu=100 tof=15";
  sendFrame(FRAME_HELLO, (const uint8_t *)text, strlen(text));
}

// ---------------------------------------------------------------- setup
void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Serial.begin(921600);  // nominal; USB CDC is not baud-limited
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);

  if (!imu.begin(0x4B, Wire)) {
    // Dead IMU: blink an SOS-ish pattern forever -- a silent head
    // that streams only ToF would corrupt captures downstream.
    while (true) { digitalWrite(LED_PIN, HIGH); delay(100);
                    digitalWrite(LED_PIN, LOW);  delay(100); }
  }
  imu.enableRotationVector(IMU_REPORT_MS);
  imu.enableAccelerometer(IMU_REPORT_MS);
  imu.enableGyro(IMU_REPORT_MS);

  if (!tof.begin()) {
    while (true) { digitalWrite(LED_PIN, HIGH); delay(400);
                   digitalWrite(LED_PIN, LOW);  delay(100); }
  }
  tof.setResolution(8 * 8);
  tof.setRangingFrequency(TOF_HZ);
  tof.startRanging();

  sendHello();
}

// ----------------------------------------------------------------- loop
// Freshest accel/gyro, refreshed by their own reports; an IMU frame
// goes out once per rotation-vector report (100 Hz), carrying the
// freshest accel/gyro, exactly as protocol.py documents.
static float acc[3] = {0, 0, 0}, gyr[3] = {0, 0, 0};
static bool haveAcc = false, haveGyr = false;

static void pollImu() {
  while (imu.dataAvailable()) {
    uint8_t id = imu.getSensorEventID();
    if (id == SENSOR_REPORTID_ACCELEROMETER) {
      acc[0] = imu.getAccelX(); acc[1] = imu.getAccelY();
      acc[2] = imu.getAccelZ(); haveAcc = true;
    } else if (id == SENSOR_REPORTID_GYROSCOPE) {
      gyr[0] = imu.getGyroX(); gyr[1] = imu.getGyroY();
      gyr[2] = imu.getGyroZ(); haveGyr = true;
    } else if (id == SENSOR_REPORTID_ROTATION_VECTOR) {
      if (!haveAcc || !haveGyr) continue;
      float payload[10];
      payload[0] = imu.getQuatW(); payload[1] = imu.getQuatX();
      payload[2] = imu.getQuatY(); payload[3] = imu.getQuatZ();
      payload[4] = acc[0]; payload[5] = acc[1]; payload[6] = acc[2];
      payload[7] = gyr[0]; payload[8] = gyr[1]; payload[9] = gyr[2];
      sendFrame(FRAME_IMU, (const uint8_t *)payload, sizeof(payload));
    }
  }
}

static void pollTof() {
  if (!tof.isDataReady()) return;
  if (!tof.getRangingData(&tofData)) return;
  int16_t zones[64];
  for (int i = 0; i < 64; i++) {
    uint8_t status = tofData.target_status[i];
    // ST: status 5 (and 9) mean a valid range; anything else is
    // "no trustworthy target" and maps to the format's -1.
    zones[i] = (status == 5 || status == 9)
                   ? tofData.distance_mm[i] : (int16_t)-1;
  }
  sendFrame(FRAME_TOF, (const uint8_t *)zones, sizeof(zones));
}

static void pollButton() {
  static int last = HIGH;
  static uint32_t lastChangeMs = 0;
  int now = digitalRead(BUTTON_PIN);
  if (now != last && (uint32_t)(millis() - lastChangeMs) > 30) {
    lastChangeMs = millis();
    last = now;
    if (now == LOW) {
      uint8_t ev = BUTTON_PRESS;
      sendFrame(FRAME_BUTTON, &ev, 1);
    }
  }
}

static void pollDownlink() {
  // PC -> head, single bytes: 'R' recording (LED on), 'S' stopped
  // (LED off), 'H' please re-send HELLO (e.g. recorder just opened
  // the port after the head booted).
  while (Serial.available()) {
    int c = Serial.read();
    if (c == 'R') digitalWrite(LED_PIN, HIGH);
    else if (c == 'S') digitalWrite(LED_PIN, LOW);
    else if (c == 'H') sendHello();
  }
}

void loop() {
  pollImu();
  pollTof();
  pollButton();
  pollDownlink();
}
