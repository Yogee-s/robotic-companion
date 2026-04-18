// ════════════════════════════════════════════════════════════════════════
//  Companion Face — ESP32 firmware for Diymore 2.8" 240×320 (ILI9341 + XPT2046)
//
//  Receives state lines from the Jetson over serial (115200 baud) and
//  renders the robot face locally. Three scenes: FACE, QUICK_GRID, MORE_LIST.
//  Emits BTN / TOUCH messages back up when the user taps.
//
//  Protocol (line-delimited ASCII):
//    RX: FACE v=+0.72 a=+0.30 talk=1 listen=0 think=0 sleep=0 gaze=-12 privacy=0
//        VISEME ahh
//        SCENE face | quickgrid | morelist
//        PRIVACY 1 | 0
//    TX: BTN mute_mic | stop_talking | sleep | more | timer | remind_me | ...
//        TOUCH x y
// ════════════════════════════════════════════════════════════════════════

#include <TFT_eSPI.h>
#include <XPT2046_Touchscreen.h>

// ── Pins (Diymore ESP32-2432S028 / CYD) ────────────────────────────────
//  Touch chip XPT2046 is on its OWN SPI bus, NOT shared with the TFT.
#define TOUCH_CLK  25   // TP CLK
#define TOUCH_CS   33   // TP CS
#define TOUCH_MOSI 32   // TP DIN
#define TOUCH_MISO 39   // TP OUT  (input-only pin, which is fine for MISO)
#define TOUCH_IRQ  36   // TP IRQ  (input-only pin)

TFT_eSPI tft = TFT_eSPI();
TFT_eSprite fb = TFT_eSprite(&tft);   // off-screen framebuffer — all draws go here
SPIClass touchSPI = SPIClass(VSPI);
XPT2046_Touchscreen ts(TOUCH_CS, TOUCH_IRQ);

// ── Colours (Catppuccin Mocha — slightly warmed) ────────────────────────
static const uint16_t CLR_BG        = 0x18C3;  // #1e1e2e — base dark
static const uint16_t CLR_BG_SOFT   = 0x2124;  // slight lift for depth band
static const uint16_t CLR_SURFACE   = 0x3188;  // #313244
static const uint16_t CLR_EYE       = 0xFFBC;  // warm cream (not pure white)
static const uint16_t CLR_EYE_SHADE = 0xE75C;  // cream shadow / iris rim
static const uint16_t CLR_PUPIL     = 0x1062;  // deep blue-black
static const uint16_t CLR_HILITE    = 0xFFFF;  // white spark on pupil
static const uint16_t CLR_MOUTH     = 0xF2CA;  // warm coral
static const uint16_t CLR_MOUTH_DK  = 0xB185;  // coral shadow
static const uint16_t CLR_BROW      = 0xDEDB;  // light cream (eyebrows)
static const uint16_t CLR_TEXT      = 0xDEDB;
static const uint16_t CLR_ACCENT    = 0x44DF;

// ── Scenes ───────────────────────────────────────────────────────────────
enum Scene { SCENE_FACE, SCENE_QUICKGRID, SCENE_MORELIST };
Scene currentScene = SCENE_FACE;

// ── Face state (updated from serial) ─────────────────────────────────────
struct FaceState {
  float valence  = 0.0f;
  float arousal  = 0.0f;
  bool  talking  = false;
  bool  listening = false;
  bool  thinking = false;
  bool  sleep    = false;
  bool  privacy  = false;
  float gaze_deg = 0.0f;
};
FaceState fs;

String currentViseme = "rest";

unsigned long lastTouchMs = 0;
const unsigned long AUTO_DISMISS_MS = 4000;

unsigned long lastBlinkTickMs = 0;
bool blinkClosed = false;

void writeReply(const String& line) {
  Serial.println(line);
}

// ── Serial RX parsing ────────────────────────────────────────────────────
String rxBuf;

void parseFaceLine(const String& body) {
  // FACE v=+0.72 a=+0.30 talk=1 listen=0 think=0 sleep=0 gaze=-12 privacy=0
  int idx = 0;
  while (idx < (int)body.length()) {
    int sp = body.indexOf(' ', idx);
    String tok = sp < 0 ? body.substring(idx) : body.substring(idx, sp);
    tok.trim();
    int eq = tok.indexOf('=');
    if (eq > 0) {
      String k = tok.substring(0, eq);
      String v = tok.substring(eq + 1);
      if      (k == "v")        fs.valence  = v.toFloat();
      else if (k == "a")        fs.arousal  = v.toFloat();
      else if (k == "talk")     fs.talking  = (v.toInt() != 0);
      else if (k == "listen")   fs.listening = (v.toInt() != 0);
      else if (k == "think")    fs.thinking = (v.toInt() != 0);
      else if (k == "sleep")    fs.sleep    = (v.toInt() != 0);
      else if (k == "gaze")     fs.gaze_deg = v.toFloat();
      else if (k == "privacy")  fs.privacy  = (v.toInt() != 0);
    }
    if (sp < 0) break;
    idx = sp + 1;
  }
}

void handleLine(const String& line) {
  if (line.startsWith("FACE ")) {
    parseFaceLine(line.substring(5));
  } else if (line.startsWith("VISEME ")) {
    currentViseme = line.substring(7);
    currentViseme.trim();
  } else if (line.startsWith("SCENE ")) {
    String s = line.substring(6); s.trim();
    if      (s == "face")       currentScene = SCENE_FACE;
    else if (s == "quickgrid")  currentScene = SCENE_QUICKGRID;
    else if (s == "morelist")   currentScene = SCENE_MORELIST;
    lastTouchMs = millis();
  } else if (line.startsWith("PRIVACY ")) {
    fs.privacy = (line.substring(8).toInt() != 0);
  }
}

void pumpSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      handleLine(rxBuf);
      rxBuf = "";
    } else if (c != '\r' && rxBuf.length() < 200) {
      rxBuf += c;
    }
  }
}

// ── Drawing: face (into the off-screen sprite fb) ───────────────────────
// ── Expression helpers ─────────────────────────────────────────────────
enum Expression { EX_NEUTRAL, EX_HAPPY, EX_EXCITED, EX_SURPRISED,
                  EX_SAD, EX_ANGRY, EX_CALM };

Expression pickExpression(float v, float a) {
  if (a > 0.55f && v > -0.2f && v < 0.3f)  return EX_SURPRISED;
  if (a > 0.45f && v > 0.25f)              return EX_EXCITED;
  if (a > 0.35f && v < -0.25f)             return EX_ANGRY;
  if (v >  0.30f)                          return EX_HAPPY;
  if (v < -0.30f)                          return EX_SAD;
  if (a < -0.20f && v > -0.10f)            return EX_CALM;
  return EX_NEUTRAL;
}

// Pixel-plotted parabola from (cx-w/2, cy) → (cx+w/2, cy) peaking at
// (cx, cy ± height). `concaveDown=true` makes the middle dip DOWN (larger
// y) — i.e. a smile; false makes it rise UP — i.e. a frown / n-shape.
static void drawCurve(int16_t cx, int16_t cy, int16_t w, int16_t height,
                      uint16_t color, bool concaveDown,
                      int16_t thickness = 3) {
  int16_t hw = w / 2;
  float invHw2 = 1.0f / (float)(hw * hw);
  int16_t sign = concaveDown ? +1 : -1;
  for (int16_t x = -hw; x <= hw; ++x) {
    int16_t yoff = (int16_t)((float)height * (1.0f - (float)(x * x) * invHw2));
    int16_t y0 = cy + sign * yoff;
    for (int16_t t = 0; t < thickness; ++t) {
      fb.drawPixel(cx + x, y0 + t, color);
    }
  }
}

static void drawEye(int16_t cx, int16_t cy, int16_t r,
                    int16_t pupilOffX, int16_t pupilOffY,
                    int16_t upperLid, int16_t lowerLid,
                    bool happySquint, float pupilScale = 0.50f) {
  if (happySquint) {
    drawCurve(cx, cy + 2, r * 2 + 4, r - 6, CLR_EYE,
              /*concaveDown=*/false, /*thickness=*/4);
    return;
  }

  // Eye white + soft rim
  fb.fillCircle(cx, cy, r, CLR_EYE);
  fb.drawCircle(cx, cy, r, CLR_EYE_SHADE);
  fb.drawCircle(cx, cy, r - 1, CLR_EYE_SHADE);

  if (upperLid > 0) fb.fillRect(cx - r, cy - r, r * 2, upperLid, CLR_BG);
  if (lowerLid > 0) fb.fillRect(cx - r, cy + r - lowerLid, r * 2, lowerLid, CLR_BG);

  int16_t irisR  = (int16_t)(r * 0.70f);
  int16_t pupilR = (int16_t)(r * pupilScale);
  fb.fillCircle(cx + pupilOffX, cy + pupilOffY, irisR, CLR_EYE_SHADE);
  fb.fillCircle(cx + pupilOffX, cy + pupilOffY, pupilR, CLR_PUPIL);

  // Specular highlights — two dots give a "wet eye" feel
  fb.fillCircle(cx + pupilOffX + pupilR / 2 - 1,
                cy + pupilOffY - pupilR / 2 + 1, 3, CLR_HILITE);
  fb.fillCircle(cx + pupilOffX - pupilR / 3,
                cy + pupilOffY + pupilR / 3, 1, CLR_HILITE);
}

static void drawClosedEye(int16_t cx, int16_t cy, int16_t r, bool smileUp) {
  // Sleeping / blink: curved line. smileUp=true means corners down, middle
  // up (peaceful closed-eye arc); else corners up, middle down.
  drawCurve(cx, cy, r * 2 + 4, r - 6, CLR_EYE,
            /*concaveDown=*/!smileUp, /*thickness=*/3);
}

static void drawBrow(int16_t cx, int16_t cy, int16_t r, Expression ex, int side) {
  const int16_t bw = r + 6;
  int16_t innerY = cy - r - 12;
  int16_t outerY = cy - r - 12;

  switch (ex) {
    case EX_HAPPY:
    case EX_EXCITED:  innerY -= 2; outerY -= 6; break;
    case EX_SAD:      innerY -= 6; outerY += 4; break;
    case EX_ANGRY:    innerY += 7; outerY -= 3; break;
    case EX_SURPRISED:innerY -= 10; outerY -= 10; break;
    case EX_CALM:     innerY += 2; outerY += 2; break;
    default: break;
  }

  int16_t xL = cx - bw, xR = cx + bw;
  int16_t yL = (side < 0) ? outerY : innerY;
  int16_t yR = (side < 0) ? innerY : outerY;
  for (int t = 0; t < 4; ++t) {
    fb.drawLine(xL, yL + t, xR, yR + t, CLR_BROW);
  }
}

static void drawMouth(int16_t cx, int16_t cy, Expression ex,
                      float valence, bool talking, const String& viseme) {
  // Open-mouth shapes during speech
  bool wantOpen = talking &&
                  (viseme == "ahh" || viseme == "oh" ||
                   viseme == "eh"  || viseme == "l");
  if (wantOpen) {
    int16_t ow = 58, oh = 22;
    if      (viseme == "ahh") { ow = 64; oh = 30; }
    else if (viseme == "oh")  { ow = 44; oh = 34; }
    else if (viseme == "eh")  { ow = 70; oh = 18; }
    else if (viseme == "l")   { ow = 50; oh = 26; }
    fb.fillRoundRect(cx - ow / 2, cy - oh / 2, ow, oh, oh / 2, CLR_MOUTH_DK);
    fb.fillRoundRect(cx - ow / 2 + 2, cy - oh / 2 + 2, ow - 4, oh - 4,
                     oh / 2 - 2, CLR_MOUTH);
    return;
  }

  if (ex == EX_SURPRISED) {
    fb.fillCircle(cx, cy, 14, CLR_MOUTH_DK);
    fb.fillCircle(cx, cy, 11, CLR_MOUTH);
    return;
  }

  // Pixel-plotted parabolic curve — geometry is deterministic, no weird arcs.
  const int16_t w = 112;
  int16_t amp = (int16_t)(valence * 18.0f);
  if (amp > 2) {
    // Smile: corners up, middle dips down.
    drawCurve(cx, cy, w, amp, CLR_MOUTH_DK, /*concaveDown=*/true, 1);
    drawCurve(cx, cy - 1, w, amp, CLR_MOUTH, /*concaveDown=*/true, 3);
  } else if (amp < -2) {
    // Frown: middle rises up.
    drawCurve(cx, cy, w, -amp, CLR_MOUTH_DK, /*concaveDown=*/false, 1);
    drawCurve(cx, cy - 1, w, -amp, CLR_MOUTH, /*concaveDown=*/false, 3);
  } else {
    // Neutral — gentle pill
    fb.fillRoundRect(cx - 26, cy - 3, 52, 6, 3, CLR_MOUTH);
  }
}

// ── Drawing: face (into the off-screen sprite fb) ───────────────────────
void drawFace() {
  fb.fillSprite(CLR_BG);
  const int16_t W = fb.width();
  const int16_t H = fb.height();
  const int16_t cx = W / 2;
  const int16_t cy = H / 2 - 6;

  const Expression ex = pickExpression(fs.valence, fs.arousal);

  int16_t eyeR = 26;
  if (ex == EX_SURPRISED) eyeR = 31;
  else if (ex == EX_EXCITED) eyeR = 28;
  else if (ex == EX_SAD || ex == EX_CALM) eyeR = 24;

  const int16_t eyeDX = 64;
  const int16_t eyeY  = cy - 16;

  int16_t gazeX = (int16_t)(fs.gaze_deg / 45.0f * (float)(eyeR * 0.4f));
  if (gazeX > eyeR - 6) gazeX = eyeR - 6;
  if (gazeX < -(eyeR - 6)) gazeX = -(eyeR - 6);
  int16_t gazeY = 0;
  if (ex == EX_SAD)   gazeY = 4;
  if (ex == EX_ANGRY) gazeY = -2;

  int16_t upperLid = 0, lowerLid = 0;
  if (ex == EX_SAD)     upperLid = 6;
  if (ex == EX_ANGRY)   upperLid = 9;
  if (ex == EX_CALM)    upperLid = 4;
  if (ex == EX_EXCITED) lowerLid = 3;
  bool happySquint = (ex == EX_HAPPY || ex == EX_EXCITED) && fs.valence > 0.55f;

  // Pupil size responds to arousal (dilates when aroused) and expression.
  //  Neutral ~0.48r, surprised ~0.60r, calm ~0.42r.
  float pupilScale = 0.48f + fs.arousal * 0.07f;
  if (ex == EX_SURPRISED) pupilScale = 0.60f;
  if (ex == EX_CALM)      pupilScale = 0.42f;
  if (pupilScale < 0.35f) pupilScale = 0.35f;
  if (pupilScale > 0.65f) pupilScale = 0.65f;

  drawBrow(cx - eyeDX, eyeY, eyeR, ex, -1);
  drawBrow(cx + eyeDX, eyeY, eyeR, ex, +1);

  for (int side = -1; side <= 1; side += 2) {
    if (blinkClosed || fs.sleep) {
      drawClosedEye(cx + side * eyeDX, eyeY, eyeR,
                    ex == EX_HAPPY || ex == EX_EXCITED || fs.sleep);
    } else {
      drawEye(cx + side * eyeDX, eyeY, eyeR,
              gazeX, gazeY, upperLid, lowerLid, happySquint, pupilScale);
    }
  }

  drawMouth(cx, cy + 58, ex, fs.valence, fs.talking, currentViseme);

  // Sleep: floating Zzz in the TOP-RIGHT corner, well clear of the eyes.
  // Smallest Z is nearest the head; each successive Z is larger and
  // floats further up-right, like the classic "drifting off" trail.
  if (fs.sleep) {
    unsigned long now = millis();
    int phase = (int)((now / 550) % 4);   // 0..3, ~2.2 s full cycle
    const int16_t baseX = W - 80;         // ~240 on 320-wide screen
    const int16_t baseY = 48;             // 48, well above eye top (~72)
    struct Z { int16_t dx, dy, size; } zs[3] = {
      {  0,   0,  9 },
      { 14, -18, 13 },
      { 32, -38, 17 },
    };
    for (int i = 0; i < 3; ++i) {
      if (phase <= i) break;
      int16_t zx = baseX + zs[i].dx;
      int16_t zy = baseY + zs[i].dy;
      int16_t s  = zs[i].size;
      // Thick "Z": top bar, diagonal, bottom bar
      for (int t = 0; t < 3; ++t) {
        fb.drawLine(zx,     zy + t,         zx + s, zy + t,         CLR_EYE);
        fb.drawLine(zx + s, zy + s - 1 - t, zx,     zy + s - 1 - t, CLR_EYE);
      }
      for (int t = 0; t < 2; ++t) {
        fb.drawLine(zx + s - t,     zy + 2, zx - t,     zy + s - 2, CLR_EYE);
        fb.drawLine(zx + s - t + 1, zy + 2, zx - t + 1, zy + s - 2, CLR_EYE);
      }
    }
  }

  if (fs.privacy) {
    fb.fillRect(0, (int16_t)(H * 0.30f), W, (int16_t)(H * 0.22f), CLR_BG);
    fb.fillRect(0, (int16_t)(H * 0.30f), W, 2, CLR_SURFACE);
    fb.fillRect(0, (int16_t)(H * 0.52f) - 2, W, 2, CLR_SURFACE);
    fb.setTextColor(CLR_TEXT, CLR_BG);
    fb.setTextSize(2);
    fb.setCursor(W / 2 - 70, (int16_t)(H * 0.40f));
    fb.print("-- privacy --");
  }
}

// ── Scene: quick grid (2×2 big icons) ───────────────────────────────────
struct Tile { const char* label; const char* action; };
const Tile QUICK_TILES[4] = {
    {"Mute mic",   "mute_mic"},
    {"Stop",       "stop_talking"},
    {"Sleep",      "sleep"},
    {"More",       "more"},
};

void drawQuickGrid() {
  fb.fillSprite(CLR_BG);
  int16_t W = fb.width(), H = fb.height();
  int16_t tw = W / 2, th = H / 2;
  for (int i = 0; i < 4; ++i) {
    int16_t x = (i % 2) * tw, y = (i / 2) * th;
    fb.fillRoundRect(x + 6, y + 6, tw - 12, th - 12, 10, CLR_SURFACE);
    fb.setTextColor(CLR_TEXT, CLR_SURFACE);
    fb.setTextSize(2);
    int16_t tx = x + tw / 2 - ((int16_t)strlen(QUICK_TILES[i].label) * 6);
    int16_t ty = y + th / 2 - 8;
    fb.setCursor(tx, ty);
    fb.print(QUICK_TILES[i].label);
  }
}

// ── Scene: more list ─────────────────────────────────────────────────────
struct Row { const char* label; const char* action; };
const Row MORE_ROWS[] = {
    {"Volume",    "volume"},
    {"Privacy",   "privacy"},
    {"Status",    "status"},
    {"Restart",   "restart"},
    {"< Back",    "back"},
};
const int MORE_COUNT = sizeof(MORE_ROWS) / sizeof(MORE_ROWS[0]);

void drawMoreList() {
  fb.fillSprite(CLR_BG);
  int16_t W = fb.width();
  int16_t H = fb.height();
  int16_t rowH = 44;
  int16_t totalH = MORE_COUNT * rowH + (MORE_COUNT - 1) * 4;
  int16_t yStart = (H - totalH) / 2;
  for (int i = 0; i < MORE_COUNT; ++i) {
    int16_t y = yStart + i * (rowH + 4);
    fb.fillRoundRect(12, y, W - 24, rowH, 10, CLR_SURFACE);
    fb.setTextColor(CLR_TEXT, CLR_SURFACE);
    fb.setTextSize(3);
    fb.setCursor(26, y + (rowH - 24) / 2);
    fb.print(MORE_ROWS[i].label);
  }
}

// ── Touch handling ──────────────────────────────────────────────────────
void handleTouch(int16_t x, int16_t y) {
  lastTouchMs = millis();
  char buf[32];
  snprintf(buf, sizeof(buf), "TOUCH %d %d", x, y);
  writeReply(String(buf));

  if (currentScene == SCENE_FACE) {
    currentScene = SCENE_QUICKGRID;
    return;
  }
  if (currentScene == SCENE_QUICKGRID) {
    int16_t W = tft.width(), H = tft.height();
    int col = x < W / 2 ? 0 : 1;
    int row = y < H / 2 ? 0 : 1;
    int i = row * 2 + col;
    const char* act = QUICK_TILES[i].action;
    if (strcmp(act, "more") == 0) {
      currentScene = SCENE_MORELIST;
    } else {
      currentScene = SCENE_FACE;
      writeReply(String("BTN ") + act);
    }
    return;
  }
  if (currentScene == SCENE_MORELIST) {
    int16_t H = tft.height();
    int16_t rowH = 44;
    int16_t totalH = MORE_COUNT * rowH + (MORE_COUNT - 1) * 4;
    int16_t yStart = (H - totalH) / 2;
    int idx = -1;
    for (int i = 0; i < MORE_COUNT; ++i) {
      int16_t ry = yStart + i * (rowH + 4);
      if (y >= ry && y < ry + rowH) { idx = i; break; }
    }
    if (idx >= 0) {
      const char* act = MORE_ROWS[idx].action;
      if (strcmp(act, "back") == 0) {
        currentScene = SCENE_FACE;
      } else {
        currentScene = SCENE_FACE;
        writeReply(String("BTN ") + act);
      }
    }
  }
}

// ── Setup & loop ─────────────────────────────────────────────────────────
bool touchReady = false;

void setup() {
  Serial.begin(115200);
  delay(50);
  Serial.println();
  Serial.println("BOOT companion_face");
  Serial.flush();

  tft.init();
  tft.setRotation(1);               // landscape, 320×240 (USB port on the right)
  tft.fillScreen(CLR_BG);

  // Off-screen framebuffer. 320×240 @16bpp would need 150 KB which a single
  // ESP32 heap allocation can't usually provide, so use 8bpp (76 KB).
  fb.setColorDepth(8);
  void* ptr = fb.createSprite(tft.width(), tft.height());
  Serial.print("FB: sprite ");
  if (ptr == nullptr) {
    Serial.print("FAILED (free heap=");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes) — screen will stay on splash");
  } else {
    Serial.print("ready, free heap=");
    Serial.println(ESP.getFreeHeap());
  }

  // Splash straight on the screen (sprite drawing will take over after first loop)
  tft.setTextColor(CLR_TEXT, CLR_BG);
  tft.setTextSize(3);
  tft.setCursor(80, 90);
  tft.println("Companion");
  tft.setTextSize(2);
  tft.setCursor(100, 130);
  tft.println("waiting...");
  Serial.println("TFT init done");
  Serial.flush();

  // Touch — its own VSPI bus on pins 25/39/32/33 (per the board pinout).
  touchSPI.begin(TOUCH_CLK, TOUCH_MISO, TOUCH_MOSI, TOUCH_CS);
  ts.begin(touchSPI);
  ts.setRotation(1);
  touchReady = true;
  Serial.println("Touch init done");
  Serial.flush();
}

void loop() {
  pumpSerial();

  // Heartbeat every 2 s so we can see from the serial monitor that loop()
  // is alive and what the current scene + face state is.
  static unsigned long lastHb = 0;
  unsigned long nowHb = millis();
  if (nowHb - lastHb > 2000) {
    lastHb = nowHb;
    char buf[80];
    snprintf(buf, sizeof(buf),
             "HB scene=%d v=%+.2f a=%+.2f talk=%d",
             (int)currentScene, fs.valence, fs.arousal, (int)fs.talking);
    Serial.println(buf);
  }

  unsigned long now = millis();

  // ── Blinks — varied timing + occasional double-blink ────────────────
  static unsigned long nextBlinkAt = 0;
  static int pendingBlinks = 0;
  if (!blinkClosed && now >= nextBlinkAt) {
    blinkClosed = true;
    lastBlinkTickMs = now;
    if (pendingBlinks == 0)
      pendingBlinks = (random(100) < 15) ? 1 : 0;
  } else if (blinkClosed && (now - lastBlinkTickMs > 110)) {
    blinkClosed = false;
    if (pendingBlinks > 0) {
      --pendingBlinks;
      nextBlinkAt = now + 140;
    } else {
      nextBlinkAt = now + 2500 + random(3000);
    }
  }

  // ── Idle gaze: pick a target, smoothly interpolate toward it ────────
  // Commanded gaze (from Jetson) takes priority — if fs.gaze_deg is being
  // driven externally, we leave it alone. Otherwise we pick a random
  // target every few seconds and lerp smoothly to it so the pupils
  // slide instead of teleporting.
  static float idleTarget  = 0.0f;
  static float idleCurrent = 0.0f;
  static unsigned long nextIdleTarget = 0;
  static unsigned long lastGazeRedraw = 0;
  static bool gazeExternal = false;

  // Detect whether gaze is being driven externally: if it was changed by
  // the serial parser since the last frame, assume "external" for 2s.
  static float lastSeenGaze = 0.0f;
  if (fabsf(fs.gaze_deg - lastSeenGaze) > 0.5f) {
    gazeExternal = true;
    lastGazeRedraw = now;
  }
  if (gazeExternal && now - lastGazeRedraw > 2000) gazeExternal = false;
  lastSeenGaze = fs.gaze_deg;

  if (!gazeExternal && currentScene == SCENE_FACE) {
    if (now >= nextIdleTarget) {
      idleTarget = (float)random(-22, 23);
      // Occasionally a quick short saccade, sometimes a longer stare.
      nextIdleTarget = now + 1200 + random(3800);
    }
    // Smooth LERP toward target (~8% per frame ≈ 120ms settle)
    idleCurrent += (idleTarget - idleCurrent) * 0.12f;
    // Snap to integer so dirty-check doesn't fire every frame
    int16_t quantised = (int16_t)idleCurrent;
    if ((int16_t)fs.gaze_deg != quantised) {
      fs.gaze_deg = (float)quantised;
    }
  }

  // Auto-dismiss overlays
  if (currentScene != SCENE_FACE && (now - lastTouchMs) > AUTO_DISMISS_MS) {
    currentScene = SCENE_FACE;
  }

  // Touch
  if (ts.tirqTouched() && ts.touched()) {
    TS_Point p = ts.getPoint();
    // XPT2046 raw values range ~200..3900; map to display coordinates.
    int16_t x = map(p.x, 240, 3800, 0, tft.width());
    int16_t y = map(p.y, 240, 3800, 0, tft.height());
    handleTouch(x, y);
    // small debounce
    while (ts.touched()) { delay(5); }
  }

  // ── Change detection: only redraw + blit when the visible state changed.
  static float  lastV = 999, lastA = 999, lastGaze = 999;
  static bool   lastTalk = false, lastListen = false, lastThink = false;
  static bool   lastSleep = false, lastPriv = false, lastBlink = false;
  static String lastViseme = "";
  static int    lastScene = -1;
  static int    lastSleepPhase = -1;

  // Sleep Zzz animation phase — triggers a redraw as the Z's grow.
  int sleepPhase = fs.sleep ? (int)((now / 600) % 4) : -1;

  bool dirty =
    fs.valence   != lastV      ||
    fs.arousal   != lastA      ||
    fs.gaze_deg  != lastGaze   ||
    fs.talking   != lastTalk   ||
    fs.listening != lastListen ||
    fs.thinking  != lastThink  ||
    fs.sleep     != lastSleep  ||
    fs.privacy   != lastPriv   ||
    blinkClosed  != lastBlink  ||
    currentViseme != lastViseme ||
    (int)currentScene != lastScene ||
    sleepPhase    != lastSleepPhase;

  if (dirty) {
    switch (currentScene) {
      case SCENE_FACE:      drawFace();      break;
      case SCENE_QUICKGRID: drawQuickGrid(); break;
      case SCENE_MORELIST:  drawMoreList();  break;
    }
    // Blit the sprite to the LCD in one continuous SPI transaction — this
    // is what eliminates the flicker. No visible wipe-and-redraw.
    fb.pushSprite(0, 0);

    lastV = fs.valence; lastA = fs.arousal; lastGaze = fs.gaze_deg;
    lastTalk = fs.talking; lastListen = fs.listening; lastThink = fs.thinking;
    lastSleep = fs.sleep; lastPriv = fs.privacy;
    lastBlink = blinkClosed; lastViseme = currentViseme;
    lastScene = (int)currentScene;
    lastSleepPhase = sleepPhase;
  }

  delay(16);  // ~60 fps max check rate — but only redraws on state change
}
