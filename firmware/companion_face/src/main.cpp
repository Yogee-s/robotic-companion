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
  // Explicit expression hint from the Jetson. When set (and not "none"),
  // it OVERRIDES the V/A-based expression pick, so we can render
  // distinctive ornaments that can't be inferred from valence/arousal
  // alone (swirl eyes for confused, lightbulb for idea, hearts for love).
  String expression = "";
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
      else if (k == "expr")     fs.expression = v;
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
enum Expression {
  // V/A-derived (legacy)
  EX_NEUTRAL, EX_HAPPY, EX_EXCITED, EX_SURPRISED,
  EX_SAD, EX_ANGRY, EX_CALM,
  // Explicitly signalled via expr= (distinctive ornament branches)
  EX_CONFUSED, EX_THINKING, EX_IDEA, EX_LOVE, EX_WINK, EX_LISTENING,
};

Expression pickExpression(float v, float a) {
  if (a > 0.55f && v > -0.2f && v < 0.3f)  return EX_SURPRISED;
  if (a > 0.45f && v > 0.25f)              return EX_EXCITED;
  if (a > 0.35f && v < -0.25f)             return EX_ANGRY;
  if (v >  0.30f)                          return EX_HAPPY;
  if (v < -0.30f)                          return EX_SAD;
  if (a < -0.20f && v > -0.10f)            return EX_CALM;
  return EX_NEUTRAL;
}

// Map the `expr=` string from the Jetson → Expression enum. Returns
// EX_NEUTRAL as a sentinel when the hint is empty / "none" — callers
// fall back to the V/A-based pickExpression in that case.
Expression expressionFromString(const String& s) {
  if (s.length() == 0 || s == "none") return EX_NEUTRAL;
  if (s == "confused")  return EX_CONFUSED;
  if (s == "thinking")  return EX_THINKING;
  if (s == "idea")      return EX_IDEA;
  if (s == "love")      return EX_LOVE;
  if (s == "wink")      return EX_WINK;
  if (s == "listening") return EX_LISTENING;
  if (s == "surprised") return EX_SURPRISED;
  if (s == "excited")   return EX_EXCITED;
  if (s == "sad")       return EX_SAD;
  if (s == "angry")     return EX_ANGRY;
  return EX_NEUTRAL;
}

// ── Ornament primitives (matching pygame backend visually) ─────────────

// Swirly "vortex" eye — Archimedean spiral drawn as line segments.
// `rot` is a radians offset applied to the spiral's start angle, so we
// can rotate it over time for a subtle spinning animation.
static void drawSpiralEye(int16_t cx, int16_t cy, int16_t r, float rot) {
  fb.fillCircle(cx, cy, r + 2, CLR_BG);
  const float TURNS = 2.1f;
  const int   STEPS = 36;
  int16_t px = cx, py = cy;
  for (int i = 0; i <= STEPS; ++i) {
    float f = (float)i / (float)STEPS;
    float t = rot + f * TURNS * 2.0f * PI;
    float rr = f * (float)r;
    int16_t nx = cx + (int16_t)(rr * cosf(t));
    int16_t ny = cy + (int16_t)(rr * sinf(t));
    if (i > 0) {
      fb.drawLine(px, py, nx, ny, CLR_EYE);
      fb.drawLine(px, py + 1, nx, ny + 1, CLR_EYE);   // thicken
    }
    px = nx; py = ny;
  }
  fb.fillCircle(cx, cy, 2, CLR_EYE);
}

// Floating '?' glyph: hook arc + stem + dot.
static void drawQuestionMark(int16_t x, int16_t y, int16_t sz,
                             uint16_t col) {
  // Hook drawn as a sequence of small lines along an arc.
  int16_t hx = x;
  int16_t hy = y + sz / 2;
  int16_t r  = sz / 2;
  int16_t prevX = hx + (int16_t)((float)r * cosf(-PI / 6));
  int16_t prevY = hy + (int16_t)((float)r * sinf(-PI / 6));
  for (int deg = -30; deg <= 180; deg += 15) {
    float a = (float)deg * PI / 180.0f;
    int16_t nx = hx + (int16_t)((float)r * cosf(a));
    int16_t ny = hy + (int16_t)((float)r * sinf(a));
    fb.drawLine(prevX, prevY, nx, ny, col);
    fb.drawLine(prevX + 1, prevY, nx + 1, ny, col);
    prevX = nx; prevY = ny;
  }
  // Stem below the hook
  fb.drawLine(x, y + sz - 4, x, y + (int16_t)(sz * 1.35f), col);
  fb.drawLine(x + 1, y + sz - 4, x + 1, y + (int16_t)(sz * 1.35f), col);
  // Dot
  fb.fillCircle(x, y + (int16_t)(sz * 1.55f), 3, col);
}

// Small cog / gear — thinking icon.
static void drawGear(int16_t cx, int16_t cy, int16_t r) {
  fb.drawCircle(cx, cy, r, CLR_BROW);
  fb.drawCircle(cx, cy, r - 1, CLR_BROW);
  for (int i = 0; i < 8; ++i) {
    float ang = (float)i * PI / 4.0f;
    int16_t ox = cx + (int16_t)((float)(r + 3) * cosf(ang));
    int16_t oy = cy + (int16_t)((float)(r + 3) * sinf(ang));
    fb.fillCircle(ox, oy, 2, CLR_BROW);
  }
  fb.fillCircle(cx, cy, max((int16_t)2, (int16_t)(r / 3)), CLR_BG);
}

// Three horizontal dots "...".
static void drawEllipsis(int16_t x, int16_t y, int16_t spacing) {
  for (int i = 0; i < 3; ++i) {
    fb.fillCircle(x + i * spacing, y, 2, CLR_EYE);
  }
}

// Lightbulb + radiating rays — idea icon.
static void drawBulb(int16_t x, int16_t y, int16_t sz) {
  const uint16_t BULB       = 0xFE40;  // warm yellow
  const uint16_t BULB_LINE  = 0x8A00;  // dark amber outline
  const uint16_t RAYS       = 0xE520;  // slightly dim yellow
  const uint16_t BASE       = 0x9CE7;  // grey base
  // Rays
  for (int i = 0; i < 8; ++i) {
    float ang = (float)i * PI / 4.0f;
    int16_t ri = (int16_t)(sz * 0.78f);
    int16_t ro = (int16_t)(sz * 1.15f);
    int16_t x1 = x + (int16_t)((float)ri * cosf(ang));
    int16_t y1 = y + (int16_t)((float)ri * sinf(ang));
    int16_t x2 = x + (int16_t)((float)ro * cosf(ang));
    int16_t y2 = y + (int16_t)((float)ro * sinf(ang));
    fb.drawLine(x1, y1, x2, y2, RAYS);
  }
  int16_t rb = (int16_t)(sz * 0.55f);
  fb.fillCircle(x, y, rb, BULB);
  fb.drawCircle(x, y, rb, BULB_LINE);
  // Base
  int16_t baseY = y + (int16_t)(rb * 0.85f);
  fb.fillRect(x - rb / 2, baseY, rb, 5, BASE);
  fb.drawFastHLine(x - rb / 2, baseY + 2, rb, BULB_LINE);
}

// Compact lightbulb — just the bulb outline, no sun rays. Used for the
// "idea" face where the bulb is placed near the head at a smaller scale.
// `glow` ∈ [0,1] pulses the bulb colour brightness for a subtle animation.
static void drawBulbCompact(int16_t x, int16_t y, int16_t sz, float glow) {
  // Body colour lerps between dim and bright yellow based on glow.
  // 0x8240 dim → 0xFE60 bright (RGB565)
  uint16_t bulb;
  {
    uint8_t r = (uint8_t)(0x20 + (uint8_t)(glow * 0x1F));  // 5 bits
    uint8_t g = (uint8_t)(0x10 + (uint8_t)(glow * 0x2F));  // 6 bits
    uint8_t b = (uint8_t)(0x00 + (uint8_t)(glow * 0x06));  // 5 bits
    bulb = (uint16_t)((r << 11) | (g << 5) | b);
  }
  const uint16_t outline = 0x8220;
  const uint16_t base    = 0x9CE7;

  int16_t rb = (int16_t)(sz * 0.55f);
  fb.fillCircle(x, y, rb, bulb);
  fb.drawCircle(x, y, rb, outline);
  // Metal base
  int16_t baseY = y + (int16_t)(rb * 0.85f);
  fb.fillRect(x - rb / 2, baseY, rb, 4, base);
  fb.drawFastHLine(x - rb / 2, baseY + 2, rb, outline);
  // Small specular highlight at upper-left of the bulb
  fb.fillCircle(x - rb / 3, y - rb / 3, max((int16_t)2, (int16_t)(rb / 4)),
                0xFFFF);
}

// Small filled heart (two top circles + triangle).
static void drawHeart(int16_t cx, int16_t cy, int16_t size, uint16_t col) {
  int16_t r = max((int16_t)2, (int16_t)(size / 2));
  fb.fillCircle(cx - r + 1, cy - r / 2, r, col);
  fb.fillCircle(cx + r - 1, cy - r / 2, r, col);
  fb.fillTriangle(cx - r - 1, cy - 1,
                  cx + r + 1, cy - 1,
                  cx,         cy + size, col);
}

// Three concentric sound-wave arcs — listening icon.
// `facing`: +1 → arcs open to the right; -1 → open to the left.
static void drawSoundArcs(int16_t cx, int16_t cy, int n,
                          int16_t spacing, int facing) {
  const uint16_t COL = 0x6CDF;  // light blue
  const float HALF = PI / 4.0f;
  const float CENTER = (facing > 0) ? 0.0f : PI;
  for (int i = 1; i <= n; ++i) {
    int16_t rr = i * spacing;
    int16_t prevX = cx + (int16_t)((float)rr * cosf(CENTER - HALF));
    int16_t prevY = cy + (int16_t)((float)rr * sinf(CENTER - HALF));
    for (int step = 1; step <= 8; ++step) {
      float a = CENTER - HALF + (2.0f * HALF) * (float)step / 8.0f;
      int16_t nx = cx + (int16_t)((float)rr * cosf(a));
      int16_t ny = cy + (int16_t)((float)rr * sinf(a));
      fb.drawLine(prevX, prevY, nx, ny, COL);
      fb.drawLine(prevX, prevY + 1, nx, ny + 1, COL);
      prevX = nx; prevY = ny;
    }
  }
}

// Zigzag "confused" mouth — a short wavy line.
static void drawWavyMouth(int16_t cx, int16_t cy, int16_t w) {
  const int STEPS = 6;
  int16_t prevX = cx - w / 2, prevY = cy - 4;
  for (int i = 1; i <= STEPS; ++i) {
    int16_t nx = cx - w / 2 + (int16_t)((float)w * (float)i / (float)STEPS);
    int16_t ny = cy + ((i & 1) ? 4 : -4);
    fb.drawLine(prevX, prevY, nx, ny, CLR_MOUTH);
    fb.drawLine(prevX, prevY + 1, nx, ny + 1, CLR_MOUTH);
    prevX = nx; prevY = ny;
  }
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

// Draws one of the "distinctive" ornament-based faces.
// `now` is millis() at frame time — used for phase-based animations.
// Returns true if it handled the expression; false → fall back to
// the generic V/A renderer.
static bool drawExplicitExpression(Expression ex,
                                   int16_t cx, int16_t cy,
                                   int16_t W, int16_t H,
                                   unsigned long now) {
  const int16_t eyeDX = 64;
  const int16_t eyeY  = cy - 16;
  const int16_t eyeR  = 26;

  switch (ex) {
    case EX_CONFUSED: {
      // Swirly eyes slowly rotating for a dizzy feel. No question marks.
      int16_t rr = (int16_t)(eyeR * 1.3f);
      float rot = (float)(now % 4000) / 4000.0f * 2.0f * PI;   // 4 s / turn
      drawSpiralEye(cx - eyeDX, eyeY, rr, rot);
      drawSpiralEye(cx + eyeDX, eyeY, rr, -rot);               // counter-rotate
      // Asymmetric brows (kept — they're free and reinforce the mood)
      fb.drawLine(cx - eyeDX - rr, eyeY - rr - 10,
                  cx - eyeDX + rr, eyeY - rr + 2, CLR_BROW);
      fb.drawLine(cx - eyeDX - rr, eyeY - rr - 11,
                  cx - eyeDX + rr, eyeY - rr + 1, CLR_BROW);
      fb.drawLine(cx + eyeDX - rr, eyeY - rr - 2,
                  cx + eyeDX + rr, eyeY - rr - 14, CLR_BROW);
      fb.drawLine(cx + eyeDX - rr, eyeY - rr - 3,
                  cx + eyeDX + rr, eyeY - rr - 15, CLR_BROW);
      // Zigzag mouth
      drawWavyMouth(cx, cy + 58, 60);
      return true;
    }
    case EX_THINKING: {
      // Eyes look up-and-right (pensive)
      for (int side = -1; side <= 1; side += 2) {
        int16_t ex2 = cx + side * eyeDX;
        fb.fillCircle(ex2, eyeY, eyeR, CLR_EYE);
        fb.drawCircle(ex2, eyeY, eyeR, CLR_EYE_SHADE);
        fb.fillCircle(ex2 + 8, eyeY - 10, (int16_t)(eyeR * 0.42f), CLR_PUPIL);
      }
      // Quizzical asymmetric brows
      fb.drawLine(cx - eyeDX - eyeR, eyeY - eyeR - 8,
                  cx - eyeDX + eyeR, eyeY - eyeR - 12, CLR_BROW);
      fb.drawLine(cx + eyeDX - eyeR, eyeY - eyeR - 18,
                  cx + eyeDX + eyeR, eyeY - eyeR - 8, CLR_BROW);
      // Neutral closed-line mouth
      fb.fillRoundRect(cx - 22, cy + 55, 44, 6, 3, CLR_MOUTH);
      // Animated ellipsis: 1 → 2 → 3 → 0 dots on a 1.6 s cycle.
      int dots = (int)((now / 400) % 4);   // 0..3
      int16_t dotX = cx - 14;
      int16_t dotY = cy - 86;
      for (int i = 0; i < dots; ++i) {
        fb.fillCircle(dotX + i * 12, dotY, 3, CLR_EYE);
      }
      return true;
    }
    case EX_IDEA: {
      // Bright wide eyes with small pupils + highlight
      for (int side = -1; side <= 1; side += 2) {
        int16_t ex2 = cx + side * eyeDX;
        fb.fillCircle(ex2, eyeY, eyeR + 1, CLR_EYE);
        fb.drawCircle(ex2, eyeY, eyeR + 1, CLR_EYE_SHADE);
        fb.fillCircle(ex2, eyeY, (int16_t)(eyeR * 0.32f), CLR_PUPIL);
        fb.fillCircle(ex2 - 6, eyeY - 6, 3, CLR_HILITE);
      }
      // Arched brows
      for (int side = -1; side <= 1; side += 2) {
        int16_t ex2 = cx + side * eyeDX;
        for (int t = 0; t < 2; ++t) {
          fb.drawLine(ex2 - eyeR, eyeY - eyeR - 12 + t,
                      ex2 - 4,     eyeY - eyeR - 18 + t, CLR_BROW);
          fb.drawLine(ex2 - 4,     eyeY - eyeR - 18 + t,
                      ex2 + eyeR,  eyeY - eyeR - 12 + t, CLR_BROW);
        }
      }
      // Big grin
      drawCurve(cx, cy + 58, 112, 18, CLR_MOUTH_DK, true, 1);
      drawCurve(cx, cy + 57, 112, 18, CLR_MOUTH, true, 4);
      // Compact lightbulb (no sun rays) placed top-right of the right eye.
      // Pulsing glow: 0.55 .. 1.0, full cycle ~1.4 s.
      float pulse = 0.55f + 0.45f *
                    (0.5f + 0.5f * cosf((float)(now % 1400) / 1400.0f * 2.0f * PI));
      drawBulbCompact(cx + eyeDX + eyeR + 8, eyeY - eyeR - 6, 18, pulse);
      return true;
    }
    case EX_WINK: {
      // Animation: mostly winking (left eye closed), but every ~2 s both
      // eyes "blink" briefly to make the wink feel alive.
      //   phase 0-1 (200 ms): both eyes briefly open
      //   phase 2-19 (1800 ms): left closed, right open (the wink pose)
      int phase = (int)((now / 100) % 20);
      bool bothOpen = (phase < 2);

      if (bothOpen) {
        // Both open: a relaxed smile face
        for (int side = -1; side <= 1; side += 2) {
          int16_t ex2 = cx + side * eyeDX;
          fb.fillCircle(ex2, eyeY, eyeR, CLR_EYE);
          fb.drawCircle(ex2, eyeY, eyeR, CLR_EYE_SHADE);
          fb.fillCircle(ex2, eyeY, (int16_t)(eyeR * 0.45f), CLR_PUPIL);
          fb.fillCircle(ex2 - 6, eyeY - 6, 3, CLR_HILITE);
        }
      } else {
        // Left closed (smile arc), right open with gleam
        drawClosedEye(cx - eyeDX, eyeY, eyeR, /*smileUp=*/true);
        fb.fillCircle(cx + eyeDX, eyeY, eyeR, CLR_EYE);
        fb.drawCircle(cx + eyeDX, eyeY, eyeR, CLR_EYE_SHADE);
        fb.fillCircle(cx + eyeDX, eyeY, (int16_t)(eyeR * 0.45f), CLR_PUPIL);
        fb.fillCircle(cx + eyeDX - 6, eyeY - 6, 3, CLR_HILITE);
      }
      // Brows
      for (int t = 0; t < 2; ++t) {
        fb.drawLine(cx - eyeDX - eyeR, eyeY - eyeR - 10 + t,
                    cx - eyeDX + eyeR, eyeY - eyeR - 10 + t, CLR_BROW);
        fb.drawLine(cx + eyeDX - eyeR, eyeY - eyeR - 14 + t,
                    cx + eyeDX + eyeR, eyeY - eyeR - 10 + t, CLR_BROW);
      }
      // Lopsided smirk (always, so the character stays consistent)
      fb.drawLine(cx - 44, cy + 62, cx,      cy + 58, CLR_MOUTH);
      fb.drawLine(cx,      cy + 58, cx + 44, cy + 50, CLR_MOUTH);
      fb.drawLine(cx - 44, cy + 63, cx,      cy + 59, CLR_MOUTH);
      fb.drawLine(cx,      cy + 59, cx + 44, cy + 51, CLR_MOUTH);
      return true;
    }
    case EX_LISTENING: {
      for (int side = -1; side <= 1; side += 2) {
        int16_t ex2 = cx + side * eyeDX;
        fb.fillCircle(ex2, eyeY, eyeR, CLR_EYE);
        fb.drawCircle(ex2, eyeY, eyeR, CLR_EYE_SHADE);
        fb.fillCircle(ex2, eyeY, (int16_t)(eyeR * 0.45f), CLR_PUPIL);
      }
      // Attentive flat brows
      for (int side = -1; side <= 1; side += 2) {
        int16_t ex2 = cx + side * eyeDX;
        for (int t = 0; t < 2; ++t) {
          fb.drawLine(ex2 - eyeR, eyeY - eyeR - 10 + t,
                      ex2 + eyeR, eyeY - eyeR - 10 + t, CLR_BROW);
        }
      }
      // Small neutral-interested mouth
      fb.fillRoundRect(cx - 18, cy + 55, 36, 6, 3, CLR_MOUTH);
      // Sound arcs animate outward: each arc's visible index cycles
      // over 4 positions, giving a "radar ping" feel.
      int step = (int)((now / 180) % 4);   // 0..3
      // Draw the 3 arcs with radii shifted by `step` for the outward motion.
      for (int side = -1; side <= 1; side += 2) {
        int16_t acx = cx + side * 108;
        for (int i = 0; i < 3; ++i) {
          int16_t rr = ((i + step) % 4 + 1) * 8;   // 8..32
          // We reuse drawSoundArcs for one arc at a time by faking n=1
          // and telling it to start at ring `rr/8`. Simpler: inline the
          // one-ring version here.
          const uint16_t COL = 0x6CDF;
          const float HALF = PI / 4.0f;
          const float CENTER = (side > 0) ? 0.0f : PI;
          int16_t prevX = acx + (int16_t)((float)rr * cosf(CENTER - HALF));
          int16_t prevY = eyeY + 4 + (int16_t)((float)rr * sinf(CENTER - HALF));
          for (int s = 1; s <= 8; ++s) {
            float a = CENTER - HALF + (2.0f * HALF) * (float)s / 8.0f;
            int16_t nx = acx + (int16_t)((float)rr * cosf(a));
            int16_t ny = eyeY + 4 + (int16_t)((float)rr * sinf(a));
            fb.drawLine(prevX, prevY, nx, ny, COL);
            fb.drawLine(prevX, prevY + 1, nx, ny + 1, COL);
            prevX = nx; prevY = ny;
          }
        }
      }
      return true;
    }
    default:
      return false;
  }
}

// ── Drawing: face (into the off-screen sprite fb) ───────────────────────
void drawFace() {
  fb.fillSprite(CLR_BG);
  const int16_t W = fb.width();
  const int16_t H = fb.height();
  const int16_t cx = W / 2;
  const int16_t cy = H / 2 - 6;

  // An explicit expression hint overrides the V/A-based default pick.
  Expression ex = expressionFromString(fs.expression);
  if (ex == EX_NEUTRAL) {
    ex = pickExpression(fs.valence, fs.arousal);
  }

  // Try the distinctive ornament path first. If it handles this
  // expression, skip the V/A renderer entirely (no generic eyes/mouth).
  if (!fs.sleep && !blinkClosed &&
      drawExplicitExpression(ex, cx, cy, W, H, millis())) {
    if (fs.privacy) {
      fb.fillRect(0, (int16_t)(H * 0.30f), W, (int16_t)(H * 0.22f), CLR_BG);
      fb.fillRect(0, (int16_t)(H * 0.30f), W, 2, CLR_SURFACE);
      fb.fillRect(0, (int16_t)(H * 0.52f) - 2, W, 2, CLR_SURFACE);
    }
    return;
  }

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
  static String lastExpr = "";
  static int    lastScene = -1;
  static int    lastSleepPhase = -1;
  static int    lastAnimPhase = -1;

  // Sleep Zzz animation phase — triggers a redraw as the Z's grow.
  int sleepPhase = fs.sleep ? (int)((now / 600) % 4) : -1;

  // Animation phase for expressions that need continuous repaints:
  // confused (rotating swirls), thinking (dot count), idea (pulsing bulb),
  // wink (blink cycle), listening (sound-arc radii). 40 ms tick ≈ 25 Hz.
  int animPhase = -1;
  if (currentScene == SCENE_FACE && !fs.sleep) {
    if (fs.expression == "confused" || fs.expression == "thinking" ||
        fs.expression == "idea"     || fs.expression == "wink"     ||
        fs.expression == "listening") {
      animPhase = (int)((now / 40) % 10000);
    }
  }

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
    fs.expression != lastExpr   ||
    (int)currentScene != lastScene ||
    sleepPhase    != lastSleepPhase ||
    animPhase     != lastAnimPhase;

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
    lastExpr = fs.expression;
    lastScene = (int)currentScene;
    lastSleepPhase = sleepPhase;
    lastAnimPhase = animPhase;
  }

  delay(16);  // ~60 fps max check rate — but only redraws on state change
}
