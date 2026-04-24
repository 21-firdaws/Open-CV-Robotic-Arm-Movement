"""
Collapsing Mini Galaxy - FINAL VERSION
=======================================
Looks EXACTLY like reference: dense tangled glowing filaments, bright core,
purple/green/white colors.

LEFT  HAND : open palm / move index finger  ->  rotates the galaxy
RIGHT HAND : pinch thumb+index              ->  zoom in/out (scale)

pip install opencv-python mediapipe numpy
python "galaxy_hands.py"
"""

import sys, subprocess, importlib, os, urllib.request, time, math, random
import numpy as np

def _ensure(imp, pip=None):
    try: importlib.import_module(imp)
    except ImportError:
        print(f"Installing {pip or imp}...")
        subprocess.check_call([sys.executable,"-m","pip","install",pip or imp,"-q"])

_ensure("cv2","opencv-python")
_ensure("mediapipe","mediapipe")
_ensure("numpy","numpy")

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),"hand_landmarker.task")
if not os.path.exists(MODEL):
    print("Downloading hand model (~5MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL)
    print("Done.")

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
GW, GH = 640, 400
CX, CY = GW // 2, GH // 2

# ════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  — mostly white, tinted with purple / green / blue-white
# ════════════════════════════════════════════════════════════════════════════
def _hsv(h, s, v):
    c=v*s; x=c*(1-abs((h*6)%2-1)); m=v-c
    r,g,b = [(c,x,0),(x,c,0),(0,c,x),(0,x,c),(x,0,c),(c,0,x)][int(h*6)%6]
    return (int((b+m)*255), int((g+m)*255), int((r+m)*255))

rng = random.Random(7)
PAL = []
for _ in range(4000):
    t = rng.random()
    if   t < 0.50: PAL.append(_hsv(0.58, rng.uniform(0.00,0.08), rng.uniform(0.90,1.00)))  # cold white
    elif t < 0.68: PAL.append(_hsv(0.72, rng.uniform(0.50,0.85), rng.uniform(0.75,1.00)))  # purple
    elif t < 0.82: PAL.append(_hsv(0.35, rng.uniform(0.30,0.65), rng.uniform(0.75,1.00)))  # green
    else:          PAL.append(_hsv(0.60, rng.uniform(0.20,0.55), rng.uniform(0.80,1.00)))  # blue

# ════════════════════════════════════════════════════════════════════════════
#  PARTICLE — stores its own short trail for filament drawing
# ════════════════════════════════════════════════════════════════════════════
TRAIL = 22   # positions remembered — longer = longer individual filaments

class Particle:
    __slots__ = ('x','y','vx','vy','col','age','life','trail','ox','oy')

    def __init__(self):
        self.trail = []
        self.age   = 0.0
        self.life  = 1.0
        self.col   = PAL[0]
        self.x=self.y=self.vx=self.vy=0.0
        self.ox=self.oy=0.0
        self._spawn(1.0, 0.0)

    def _spawn(self, scale, spin):
        a   = random.uniform(0, 2*math.pi)
        r   = abs(random.gauss(0, 4.0))   # spawn VERY close to centre
        self.x  = CX + r*math.cos(a)
        self.y  = CY + r*math.sin(a)
        # outward velocity, nearly radial
        spd = random.uniform(0.5, 2.0) * scale
        dev = random.gauss(0, 0.12)        # small angular deviation
        # add spin offset so left-hand rotation works
        self.vx = math.cos(a + dev + spin*0.5) * spd
        self.vy = math.sin(a + dev + spin*0.5) * spd
        self.col  = random.choice(PAL)
        self.age  = 0.0
        self.life = random.uniform(1.2, 2.6)
        self.trail = [(self.x, self.y)] * TRAIL
        self.ox, self.oy = self.x, self.y

    def step(self, scale, spin, dt):
        self.ox, self.oy = self.x, self.y
        self.age += dt

        dx   = CX - self.x
        dy   = CY - self.y
        dist = math.hypot(dx, dy) + 1e-4

        # ── gravity pulls filaments back toward centre ────────────────────
        g = 70.0 * scale / (dist + 4)
        self.vx += g*(dx/dist)*dt
        self.vy += g*(dy/dist)*dt

        # ── spin: tangential force from left-hand gesture ─────────────────
        if abs(spin) > 0.005:
            # perpendicular to radius vector
            nx = -dy / dist
            ny =  dx / dist
            kick = spin * 220.0 * dt
            self.vx += nx * kick
            self.vy += ny * kick

        # ── no chaos turbulence (clean dense look) ────────────────────────
        # speed cap — CRITICAL: keeps filaments short and packed at centre
        spd = math.hypot(self.vx, self.vy)
        cap = 2.2 * scale
        if spd > cap:
            self.vx = self.vx/spd * cap
            self.vy = self.vy/spd * cap

        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60

        self.trail.pop(0)
        self.trail.append((self.x, self.y))

        if self.age > self.life or dist > max(GW,GH)*0.66:
            self._spawn(scale, spin)

    def paint(self, layer):
        af  = max(0.0, 1.0 - self.age/self.life)
        alp = af * af
        n   = len(self.trail)
        h, w = layer.shape[:2]
        for i in range(1, n):
            t   = i / n               # 0=tail → 1=head
            a   = alp * (t ** 2.0)    # head is bright, tail fades
            if a < 0.02: continue
            col = tuple(int(c*a) for c in self.col)
            x0,y0 = int(self.trail[i-1][0]), int(self.trail[i-1][1])
            x1,y1 = int(self.trail[i  ][0]), int(self.trail[i  ][1])
            if (0<=x0<w and 0<=y0<h) or (0<=x1<w and 0<=y1<h):
                cv2.line(layer,(x0,y0),(x1,y1),col,1,cv2.LINE_AA)
        # bright sparkle at head
        hx,hy = int(self.x), int(self.y)
        if 0<=hx<w and 0<=hy<h and alp>0.35:
            b = tuple(min(255,int(c*alp)+100) for c in self.col)
            cv2.circle(layer,(hx,hy),1,b,-1)

# ════════════════════════════════════════════════════════════════════════════
#  GALAXY
# ════════════════════════════════════════════════════════════════════════════
class Galaxy:
    N = 1600   # particle count — more = denser filament web

    def __init__(self):
        self.scale = 1.0
        self.spin  = 0.0
        self.ps    = [Particle() for _ in range(self.N)]
        # pre-age so the ball looks full immediately
        for p in self.ps:
            p.age = random.uniform(0, p.life * 0.8)

    def update(self, scale, spin, dt):
        self.scale = scale
        self.spin  = spin
        for p in self.ps:
            p.step(scale, spin, dt)

    def render(self):
        # fresh black canvas every frame (no persistent = no swirl artifact)
        layer = np.zeros((GH, GW, 3), dtype=np.float32)
        for p in self.ps:
            p.paint(layer)

        # ── additive core glow — NO opaque fill ──────────────────────────
        cr   = max(3, int(6 * self.scale))
        core = np.zeros_like(layer)
        # outer soft halo (very dim)
        cv2.circle(core,(CX,CY), int(cr*9),  (3,  5,  2),   -1)
        cv2.circle(core,(CX,CY), int(cr*4),  (20, 30, 12),  -1)
        cv2.circle(core,(CX,CY), int(cr*2),  (80,100, 55),  -1)
        cv2.circle(core,(CX,CY), cr+1,       (200,220,160), -1)
        cv2.circle(core,(CX,CY), cr,         (255,255,230), -1)
        # blur the halo only, then add
        halo = cv2.GaussianBlur(core, (41,41), 0)
        # keep sharp bright dot on top
        cv2.circle(halo,(CX,CY), max(2,cr-1),(255,255,240), -1)
        layer = layer + halo.astype(np.float32)

        # ── bloom ─────────────────────────────────────────────────────────
        b1 = cv2.GaussianBlur(layer,(3,3), 0)
        b2 = cv2.GaussianBlur(layer,(9,9), 0)
        out = layer + b1*0.40 + b2*0.18
        np.clip(out,0,255,out=out)
        return out.astype(np.uint8)

# ════════════════════════════════════════════════════════════════════════════
#  HAND TRACKING
# ════════════════════════════════════════════════════════════════════════════
class Hands:
    TIP=8; THUMB=4; PINKY_MCP=17; INDEX_MCP=5

    def __init__(self):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._lmk  = mp_vision.HandLandmarker.create_from_options(opts)
        self._ts   = 0
        self._ss   = 1.0   # smoothed scale
        self._spin = 0.0   # smoothed spin
        self.scale = 1.0
        self.spin  = 0.0
        # HUD overlay points
        self.l_dot  = None   # left hand: single index tip dot
        self.r_pt1  = None   # right hand: thumb dot
        self.r_pt2  = None   # right hand: index dot
        # store previous index-tip x to compute delta for rotation
        self._prev_lx = None

    def process(self, bgr):
        h, w = bgr.shape[:2]
        img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        self._ts += 33
        res = self._lmk.detect_for_video(img, self._ts)

        rs   = self._ss
        spin = self._spin
        self.l_dot = self.r_pt1 = self.r_pt2 = None

        if res.hand_landmarks:
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                tip   = lms[self.TIP]
                thumb = lms[self.THUMB]
                wrist = lms[0]
                imcp  = lms[self.INDEX_MCP]

                tx  = int(tip.x   * w);  ty  = int(tip.y   * h)
                thx = int(thumb.x * w);  thy = int(thumb.y * h)

                # wrist.x > 0.5 in flipped image = user's LEFT hand (on right side)
                if wrist.x > 0.5:
                    # ── LEFT HAND → ROTATION ──────────────────────────────
                    # Use horizontal offset of index tip from palm centre
                    palm_cx = (imcp.x + wrist.x) * 0.5
                    offset  = (tip.x - palm_cx) * 5.0   # amplify
                    spin    = float(np.clip(offset, -1.5, 1.5))
                    self.l_dot = (tx, ty)
                else:
                    # ── RIGHT HAND → SCALE (pinch) ────────────────────────
                    d  = math.hypot(tx-thx, ty-thy)
                    # pinch closed (d~0) = scale 0.25, fully spread (d~200) = scale 3.0
                    rs = float(np.clip(0.25 + d/175.0 * 2.75, 0.25, 3.0))
                    self.r_pt1 = (thx, thy)
                    self.r_pt2 = (tx,  ty)

        # exponential smoothing
        a = 0.14
        self._ss   += a*(rs   - self._ss)
        self._spin += a*(spin - self._spin)
        self.scale = round(self._ss,   2)
        self.spin  = round(self._spin, 3)

# ════════════════════════════════════════════════════════════════════════════
#  HUD
# ════════════════════════════════════════════════════════════════════════════
def draw_hud(frame, hands):
    h, w = frame.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX
    W = (255,255,255); K = (0,0,0)

    def txt(s, x, y, sz=1.05):
        cv2.putText(frame,s,(x+2,y+2),F,sz,K,4,cv2.LINE_AA)
        cv2.putText(frame,s,(x,  y  ),F,sz,W,2,cv2.LINE_AA)

    # Rotation arrow indicator
    sp = hands.spin
    spin_label = f"Spin: {'>>>' if sp>0.15 else ('<<<' if sp<-0.15 else ' — ')}"
    txt(spin_label,            28,    h-58)
    txt(f"Scale: {hands.scale:.2f}", w-242, h-58)

    # Left hand: single dot
    if hands.l_dot:
        cv2.circle(frame,hands.l_dot,11,W,-1)
        cv2.circle(frame,hands.l_dot,11,(90,90,90),2)

    # Right hand: two dots + line
    if hands.r_pt1 and hands.r_pt2:
        cv2.line(frame,hands.r_pt1,hands.r_pt2,W,2,cv2.LINE_AA)
        for pt in (hands.r_pt1, hands.r_pt2):
            cv2.circle(frame,pt,11,W,-1)
            cv2.circle(frame,pt,11,(90,90,90),2)

# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    CW, CH = 640, 480

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CH)
    cap.set(cv2.CAP_PROP_FPS, 30)

    galaxy = Galaxy()
    hands  = Hands()
    prev   = time.perf_counter()

    print("\n"+"="*50)
    print("  Collapsing Mini Galaxy — Hand Gesture Control")
    print("="*50)
    print("  LEFT  hand  tilt index L/R  ->  Rotate/spin")
    print("  RIGHT hand  pinch/spread    ->  Scale (zoom)")
    print("  Q / ESC  ->  quit")
    print("="*50+"\n")

    F     = cv2.FONT_HERSHEY_SIMPLEX
    TITLE = "Collapsing mini galaxy with my hands"

    while True:
        now  = time.perf_counter()
        dt   = min(now-prev, 0.05)
        prev = now

        ok, cam = cap.read()
        if not ok: time.sleep(0.03); continue

        cam = cv2.flip(cam,1)
        hands.process(cam)
        galaxy.update(hands.scale, hands.spin, dt)

        gal = galaxy.render()
        cam = cv2.resize(cam,(CW,CH))
        draw_hud(cam, hands)

        out = np.vstack([gal, cam])
        tw  = CW//2 - 298
        cv2.putText(out,TITLE,(tw+2,44),F,0.88,(0,0,0),     3,cv2.LINE_AA)
        cv2.putText(out,TITLE,(tw,  44),F,0.88,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow("Mini Galaxy", out)
        if cv2.waitKey(1) & 0xFF in (ord('q'),ord('Q'),27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")

if __name__ == "__main__":
    main()