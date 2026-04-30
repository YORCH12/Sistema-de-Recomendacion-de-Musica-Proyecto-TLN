import os
import threading
import queue
import time
import re
import json
import math
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import warnings

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import tkinter as tk
import customtkinter as ctk
from PIL import Image

from TLN_2 import SistemaRecomendacion, preprocesar_letra

CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID",     "cdd38f1958794ce8b9b6e0e6b58df1ed")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "fa7876458a4143608a30a1fa4b17ff76")
REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI",  "http://127.0.0.1:3000")
SCOPE = (
    "user-modify-playback-state user-read-playback-state "
    "user-read-currently-playing playlist-read-private "
    "playlist-read-collaborative user-top-read user-library-read"
)

cc_manager   = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
spotify_data = spotipy.Spotify(client_credentials_manager=cc_manager, retries=0, requests_timeout=5)
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI, scope=SCOPE
    ),
    retries=0, requests_timeout=5
)

def obtener_letra(artista: str, cancion: str) -> str:
    url = f"https://api.lyrics.ovh/v1/{artista}/{cancion}"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json().get("lyrics", "")
    except Exception:
        pass
    return ""

def normalizar_scores(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

def normalizar_titulo(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(r"\b(feat|ft)\.?\s+[^-–]+", " ", t)
    t = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalizar_artista(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

NLP_WEIGHT       = 0.75
MIN_SIMILARITY   = 0.05
TOP_N_DEFAULT    = 8
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class MotorRecomendacion:
    def __init__(self):
        self.df_canciones   = pd.DataFrame()
        self.scaler         = MinMaxScaler()

    def cargar_datos(self):
        tracks = []
        try:
            for rango in ["short_term", "medium_term", "long_term"]:
                items = sp.current_user_top_tracks(limit=50, time_range=rango).get("items", [])
                tracks.extend(items)
            for offset in range(0, 500, 50):
                saved = sp.current_user_saved_tracks(limit=50, offset=offset)
                items = saved.get("items", [])
                if not items:
                    break
                tracks.extend(item["track"] for item in items if item.get("track"))
        except Exception as e:
            print(f"[Spotify] Error cargando datos: {e}")

        datos = []
        for track in tracks:
            if not track or not track.get("uri"):
                continue
            datos.append({
                "id":             track["id"],
                "uri":            track["uri"],
                "nombre":         track["name"],
                "artista":        track["artists"][0]["name"],
                "album":          track["album"]["images"][0]["url"] if track["album"]["images"] else "",
                "popularidad":    track.get("popularity", 0),
                "genero_busqueda": "Tus Favoritos",
            })

        self.df_canciones = (
            pd.DataFrame(datos)
            .drop_duplicates(subset=["id"])
            .reset_index(drop=True)
        )

        if self.df_canciones.empty:
            return

        self.df_canciones["pop_norm"] = self.scaler.fit_transform(
            self.df_canciones[["popularidad"]]
        )

    def obtener_recomendaciones(self, track_id, motor_nlp, top_n=TOP_N_DEFAULT):
        df = self.df_canciones
        pop_scores = df["pop_norm"].values.copy()
        nlp_scores = self._calcular_scores_nlp(track_id, df, motor_nlp)

        if nlp_scores is not None:
            pop_norm  = normalizar_scores(pop_scores)
            nlp_norm  = normalizar_scores(nlp_scores)
            score_final = NLP_WEIGHT * nlp_norm + (1 - NLP_WEIGHT) * pop_norm
        else:
            score_final = normalizar_scores(pop_scores)

        mascara_propia = df["id"] != track_id
        mascara_umbral = score_final >= MIN_SIMILARITY
        mascara = mascara_propia & mascara_umbral

        if mascara.sum() == 0:
            mascara = mascara_propia

        candidatos = df[mascara].copy()
        candidatos["similitud"] = score_final[mascara]

        return (
            candidatos
            .sort_values("similitud", ascending=False)
            .head(top_n)
            .to_dict("records")
        )

    def _calcular_scores_nlp(self, track_id, df, motor_nlp):
        if motor_nlp is None or motor_nlp.df.empty or motor_nlp.matriz_tfidf is None:
            return None

        fila_actual = df[df["id"] == track_id]
        if fila_actual.empty:
            return None
        nombre_actual = fila_actual.iloc[0]["nombre"]
        artista_actual = fila_actual.iloc[0].get("artista", "")

        objetivo_titulo = normalizar_titulo(nombre_actual)
        objetivo_artista = normalizar_artista(artista_actual)

        with motor_nlp.lock:
            df_nlp = motor_nlp.df.copy()
            if df_nlp.empty or "Cancion" not in df_nlp.columns:
                return None

            df_nlp["titulo_norm"] = df_nlp["Cancion"].astype(str).map(normalizar_titulo)
            if "Artista" in df_nlp.columns:
                df_nlp["artista_norm"] = df_nlp["Artista"].astype(str).map(normalizar_artista)
            else:
                df_nlp["artista_norm"] = ""

            exactos = df_nlp[
                (df_nlp["titulo_norm"] == objetivo_titulo) &
                (df_nlp["artista_norm"] == objetivo_artista)
            ]
            if exactos.empty:
                exactos = df_nlp[df_nlp["titulo_norm"] == objetivo_titulo]
            if exactos.empty:
                return None

            idx_ref = int(exactos.index[0])
            vec_actual = motor_nlp.matriz_tfidf[idx_ref]
            all_scores_nlp = cosine_similarity(vec_actual, motor_nlp.matriz_tfidf)[0]

            scores = np.zeros(len(df))
            for i, row in df.iterrows():
                tit = normalizar_titulo(row.get("nombre", ""))
                art = normalizar_artista(row.get("artista", ""))

                cand = df_nlp[(df_nlp["titulo_norm"] == tit) & (df_nlp["artista_norm"] == art)]
                if cand.empty:
                    cand = df_nlp[df_nlp["titulo_norm"] == tit]
                if not cand.empty:
                    idxs = cand.index.to_numpy(dtype=int)
                    scores[i] = float(np.max(all_scores_nlp[idxs]))

        return scores

def background_nlp_enricher(df_canciones, motor_nlp):
    for _, track in df_canciones.iterrows():
        cancion = track["nombre"]
        artista = track["artista"]

        with motor_nlp.lock:
            ya_existe = not motor_nlp.df[
                motor_nlp.df["Cancion"].str.lower() == cancion.lower()
            ].empty

        if ya_existe:
            continue

        letra = obtener_letra(artista, cancion)
        if not letra:
            time.sleep(1.5)
            continue

        nueva_fila = pd.DataFrame([{
            "Artista":         artista,
            "Cancion":         cancion,
            "Album":           "Spotify Sync",
            "Genero":          track.get("genero_busqueda", "Tus Favoritos"),
            "Letra":           letra,
            "letra_procesada": preprocesar_letra(letra),
        }])

        with motor_nlp.lock:
            motor_nlp.df = pd.concat([motor_nlp.df, nueva_fila], ignore_index=True)
            try:
                motor_nlp.df.to_csv("corpus_letras.csv", index=False, encoding="utf-8")
            except Exception:
                pass
            motor_nlp.matriz_tfidf    = motor_nlp.vectorizer.fit_transform(
                motor_nlp.df["letra_procesada"]
            )
            motor_nlp.matriz_similitud = cosine_similarity(motor_nlp.matriz_tfidf)

        time.sleep(1.5)

class AudioVisualizer(tk.Canvas):

    NEON_PINK   = "#FF1E6E"
    NEON_CYAN   = "#00F5E9"
    NEON_PURPLE = "#B442FF"
    NEON_GOLD   = "#FFD166"

    def __init__(self, master, width=380, height=90, **kw):
        super().__init__(master, width=width, height=height,
                         bg="#07060F", highlightthickness=0, **kw)
        self.w = width
        self.h = height
        self._bars  = 38
        self._phase = 0.0
        self._particles: list[dict] = []
        self._is_playing = False
        self._init_particles()
        self._animate()

    def _init_particles(self):
        for _ in range(14):
            self._particles.append(self._new_particle())

    def _new_particle(self) -> dict:
        import random
        return {
            "x": random.uniform(0, self.w),
            "y": random.uniform(0, self.h),
            "vx": random.uniform(-0.4, 0.4),
            "vy": random.uniform(-0.6, -0.1),
            "r": random.uniform(1.5, 4.0),
            "alpha": random.uniform(0.4, 1.0),
            "color": random.choice([self.NEON_PINK, self.NEON_CYAN, self.NEON_PURPLE]),
            "life": random.uniform(0.0, 1.0),
        }

    def set_playing(self, playing: bool):
        self._is_playing = playing

    def _animate(self):
        self.delete("all")
        self._phase += 0.07 if self._is_playing else 0.015

        bar_w    = self.w / self._bars
        bar_gap  = bar_w * 0.35
        bar_real = bar_w - bar_gap

        for i in range(self._bars):

            amp = self._is_playing
            h1 = math.sin(self._phase * 1.3 + i * 0.45) * 0.5 + 0.5
            h2 = math.sin(self._phase * 0.7 + i * 0.22 + 1.2) * 0.3 + 0.3
            h3 = math.sin(self._phase * 2.1 + i * 0.80) * 0.2 + 0.2
            raw = (h1 * 0.5 + h2 * 0.3 + h3 * 0.2) if self._is_playing else (h1 * 0.12 + h2 * 0.06)
            bh = max(3, raw * (self.h - 8))

            x0 = i * bar_w + bar_gap / 2
            x1 = x0 + bar_real
            y0 = self.h - bh
            y1 = self.h

            t = i / max(1, self._bars - 1)
            r = int(255 * (1 - t) + 0 * t)
            g = int(30 * (1 - t) + 245 * t)
            b = int(110 * (1 - t) + 233 * t)
            color = f"#{r:02x}{g:02x}{b:02x}"

            self.create_rectangle(x0 - 1, y0 - 2, x1 + 1, y1, fill=color,
                                   outline="", stipple="gray25")
            self.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

            if bh > 6:
                self.create_rectangle(x0, y0, x1, y0 + 3,
                                      fill="#FFFFFF", outline="", stipple="gray50")

        for p in self._particles:
            speed = 1.5 if self._is_playing else 0.3
            p["x"]  += p["vx"] * speed
            p["y"]  += p["vy"] * speed
            p["life"] += 0.008 * speed

            if p["life"] >= 1.0 or p["y"] < -5 or p["x"] < -5 or p["x"] > self.w + 5:
                p.update(self._new_particle())
                p["y"] = self.h + 2

            r = int(p["r"])
            if r >= 1:
                self.create_oval(
                    p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r,
                    fill=p["color"], outline="", stipple="gray75"
                )

        self.after(33, self._animate)

class WaveBadge(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, width=36, height=20, bg="#0E0E1A",
                         highlightthickness=0, **kw)
        self._phase = 0.0
        self._anim()

    def _anim(self):
        self.delete("all")
        self._phase += 0.12
        pts = []
        for x in range(36):
            y = 10 + math.sin(self._phase + x * 0.45) * 5
            pts.extend([x, y])
        if len(pts) >= 4:
            self.create_line(pts, fill="#FF1E6E", width=1.5, smooth=True)
        self.after(40, self._anim)

class RotatingCoverCanvas(tk.Canvas):
    def __init__(self, master, size=260, **kw):
        super().__init__(master, width=size, height=size,
                         bg="#07060F", highlightthickness=0, **kw)
        self.size    = size
        self._angle  = 0.0
        self._img_tk = None
        self._playing = False
        self._draw()

    def set_playing(self, playing: bool):
        self._playing = playing

    def set_image(self, pil_img: Image.Image):

        s = self.size - 20
        pil_img = pil_img.resize((s, s), Image.LANCZOS)
        mask = Image.new("L", (s, s), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, s, s), fill=255)
        result = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        result.paste(pil_img, (0, 0))
        result.putalpha(mask)
        from PIL import ImageTk
        self._img_pil = result
        self._update_img_tk()

    def _update_img_tk(self):
        if hasattr(self, "_img_pil"):
            from PIL import ImageTk
            self._img_tk = ImageTk.PhotoImage(self._img_pil)

    def _draw(self):
        self.delete("all")
        cx = cy = self.size // 2
        r_outer = self.size // 2 - 4
        r_inner = r_outer - 5

        segs = 72
        for i in range(segs):
            a0 = math.radians(self._angle + i * (360 / segs))
            a1 = math.radians(self._angle + (i + 1) * (360 / segs))
            t  = i / segs

            rr = int(255 * (1 - t) + 0 * t)
            gg = int(30  * (1 - t) + 245 * t)
            bb = int(110 * (1 - t) + 233 * t)
            col = f"#{rr:02x}{gg:02x}{bb:02x}"

            x0 = cx + r_outer * math.cos(a0)
            y0 = cy + r_outer * math.sin(a0)
            x1 = cx + r_outer * math.cos(a1)
            y1 = cy + r_outer * math.sin(a1)
            self.create_line(x0, y0, x1, y1, fill=col, width=4)

        if self._img_tk:
            self.create_image(cx, cy, image=self._img_tk, anchor="center")
        else:
            self.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                             fill="#1A1830", outline="")
            self.create_text(cx, cy, text="♪", fill="#FF1E6E", font=("Arial", 48))

        gx = cx + r_outer * math.cos(math.radians(self._angle))
        gy = cy + r_outer * math.sin(math.radians(self._angle))
        self.create_oval(gx - 4, gy - 4, gx + 4, gy + 4, fill="#FFFFFF", outline="")

        if self._playing:
            self._angle = (self._angle + 0.6) % 360
        else:
            self._angle = (self._angle + 0.08) % 360

        self.after(16, self._draw)

class SpotifyDesktopApp(ctk.CTk):

    C_BG         = "#07060F"
    C_SIDEBAR    = "#0B0A17"
    C_CARD       = "#0E0D1C"
    C_CARD_HOV   = "#181630"
    C_BORDER     = "#1F1D3A"
    C_BORDER_HOV = "#FF1E6E"
    C_PINK       = "#FF1E6E"
    C_CYAN       = "#00F5E9"
    C_PURPLE     = "#B442FF"
    C_GOLD       = "#FFD166"
    C_TEXT       = "#EAE8FF"
    C_MUTED      = "#5E5B85"
    C_PROGRESS   = "#1A1838"

    def __init__(self, motor_content: MotorRecomendacion, motor_nlp: SistemaRecomendacion):
        super().__init__()
        self.motor_content        = motor_content
        self.motor_nlp            = motor_nlp
        self.currently_playing_id = None
        self.image_queue          = queue.Queue()
        self.openai_api_key       = os.getenv("OPENAI_API_KEY", "").strip()
        self.explanation_cache    = {}
        self.current_rendered_tracks = []

        self.title("")
        self.geometry("1360x880")
        self.minsize(1100, 700)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color=self.C_BG)
        try:
            self.attributes("-alpha", 0.97)
        except Exception:
            pass

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._generar_iconos_media()
        self._build_sidebar()
        self._build_main()

        self.process_image_queue()
        if not self.motor_content.df_canciones.empty:
            sample = self.motor_content.df_canciones.sample(
                min(12, len(self.motor_content.df_canciones))
            ).to_dict("records")
            self.render_grid(sample)
        self.check_state()

    def _build_sidebar(self):
        self.left_panel = ctk.CTkFrame(self, width=360, fg_color=self.C_SIDEBAR,
                                       corner_radius=0, border_width=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_propagate(False)
        self.left_panel.grid_rowconfigure(99, weight=1)

        logo_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        logo_frame.pack(fill="x", padx=28, pady=(28, 0))

        logo_dot = tk.Canvas(logo_frame, width=10, height=10,
                              bg=self.C_SIDEBAR, highlightthickness=0)
        logo_dot.pack(side="left", padx=(0, 8))
        logo_dot.create_oval(0, 0, 10, 10, fill=self.C_PINK, outline="")
        self._pulse_logo(logo_dot)

        ctk.CTkLabel(logo_frame, text="",
                     font=("Courier New", 18, "bold"), text_color=self.C_TEXT).pack(side="left")
        ctk.CTkLabel(logo_frame, text="",
                     font=("Courier New", 18, "bold"), text_color=self.C_PINK).pack(side="left")

        sep = tk.Canvas(self.left_panel, height=1, bg=self.C_SIDEBAR, highlightthickness=0)
        sep.pack(fill="x", padx=24, pady=(16, 0))
        sep.create_line(0, 0, 400, 0, fill=self.C_BORDER, width=1)

        cover_wrapper = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        cover_wrapper.pack(pady=(20, 10))

        self.cover_canvas = RotatingCoverCanvas(cover_wrapper, size=260)
        self.cover_canvas.pack()

        self.now_playing_label = ctk.CTkLabel(
            self.left_panel, text="Ninguna Pista Activa",
            font=("Courier New", 17, "bold"), text_color=self.C_TEXT, wraplength=300
        )
        self.now_playing_label.pack(pady=(14, 2), padx=20)

        self.now_playing_artist = ctk.CTkLabel(
            self.left_panel, text="Esperando reproducción...",
            font=("Helvetica", 12), text_color=self.C_MUTED
        )
        self.now_playing_artist.pack(padx=20)

        prog_bg = ctk.CTkFrame(self.left_panel, fg_color=self.C_PROGRESS,
                               height=4, corner_radius=2)
        prog_bg.pack(fill="x", padx=28, pady=(18, 6))
        self.progress_bar = ctk.CTkProgressBar(
            self.left_panel, height=3, corner_radius=2,
            fg_color=self.C_PROGRESS, progress_color=self.C_PINK,
            border_width=0
        )
        self.progress_bar.pack(fill="x", padx=28, pady=(0, 10))
        self.progress_bar.set(0.0)

        controls = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        controls.pack(pady=8)

        self.btn_prev = ctk.CTkButton(
            controls, text="", image=self.icon_prev, width=42, height=42,
            fg_color="transparent", hover_color=self.C_CARD_HOV, corner_radius=21,
            command=self.prev_track
        )
        self.btn_prev.pack(side="left", padx=10)

        self.btn_play_pause = ctk.CTkButton(
            controls, text="", image=self.icon_play_large, width=70, height=70,
            corner_radius=35, fg_color=self.C_PINK, hover_color="#CC1458",
            command=self.toggle_playback
        )
        self.btn_play_pause.pack(side="left", padx=14)

        self.btn_next = ctk.CTkButton(
            controls, text="", image=self.icon_next, width=42, height=42,
            fg_color="transparent", hover_color=self.C_CARD_HOV, corner_radius=21,
            command=self.next_track
        )
        self.btn_next.pack(side="left", padx=10)

        viz_label = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        viz_label.pack(fill="x", padx=28, pady=(20, 4))
        ctk.CTkLabel(viz_label, text="AUDIO WAVEFORM",
                     font=("Courier New", 9, "bold"), text_color=self.C_MUTED).pack(side="left")
        ctk.CTkLabel(viz_label, text="● LIVE",
                     font=("Courier New", 9, "bold"), text_color=self.C_PINK).pack(side="right")

        self.audio_viz = AudioVisualizer(self.left_panel, width=304, height=80)
        self.audio_viz.pack(padx=28)

        nlp_header = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        nlp_header.pack(fill="x", padx=28, pady=(18, 6))
        ctk.CTkLabel(nlp_header, text="◈ NLP ANALYSIS",
                     font=("Courier New", 10, "bold"), text_color=self.C_CYAN).pack(side="left")

        self.txt_insight = ctk.CTkTextbox(
            self.left_panel, width=304, height=160,
            fg_color="#0A0918", text_color="#A8A5CC",
            font=("Courier New", 11),
            border_color=self.C_BORDER, border_width=1,
            corner_radius=12, scrollbar_button_color=self.C_BORDER
        )
        self.txt_insight.pack(padx=28, pady=(0, 16))
        self.txt_insight.insert("1.0", "> Iniciando análisis lírico...\n> Cargando modelo TF-IDF...\n> En espera de reproducción.")
        self.txt_insight.configure(state="disabled")

    def _build_main(self):
        self.main_outer = ctk.CTkFrame(self, fg_color=self.C_BG, corner_radius=0)
        self.main_outer.grid(row=0, column=1, sticky="nsew")
        self.main_outer.grid_rowconfigure(1, weight=1)
        self.main_outer.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.main_outer, fg_color=self.C_BG, height=110)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.grid_propagate(False)

        ctk.CTkLabel(header, text="RECOMENDACIONES",
                     font=("Courier New", 10, "bold"), text_color=self.C_PINK
                     ).place(x=44, y=26)

        self.lbl_subtitle = ctk.CTkLabel(
            header, text="Basado en tu historial de escucha",
            font=("Courier New", 24, "bold"), text_color=self.C_TEXT,
            wraplength=750, justify="left"
        )
        self.lbl_subtitle.place(x=44, y=48)

        sep2 = tk.Canvas(self.main_outer, height=2, bg=self.C_BG, highlightthickness=0)
        sep2.grid(row=0, column=0, sticky="ew", padx=44)
        self._draw_neon_line(sep2)

        self.main_frame = ctk.CTkScrollableFrame(
            self.main_outer, fg_color=self.C_BG, corner_radius=0,
            scrollbar_button_color=self.C_BORDER,
            scrollbar_button_hover_color=self.C_PINK
        )
        self.main_frame.grid(row=1, column=0, sticky="nsew")

        self.grid_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.grid_frame.pack(fill="both", expand=True, padx=44, pady=(10, 30))

    def _generar_iconos_media(self):
        def draw_play(size, fill=(255, 30, 110)):
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            m = size * 0.06
            poly = [
                (size * 0.38, size * 0.28),
                (size * 0.38, size * 0.72),
                (size * 0.72, size * 0.50)
            ]
            d.polygon(poly, fill=(255, 255, 255, 230))
            return img

        def draw_pause(size):
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rectangle([size*0.30, size*0.28, size*0.44, size*0.72], fill=(255,255,255,230))
            d.rectangle([size*0.56, size*0.28, size*0.70, size*0.72], fill=(255,255,255,230))
            return img

        def draw_skip(size, forward=True):
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            if forward:
                d.polygon([(size*.20,size*.25),(size*.20,size*.75),(size*.55,size*.50)], fill=(255,255,255,210))
                d.rectangle([size*.58,size*.25,size*.72,size*.75], fill=(255,255,255,210))
            else:
                d.polygon([(size*.80,size*.25),(size*.80,size*.75),(size*.45,size*.50)], fill=(255,255,255,210))
                d.rectangle([size*.28,size*.25,size*.42,size*.75], fill=(255,255,255,210))
            return img

        pl64 = draw_play(64)
        pa64 = draw_pause(64)
        pl36 = draw_play(36)

        self.icon_play_large  = ctk.CTkImage(pl64, pl64, (64, 64))
        self.icon_pause_large = ctk.CTkImage(pa64, pa64, (64, 64))
        self.icon_play_small  = ctk.CTkImage(pl36, pl36, (36, 36))
        s32n = draw_skip(32, False)
        s32f = draw_skip(32, True)
        self.icon_prev = ctk.CTkImage(s32n, s32n, (32, 32))
        self.icon_next = ctk.CTkImage(s32f, s32f, (32, 32))

    def _pulse_logo(self, canvas, r=0):

        alpha = abs(math.sin(time.time() * 2))
        c = int(200 + 55 * alpha)
        color = f"#{c:02x}10{int(40 + 30*alpha):02x}"
        canvas.delete("all")
        canvas.create_oval(0, 0, 10, 10, fill=color, outline="")
        canvas.after(50, lambda: self._pulse_logo(canvas))

    def _draw_neon_line(self, canvas):

        canvas.delete("all")
        w = canvas.winfo_width() or 900
        segs = 60
        for i in range(segs):
            t = i / segs
            rr = int(255 * (1-t) + 0 * t)
            gg = int(30  * (1-t) + 245 * t)
            bb = int(110 * (1-t) + 233 * t)
            x0 = int(t * w)
            x1 = int((i+1)/segs * w)
            canvas.create_line(x0, 1, x1, 1, fill=f"#{rr:02x}{gg:02x}{bb:02x}", width=2)
        canvas.after(100, lambda: self._draw_neon_line(canvas))

    def _pulsar_badge(self, frame, label, _bright=True):

        try:
            if _bright:
                frame.configure(fg_color="#B442FF")
                label.configure(text_color="#FFFFFF")
            else:
                frame.configure(fg_color="#7A1ABF")
                label.configure(text_color="#E0C8FF")
            frame.after(600, lambda: self._pulsar_badge(frame, label, not _bright))
        except Exception:
            pass

    def process_image_queue(self):
        while not self.image_queue.empty():
            try:
                item = self.image_queue.get_nowait()
                if len(item) == 3:
                    ctk_img, label_widget, is_cover = item
                    if is_cover and ctk_img:
                        try:
                            label_widget.set_image(ctk_img)
                        except Exception:
                            pass
                    elif ctk_img:
                        try:
                            label_widget.configure(image=ctk_img, text="")
                        except Exception:
                            pass
                else:
                    ctk_img, label_widget = item
                    try:
                        label_widget.configure(
                            image=ctk_img if ctk_img else None,
                            text="" if ctk_img else "♪"
                        )
                    except Exception:
                        pass
            except queue.Empty:
                break
        self.after(100, self.process_image_queue)

    def update_insight(self, texto: str):
        self.txt_insight.configure(state="normal")
        self.txt_insight.delete("1.0", "end")
        self.txt_insight.insert("1.0", texto)
        self.txt_insight.configure(state="disabled")

    def fetch_image(self, url: str, label_widget, size=(110, 110), is_cover=False):
        try:
            response = requests.get(url, timeout=5)
            pil_img = Image.open(BytesIO(response.content)).convert("RGB")
            if is_cover:

                self.image_queue.put((pil_img, label_widget, True))
            else:
                pil_img = pil_img.resize(size, Image.LANCZOS)
                ctk_img = ctk.CTkImage(pil_img, pil_img, size)
                self.image_queue.put((ctk_img, label_widget, False))
        except Exception:
            self.image_queue.put((None, label_widget, False))

    def toggle_playback(self):
        try:
            current = sp.current_playback()
            if current and current["is_playing"]:
                sp.pause_playback()
                self.audio_viz.set_playing(False)
                self.cover_canvas.set_playing(False)
                self.btn_play_pause.configure(image=self.icon_play_large)
            else:
                sp.start_playback()
                self.audio_viz.set_playing(True)
                self.cover_canvas.set_playing(True)
                self.btn_play_pause.configure(image=self.icon_pause_large)
        except Exception:
            pass

    def next_track(self):
        try:
            sp.next_track()
        except Exception:
            pass

    def prev_track(self):
        try:
            sp.previous_track()
        except Exception:
            pass

    def play_track_uri(self, uri):
        if not uri or not isinstance(uri, str):
            self._notificar_error_reproduccion("No hay URI de Spotify para esta pista.")
            return
        uri = uri.strip()
        if not uri.startswith("spotify:track:"):
            self._notificar_error_reproduccion("URI inválida (se espera spotify:track:…).")
            return

        device_id = self._seleccionar_dispositivo_spotify()

        try:
            if device_id:
                try:
                    sp.transfer_playback(device_id=device_id, force_play=True)
                except Exception:
                    pass
                sp.start_playback(device_id=device_id, uris=[uri])
            else:
                sp.start_playback(uris=[uri])
        except SpotifyException as e:
            if e.http_status == 404:
                self._notificar_error_reproduccion(
                    "No hay reproductor activo. Abre Spotify en tu PC o móvil."
                )
            elif e.http_status == 403:
                self._notificar_error_reproduccion(
                    "Spotify no permitió el control. Requiere cuenta Premium."
                )
            else:
                self._notificar_error_reproduccion(e.msg if getattr(e, "msg", None) else str(e))
        except Exception as e:
            self._notificar_error_reproduccion(str(e))

    def _seleccionar_dispositivo_spotify(self):
        try:
            data = sp.devices()
            devices = data.get("devices") or []
            if not devices:
                return None
            for d in devices:
                if d.get("is_active") and d.get("id") and not d.get("is_restricted"):
                    return str(d["id"])
            preferidos = ("Computer", "Smartphone", "Tablet", "TV", "Speaker")
            for tipo in preferidos:
                for d in devices:
                    if d.get("type") == tipo and d.get("id") and not d.get("is_restricted"):
                        return str(d["id"])
            for d in devices:
                if d.get("id") and not d.get("is_restricted"):
                    return str(d["id"])
        except Exception:
            pass
        return None

    def _notificar_error_reproduccion(self, mensaje: str):
        print(f"[Spotify reproducción] {mensaje}")
        try:
            prev = self.lbl_subtitle.cget("text")
            corto = mensaje if len(mensaje) <= 140 else mensaje[:137] + "…"
            self.lbl_subtitle.configure(text=f"⚠ {corto}", text_color=self.C_GOLD)
            def restaurar():
                try:
                    self.lbl_subtitle.configure(text=prev, text_color=self.C_TEXT)
                except Exception:
                    pass
            self.after(9000, restaurar)
        except Exception:
            pass

    def show_track_detail_modal(self, track: dict):
        win = ctk.CTkToplevel(self)
        win.title("")
        win.geometry("600x580")
        win.minsize(500, 460)
        win.configure(fg_color="#09081A")
        win.transient(self)
        win.grab_set()

        def _cerrar():
            win.grab_release()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _cerrar)

        title_bar = ctk.CTkFrame(win, fg_color="#0E0D22", height=48, corner_radius=0)
        title_bar.pack(fill="x")
        ctk.CTkLabel(title_bar, text="◈  DETALLE DE RECOMENDACIÓN",
                     font=("Courier New", 11, "bold"),
                     text_color=self.C_CYAN).pack(side="left", padx=20, pady=12)

        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent", corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        nombre_completo = track.get("nombre") or "—"
        artista         = track.get("artista") or "—"

        if track.get("album"):
            modal_cover = ctk.CTkLabel(scroll, text="♪", width=80, height=80,
                                       fg_color=self.C_CARD, corner_radius=10,
                                       font=("Arial", 28))
            modal_cover.pack(anchor="w", padx=16, pady=(12, 0))
            threading.Thread(
                target=self.fetch_image,
                args=(track["album"], modal_cover, (80, 80)),
                daemon=True
            ).start()

        ctk.CTkLabel(scroll, text=nombre_completo,
                     font=("Courier New", 18, "bold"), text_color=self.C_TEXT,
                     wraplength=540, justify="left").pack(anchor="w", padx=16, pady=(10, 2))

        ctk.CTkLabel(scroll, text=artista,
                     font=("Helvetica", 13), text_color=self.C_MUTED,
                     wraplength=540, justify="left").pack(anchor="w", padx=16, pady=(0, 10))

        if "similitud" in track:
            try:
                raw = float(track["similitud"])
                pct = self._score_a_porcentaje(raw)
                score_row = ctk.CTkFrame(scroll, fg_color=self.C_CARD, corner_radius=8)
                score_row.pack(fill="x", padx=16, pady=(0, 10))
                ctk.CTkLabel(score_row, text=f"  MATCH SCORE  {pct}%  ",
                             font=("Courier New", 12, "bold"),
                             text_color=self.C_GOLD).pack(side="left", pady=8)
                bar = ctk.CTkProgressBar(score_row, height=6, width=200,
                                         fg_color=self.C_BORDER, progress_color=self.C_GOLD,
                                         corner_radius=3)
                bar.pack(side="left", padx=12, pady=8)
                bar.set(min(1.0, pct / 100))
                ctk.CTkLabel(score_row, text=f"raw: {min(1.0,max(0.0,raw)):.4f}  ",
                             font=("Courier New", 10), text_color=self.C_MUTED).pack(side="right", pady=8)
            except Exception:
                pass

        origen = track.get("genero_busqueda")
        if origen:
            ctk.CTkLabel(scroll, text=f"Origen: {origen}",
                         font=("Courier New", 10), text_color=self.C_MUTED).pack(anchor="w", padx=16, pady=(0, 8))

        ctk.CTkLabel(scroll, text="ANÁLISIS DE RECOMENDACIÓN",
                     font=("Courier New", 10, "bold"),
                     text_color=self.C_CYAN).pack(anchor="w", padx=16, pady=(10, 4))

        desc_completa = track.get("motivo_recomendacion",
                                   "Sin descripción generada para esta recomendación.")
        tb = ctk.CTkTextbox(scroll, width=540, height=200,
                             fg_color="#0A0918", text_color="#9A97C0",
                             font=("Courier New", 11), wrap="word",
                             border_color=self.C_BORDER, border_width=1,
                             corner_radius=10)
        tb.pack(fill="x", padx=16, pady=(0, 12))
        tb.insert("1.0", desc_completa)
        tb.configure(state="disabled")

        extras = []
        if "popularidad" in track and track["popularidad"] is not None:
            extras.append(f"Popularidad Spotify: {track['popularidad']}")
        if track.get("id"):
            extras.append(f"Track ID: {track['id']}")
        uri = track.get("uri")
        if uri:
            extras.append(f"URI: {uri}")

        if extras:
            ctk.CTkLabel(scroll, text="METADATA",
                         font=("Courier New", 10, "bold"),
                         text_color=self.C_CYAN).pack(anchor="w", padx=16, pady=(8, 4))
            ctk.CTkLabel(scroll, text="\n".join(extras),
                         font=("Courier New", 10), text_color=self.C_MUTED,
                         wraplength=540, justify="left").pack(anchor="w", padx=16, pady=(0, 16))

        footer = ctk.CTkFrame(win, fg_color="#0E0D22", height=60, corner_radius=0)
        footer.pack(fill="x", side="bottom")
        if uri:
            ctk.CTkButton(footer, text="▶  REPRODUCIR AHORA",
                          font=("Courier New", 11, "bold"),
                          fg_color=self.C_PINK, hover_color="#CC1458",
                          text_color="#FFFFFF", corner_radius=8, height=36,
                          command=lambda u=uri: self.play_track_uri(u)
                          ).pack(side="left", padx=20, pady=12)
        ctk.CTkButton(footer, text="CERRAR",
                      font=("Courier New", 11, "bold"),
                      fg_color="transparent", hover_color=self.C_CARD_HOV,
                      text_color=self.C_MUTED, corner_radius=8, height=36,
                      border_color=self.C_BORDER, border_width=1,
                      command=_cerrar).pack(side="left", pady=12)

        win.update_idletasks()
        x = max(0, (win.winfo_screenwidth() // 2) - (win.winfo_width() // 2))
        y = max(0, (win.winfo_screenheight() // 2) - (win.winfo_height() // 2))
        win.geometry(f"+{x}+{y}")
        win.focus_force()

    def show_loading(self, mensaje: str = "Analizando..."):

        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        self._loading_active = True
        container = ctk.CTkFrame(self.grid_frame, fg_color="transparent")
        container.pack(expand=True, fill="both", pady=60)

        spin_cnv = tk.Canvas(container, width=80, height=80,
                             bg=self.C_BG, highlightthickness=0)
        spin_cnv.pack(pady=(30, 16))

        lbl_msg = ctk.CTkLabel(container, text=mensaje,
                               font=("Courier New", 13, "bold"),
                               text_color=self.C_CYAN)
        lbl_msg.pack()

        lbl_dots = ctk.CTkLabel(container, text="",
                                font=("Courier New", 11), text_color=self.C_MUTED)
        lbl_dots.pack(pady=4)

        hints = [
            "Procesando vectores TF-IDF...",
            "Calculando similitud coseno...",
            "Buscando en corpus lírico...",
            "Construyendo recomendaciones...",
            "Analizando sentimiento...",
        ]
        lbl_hint = ctk.CTkLabel(container, text=hints[0],
                                font=("Courier New", 10), text_color=self.C_MUTED)
        lbl_hint.pack(pady=2)

        self._spin_angle   = 0.0
        self._spin_dots    = 0
        self._spin_hint_i  = 0

        def _animate_spinner():
            if not getattr(self, "_loading_active", False):
                return
            try:
                spin_cnv.winfo_exists()
            except Exception:
                return

            spin_cnv.delete("all")
            cx = cy = 40
            r = 28
            segs = 12
            for i in range(segs):
                a0 = math.radians(self._spin_angle + i * (360 / segs))
                a1 = math.radians(self._spin_angle + (i + 1) * (360 / segs) - 2)

                brightness = i / segs
                rr = int(0   + 255 * brightness)
                gg = int(245 * brightness)
                bb = int(233 * (1 - brightness) + 110 * brightness)
                col = f"#{rr:02x}{gg:02x}{bb:02x}"
                w = max(1, int(4 * brightness))
                x0 = cx + r * math.cos(a0); y0 = cy + r * math.sin(a0)
                x1 = cx + r * math.cos(a1); y1 = cy + r * math.sin(a1)
                spin_cnv.create_line(x0, y0, x1, y1, fill=col, width=w,
                                     capstyle="round")

            pulse = abs(math.sin(self._spin_angle * 0.05))
            pr = int(4 + 4 * pulse)
            pc = int(255 * pulse)
            spin_cnv.create_oval(cx-pr, cy-pr, cx+pr, cy+pr,
                                 fill=f"#{pc:02x}30{pc//2:02x}", outline="")

            self._spin_angle = (self._spin_angle + 6) % 360
            self._spin_dots  = (self._spin_dots + 1) % 12
            dots_txt = "·" * (self._spin_dots // 4 + 1)

            try:
                lbl_dots.configure(text=dots_txt)
            except Exception:
                return

            if self._spin_dots == 0:
                self._spin_hint_i = (self._spin_hint_i + 1) % len(hints)
                try:
                    lbl_hint.configure(text=hints[self._spin_hint_i])
                except Exception:
                    return

            self.after(33, _animate_spinner)

        _animate_spinner()

    def hide_loading(self):

        self._loading_active = False

    def render_grid(self, tracks: list[dict]):
        self.hide_loading()
        self.current_rendered_tracks = tracks

        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        for i, track in enumerate(tracks):
            self._build_track_card(track, i)

    def _build_track_card(self, track: dict, idx: int):

        es_match_emocional = track.get("score_emocion", 0) == 1

        C_CARD_BG  = "#1A0E2E" if es_match_emocional else self.C_CARD
        C_BORDER_N = self.C_PURPLE if es_match_emocional else self.C_BORDER
        C_TITLE    = "#D4AAFF" if es_match_emocional else self.C_TEXT

        card = ctk.CTkFrame(
            self.grid_frame, fg_color=C_CARD_BG,
            border_color=C_BORDER_N,
            border_width=2 if es_match_emocional else 1,
            corner_radius=16
        )
        card.pack(fill="x", pady=6 if es_match_emocional else 5)
        card.grid_columnconfigure(3, weight=1)
        card.grid_columnconfigure(4, weight=2)

        accent_bar_color = self.C_PURPLE if es_match_emocional else "transparent"
        accent_bar = ctk.CTkFrame(card, fg_color=accent_bar_color,
                                   width=4, corner_radius=4)
        accent_bar.grid(row=0, column=0, rowspan=3, padx=(0, 0), pady=6, sticky="ns")

        idx_bg = C_CARD_BG
        idx_cnv = tk.Canvas(card, width=30, height=30,
                             bg=idx_bg, highlightthickness=0)
        idx_cnv.grid(row=0, column=1, rowspan=3, padx=(8, 0), pady=16, sticky="ns")
        idx_color = self.C_PURPLE if es_match_emocional else self.C_MUTED
        idx_cnv.create_text(15, 15, text=f"{idx+1:02d}",
                            fill=idx_color, font=("Courier New", 9, "bold"))

        img_lbl = ctk.CTkLabel(card, text="♪", width=64, height=64,
                               fg_color="#1C1040" if es_match_emocional else "#151830",
                               corner_radius=10,
                               font=("Arial", 22), text_color=self.C_MUTED)
        img_lbl.grid(row=0, column=2, rowspan=3, padx=(8, 12), pady=12)

        if track.get("album"):
            threading.Thread(
                target=self.fetch_image,
                args=(track["album"], img_lbl, (64, 64)),
                daemon=True
            ).start()

        info_col = ctk.CTkFrame(card, fg_color="transparent")
        info_col.grid(row=0, column=3, rowspan=3, padx=(0, 8), pady=10, sticky="nsew")
        info_col.grid_columnconfigure(0, weight=1)

        if es_match_emocional:

            motivo_raw = track.get("motivo_recomendacion", "")
            emo_label = "MISMO SENTIMIENTO"
            for clave, nombre in [("ALEGRÍA","ALEGRÍA"), ("TRISTEZA","TRISTEZA"),
                                   ("ENOJO","ENOJO"), ("MIEDO","MIEDO"),
                                   ("SORPRESA","SORPRESA"), ("DISGUSTO","DISGUSTO")]:
                if clave in motivo_raw.upper():
                    emo_label = clave
                    break

            badge_frame = ctk.CTkFrame(info_col, fg_color=self.C_PURPLE,
                                        corner_radius=8, height=22)
            badge_frame.grid(row=0, column=0, sticky="w", pady=(0, 3))
            badge_frame.grid_propagate(False)

            badge_lbl = ctk.CTkLabel(
                badge_frame,
                text=f"  ◈ {emo_label}  ",
                font=("Courier New", 9, "bold"),
                text_color="#FFFFFF"
            )
            badge_lbl.pack(side="left", pady=2)

            self._pulsar_badge(badge_frame, badge_lbl)
            title_row = 1
        else:
            title_row = 0

        t_lbl = ctk.CTkLabel(
            info_col, text=track["nombre"],
            font=("Courier New", 14, "bold"), text_color=C_TITLE,
            anchor="w", justify="left", wraplength=260
        )
        t_lbl.grid(row=title_row, column=0, sticky="ew", pady=(2, 0))

        a_lbl = ctk.CTkLabel(
            info_col, text=track.get("artista", ""),
            font=("Helvetica", 11), text_color=self.C_MUTED,
            anchor="w", justify="left"
        )
        a_lbl.grid(row=title_row + 1, column=0, sticky="ew")

        wave_badge = WaveBadge(info_col)
        wave_badge.grid(row=title_row + 2, column=0, sticky="w", pady=(2, 0))

        descripcion = track.get("motivo_recomendacion", "")

        descripcion = re.sub(r"(?i)Coincide en un.*", "", descripcion).strip()
        descripcion = re.sub(r"(?i)La puntuación refleja.*", "", descripcion).strip()

        if es_match_emocional:
            descripcion = descripcion.replace("◈ Comparte tu misma emoción", "").lstrip(" .,()")

            for emo in ["ALEGRÍA", "TRISTEZA", "ENOJO", "MIEDO", "SORPRESA", "DISGUSTO", "OTRO"]:
                if descripcion.upper().startswith(emo):
                    descripcion = descripcion[len(emo):].lstrip(" .)-:")
                    break

            if descripcion.startswith("omparte"):
                descripcion = "C" + descripcion[1:]

            desc_color = "#9E85CC"
        else:
            desc_color = self.C_MUTED

        desc_lbl = ctk.CTkLabel(
            card, text=descripcion,
            font=("Helvetica", 11), text_color=desc_color,
            justify="left", anchor="w", wraplength=550
        )
        desc_lbl.grid(row=0, column=4, rowspan=3, padx=(4, 8), pady=10, sticky="nsew")

        if "similitud" in track:
            try:
                pct = self._score_a_porcentaje(float(track["similitud"]))
                match_cnv = tk.Canvas(card, width=54, height=54,
                                      bg=C_CARD_BG, highlightthickness=0)
                match_cnv.grid(row=0, column=5, rowspan=3, padx=(4, 6), pady=17)
                ring_color = self.C_PURPLE if es_match_emocional else self.C_GOLD
                match_cnv.create_oval(2, 2, 52, 52, outline=ring_color, width=2 if es_match_emocional else 1)
                match_cnv.create_text(27, 20, text=f"{pct}%",
                                      fill=ring_color, font=("Courier New", 10, "bold"))
                match_cnv.create_text(27, 36, text="match",
                                      fill=self.C_MUTED, font=("Courier New", 7))
            except Exception:
                pass

        uri_track = track.get("uri")
        if uri_track:
            play_bg    = self.C_PURPLE if es_match_emocional else self.C_PINK
            play_hover = "#8A22CC" if es_match_emocional else "#CC1458"
            btn_play = ctk.CTkButton(
                card, text="", image=self.icon_play_small, width=42, height=42,
                fg_color=play_bg, hover_color=play_hover, corner_radius=21,
                command=lambda u=uri_track: self.play_track_uri(u)
            )
            btn_play.grid(row=0, column=6, rowspan=3, padx=(4, 14), pady=23)

        C_HOVER_BG  = "#251040" if es_match_emocional else self.C_CARD_HOV
        C_LEAVE_BG  = C_CARD_BG
        C_LEAVE_BD  = C_BORDER_N

        def on_enter(e, c=card, tl=t_lbl, ic=idx_cnv):
            c.configure(fg_color=C_HOVER_BG, border_color=self.C_PURPLE if es_match_emocional else self.C_PINK)
            tl.configure(text_color=self.C_PURPLE if es_match_emocional else self.C_PINK)
            ic.configure(bg=C_HOVER_BG)
            ic.itemconfig("all", fill=self.C_PURPLE if es_match_emocional else self.C_PINK)

        def on_leave(e, c=card, tl=t_lbl, ic=idx_cnv):
            c.configure(fg_color=C_LEAVE_BG, border_color=C_LEAVE_BD)
            tl.configure(text_color=C_TITLE)
            ic.configure(bg=C_LEAVE_BG)
            ic.itemconfig("all", fill=idx_color)

        def abrir_modal(_e=None, t=dict(track)):
            self.show_track_detail_modal(t)

        for widget in [card, img_lbl, t_lbl, a_lbl, desc_lbl, info_col]:
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", abrir_modal)

    def _score_a_porcentaje(self, score: float) -> int:
        try:
            v = max(0.0, min(1.0, float(score)))
            p50 = getattr(self, "_pct_p50", 0.10)
            p90 = getattr(self, "_pct_p90", 0.30)
            if p90 <= p50:
                return int(round(v * 100))
            if v <= p50:
                mapped = (v / max(p50, 1e-9)) * 55.0
            elif v >= p90:
                mapped = 90.0 + min(10.0, ((v - p90) / max(1 - p90, 1e-9)) * 10.0)
            else:
                mapped = 55.0 + ((v - p50) / (p90 - p50)) * 35.0
            return int(round(max(1.0, min(99.0, mapped))))
        except Exception:
            return int(round(max(0.0, min(1.0, float(score))) * 100))

    def _actualizar_percentiles_visuales(self, tracks: list[dict]):
        vals = [float(t.get("similitud", 0.0)) for t in tracks if "similitud" in t]
        if len(vals) >= 3:
            arr = np.array(vals, dtype=float)
            self._pct_p50 = float(np.quantile(arr, 0.50))
            self._pct_p90 = float(np.quantile(arr, 0.90))

    def fetch_spotify_uri(self, cancion: str, artista: str):
        try:
            res = spotify_data.search(q=f"track:{cancion} artist:{artista}", type="track", limit=1)
            if res["tracks"]["items"]:
                t = res["tracks"]["items"][0]
                return {
                    "uri":             t["uri"],
                    "album":           t["album"]["images"][0]["url"] if t["album"]["images"] else "",
                    "nombre":          cancion,
                    "artista":         artista,
                    "genero_busqueda": "NLP Corpus",
                }
        except Exception:
            pass
        return None

    def _resolver_nombre_en_corpus(self, nombre_actual: str):
        if not nombre_actual:
            return None
        with self.motor_nlp.lock:
            if self.motor_nlp.df.empty or "Cancion" not in self.motor_nlp.df.columns:
                return None
            canciones = self.motor_nlp.df["Cancion"].dropna().astype(str).tolist()

        for cancion in canciones:
            if cancion.lower() == nombre_actual.lower():
                return cancion

        objetivo = normalizar_titulo(nombre_actual)
        if not objetivo:
            return None

        mejor_match = None
        for cancion in canciones:
            cand = normalizar_titulo(cancion)
            if not cand:
                continue
            if cand == objetivo:
                return cancion
            if objetivo in cand or cand in objetivo:
                if mejor_match is None or len(cand) > len(normalizar_titulo(mejor_match)):
                    mejor_match = cancion

        return mejor_match

    def _terminos_clave_song(self, nombre_cancion: str, top_n: int = 8):
        nombre_corpus = self._resolver_nombre_en_corpus(nombre_cancion) or nombre_cancion
        try:
            terms = self.motor_nlp.mostrar_terminos_clave(nombre_corpus, top_n=top_n)
            return [t["termino"] for t in terms if isinstance(t, dict) and t.get("termino")]
        except Exception:
            return []

    def _motivos_locales(self, nombre_actual: str, tracks: list[dict],
                          artista_base=None) -> list[dict]:
        base_terms = self._terminos_clave_song(nombre_actual, top_n=10)
        base_set = set(base_terms)
        out = []
        for tr in tracks:
            nombre_rec = tr.get("nombre", "")
            artista_rec = tr.get("artista", "")
            rec_terms = self._terminos_clave_song(nombre_rec, top_n=10)
            rec_set = set(rec_terms)
            compartidos = [t for t in base_terms if t in rec_set][:5]
            solo_base = [t for t in base_terms if t not in rec_set][:3]
            solo_rec  = [t for t in rec_terms  if t not in base_set][:3]

            score_raw = tr.get("similitud")
            pct_txt = ""
            if score_raw is not None:
                try:
                    pct = self._score_a_porcentaje(float(score_raw))
                    pct_txt = f" Coincide en un {pct}% según la escala de esta lista."
                except Exception:
                    pct_txt = ""

            modo = ""
            if tr.get("genero_busqueda") == "NLP Corpus":
                modo = (" La puntuación refleja similitud coseno entre vectores TF-IDF "
                        "(unigramas y bigramas) de las letras.")
            elif "similitud" in tr:
                modo = (" La puntuación combina afinidad lírica (TF-IDF cuando hay letra en el corpus) "
                        "con popularidad relativa dentro de tus canciones cargadas.")

            ref = f"'{nombre_actual[:22]}'"
            if artista_base:
                ref = f"'{nombre_actual[:18]}' de {artista_base.split()[0][:12]}"

            if compartidos:
                nucleo = (f"Comparte vocabulario lírico fuerte con {ref}: "
                          f"{', '.join(compartidos)}.")
            elif base_terms or rec_terms:
                nucleo = (f"El modelo ve afinidad temática con {ref} aunque no hay "
                          f"muchos términos TF-IDF exactamente iguales en el top.")
            else:
                nucleo = "Recomendada por el ranking híbrido (letras y/o popularidad)."

            detalle = ""
            if solo_base:
                detalle += f" En lo que escuchas destacan: {', '.join(solo_base)}."
            if solo_rec:
                detalle += f" En esta pista resaltan: {', '.join(solo_rec)}."
            if artista_rec:
                detalle += f" Artista: {artista_rec.split()[0][:18]}."

            motivo = (nucleo + detalle + pct_txt + modo).strip()
            tr["motivo_recomendacion"] = motivo
            out.append(tr)
        return out

    def _motivos_gpt(self, nombre_actual: str, tracks: list[dict]) -> list[dict]:
        if not self.openai_api_key:
            return self._motivos_locales(nombre_actual, tracks)

        payload_items = []
        for tr in tracks:
            key = (nombre_actual.lower(), tr.get("nombre", "").lower(), tr.get("artista", "").lower())
            if key in self.explanation_cache:
                tr["motivo_recomendacion"] = self.explanation_cache[key]
                continue
            rec_terms  = self._terminos_clave_song(tr.get("nombre", ""), top_n=8)
            base_terms = self._terminos_clave_song(nombre_actual, top_n=8)
            compartidos = [t for t in base_terms if t in set(rec_terms)][:4]
            payload_items.append({
                "nombre": tr.get("nombre", ""),
                "artista": tr.get("artista", ""),
                "match_pct": int(round(max(0.0, min(1.0, float(tr.get("similitud", 0.0)))) * 100)),
                "terminos_compartidos": compartidos,
            })

        if not payload_items:
            return tracks

        try:
            prompt = {
                "cancion_base": nombre_actual,
                "recomendaciones": payload_items,
                "instrucciones": (
                    "Devuelve SOLO JSON válido con una lista de objetos con campos "
                    "nombre, artista, motivo. El motivo debe tener 1 sola oración, "
                    "en español, máximo 16 palabras, explicando por qué se recomendó "
                    "según similitud lírica/temas."
                ),
            }
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_api_key}",
                         "Content-Type": "application/json"},
                json={"model": OPENAI_MODEL, "temperature": 0.4,
                      "messages": [
                          {"role": "system", "content": "Eres un asistente que responde solo JSON válido."},
                          {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                      ]},
                timeout=10,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            data = json.loads(content)
            motivos_map = {
                (d.get("nombre","").lower(), d.get("artista","").lower()): d.get("motivo","")
                for d in data if isinstance(d, dict)
            }
            for tr in tracks:
                key_song  = (tr.get("nombre","").lower(), tr.get("artista","").lower())
                cache_key = (nombre_actual.lower(), key_song[0], key_song[1])
                motivo = motivos_map.get(key_song, "")
                if not motivo:
                    motivo = self._motivos_locales(nombre_actual, [tr])[0]["motivo_recomendacion"]
                self.explanation_cache[cache_key] = motivo
                tr["motivo_recomendacion"] = motivo
        except Exception:
            return self._motivos_locales(nombre_actual, tracks)

        return tracks

    def display_nlp_explainability(self, nombre_actual: str):
        try:
            nombre_corpus = self._resolver_nombre_en_corpus(nombre_actual)
            if not nombre_corpus:
                return
            terminos = self.motor_nlp.mostrar_terminos_clave(nombre_corpus, top_n=7)
            if not terminos:
                return

            insight = f"> TF-IDF ANÁLISIS\n> Track: {nombre_corpus[:22]}\n\n"
            for t in terminos:
                bar = "█" * int(t['score'] * 20)
                insight += f"  {t['termino'][:14]:<14} {bar} {t['score']:.3f}\n"

            letra_texto = ""
            with self.motor_nlp.lock:
                df = self.motor_nlp.df
                letra_row = df[df["Cancion"].str.lower() == nombre_corpus.lower()]
                if not letra_row.empty and "Letra" in letra_row.columns:
                    letra_texto = str(letra_row["Letra"].iloc[0])

            if letra_texto:
                threading.Thread(
                    target=self._analizar_y_mostrar_sentimiento,
                    args=(letra_texto, insight),
                    daemon=True
                ).start()
            else:
                insight += "\n> Vocabulario compartido detectado."
                self.update_insight(insight)

        except Exception as e:
            print(f"[XAI] {e}")

    def _analizar_y_mostrar_sentimiento(self, letra: str, base_insight: str):
        from TLN_2 import analizar_sentimiento_letra, SENTIMIENTO_DISPONIBLE
        if not SENTIMIENTO_DISPONIBLE:
            self.update_insight(base_insight + "\n\n[pysentimiento no instalado]")
            return

        self.update_insight(base_insight + "\n\n> Analizando sentimiento... ⏳")
        sent = analizar_sentimiento_letra(letra)
        if sent:
            map_polaridad = {"POS": "POSITIVO ▲", "NEG": "NEGATIVO ▼", "NEU": "NEUTRAL ─"}
            map_emocion = {
                "joy": "ALEGRÍA", "sadness": "TRISTEZA",
                "anger": "ENOJO", "fear": "MIEDO",
                "surprise": "SORPRESA", "disgust": "DISGUSTO", "others": "OTRA"
            }
            pol = map_polaridad.get(sent["polaridad"], sent["polaridad"])
            emo = map_emocion.get(sent["emocion"], sent["emocion"])
            nuevo = (base_insight +
                     f"\n> SENTIMENT ANALYSIS\n"
                     f"  Polaridad : {pol} {sent['prob_sentimiento']*100:.1f}%\n"
                     f"  Emoción   : {emo} {sent['prob_emocion']*100:.1f}%")
            self.update_insight(nuevo)

            if hasattr(self, 'current_rendered_tracks') and self.current_rendered_tracks:
                threading.Thread(
                    target=self._reordenar_por_emocion,
                    args=(sent["emocion"], self.current_rendered_tracks.copy(), map_emocion),
                    daemon=True
                ).start()
        else:
            self.update_insight(base_insight + "\n\n> Análisis de sentimiento no disponible.")

    def _reordenar_por_emocion(self, emocion_objetivo: str, tracks: list[dict],
                                 map_emocion: dict):
        from TLN_2 import analizar_sentimiento_letra

        tracks_a_analizar = tracks[:8]
        restantes = tracks[8:]
        tracks_con_emocion = []

        for tr in tracks_a_analizar:
            letra_texto = ""
            with self.motor_nlp.lock:
                df = self.motor_nlp.df
                match = df[df["Cancion"].str.lower() == tr["nombre"].lower()]
                if not match.empty and "Letra" in match.columns:
                    letra_texto = str(match["Letra"].iloc[0])

            if letra_texto:
                sent = analizar_sentimiento_letra(letra_texto)
                tr["emocion_calculada"] = sent.get("emocion", "unknown")
            else:
                tr["emocion_calculada"] = "unknown"

            if tr["emocion_calculada"] == emocion_objetivo:
                tr["score_emocion"] = 1
                emo_es = map_emocion.get(emocion_objetivo, emocion_objetivo)
                if "Comparte tu misma emoción" not in tr.get("motivo_recomendacion", ""):
                    tr["motivo_recomendacion"] = (
                        f"◈ Comparte tu misma emoción ({emo_es}). " +
                        tr.get("motivo_recomendacion", "")
                    )
            else:
                tr["score_emocion"] = 0

            tracks_con_emocion.append(tr)

        tracks_con_emocion.sort(key=lambda x: x.get("score_emocion", 0), reverse=True)
        tracks_finales = tracks_con_emocion + restantes
        self.after(0, lambda: self.render_grid(tracks_finales))

    def check_state(self):
        try:
            current = sp.current_playback()
            if current and current.get("is_playing") and current.get("item"):
                item          = current["item"]
                track_id      = item["id"]
                nombre_actual = item["name"]
                artista_actual = item["artists"][0]["name"]
                is_playing    = current["is_playing"]

                self.now_playing_label.configure(text=nombre_actual)
                self.now_playing_artist.configure(text=artista_actual)

                self.audio_viz.set_playing(is_playing)
                self.cover_canvas.set_playing(is_playing)

                if is_playing:
                    self.btn_play_pause.configure(image=self.icon_pause_large)
                else:
                    self.btn_play_pause.configure(image=self.icon_play_large)

                prog = current.get("progress_ms", 0)
                dur  = item.get("duration_ms", 1)
                if dur > 0:
                    self.progress_bar.set(prog / dur)

                if self.currently_playing_id != track_id:
                    self.currently_playing_id = track_id
                    album_url = item["album"]["images"][0]["url"] if item["album"]["images"] else ""
                    if album_url:
                        threading.Thread(
                            target=self.fetch_image,
                            args=(album_url, self.cover_canvas, (240, 240), True),
                            daemon=True
                        ).start()

                    self.show_loading(f"Buscando recomendaciones\npara '{nombre_actual[:28]}'")
                    threading.Thread(
                        target=self._actualizar_recomendaciones,
                        args=(track_id, nombre_actual, artista_actual, current),
                        daemon=True
                    ).start()
            else:
                self.now_playing_label.configure(text="Ninguna Pista Activa")
                self.now_playing_artist.configure(text="Esperando reproducción...")
                self.audio_viz.set_playing(False)
                self.cover_canvas.set_playing(False)
                self.btn_play_pause.configure(image=self.icon_play_large)
                self.currently_playing_id = None
        except Exception as e:
            print(f"[UI] {e}")

        self.after(4000, self.check_state)

    def _actualizar_recomendaciones(self, track_id, nombre_actual, artista_actual, current):

        with self.motor_nlp.lock:
            en_corpus = not self.motor_nlp.df[
                self.motor_nlp.df["Cancion"].str.lower() == nombre_actual.lower()
            ].empty

        if not en_corpus:
            letra = obtener_letra(artista_actual, nombre_actual)
            if letra:
                nueva_fila = pd.DataFrame([{
                    "Artista":         artista_actual,
                    "Cancion":         nombre_actual,
                    "Album":           current["item"]["album"]["name"],
                    "Genero":          "Dinámico",
                    "Letra":           letra,
                    "letra_procesada": preprocesar_letra(letra),
                }])
                with self.motor_nlp.lock:
                    self.motor_nlp.df = pd.concat(
                        [self.motor_nlp.df, nueva_fila], ignore_index=True
                    )
                    try:
                        self.motor_nlp.df.to_csv("corpus_letras.csv", index=False, encoding="utf-8")
                    except Exception:
                        pass
                    self.motor_nlp.matriz_tfidf = self.motor_nlp.vectorizer.fit_transform(
                        self.motor_nlp.df["letra_procesada"]
                    )
                    self.motor_nlp.matriz_similitud = cosine_similarity(self.motor_nlp.matriz_tfidf)

        nombre_corpus = self._resolver_nombre_en_corpus(nombre_actual) or nombre_actual
        nlp_recs = self.motor_nlp.recomendar(nombre_corpus, k=TOP_N_DEFAULT)
        if nlp_recs:
            ui_tracks = []
            for r in nlp_recs:
                spotify_info = self.fetch_spotify_uri(r["Cancion"], r["Artista"])
                if spotify_info:
                    spotify_info["similitud"] = r["Similitud"]
                    ui_tracks.append(spotify_info)

            if ui_tracks:
                self._actualizar_percentiles_visuales(ui_tracks)
                tracks_final = self._motivos_locales(nombre_actual, ui_tracks,
                                                      artista_base=artista_actual)
                def _ui_nlp(tf=tracks_final, nc=nombre_corpus, na=nombre_actual):
                    self.lbl_subtitle.configure(text=f"Recomendaciones para: '{na}'")
                    self.display_nlp_explainability(nc)
                    self.render_grid(tf)
                self.after(0, _ui_nlp)
                return

        recomendadas = self.motor_content.obtener_recomendaciones(
            track_id, self.motor_nlp, top_n=TOP_N_DEFAULT
        )
        tiene_nlp = any(r.get("similitud", 0) > MIN_SIMILARITY for r in recomendadas)
        self._actualizar_percentiles_visuales(recomendadas)
        tracks_final = self._motivos_locales(nombre_actual, recomendadas,
                                              artista_base=artista_actual)

        if tiene_nlp:
            def _ui_hybrid(tf=tracks_final, na=nombre_actual):
                self.lbl_subtitle.configure(text=f"Recomendación Híbrida (NLP + Pop): '{na}'")
                self.display_nlp_explainability(na)
                self.render_grid(tf)
            self.after(0, _ui_hybrid)
        else:
            def _ui_pop(tf=tracks_final, na=nombre_actual):
                self.lbl_subtitle.configure(text=f"Recomendación por popularidad: '{na}'")
                self.update_insight(
                    f"> Sin letra en corpus:\n> {na[:20]}\n\n"
                    "> Usando similitud de popularidad\n> normalizada como fallback."
                )
                self.render_grid(tf)
            self.after(0, _ui_pop)

if __name__ == "__main__":
    motor_content = MotorRecomendacion()
    motor_content.cargar_datos()

    motor_nlp = SistemaRecomendacion("corpus_letras.csv")
    motor_nlp.lock = threading.Lock()

    threading.Thread(
        target=background_nlp_enricher,
        args=(motor_content.df_canciones, motor_nlp),
        daemon=True,
    ).start()

    app = SpotifyDesktopApp(motor_content, motor_nlp)
    app.mainloop()
