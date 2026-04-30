import os
import threading
import queue
import time
import re
import json
import requests
from io import BytesIO

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

import customtkinter as ctk
from PIL import Image

# ── Importamos nuestro motor NLP de Letras
from TLN_2 import SistemaRecomendacion, preprocesar_letra

# ─────────────────────────────────────────
# CREDENCIALES — usar variables de entorno
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────
def obtener_letra(artista: str, cancion: str) -> str:
    """Busca la letra en lyrics.ovh."""
    url = f"https://api.lyrics.ovh/v1/{artista}/{cancion}"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json().get("lyrics", "")
    except Exception:
        pass
    return ""


def normalizar_scores(scores: np.ndarray) -> np.ndarray:
    """Normaliza un array de scores al rango [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def normalizar_titulo(texto: str) -> str:
    """
    Normaliza títulos para mejorar matching entre Spotify y corpus NLP.
    Elimina sufijos típicos como "(Remastered ...)", "[Live]" y "feat.".
    """
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(r"\b(feat|ft)\.?\s+[^-–]+", " ", t)
    t = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalizar_artista(texto: str) -> str:
    """Normaliza nombre de artista para matching robusto."""
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ─────────────────────────────────────────
# MOTOR DE RECOMENDACIÓN HÍBRIDO
# ─────────────────────────────────────────
# MEJORA PRINCIPAL: se fusionan el score NLP (similitud coseno sobre TF-IDF
# de letras) con el score de popularidad usando un peso configurable.
# Esto elimina el problema original donde features triviales hacen que
# la similitud coseno no discrimine nada.
#
#   score_final = α * score_nlp + (1-α) * score_popularidad
#
# α = NLP_WEIGHT (0.0 → solo popularidad, 1.0 → solo NLP)
# ─────────────────────────────────────────
NLP_WEIGHT       = 0.75   # Peso del score NLP vs popularidad
MIN_SIMILARITY   = 0.05   # Umbral mínimo para filtrar ruido (coseno ≥ 0.05)
TOP_N_DEFAULT    = 8
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class MotorRecomendacion:
    def __init__(self):
        self.df_canciones   = pd.DataFrame()
        self.scaler         = MinMaxScaler()

    # ── Carga de datos desde Spotify ─────────────────────────────────────────
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

        # Normalizar popularidad con MinMaxScaler (más robusto que división manual)
        self.df_canciones["pop_norm"] = self.scaler.fit_transform(
            self.df_canciones[["popularidad"]]
        )

    # ── Recomendación híbrida: NLP + popularidad ─────────────────────────────
    def obtener_recomendaciones(
        self,
        track_id:  str,
        motor_nlp: "SistemaRecomendacion",
        top_n:     int = TOP_N_DEFAULT,
    ) -> list[dict]:
        """
        Fusiona scores NLP (letras) con popularidad normalizada.
        Si el motor NLP no tiene datos suficientes, usa solo popularidad.
        """
        df = self.df_canciones

        # Score de popularidad para todas las canciones
        pop_scores = df["pop_norm"].values.copy()

        # Score NLP: similitud coseno entre la canción actual y el corpus NLP
        nlp_scores = self._calcular_scores_nlp(track_id, df, motor_nlp)

        if nlp_scores is not None:
            # Normalizar ambos scores al mismo rango antes de combinar
            pop_norm  = normalizar_scores(pop_scores)
            nlp_norm  = normalizar_scores(nlp_scores)
            score_final = NLP_WEIGHT * nlp_norm + (1 - NLP_WEIGHT) * pop_norm
        else:
            # Fallback limpio: solo popularidad
            score_final = normalizar_scores(pop_scores)

        # Excluir la canción actual y aplicar umbral mínimo
        mascara_propia = df["id"] != track_id
        mascara_umbral = score_final >= MIN_SIMILARITY
        mascara = mascara_propia & mascara_umbral

        if mascara.sum() == 0:
            # Si nada supera el umbral, relajamos el filtro
            mascara = mascara_propia

        candidatos = df[mascara].copy()
        candidatos["similitud"] = score_final[mascara]

        return (
            candidatos
            .sort_values("similitud", ascending=False)
            .head(top_n)
            .to_dict("records")
        )

    def _calcular_scores_nlp(
        self,
        track_id:  str,
        df:        pd.DataFrame,
        motor_nlp: "SistemaRecomendacion",
    ) -> np.ndarray | None:
        """
        Calcula similitud coseno entre la canción actual y TODAS las canciones
        del df usando la matriz TF-IDF del motor NLP. Retorna None si no hay
        datos NLP suficientes.
        """
        if motor_nlp is None or motor_nlp.df.empty or motor_nlp.matriz_tfidf is None:
            return None

        # Canción de referencia desde Spotify
        fila_actual = df[df["id"] == track_id]
        if fila_actual.empty:
            return None
        nombre_actual = fila_actual.iloc[0]["nombre"]
        artista_actual = fila_actual.iloc[0].get("artista", "")

        objetivo_titulo = normalizar_titulo(nombre_actual)
        objetivo_artista = normalizar_artista(artista_actual)

        # Buscar en el corpus NLP
        with motor_nlp.lock:
            df_nlp = motor_nlp.df.copy()
            if df_nlp.empty or "Cancion" not in df_nlp.columns:
                return None

            # Índice robusto por (titulo_norm, artista_norm). Fallback por título.
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

            # Coseno contra TODO el corpus NLP (más estable y rápido que loop por track).
            all_scores_nlp = cosine_similarity(vec_actual, motor_nlp.matriz_tfidf)[0]

            # Mapeo de Spotify track -> mejor score NLP usando título/artista normalizados
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


# ─────────────────────────────────────────
# ENRICHER EN BACKGROUND (con lock)
# ─────────────────────────────────────────
def background_nlp_enricher(df_canciones: pd.DataFrame, motor_nlp: "SistemaRecomendacion"):
    """
    Descarga letras para las canciones del df que aún no están en el corpus,
    y actualiza la matriz TF-IDF de forma thread-safe usando motor_nlp.lock.
    """
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
            # Guardado incremental — solo escribimos si la sesión es estable
            try:
                motor_nlp.df.to_csv("corpus_letras.csv", index=False, encoding="utf-8")
            except Exception:
                pass
            # Re-fit solo dentro del lock para evitar race condition
            motor_nlp.matriz_tfidf    = motor_nlp.vectorizer.fit_transform(
                motor_nlp.df["letra_procesada"]
            )
            motor_nlp.matriz_similitud = cosine_similarity(motor_nlp.matriz_tfidf)

        time.sleep(1.5)


# ─────────────────────────────────────────
# INTERFAZ GRÁFICA
# ─────────────────────────────────────────
class SpotifyDesktopApp(ctk.CTk):
    def __init__(self, motor_content: MotorRecomendacion, motor_nlp: SistemaRecomendacion):
        super().__init__()
        self.motor_content        = motor_content
        self.motor_nlp            = motor_nlp
        self.currently_playing_id = None
        self.image_queue          = queue.Queue()
        self.openai_api_key       = os.getenv("OPENAI_API_KEY", "").strip()
        self.explanation_cache    = {}

        # ── Configuración de ventana ──────────────────────────────────────────
        self.title("S P O T I F Y  A I  —  Premium Gold")
        self.geometry("1200x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color="#0C0C0A")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Paleta
        self.color_accent     = "#A39422"
        self.color_sidebar    = "#080806"
        self.color_main       = "#141412"
        self.color_card       = "#1C1C19"
        self.color_text_muted = "#8E8E89"

        # ── Sidebar ───────────────────────────────────────────────────────────
        self.sidebar_frame = ctk.CTkFrame(self, width=260, fg_color=self.color_sidebar, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            self.sidebar_frame, text="⚡ SPOTIFY AI",
            font=("Courier", 18, "bold"), text_color=self.color_accent
        ).pack(pady=(30, 40), padx=20, anchor="w")

        for txt in ["MY MUSIC", "LOCAL"]:
            ctk.CTkLabel(self.sidebar_frame, text=txt, font=("Inter", 11, "bold"),
                         text_color="#FFFFFF").pack(pady=(20, 10), padx=20, anchor="w")

        ctk.CTkLabel(
            self.sidebar_frame, text="▶ Tracks NLP",
            font=("Inter", 13, "bold"), text_color=self.color_accent
        ).pack(pady=(0, 10), padx=20, anchor="w")

        # Sección NLP Insights
        ctk.CTkLabel(
            self.sidebar_frame, text="🧠 NLP INSIGHTS",
            font=("Inter", 11, "bold"), text_color="#FFFFFF"
        ).pack(pady=(40, 10), padx=20, anchor="w")

        self.txt_insight = ctk.CTkTextbox(
            self.sidebar_frame, width=220, height=250,
            fg_color="#181815", text_color=self.color_text_muted,
            font=("Consolas", 12), border_color=self.color_accent, border_width=1
        )
        self.txt_insight.pack(padx=20, anchor="w")
        self.txt_insight.insert("1.0", "> Esperando reproducción\n> para extraer términos\n> TF-IDF...")
        self.txt_insight.configure(state="disabled")

        # ── Main content ──────────────────────────────────────────────────────
        self.main_frame = ctk.CTkScrollableFrame(self, fg_color=self.color_main, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(
            self.main_frame, text="VIBRACIONES PERSONALIZADAS",
            font=("Inter", 11, "bold"), text_color=self.color_accent
        ).grid(row=0, column=0, pady=(30, 5), padx=40, sticky="w")

        self.lbl_subtitle = ctk.CTkLabel(
            self.main_frame, text="Recomendaciones Basadas en tus Gustos",
            font=("Inter", 24, "bold"), text_color="#FFFFFF"
        )
        self.lbl_subtitle.grid(row=1, column=0, pady=(0, 30), padx=40, sticky="w")

        self.grid_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.grid_frame.grid(row=2, column=0, sticky="nsew", padx=30)

        # ── Player inferior ───────────────────────────────────────────────────
        self.player_frame = ctk.CTkFrame(self, height=90, fg_color=self.color_sidebar, corner_radius=0)
        self.player_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.player_frame.pack_propagate(False)

        self.now_playing_label = ctk.CTkLabel(
            self.player_frame, text="Ninguna Pista Activa",
            font=("Inter", 14, "bold"), text_color="#FFFFFF"
        )
        self.now_playing_label.pack(side="left", padx=40)

        controls = ctk.CTkFrame(self.player_frame, fg_color="transparent")
        controls.pack(side="left", expand=True)

        ctk.CTkButton(
            controls, text="⏯", width=50, height=50, corner_radius=25,
            font=("Inter", 24), fg_color=self.color_accent,
            hover_color="#C2B12F", text_color="#000000",
            command=self.toggle_playback
        ).pack(side="left", padx=15)

        ctk.CTkButton(
            controls, text="⏭", width=40, height=40,
            font=("Inter", 20), fg_color="transparent",
            hover_color="#20201A", text_color=self.color_text_muted,
            command=self.next_track
        ).pack(side="left", padx=10)

        # Mostrar canciones iniciales aleatorias
        self.process_image_queue()
        if not self.motor_content.df_canciones.empty:
            sample = self.motor_content.df_canciones.sample(
                min(12, len(self.motor_content.df_canciones))
            ).to_dict("records")
            self.render_grid(sample)
        self.check_state()

    # ── Utilidades de UI ──────────────────────────────────────────────────────
    def process_image_queue(self):
        while not self.image_queue.empty():
            try:
                ctk_img, label_widget = self.image_queue.get_nowait()
                try:
                    label_widget.configure(image=ctk_img if ctk_img else None,
                                           text="" if ctk_img else "No Image")
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

    def fetch_image(self, url: str, label_widget):
        try:
            response = requests.get(url, timeout=5)
            img_data = Image.open(BytesIO(response.content)).resize((120, 120))
            ctk_img  = ctk.CTkImage(light_image=img_data, dark_image=img_data, size=(120, 120))
            self.image_queue.put((ctk_img, label_widget))
        except Exception:
            self.image_queue.put((None, label_widget))

    # ── Controles Spotify ──────────────────────────────────────────────────────
    def toggle_playback(self):
        try:
            current = sp.current_playback()
            if current and current["is_playing"]:
                sp.pause_playback()
            else:
                sp.start_playback()
        except Exception:
            pass

    def next_track(self):
        try:
            sp.next_track()
        except Exception:
            pass

    def play_track_uri(self, uri: str | None):
        """
        Inicia reproducción de una pista por URI.
        Spotify Web API suele exigir un dispositivo con la app abierta; si no hay
        reproductor activo, start_playback falla (404). Se elige dispositivo y
        se intenta transfer_playback antes de reproducir.
        """
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
                except SpotifyException:
                    pass
                except Exception:
                    pass
                sp.start_playback(device_id=device_id, uris=[uri])
            else:
                sp.start_playback(uris=[uri])
        except SpotifyException as e:
            if e.http_status == 404:
                self._notificar_error_reproduccion(
                    "No hay reproductor activo. Abre Spotify en tu PC o móvil, "
                    "reproduce cualquier canción unos segundos y vuelve a pulsar Reproducir."
                )
            elif e.http_status == 403:
                self._notificar_error_reproduccion(
                    "Spotify no permitió el control de reproducción. "
                    "Suele requerir cuenta Premium y permisos de la app."
                )
            else:
                self._notificar_error_reproduccion(
                    e.msg if getattr(e, "msg", None) else str(e)
                )
        except Exception as e:
            self._notificar_error_reproduccion(str(e))

    def _seleccionar_dispositivo_spotify(self) -> str | None:
        """Devuelve id de dispositivo preferido (activo o el primero disponible)."""
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
                    if (
                        d.get("type") == tipo
                        and d.get("id")
                        and not d.get("is_restricted")
                    ):
                        return str(d["id"])
            for d in devices:
                if d.get("id") and not d.get("is_restricted"):
                    return str(d["id"])
        except Exception:
            pass
        return None

    def _notificar_error_reproduccion(self, mensaje: str) -> None:
        print(f"[Spotify reproducción] {mensaje}")
        try:
            prev = self.lbl_subtitle.cget("text")
            corto = mensaje if len(mensaje) <= 140 else mensaje[:137] + "…"
            self.lbl_subtitle.configure(text=f"⚠ {corto}")

            def restaurar():
                try:
                    self.lbl_subtitle.configure(text=prev)
                except Exception:
                    pass

            self.after(9000, restaurar)
        except Exception:
            pass

    def show_track_detail_modal(self, track: dict):
        """Ventana modal con toda la información de la pista y descripción completa."""
        win = ctk.CTkToplevel(self)
        win.title("Detalle de la recomendación")
        win.geometry("560x520")
        win.minsize(480, 400)
        win.configure(fg_color=self.color_main)
        win.transient(self)
        win.grab_set()

        def _cerrar():
            win.grab_release()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _cerrar)

        header = ctk.CTkFrame(win, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(16, 8))

        ctk.CTkLabel(
            header, text="Detalle",
            font=("Inter", 11, "bold"), text_color=self.color_accent
        ).pack(anchor="w")

        scroll = ctk.CTkScrollableFrame(
            win, fg_color=self.color_card, corner_radius=12, border_width=1,
            border_color="#2A2A26"
        )
        scroll.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        nombre_completo = track.get("nombre") or "—"
        artista = track.get("artista") or "—"

        ctk.CTkLabel(
            scroll, text=nombre_completo,
            font=("Inter", 18, "bold"), text_color="#FFFFFF", wraplength=500, justify="left"
        ).pack(anchor="w", padx=16, pady=(16, 4))
        ctk.CTkLabel(
            scroll, text=artista,
            font=("Inter", 14), text_color=self.color_text_muted, wraplength=500, justify="left"
        ).pack(anchor="w", padx=16, pady=(0, 12))

        if "similitud" in track:
            try:
                raw = float(track["similitud"])
                raw_clamped = max(0.0, min(1.0, raw))
                pct = self._score_a_porcentaje(raw)
                match_txt = (
                    f"Coincidencia: {pct}% (escala visual de esta lista) · "
                    f"score bruto del modelo: {raw_clamped:.4f}"
                )
            except Exception:
                match_txt = f"Coincidencia: {track.get('similitud')}"
            ctk.CTkLabel(
                scroll, text=match_txt,
                font=("Inter", 12), text_color="#FFFFFF", wraplength=500, justify="left"
            ).pack(anchor="w", padx=16, pady=(0, 8))

        origen = track.get("genero_busqueda")
        if origen:
            ctk.CTkLabel(
                scroll, text=f"Origen: {origen}",
                font=("Inter", 11), text_color=self.color_text_muted, wraplength=500, justify="left"
            ).pack(anchor="w", padx=16, pady=(0, 8))

        ctk.CTkLabel(
            scroll, text="Descripción (completa)",
            font=("Inter", 12, "bold"), text_color="#FFFFFF"
        ).pack(anchor="w", padx=16, pady=(12, 4))

        desc_completa = track.get(
            "motivo_recomendacion",
            "Sin descripción generada para esta recomendación.",
        )
        tb = ctk.CTkTextbox(
            scroll, width=500, height=220,
            fg_color="#181815", text_color=self.color_text_muted,
            font=("Inter", 12), wrap="word", border_color=self.color_accent, border_width=1
        )
        tb.pack(fill="x", padx=16, pady=(0, 12))
        tb.insert("1.0", desc_completa)
        tb.configure(state="disabled")

        extras: list[str] = []
        if "popularidad" in track and track["popularidad"] is not None:
            extras.append(f"Popularidad (Spotify): {track['popularidad']}")
        if track.get("id"):
            extras.append(f"ID de pista: {track['id']}")
        uri = track.get("uri")
        if uri:
            extras.append(f"URI: {uri}")
        alb_url = track.get("album")
        if isinstance(alb_url, str) and alb_url.startswith("http"):
            extras.append(f"Portada (URL): {alb_url}")

        if extras:
            ctk.CTkLabel(
                scroll, text="Información adicional",
                font=("Inter", 12, "bold"), text_color="#FFFFFF"
            ).pack(anchor="w", padx=16, pady=(8, 4))
            ctk.CTkLabel(
                scroll, text="\n".join(extras),
                font=("Consolas", 11), text_color=self.color_text_muted,
                wraplength=500, justify="left"
            ).pack(anchor="w", padx=16, pady=(0, 16))

        footer = ctk.CTkFrame(win, fg_color="transparent")
        footer.pack(fill="x", padx=20, pady=(0, 16))

        if uri:
            ctk.CTkButton(
                footer, text="▶ Reproducir en Spotify",
                fg_color=self.color_accent, hover_color="#C2B12F", text_color="#000000",
                command=lambda u=uri: self.play_track_uri(u),
            ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            footer, text="Cerrar",
            fg_color="transparent", hover_color="#20201A",
            text_color=self.color_text_muted, command=_cerrar,
        ).pack(side="left")

        win.update_idletasks()
        x = max(0, (win.winfo_screenwidth() // 2) - (win.winfo_width() // 2))
        y = max(0, (win.winfo_screenheight() // 2) - (win.winfo_height() // 2))
        win.geometry(f"+{x}+{y}")
        win.focus_force()

    # ── Grid de tarjetas ──────────────────────────────────────────────────────
    def render_grid(self, tracks: list[dict]):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        columns = 4
        for i, track in enumerate(tracks):
            row_idx = i // columns
            col_idx = i % columns

            card = ctk.CTkFrame(
                self.grid_frame, fg_color=self.color_card,
                corner_radius=12, border_width=2,
                border_color=self.color_card, cursor="hand2"
            )
            card.grid(row=row_idx, column=col_idx, padx=15, pady=15, sticky="n")

            def on_enter(e, c=card): c.configure(border_color=self.color_accent)
            def on_leave(e, c=card): c.configure(border_color=self.color_card)

            def abrir_modal(_e=None, t=dict(track)):
                self.show_track_detail_modal(t)

            for widget in [card]:
                widget.bind("<Enter>", on_enter)
                widget.bind("<Leave>", on_leave)
                widget.bind("<Button-1>", abrir_modal)

            img_lbl = ctk.CTkLabel(card, text="Loading...", width=140, height=140,
                                   fg_color="#111111", corner_radius=8)
            img_lbl.pack(pady=(15, 12), padx=15)
            img_lbl.bind("<Enter>", on_enter)
            img_lbl.bind("<Leave>", on_leave)
            img_lbl.bind("<Button-1>", abrir_modal)

            if track.get("album"):
                threading.Thread(
                    target=self.fetch_image, args=(track["album"], img_lbl), daemon=True
                ).start()

            nombre = track["nombre"]
            if len(nombre) > 16:
                nombre = nombre[:14] + "…"

            title_lbl = ctk.CTkLabel(card, text=nombre, font=("Inter", 14, "bold"), text_color="#FFFFFF")
            title_lbl.pack(anchor="w", padx=15)
            title_lbl.bind("<Enter>", on_enter)
            title_lbl.bind("<Leave>", on_leave)
            title_lbl.bind("<Button-1>", abrir_modal)

            descripcion = track.get(
                "motivo_recomendacion",
                "Coincidencia calculada por similitud de letras (TF-IDF) y/o popularidad.",
            )
            if len(descripcion) > 220:
                descripcion = descripcion[:217] + "…"
            desc_lbl = ctk.CTkLabel(
                card, text=descripcion, font=("Inter", 10),
                text_color=self.color_text_muted, justify="left", wraplength=240
            )
            desc_lbl.pack(anchor="w", padx=15, pady=(2, 8))
            desc_lbl.bind("<Enter>", on_enter)
            desc_lbl.bind("<Leave>", on_leave)
            desc_lbl.bind("<Button-1>", abrir_modal)

            # Fila inferior: porcentaje / etiqueta a la izquierda, Reproducir a la derecha
            if "similitud" in track:
                pct = self._score_a_porcentaje(float(track["similitud"]))
                tag = f"{pct}% MATCH"
            else:
                tag = track.get("genero_busqueda", "RECOMENDADO")

            row_bottom = ctk.CTkFrame(card, fg_color="transparent")
            row_bottom.pack(fill="x", padx=15, pady=(0, 15))

            match_lbl = ctk.CTkLabel(
                row_bottom, text=tag, font=("Inter", 11, "bold"),
                text_color=self.color_accent
            )
            match_lbl.pack(side="left", anchor="w")
            match_lbl.bind("<Enter>", on_enter)
            match_lbl.bind("<Leave>", on_leave)
            match_lbl.bind("<Button-1>", abrir_modal)

            uri_track = track.get("uri")
            if uri_track:
                def _reproducir(_u=uri_track):
                    self.play_track_uri(_u)

                btn_play = ctk.CTkButton(
                    row_bottom, text="Reproducir", width=92, height=28,
                    font=("Inter", 11, "bold"),
                    fg_color=self.color_accent, hover_color="#C2B12F",
                    text_color="#000000", corner_radius=6,
                    command=_reproducir,
                )
                btn_play.pack(side="right", anchor="e")

    def _score_a_porcentaje(self, score: float) -> int:
        """
        Convierte score [0,1] a porcentaje visual calibrado por percentiles del lote.
        Evita que coincidencias buenas se vean artificialmente "bajas".
        """
        try:
            v = max(0.0, min(1.0, float(score)))
            p50 = getattr(self, "_pct_p50", 0.10)
            p90 = getattr(self, "_pct_p90", 0.30)
            if p90 <= p50:
                return int(round(v * 100))

            # mapea p50->55% y p90->90%, con saturación suave
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

    # ── Búsqueda de URI en Spotify ─────────────────────────────────────────────
    def fetch_spotify_uri(self, cancion: str, artista: str) -> dict | None:
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

    # ── Explicabilidad TF-IDF (XAI) ───────────────────────────────────────────
    def _resolver_nombre_en_corpus(self, nombre_actual: str) -> str | None:
        """
        Busca la mejor coincidencia de título en el corpus NLP.
        Primero intenta exacto (case-insensitive) y después título normalizado.
        """
        if not nombre_actual:
            return None

        with self.motor_nlp.lock:
            if self.motor_nlp.df.empty or "Cancion" not in self.motor_nlp.df.columns:
                return None
            canciones = self.motor_nlp.df["Cancion"].dropna().astype(str).tolist()

        # 1) Match exacto (rápido)
        for cancion in canciones:
            if cancion.lower() == nombre_actual.lower():
                return cancion

        # 2) Match normalizado (robusto)
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

    def _terminos_clave_song(self, nombre_cancion: str, top_n: int = 8) -> list[str]:
        nombre_corpus = self._resolver_nombre_en_corpus(nombre_cancion) or nombre_cancion
        try:
            terms = self.motor_nlp.mostrar_terminos_clave(nombre_corpus, top_n=top_n)
            return [t["termino"] for t in terms if isinstance(t, dict) and t.get("termino")]
        except Exception:
            return []

    def _motivos_locales(
        self,
        nombre_actual: str,
        tracks: list[dict],
        artista_base: str | None = None,
    ) -> list[dict]:
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
            solo_rec = [t for t in rec_terms if t not in base_set][:3]

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
                modo = (
                    " La puntuación refleja similitud coseno entre vectores TF-IDF "
                    "(unigramas y bigramas) de las letras."
                )
            elif "similitud" in tr:
                modo = (
                    " La puntuación combina afinidad lírica (TF-IDF cuando hay letra en el corpus) "
                    "con popularidad relativa dentro de tus canciones cargadas."
                )

            ref = f"'{nombre_actual[:22]}'"
            if artista_base:
                ref = f"'{nombre_actual[:18]}' de {artista_base.split()[0][:12]}"

            if compartidos:
                nucleo = (
                    f"Comparte vocabulario lírico fuerte con {ref}: "
                    f"{', '.join(compartidos)}."
                )
            elif base_terms or rec_terms:
                nucleo = (
                    f"El modelo ve afinidad temática con {ref} aunque no hay "
                    f"muchos términos TF-IDF exactamente iguales en el top."
                )
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

        # Construimos datos ligeros para explicación; usamos cache para evitar costos repetidos.
        payload_items = []
        for tr in tracks:
            key = (nombre_actual.lower(), tr.get("nombre", "").lower(), tr.get("artista", "").lower())
            if key in self.explanation_cache:
                tr["motivo_recomendacion"] = self.explanation_cache[key]
                continue

            rec_terms = self._terminos_clave_song(tr.get("nombre", ""), top_n=8)
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
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0.4,
                    "messages": [
                        {"role": "system", "content": "Eres un asistente que responde solo JSON válido."},
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                    ],
                },
                timeout=10,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            data = json.loads(content)
            motivos_map = {
                (d.get("nombre", "").lower(), d.get("artista", "").lower()): d.get("motivo", "")
                for d in data if isinstance(d, dict)
            }

            for tr in tracks:
                key_song = (tr.get("nombre", "").lower(), tr.get("artista", "").lower())
                cache_key = (nombre_actual.lower(), key_song[0], key_song[1])
                motivo = motivos_map.get(key_song, "")
                if not motivo:
                    local = self._motivos_locales(nombre_actual, [tr])[0]["motivo_recomendacion"]
                    motivo = local
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

            insight = f"🔍 Análisis TF-IDF:\nCanción: {nombre_corpus[:20]}\n\nTérminos clave:\n"
            for t in terminos:
                insight += f"• {t['termino']}: {t['score']:.3f}\n"
            insight += "\n*Recomendaciones comparten este vocabulario."
            self.update_insight(insight)
        except Exception as e:
            print(f"[XAI] {e}")

    # ── Loop de polling del estado de Spotify ─────────────────────────────────
    def check_state(self):
        try:
            current = sp.current_playback()
            if current and current.get("is_playing") and current.get("item"):
                item           = current["item"]
                track_id       = item["id"]
                nombre_actual  = item["name"]
                artista_actual = item["artists"][0]["name"]

                self.now_playing_label.configure(text=f"{artista_actual} — {nombre_actual}")

                if self.currently_playing_id != track_id:
                    self.currently_playing_id = track_id
                    self._actualizar_recomendaciones(track_id, nombre_actual, artista_actual, current)
            else:
                self.now_playing_label.configure(text="Ninguna Pista Activa")
                self.currently_playing_id = None
        except Exception as e:
            print(f"[UI] {e}")

        self.after(4000, self.check_state)

    def _actualizar_recomendaciones(
        self, track_id: str, nombre_actual: str, artista_actual: str, current: dict
    ):
        """Lógica de actualización extraída para mantener check_state limpio."""
        # Intentar enriquecer la canción actual si no está en el corpus
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

        # Intento 1: recomendación NLP directa (mismo flujo que desktop_app.py)
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
                self.lbl_subtitle.configure(
                    text=f"Recomendación NLP (TF-IDF Letras): '{nombre_actual}'"
                )
                self.display_nlp_explainability(nombre_corpus)
                self._actualizar_percentiles_visuales(ui_tracks)
                tracks_final = self._motivos_locales(
                    nombre_actual, ui_tracks, artista_base=artista_actual
                )
                self.render_grid(tracks_final)
                return

        # Intento 2 (fallback): recomendación híbrida NLP + popularidad
        recomendadas = self.motor_content.obtener_recomendaciones(
            track_id, self.motor_nlp, top_n=TOP_N_DEFAULT
        )

        # Si hay resultados del corpus propio, enriquecer con URI de Spotify
        tiene_nlp = any(
            r.get("similitud", 0) > MIN_SIMILARITY for r in recomendadas
        )

        if tiene_nlp:
            self.lbl_subtitle.configure(
                text=f"Recomendación Híbrida (NLP + Popularidad): '{nombre_actual}'"
            )
            self.display_nlp_explainability(nombre_actual)
        else:
            self.lbl_subtitle.configure(
                text=f"Recomendación Acústica (sin letra NLP): '{nombre_actual}'"
            )
            self.update_insight(
                f"🎶 Sin letra disponible:\n{nombre_actual[:15]}\n\n"
                "Usando similitud de popularidad normalizada."
            )

        self._actualizar_percentiles_visuales(recomendadas)
        tracks_final = self._motivos_locales(
            nombre_actual, recomendadas, artista_base=artista_actual
        )
        self.render_grid(tracks_final)


# ─────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────
if __name__ == "__main__":
    motor_content = MotorRecomendacion()
    motor_content.cargar_datos()

    motor_nlp = SistemaRecomendacion("corpus_letras.csv")
    # Añadir lock al motor NLP para uso thread-safe
    motor_nlp.lock = threading.Lock()

    threading.Thread(
        target=background_nlp_enricher,
        args=(motor_content.df_canciones, motor_nlp),
        daemon=True,
    ).start()

    app = SpotifyDesktopApp(motor_content, motor_nlp)
    app.mainloop()