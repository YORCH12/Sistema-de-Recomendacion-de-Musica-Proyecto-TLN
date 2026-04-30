import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import threading
import requests
import queue
from io import BytesIO

# Importamos nuestro motor NLP de Letras (TLN_1.py)
from TLN_1 import SistemaRecomendacion, preprocesar_letra

# UI Desktop Library
import customtkinter as ctk
from PIL import Image

client_id = 'cdd38f1958794ce8b9b6e0e6b58df1ed'
client_secret = 'fa7876458a4143608a30a1fa4b17ff76'
redirect_uri = 'http://127.0.0.1:3000'
scope = 'user-modify-playback-state user-read-playback-state user-read-currently-playing playlist-read-private playlist-read-collaborative user-top-read user-library-read'

cc_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify_data = spotipy.Spotify(client_credentials_manager=cc_manager, retries=0, requests_timeout=5)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope), retries=0, requests_timeout=5)

def obtener_letra(artista, cancion):
    """Busca la letra dinámicamente en lyrics.ovh"""
    url = f"https://api.lyrics.ovh/v1/{artista}/{cancion}"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json().get('lyrics', '')
    except: pass
    return ''

class MotorRecomendacion:
    def __init__(self):
        self.df_canciones = pd.DataFrame()
        self.features_lista = ['danceability']

    def cargar_datos(self):
        datos = []
        tracks = []
        try:
            for tr in ['short_term', 'medium_term', 'long_term']:
                top_tracks = sp.current_user_top_tracks(limit=50, time_range=tr)
                tracks.extend(top_tracks.get('items', []))
            
            for offset in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
                saved = sp.current_user_saved_tracks(limit=50, offset=offset)
                items = saved.get('items', [])
                if not items: break
                for item in items:
                    tracks.append(item['track'])
                
            for track in tracks:
                if track is None or track.get('uri') is None:
                    continue
                datos.append({
                    'id': track['id'],
                    'uri': track['uri'],
                    'nombre': track['name'],
                    'artista': track['artists'][0]['name'],
                    'album': track['album']['images'][0]['url'] if track['album']['images'] else '',
                    'popularidad': track.get('popularity', 0),
                    'genero_busqueda': 'Tus Favoritos'
                })
        except Exception as e:
            pass
            
        self.df_canciones = pd.DataFrame(datos).drop_duplicates(subset=['id']).reset_index(drop=True)
        
        if len(self.df_canciones) == 0:
            return

        pop_max = self.df_canciones['popularidad'].max()
        if pop_max == 0: pop_max = 1
        self.df_canciones['pop_norm'] = self.df_canciones['popularidad'] / pop_max
        dummies_gen = pd.get_dummies(self.df_canciones['genero_busqueda'], prefix='gen')
        self.features_lista = ['pop_norm'] + list(dummies_gen.columns)
        self.df_canciones = pd.concat([self.df_canciones, dummies_gen], axis=1)

    def obtener_recomendaciones(self, track_id, top_n=8):
        if track_id not in self.df_canciones['id'].values:
            return self.df_canciones.sample(top_n).to_dict('records')
            
        idx = self.df_canciones.index[self.df_canciones['id'] == track_id].tolist()[0]
        matrices = self.df_canciones[self.features_lista].fillna(0)
        similitudes = cosine_similarity(matrices.iloc[[idx]], matrices)[0]
        
        canciones_similares = self.df_canciones.copy()
        canciones_similares['similitud'] = similitudes
        canciones_similares = canciones_similares[canciones_similares['id'] != track_id]
        
        recomendadas = canciones_similares.sort_values(by='similitud', ascending=False).head(top_n)
        return recomendadas.to_dict('records')


class SpotifyDesktopApp(ctk.CTk):
    def __init__(self, motor_content, motor_nlp):
        super().__init__()
        self.motor_content = motor_content
        self.motor_nlp = motor_nlp
        self.currently_playing_name = None
        
        # Cola thread-safe para imágenes
        self.image_queue = queue.Queue()

        # Configuración de Ventana
        self.title("S P O T I F Y  A I  -  Premium Gold")
        self.geometry("1200x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color="#0C0C0A")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Paleta de Colores
        self.color_accent = "#A39422" # Amarillo/Oro solicitado
        self.color_sidebar = "#080806"
        self.color_main = "#141412"
        self.color_card = "#1C1C19"
        self.color_text_muted = "#8E8E89"

        # 1. Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=260, fg_color=self.color_sidebar, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.brand_label = ctk.CTkLabel(self.sidebar_frame, text="⚡ SPOTIFY AI", font=("Courier", 18, "bold"), text_color=self.color_accent)
        self.brand_label.pack(pady=(30, 40), padx=20, anchor="w")
        
        self.lbl_mymusic = ctk.CTkLabel(self.sidebar_frame, text="MY MUSIC", font=("Inter", 11, "bold"), text_color="#FFFFFF")
        self.lbl_mymusic.pack(pady=(0, 10), padx=20, anchor="w")
        self.lbl_local = ctk.CTkLabel(self.sidebar_frame, text="LOCAL", font=("Inter", 11, "bold"), text_color="#FFFFFF")
        self.lbl_local.pack(pady=(20, 10), padx=20, anchor="w")
        self.lbl_tracksnlp = ctk.CTkLabel(self.sidebar_frame, text="▶ Tracks NLP", font=("Inter", 13, "bold"), text_color=self.color_accent)
        self.lbl_tracksnlp.pack(pady=(0, 10), padx=20, anchor="w")
        
        # 1.1 Sidebar: Explicabilidad (XAI)
        self.lbl_insight = ctk.CTkLabel(self.sidebar_frame, text="🧠 NLP INSIGHTS", font=("Inter", 11, "bold"), text_color="#FFFFFF")
        self.lbl_insight.pack(pady=(40, 10), padx=20, anchor="w")
        
        self.txt_insight = ctk.CTkTextbox(self.sidebar_frame, width=220, height=250, fg_color="#181815", text_color=self.color_text_muted, font=("Consolas", 12), border_color=self.color_accent, border_width=1)
        self.txt_insight.pack(padx=20, anchor="w")
        self.txt_insight.insert("1.0", "> Esperando reproducción\n> para extraer términos\n> TF-IDF...")
        self.txt_insight.configure(state="disabled")

        # 2. Main Content Frame
        self.main_frame = ctk.CTkScrollableFrame(self, fg_color=self.color_main, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        self.lbl_breadcrumb = ctk.CTkLabel(self.main_frame, text="VIBRACIONES PERSONALIZADAS", font=("Inter", 11, "bold"), text_color=self.color_accent)
        self.lbl_breadcrumb.grid(row=0, column=0, pady=(30, 5), padx=40, sticky="w")
        
        self.lbl_subtitle = ctk.CTkLabel(self.main_frame, text="Recomendaciones Basadas en tus Gustos", font=("Inter", 24, "bold"), text_color="#FFFFFF")
        self.lbl_subtitle.grid(row=1, column=0, pady=(0, 30), padx=40, sticky="w")
        
        self.grid_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.grid_frame.grid(row=2, column=0, sticky="nsew", padx=30)
        
        # 3. Bottom Player
        self.player_frame = ctk.CTkFrame(self, height=90, fg_color=self.color_sidebar, corner_radius=0)
        self.player_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.player_frame.pack_propagate(False)
        
        self.now_playing_label = ctk.CTkLabel(self.player_frame, text="Ninguna Pista Activa", font=("Inter", 14, "bold"), text_color="#FFFFFF")
        self.now_playing_label.pack(side="left", padx=40)
        
        self.controls_frame = ctk.CTkFrame(self.player_frame, fg_color="transparent")
        self.controls_frame.pack(side="left", expand=True)
        
        self.btn_playpause = ctk.CTkButton(self.controls_frame, text="⏯", width=50, height=50, corner_radius=25, font=("Inter", 24), fg_color=self.color_accent, hover_color="#C2B12F", text_color="#000000", command=self.toggle_playback)
        self.btn_playpause.pack(side="left", padx=15)
        
        self.btn_next = ctk.CTkButton(self.controls_frame, text="⏭", width=40, height=40, font=("Inter", 20), fg_color="transparent", hover_color="#20201A", text_color=self.color_text_muted, command=self.next_track)
        self.btn_next.pack(side="left", padx=10)
        
        self.process_image_queue()
        
        if len(self.motor_content.df_canciones) > 0:
            self.render_grid(self.motor_content.df_canciones.sample(min(12, len(self.motor_content.df_canciones))).to_dict('records'))
        self.check_state()

    def process_image_queue(self):
        while not self.image_queue.empty():
            try:
                ctk_img, label_widget = self.image_queue.get_nowait()
                try:
                    if ctk_img:
                        label_widget.configure(image=ctk_img, text="")
                    else:
                        label_widget.configure(text="No Image")
                except:
                    pass # Evita crashes si se destruyó el widget
            except queue.Empty:
                break
        self.after(100, self.process_image_queue)

    def update_insight(self, texto):
        self.txt_insight.configure(state="normal")
        self.txt_insight.delete("1.0", "end")
        self.txt_insight.insert("1.0", texto)
        self.txt_insight.configure(state="disabled")

    def toggle_playback(self):
        try:
            current = sp.current_playback()
            if current and current['is_playing']:
                sp.pause_playback()
            else:
                sp.start_playback()
        except: pass

    def next_track(self):
        try:
            sp.next_track()
        except: pass

    def play_track_uri(self, uri):
        try:
            sp.start_playback(uris=[uri])
        except: pass

    def fetch_image(self, url, label_widget):
        try:
            response = requests.get(url, timeout=5)
            img_data = Image.open(BytesIO(response.content)).resize((120, 120))
            ctk_img = ctk.CTkImage(light_image=img_data, dark_image=img_data, size=(120, 120))
            self.image_queue.put((ctk_img, label_widget))
        except:
            self.image_queue.put((None, label_widget))

    def render_grid(self, tracks):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
            
        columns = 4
        for i, track in enumerate(tracks):
            row = i // columns
            col = i % columns
            
            card = ctk.CTkFrame(self.grid_frame, fg_color=self.color_card, corner_radius=12, border_width=2, border_color=self.color_card, cursor="hand2")
            card.grid(row=row, column=col, padx=15, pady=15, sticky="n")
            
            # Efecto Hover Dinámico
            def on_enter(e, c=card): c.configure(border_color=self.color_accent)
            def on_leave(e, c=card): c.configure(border_color=self.color_card)
            
            card.bind("<Enter>", on_enter)
            card.bind("<Leave>", on_leave)
            card.bind("<Button-1>", lambda e, u=track['uri']: self.play_track_uri(u))
            
            img_lbl = ctk.CTkLabel(card, text="Loading...", width=140, height=140, fg_color="#111111", corner_radius=8)
            img_lbl.pack(pady=(15, 12), padx=15)
            img_lbl.bind("<Enter>", on_enter)
            img_lbl.bind("<Leave>", on_leave)
            img_lbl.bind("<Button-1>", lambda e, u=track['uri']: self.play_track_uri(u))
            
            if track.get('album'):
                threading.Thread(target=self.fetch_image, args=(track['album'], img_lbl), daemon=True).start()
            
            nombre = track['nombre']
            if len(nombre) > 16: nombre = nombre[:14] + "..."
            title_lbl = ctk.CTkLabel(card, text=nombre, font=("Inter", 14, "bold"), text_color="#FFFFFF")
            title_lbl.pack(anchor="w", padx=15)
            title_lbl.bind("<Enter>", on_enter)
            title_lbl.bind("<Leave>", on_leave)
            title_lbl.bind("<Button-1>", lambda e, u=track['uri']: self.play_track_uri(u))
            
            tag = track.get('genero_busqueda', 'RECOMENDADO')
            if 'similitud' in track:
                tag = f"{int(track['similitud']*100)}% MATCH (NLP)"
            
            artist_lbl = ctk.CTkLabel(card, text=tag, font=("Inter", 11, "bold"), text_color=self.color_accent)
            artist_lbl.pack(anchor="w", padx=15, pady=(0, 15))
            artist_lbl.bind("<Enter>", on_enter)
            artist_lbl.bind("<Leave>", on_leave)
            artist_lbl.bind("<Button-1>", lambda e, u=track['uri']: self.play_track_uri(u))

    def fetch_spotify_uri(self, cancion, artista):
        try:
            res = spotify_data.search(q=f"track:{cancion} artist:{artista}", type='track', limit=1)
            if res['tracks']['items']:
                t = res['tracks']['items'][0]
                return {
                    'uri': t['uri'],
                    'album': t['album']['images'][0]['url'] if t['album']['images'] else '',
                    'nombre': cancion,
                    'artista': artista,
                    'genero_busqueda': 'NLP Corpus'
                }
        except: pass
        return None

    def display_nlp_explainability(self, nombre_actual):
        try:
            matches = self.motor_nlp.df[self.motor_nlp.df['Cancion'].str.lower() == nombre_actual.lower()]
            if not matches.empty:
                idx = matches.index[0]
                vector = self.motor_nlp.matriz_tfidf[idx]
                terminos = self.motor_nlp.vectorizer.get_feature_names_out()
                
                scores_idx = vector.toarray()[0].argsort()[::-1][:7]
                
                insight = f"🔍 Análisis TF-IDF:\nCanción: {nombre_actual[:15]}\n\nTérminos con mayor peso:\n"
                for i in scores_idx:
                    if vector[0, i] > 0:
                        insight += f"• {terminos[i]}: {vector[0, i]:.3f}\n"
                
                insight += "\n*Las recomendaciones comparten este vocabulario."
                self.update_insight(insight)
        except Exception as e:
            pass

    def check_state(self):
        try:
            current = sp.current_playback()
            if current and current.get('is_playing') and current.get('item'):
                track_id = current['item']['id']
                nombre_actual = current['item']['name']
                artista_actual = current['item']['artists'][0]['name']
                
                self.now_playing_label.configure(text=f"{artista_actual} - {nombre_actual}")
                
                if self.currently_playing_name != nombre_actual:
                    self.currently_playing_name = nombre_actual
                    
                    # 1. Intentamos usar el motor NLP (Letras TF-IDF)
                    nlp_recs = self.motor_nlp.recomendar(nombre_actual, k=8)
                    
                    if not nlp_recs:
                        letra = obtener_letra(artista_actual, nombre_actual)
                        if letra:
                            nueva_fila = pd.DataFrame([{
                                'Artista': artista_actual,
                                'Cancion': nombre_actual,
                                'Album': current['item']['album']['name'],
                                'Genero': 'Dinámico',
                                'Letra': letra,
                                'letra_procesada': preprocesar_letra(letra)
                            }])
                            self.motor_nlp.df = pd.concat([self.motor_nlp.df, nueva_fila], ignore_index=True)
                            self.motor_nlp.df.to_csv('corpus_letras.csv', index=False, encoding='utf-8')
                            
                            self.motor_nlp.matriz_tfidf = self.motor_nlp.vectorizer.fit_transform(self.motor_nlp.df['letra_procesada'])
                            self.motor_nlp.matriz_similitud = cosine_similarity(self.motor_nlp.matriz_tfidf)
                            
                            nlp_recs = self.motor_nlp.recomendar(nombre_actual, k=8)
                    
                    if nlp_recs:
                        self.lbl_subtitle.configure(text=f"Recomendación Dinámica: TF-IDF NLP (Letras) de '{nombre_actual}'")
                        self.display_nlp_explainability(nombre_actual)
                        ui_tracks = []
                        for r in nlp_recs:
                            spotify_info = self.fetch_spotify_uri(r['Cancion'], r['Artista'])
                            if spotify_info:
                                spotify_info['similitud'] = r['Similitud']
                                ui_tracks.append(spotify_info)
                        if ui_tracks:
                            self.render_grid(ui_tracks)
                    else:
                        self.lbl_subtitle.configure(text=f"Recomendación Acústica: No hay Letra NLP para '{nombre_actual}'")
                        self.update_insight(f"🎶 Análisis Acústico:\nCanción: {nombre_actual[:15]}\n\nLetra no disponible.\nSimilitud calculada por popularidad y One-Hot Encoding.")
                        recomendadas = self.motor_content.obtener_recomendaciones(track_id, top_n=8)
                        self.render_grid(recomendadas)
            else:
                self.now_playing_label.configure(text="Ninguna Pista Activa")
                self.currently_playing_name = None
        except Exception as e:
            print(f"[UI ERROR] {e}")
            
        self.after(4000, self.check_state)


def background_nlp_enricher(df_canciones, motor_nlp):
    import time
    
    for idx, track in df_canciones.iterrows():
        cancion = track['nombre']
        artista = track['artista']
        
        if motor_nlp.df[motor_nlp.df['Cancion'].str.lower() == cancion.lower()].empty:
            letra = obtener_letra(artista, cancion)
            if letra:
                nueva_fila = pd.DataFrame([{
                    'Artista': artista,
                    'Cancion': cancion,
                    'Album': "Spotify Sync",
                    'Genero': track.get('genero_busqueda', 'Tus Favoritos'),
                    'Letra': letra,
                    'letra_procesada': preprocesar_letra(letra)
                }])
                motor_nlp.df = pd.concat([motor_nlp.df, nueva_fila], ignore_index=True)
                motor_nlp.df.to_csv('corpus_letras.csv', index=False, encoding='utf-8')
                motor_nlp.matriz_tfidf = motor_nlp.vectorizer.fit_transform(motor_nlp.df['letra_procesada'])
                motor_nlp.matriz_similitud = cosine_similarity(motor_nlp.matriz_tfidf)
            time.sleep(1.5)


if __name__ == "__main__":
    motor_content = MotorRecomendacion()
    motor_content.cargar_datos()
    
    motor_nlp = SistemaRecomendacion('corpus_letras.csv')
    
    threading.Thread(target=background_nlp_enricher, args=(motor_content.df_canciones, motor_nlp), daemon=True).start()
    
    app = SpotifyDesktopApp(motor_content, motor_nlp)
    app.mainloop()
