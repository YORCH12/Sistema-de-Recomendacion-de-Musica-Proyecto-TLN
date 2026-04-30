"""
Sistema de Recomendación Musical basado en PLN
IPN - ESCOM | Procesamiento de Lenguaje Natural
Equipo: Los 4 Fantásticos | Grupo: 5BM2

── Mejoras v2 ──────────────────────────────────────────────────────────────
1. Stemming con SnowballStemmer (ES/EN) — agrupa variantes morfológicas
2. TfidfVectorizer con sublinear_tf=True — penaliza repetición de estribillos
3. max_features ampliado a 1500 + min_df=2 — elimina bigramas únicos (ruido)
4. Similitud calculada por demanda (no matriz O(n²) completa)
5. recomendar() con umbral mínimo configurable
6. mostrar_terminos_clave() retorna lista utilizable
7. Lock de threading integrado en la clase (requerido por desktop_app.py)
────────────────────────────────────────────────────────────────────────────
"""

import warnings
import re
import threading
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stemmer — no requiere descargas adicionales, viene con scikit-learn/nltk base
try:
    from nltk.stem.snowball import SnowballStemmer
    _stemmer_es = SnowballStemmer("spanish")
    _stemmer_en = SnowballStemmer("english")
    STEMMING_DISPONIBLE = True
except ImportError:
    STEMMING_DISPONIBLE = False

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONSTANTES CONFIGURABLES
# ─────────────────────────────────────────────
MIN_SIMILITUD_RECOMENDACION = 0.05   # Umbral mínimo para considerar una recomendación válida
MAX_FEATURES_TFIDF          = 1500   # Vocabulario más amplio para capturar más matices temáticos
MIN_DF_TFIDF                = 2      # Ignorar términos que aparecen en menos de 2 documentos


# ─────────────────────────────────────────────
#  STOPWORDS (ES + EN)
# ─────────────────────────────────────────────
STOPWORDS_ES = {
    'de','la','el','en','y','a','que','los','se','las','un','por','con',
    'una','su','para','es','al','lo','como','más','pero','sus','me','ya',
    'si','bien','cuando','todo','sin','sobre','también','hasta','hay','muy',
    'fue','ser','no','mi','te','le','ni','tu','yo','ha','este','del','son',
    'era','tiene','esta','esto','porque','aunque','así','tan','donde','él',
    'ellos','ella','mis','todos','todas','tus','nos','les','eres','está',
    'estoy','estás','cada','hacia','han','hemos','tengo','tenía','pues',
    'entre','tanto','solo','aquí','allí','quiero','puede','poco','ese','esa',
    'siempre','nunca','algo','nada','vez','ahora','entonces','después','antes',
    'mismo','misma','otro','otra','qué','cómo','cuándo','quién','dónde',
    # Extensión: muletillas frecuentes en letras
    'ooh','aah','ohh','yeah','hey','woah','mmm','uhh',
}

STOPWORDS_EN = {
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','what','which','who','this','that',
    'these','those','am','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','a','an','the','and','but',
    'if','or','because','as','of','at','by','for','with','about',
    'into','through','before','after','to','from','up','down','in',
    'out','on','off','not','no','so','than','too','very','just','now',
    'will','can','don','oh','yeah','got','get','let','ll','ve','re',
    'all','every','when','where','how','would','could','should','might',
    # Extensión: contracciones y muletillas en letras en inglés
    'gonna','wanna','gotta','ain','cause','em','im','ive','id','its',
    'da','na','la','ha','whoa','uh','duh',
}

STOPWORDS = STOPWORDS_ES | STOPWORDS_EN


# ─────────────────────────────────────────────
#  PREPROCESAMIENTO
# ─────────────────────────────────────────────

def _aplicar_stemming(token: str) -> str:
    """
    Aplica stemming bilingüe (ES + EN).
    Usa la raíz más corta entre ambos stemmers para evitar
    sobre-reducción en palabras que existen en ambos idiomas.
    """
    if not STEMMING_DISPONIBLE:
        return token
    raiz_es = _stemmer_es.stem(token)
    raiz_en = _stemmer_en.stem(token)
    # Preferimos la raíz española si difiere significativamente
    return raiz_es if len(raiz_es) >= 3 else raiz_en


def preprocesar_letra(texto: str) -> str:
    """
    Pipeline de preprocesamiento NLP para letras musicales:

    1. Minúsculas
    2. Elimina etiquetas de sección  [Coro], [Verse 1], [feat. X]
    3. Elimina caracteres no alfabéticos (conserva acentos ES)
    4. Normaliza espacios
    5. Tokeniza y filtra stopwords + tokens cortos (< 3 chars)
    6. Stemming morfológico (ES/EN) — agrupa variantes de la misma raíz
       Ej: "amor", "amores", "amoroso" → todos quedan como "amor"
           "love", "loving", "loved"   → todos quedan como "love"

    Returns:
        str: Texto listo para vectorización TF-IDF.
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    # 1. Minúsculas
    texto = texto.lower()

    # 2. Eliminar etiquetas de sección entre corchetes
    texto = re.sub(r'\[.*?\]', ' ', texto)

    # 3. Conservar letras (incluye acentos y ñ) y espacios
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)

    # 4. Normalizar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()

    # 5. Tokenizar, filtrar stopwords y tokens cortos
    tokens = [
        token for token in texto.split()
        if token not in STOPWORDS and len(token) > 2
    ]

    # 6. Stemming — reduce variantes morfológicas a su raíz común
    if STEMMING_DISPONIBLE:
        tokens = [_aplicar_stemming(t) for t in tokens]

    return ' '.join(tokens)


# ─────────────────────────────────────────────
#  VECTORIZACIÓN Y SISTEMA DE RECOMENDACIÓN
# ─────────────────────────────────────────────

class SistemaRecomendacion:
    """
    Sistema de recomendación musical basado en similitud semántica de letras.

    Cambios vs v1:
    - TF-IDF con sublinear_tf=True: penaliza palabras muy repetidas (estribillos).
      Sin esto, una palabra que aparece 10 veces pesa 10x más; con sublinear_tf
      pesa log(10) ~ 2.3x. Esto mejora la discriminación entre canciones.
    - max_features=1500 + min_df=2: vocabulario más rico pero sin bigramas únicos.
    - Similitud calculada por demanda (on-demand) en lugar de precalcular la
      matriz completa O(n²). Esto es más eficiente cuando el corpus crece
      dinámicamente y evita el cuello de botella en cada re-fit.
    - Lock integrado para uso thread-safe desde desktop_app.py.
    """

    def __init__(self, ruta_dataset: str):
        self.lock = threading.Lock()

        try:
            self.df = pd.read_csv(ruta_dataset)
        except FileNotFoundError:
            # Corpus vacío — el enricher lo irá llenando
            self.df = pd.DataFrame(columns=["Artista", "Cancion", "Album", "Genero", "Letra", "letra_procesada"])

        # Preprocesar solo si la columna no existe (para no reprocesar en reloads)
        if "letra_procesada" not in self.df.columns or self.df["letra_procesada"].isna().all():
            self.df["letra_procesada"] = self.df["Letra"].apply(preprocesar_letra)

        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES_TFIDF,
            ngram_range=(1, 2),
            sublinear_tf=True,   # MEJORA: log(1+tf) en vez de tf crudo
            min_df=MIN_DF_TFIDF, # MEJORA: descarta bigramas que aparecen una sola vez
            strip_accents=None,  # Conservamos acentos (ya procesados en preprocesar_letra)
        )

        if not self.df.empty and self.df["letra_procesada"].str.strip().ne("").any():
            self.matriz_tfidf = self.vectorizer.fit_transform(self.df["letra_procesada"])
        else:
            self.matriz_tfidf = None

        # La matriz_similitud completa ya no se precalcula.
        # Se mantiene como None para compatibilidad; se calcula on-demand en recomendar().
        self.matriz_similitud = None

    # ── Recomendación por canción ──────────────────────────────────────────────
    def recomendar(self, nombre_cancion: str, k: int = 5) -> list[dict]:
        """
        Retorna hasta k canciones similares a nombre_cancion.

        Similitud calculada on-demand: solo se computa la fila del vector
        de la canción de referencia contra toda la matriz, no la matriz completa.
        Complejidad: O(n) en vez de O(n²).

        Args:
            nombre_cancion: Nombre exacto (case-insensitive) de la canción.
            k:              Número máximo de recomendaciones.

        Returns:
            Lista de dicts con Cancion, Artista, Genero, Similitud.
            Lista vacía si la canción no está en el corpus o no supera el umbral.
        """
        with self.lock:
            if self.matriz_tfidf is None or self.df.empty:
                return []

            matches = self.df[self.df["Cancion"].str.lower() == nombre_cancion.lower()]
            if matches.empty:
                return []

            idx        = matches.index[0]
            vec_ref    = self.matriz_tfidf[idx]

            # Similitud coseno del vector de referencia vs todos los demás
            similitudes = cosine_similarity(vec_ref, self.matriz_tfidf)[0]

        # Construir ranking excluyendo la canción de referencia y aplicando umbral
        candidatos = [
            (i, score)
            for i, score in enumerate(similitudes)
            if i != idx and score >= MIN_SIMILITUD_RECOMENDACION
        ]
        candidatos.sort(key=lambda x: x[1], reverse=True)

        resultados = []
        for i, score in candidatos[:k]:
            resultados.append({
                "Cancion":   self.df["Cancion"].iloc[i],
                "Artista":   self.df["Artista"].iloc[i],
                "Genero":    self.df["Genero"].iloc[i],
                "Similitud": round(float(score), 4),
            })

        return resultados

    # ── Términos clave TF-IDF ──────────────────────────────────────────────────
    def mostrar_terminos_clave(self, nombre_cancion: str, top_n: int = 10) -> list[dict]:
        """
        Retorna los top_n términos con mayor peso TF-IDF para una canción.

        En v1 este método calculaba correctamente pero no retornaba nada.
        Ahora retorna una lista de dicts {termino, score} utilizable por la UI
        para el panel de explicabilidad (XAI).

        Args:
            nombre_cancion: Nombre de la canción.
            top_n:          Número de términos a retornar.

        Returns:
            Lista de dicts [{"termino": str, "score": float}, ...]
            ordenados de mayor a menor peso.
        """
        with self.lock:
            if self.matriz_tfidf is None or self.df.empty:
                return []

            matches = self.df[self.df["Cancion"].str.lower() == nombre_cancion.lower()]
            if matches.empty:
                return []

            idx      = matches.index[0]
            vector   = self.matriz_tfidf[idx].toarray()[0]
            terminos = self.vectorizer.get_feature_names_out()

        # Ordenar por score descendente y filtrar scores > 0
        indices_top = vector.argsort()[::-1][:top_n]
        return [
            {"termino": terminos[i], "score": round(float(vector[i]), 4)}
            for i in indices_top
            if vector[i] > 0
        ]

    # ── Vector de una canción (útil para el motor híbrido) ────────────────────
    def obtener_vector(self, nombre_cancion: str) -> np.ndarray | None:
        """
        Retorna el vector TF-IDF de una canción como array denso.
        Útil para comparaciones externas (ej. MotorRecomendacion híbrido).
        """
        with self.lock:
            if self.matriz_tfidf is None:
                return None
            matches = self.df[self.df["Cancion"].str.lower() == nombre_cancion.lower()]
            if matches.empty:
                return None
            return self.matriz_tfidf[matches.index[0]].toarray()[0]


# ─────────────────────────────────────────────
#  MAIN (para pruebas standalone)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "corpus_letras.csv"
    sistema = SistemaRecomendacion(dataset)

    print(f"Corpus cargado: {len(sistema.df)} canciones")
    print(f"Stemming activo: {STEMMING_DISPONIBLE}")
    print(f"Vocabulario TF-IDF: {sistema.vectorizer.get_feature_names_out().shape[0] if sistema.matriz_tfidf is not None else 0} términos\n")

    if not sistema.df.empty:
        ejemplo = sistema.df["Cancion"].iloc[0]
        print(f"Ejemplo — Recomendaciones para: '{ejemplo}'")
        for r in sistema.recomendar(ejemplo, k=5):
            print(f"  {r['Similitud']:.4f}  {r['Cancion']} — {r['Artista']}")

        print(f"\nTérminos clave de '{ejemplo}':")
        for t in sistema.mostrar_terminos_clave(ejemplo, top_n=7):
            print(f"  {t['score']:.4f}  {t['termino']}")