
import warnings
import re
import threading
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.stem.snowball import SnowballStemmer
    _stemmer_es = SnowballStemmer("spanish")
    _stemmer_en = SnowballStemmer("english")
    STEMMING_DISPONIBLE = True
except ImportError:
    STEMMING_DISPONIBLE = False

try:
    import pysentimiento
    _analyzer_sentiment = None
    _analyzer_emotion = None
    SENTIMIENTO_DISPONIBLE = True
except ImportError:
    SENTIMIENTO_DISPONIBLE = False

def obtener_analizadores():
    global _analyzer_sentiment, _analyzer_emotion
    if not SENTIMIENTO_DISPONIBLE:
        return None, None
    if _analyzer_sentiment is None:
        from pysentimiento import create_analyzer
        _analyzer_sentiment = create_analyzer(task="sentiment", lang="es")
    if _analyzer_emotion is None:
        from pysentimiento import create_analyzer
        _analyzer_emotion = create_analyzer(task="emotion", lang="es")
    return _analyzer_sentiment, _analyzer_emotion

def analizar_sentimiento_letra(letra: str) -> dict:
    if not SENTIMIENTO_DISPONIBLE or not isinstance(letra, str) or not letra.strip():
        return {}

    analyzer_sent, analyzer_emot = obtener_analizadores()
    if not analyzer_sent or not analyzer_emot:
        return {}

    letra_corta = letra[:1500]
    try:
        res_sent = analyzer_sent.predict(letra_corta)
        res_emot = analyzer_emot.predict(letra_corta)
        emocion_out = res_emot.output
        prob_emocion = res_emot.probas[emocion_out]

        if emocion_out == "others":
            otras_emociones = {k: v for k, v in res_emot.probas.items() if k != "others"}
            if otras_emociones:
                emocion_out = max(otras_emociones, key=otras_emociones.get)
                prob_emocion = otras_emociones[emocion_out]

        return {
            "polaridad": res_sent.output,
            "emocion": emocion_out,
            "prob_sentimiento": res_sent.probas[res_sent.output],
            "prob_emocion": prob_emocion
        }
    except Exception as e:
        print(f"[Sentimiento] Error: {e}")
        return {}

warnings.filterwarnings("ignore")

MIN_SIMILITUD_RECOMENDACION = 0.05
MAX_FEATURES_TFIDF          = 1500
MIN_DF_TFIDF                = 2

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

    'gonna','wanna','gotta','ain','cause','em','im','ive','id','its',
    'da','na','la','ha','whoa','uh','duh',
}

STOPWORDS = STOPWORDS_ES | STOPWORDS_EN

def _aplicar_stemming(token: str) -> str:

    if not STEMMING_DISPONIBLE:
        return token
    raiz_es = _stemmer_es.stem(token)
    raiz_en = _stemmer_en.stem(token)

    return raiz_es if len(raiz_es) >= 3 else raiz_en

def preprocesar_letra(texto: str) -> str:

    if not isinstance(texto, str) or not texto.strip():
        return ""

    texto = texto.lower()

    texto = re.sub(r'\[.*?\]', ' ', texto)

    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)

    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = [
        token for token in texto.split()
        if token not in STOPWORDS and len(token) > 2
    ]

    if STEMMING_DISPONIBLE:
        tokens = [_aplicar_stemming(t) for t in tokens]

    return ' '.join(tokens)

class SistemaRecomendacion:

    def __init__(self, ruta_dataset: str):
        self.lock = threading.Lock()

        try:
            self.df = pd.read_csv(ruta_dataset)
        except FileNotFoundError:

            self.df = pd.DataFrame(columns=["Artista", "Cancion", "Album", "Genero", "Letra", "letra_procesada"])

        if "letra_procesada" not in self.df.columns or self.df["letra_procesada"].isna().all():
            self.df["letra_procesada"] = self.df["Letra"].apply(preprocesar_letra)

        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES_TFIDF,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=MIN_DF_TFIDF,
            strip_accents=None,
        )

        if not self.df.empty and self.df["letra_procesada"].str.strip().ne("").any():
            self.matriz_tfidf = self.vectorizer.fit_transform(self.df["letra_procesada"])
        else:
            self.matriz_tfidf = None

        self.matriz_similitud = None

    def recomendar(self, nombre_cancion: str, k: int = 5) -> list[dict]:

        with self.lock:
            if self.matriz_tfidf is None or self.df.empty:
                return []

            matches = self.df[self.df["Cancion"].str.lower() == nombre_cancion.lower()]
            if matches.empty:
                return []

            idx        = matches.index[0]
            vec_ref    = self.matriz_tfidf[idx]

            similitudes = cosine_similarity(vec_ref, self.matriz_tfidf)[0]

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

    def mostrar_terminos_clave(self, nombre_cancion: str, top_n: int = 10) -> list[dict]:

        with self.lock:
            if self.matriz_tfidf is None or self.df.empty:
                return []

            matches = self.df[self.df["Cancion"].str.lower() == nombre_cancion.lower()]
            if matches.empty:
                return []

            idx      = matches.index[0]
            vector   = self.matriz_tfidf[idx].toarray()[0]
            terminos = self.vectorizer.get_feature_names_out()

        indices_top = vector.argsort()[::-1][:top_n]
        return [
            {"termino": terminos[i], "score": round(float(vector[i]), 4)}
            for i in indices_top
            if vector[i] > 0
        ]

    def obtener_vector(self, nombre_cancion: str) -> np.ndarray | None:

        with self.lock:
            if self.matriz_tfidf is None:
                return None
            matches = self.df[self.df["Cancion"].str.lower() == nombre_cancion.lower()]
            if matches.empty:
                return None
            return self.matriz_tfidf[matches.index[0]].toarray()[0]

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
