
import warnings
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

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
    'mismo','misma','otro','otra','qué','cómo','cuándo','quién','dónde'
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
    'all','every','when','where','how','would','could','should','might'
}

STOPWORDS = STOPWORDS_ES | STOPWORDS_EN

def preprocesar_letra(texto: str) -> str:

    texto = texto.lower()

    texto = re.sub(r'\[.*?\]', ' ', texto)

    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)

    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = [
        token for token in texto.split()
        if token not in STOPWORDS and len(token) > 2
    ]

    return ' '.join(tokens)

class SistemaRecomendacion:

    def __init__(self, ruta_dataset: str):
        self.df = pd.read_csv(ruta_dataset)
        self.df['letra_procesada'] = self.df['Letra'].apply(preprocesar_letra)
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2)
        )
        self.matriz_tfidf = self.vectorizer.fit_transform(self.df['letra_procesada'])
        self.matriz_similitud = cosine_similarity(self.matriz_tfidf)

    def recomendar(self, nombre_cancion: str, k: int = 5) -> list:
        matches = self.df[self.df['Cancion'].str.lower() == nombre_cancion.lower()]

        if matches.empty:
            return []

        idx = matches.index[0]
        cancion_base = self.df.iloc[idx]

        scores = list(enumerate(self.matriz_similitud[idx]))
        scores = sorted(
            [(i, s) for i, s in scores if i != idx],
            key=lambda x: x[1],
            reverse=True
        )[:k]

        resultados = []
        for i, score in scores:
            resultados.append({
                'Cancion': self.df['Cancion'].iloc[i],
                'Artista': self.df['Artista'].iloc[i],
                'Genero':  self.df['Genero'].iloc[i],
                'Similitud': round(float(score), 4)
            })

        return resultados

    def mostrar_terminos_clave(self, nombre_cancion: str, top_n: int = 10):
        matches = self.df[self.df['Cancion'].str.lower() == nombre_cancion.lower()]
        if matches.empty:
            return

        idx = matches.index[0]
        vector = self.matriz_tfidf[idx]
        terminos = self.vectorizer.get_feature_names_out()

        scores_idx = vector.toarray()[0].argsort()[::-1][:top_n]
        for i in scores_idx:
            if vector[0, i] > 0:
                pass

if __name__ == "__main__":
    pass
