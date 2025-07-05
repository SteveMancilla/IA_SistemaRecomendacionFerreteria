from firebase.config import init_firebase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import joblib

# Inicializar Firebase
db = init_firebase()

# Obtener productos desde Firestore
docs = db.collection("productos").stream()
productos = []
for doc in docs:
    data = doc.to_dict()
    productos.append({
        "nombre": data["nombre"],
        "descripcion": data["descripcion"]
    })

df = pd.DataFrame(productos)

# Vectorizar descripciones
tfidf = TfidfVectorizer(stop_words=None)
X = tfidf.fit_transform(df['descripcion'])

# Modelo de vecinos más cercanos
modelo = NearestNeighbors(n_neighbors=5, metric='cosine')
modelo.fit(X)

# Guardar modelo y vectorizador
joblib.dump(tfidf, "modelo/vectorizador.joblib")
joblib.dump(modelo, "modelo/recomendador.joblib")
df.to_csv("modelo/productos.csv", index=False)

print("✅ Modelo entrenado y guardado")
