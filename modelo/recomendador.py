import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load("modelo/recomendador.joblib")
vectorizador = joblib.load("modelo/vectorizador.joblib")
df_productos = pd.read_csv("modelo/productos.csv")

def recomendar_productos(descripciones_compradas: list[str], top_n=5):
    texto = " ".join(descripciones_compradas)
    vector = vectorizador.transform([texto])
    distancias, indices = modelo.kneighbors(vector, n_neighbors=top_n)

    recomendados = df_productos.iloc[indices[0]]
    return recomendados[['nombre', 'descripcion']].to_dict(orient='records')
