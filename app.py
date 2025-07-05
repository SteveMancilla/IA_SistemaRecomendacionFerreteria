from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Cargar modelo y vectorizador
modelo = joblib.load('modelo/recomendador.joblib')
vectorizador = joblib.load('modelo/vectorizador.joblib')

# Cargar productos desde CSV
df = pd.read_csv('modelo/productos.csv')
productos = df.to_dict(orient='records')  # Convierte cada fila en un dict

app = Flask(__name__)

@app.route('/recomendar', methods=['POST'])
def recomendar():
    datos = request.json
    productos_comprados = datos.get('productos', [])

    if not productos_comprados:
        return jsonify({'error': 'Debes proporcionar productos'}), 400

    # Obtener descripciones de los productos comprados
    descripciones_compradas = [
        p['descripcion'] for p in productos if p['nombre'] in productos_comprados
    ]

    if not descripciones_compradas:
        return jsonify({'recomendaciones': []})

    # Unir descripciones y vectorizar
    texto_combinado = ' '.join(descripciones_compradas)
    vector_usuario = vectorizador.transform([texto_combinado])
    vectores_productos = vectorizador.transform([p['descripcion'] for p in productos])
    similitudes = np.dot(vectores_productos, vector_usuario.T).toarray().flatten()

    # Ordenar los productos más similares
    indices_ordenados = similitudes.argsort()[::-1]
    recomendaciones = []
    for idx in indices_ordenados:
        nombre = productos[idx]['nombre']
        if nombre not in productos_comprados and nombre not in recomendaciones:
            recomendaciones.append(nombre)
        if len(recomendaciones) == 5:
            break

    return jsonify({'recomendaciones': recomendaciones})

if __name__ == '__main__':
    app.run(debug=True)