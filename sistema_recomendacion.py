from modelo.recomendador import recomendar_productos

compras_usuario = [
    "Lubricante de alta calidad para proteger piezas móviles de herramientas y maquinaria.",
    "Aceite para motor con aditivos antioxidantes."
]

recomendaciones = recomendar_productos(compras_usuario)
for r in recomendaciones:
    print(f"✅ {r['nombre']}: {r['descripcion']}")
