import cv2
from ultralytics import YOLO

# Cargar el modelo preentrenado
modelo = YOLO('yolov8n.pt')

# Iniciar la captura de video de la cámara web
captura = cv2.VideoCapture(0)
if not captura.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    ret, fotograma = captura.read()
    if not ret:
        print("Error: No se pudo leer el fotograma de la cámara.")
        break

    # Realizar detección
    resultados = modelo(fotograma)
    if len(resultados) > 0 and hasattr(resultados[0], 'boxes'):
        for caja in resultados[0].boxes:
            xyxy = caja.xyxy.cpu().numpy().flatten()  # Asegura convertir a numpy y aplanar el array
            x1, y1, x2, y2 = map(int, xyxy)  # Convierte las coordenadas a enteros
            confianza = caja.conf.item()  # Convierte el tensor de confianza a un valor escalar
            id_clase = int(caja.cls.item())  # Convierte el tensor de clase a un entero
            if id_clase == 67:  # Suponiendo que '67' es el ID de 'cell phone'
                cv2.rectangle(fotograma, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(fotograma, f'Teléfono móvil {confianza:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el fotograma
    cv2.imshow('Video', fotograma)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
captura.release()
cv2.destroyAllWindows()
