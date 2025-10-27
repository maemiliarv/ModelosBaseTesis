import os
import pydicom
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def apply_filters(image):
    """Aplica filtro de mediana [3x3] y CLAHE"""
    # Normalizar a uint8
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image_uint8 = image_normalized.astype(np.uint8)
    
    # Filtro de mediana 3x3
    median_filtered = cv2.medianBlur(image_uint8, 3)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(median_filtered)
    
    return clahe_applied

def get_patient_info(filename):
    """Extrae información del paciente del nombre del archivo"""
    parts = filename.replace('.dcm', '').split('_')
    patient_id = parts[1]
    breast = parts[2]  # LEFT o RIGHT
    view = parts[3]  # CC o MLO
    lesion_num = parts[4] if len(parts) > 4 else '1'
    
    return patient_id, breast, view, lesion_num

def extract_roi_with_padding(full_image, roi_image, padding=100):
    """
    Extrae ROI de la imagen completa basándose en la máscara del ROI
    con padding especificado
    """
    # Normalizar roi_image para crear máscara
    roi_norm = cv2.normalize(roi_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Crear máscara binaria (umbral para detectar la región de interés)
    _, mask = cv2.threshold(roi_norm, 10, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Si no hay contornos, usar toda la imagen ROI
        return full_image, (0, 0, full_image.shape[1], full_image.shape[0])
    
    # Obtener el bounding box del contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Añadir padding
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(full_image.shape[1], x + w + padding)
    y_end = min(full_image.shape[0], y + h + padding)
    
    # Extraer ROI con padding
    roi_cropped = full_image[y_start:y_end, x_start:x_end]
    
    return roi_cropped, (x_start, y_start, x_end - x_start, y_end - y_start)

def process_ddsm(base_path):
    """
    Procesa la base de datos DDSM
    """
    base_path = Path(base_path)
    roi_sizes = []
    
    for category in ['Benignas', 'Malignas']:
        category_path = base_path / 'Masas' / 'DDSM' / category
        full_mammo_path = category_path / 'FullMammogram'
        roi_path = category_path / 'ROI'
        cropped_path = category_path / 'Cropped'
        
        # Crear carpeta Cropped
        cropped_path.mkdir(parents=True, exist_ok=True)
        
        # Obtener todos los archivos ROI
        roi_files = list(roi_path.glob('*.dcm'))
        
        print(f"\nProcesando {category}...")
        print(f"Total de ROIs encontrados: {len(roi_files)}")
        
        for roi_file in roi_files:
            try:
                # Obtener información del archivo ROI
                patient_id, breast, view, lesion_num = get_patient_info(roi_file.name)
                
                # Construir nombre del archivo de mamografía completa
                full_mammo_name = f"P_{patient_id}_{breast}_{view}.dcm"
                full_mammo_file = full_mammo_path / full_mammo_name
                
                if not full_mammo_file.exists():
                    print(f"Advertencia: No se encontró {full_mammo_name}")
                    continue
                
                # Leer archivos DICOM
                full_dcm = pydicom.dcmread(full_mammo_file)
                roi_dcm = pydicom.dcmread(roi_file)
                
                full_image = full_dcm.pixel_array.astype(np.float32)
                roi_image = roi_dcm.pixel_array.astype(np.float32)
                
                # Extraer ROI con padding
                roi_cropped, bbox = extract_roi_with_padding(full_image, roi_image, padding=100)
                
                # Aplicar filtros
                roi_filtered = apply_filters(roi_cropped)
                
                # Guardar tamaño
                roi_sizes.append(roi_filtered.shape)
                
                # Guardar imagen procesada
                output_name = f"P_{patient_id}_{breast}_{view}_{lesion_num}.png"
                output_path = cropped_path / output_name
                cv2.imwrite(str(output_path), roi_filtered)
                
                print(f"Procesado: {output_name} - Tamaño: {roi_filtered.shape}")
                
            except Exception as e:
                print(f"Error procesando {roi_file.name}: {str(e)}")
    
    return roi_sizes

def create_histogram(sizes, output_path, database_name):
    """Crea histograma de tamaños de ROIs"""
    widths = [s[1] for s in sizes]
    heights = [s[0] for s in sizes]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Análisis de Tamaños ROI - {database_name}', fontsize=16)
    
    # Histograma de anchos
    axes[0, 0].hist(widths, bins=30, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Ancho (píxeles)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Anchos')
    axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Media: {np.mean(widths):.0f}')
    axes[0, 0].legend()
    
    # Histograma de alturas
    axes[0, 1].hist(heights, bins=30, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Alto (píxeles)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Alturas')
    axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Media: {np.mean(heights):.0f}')
    axes[0, 1].legend()
    
    # Scatter plot
    axes[1, 0].scatter(widths, heights, alpha=0.5)
    axes[1, 0].set_xlabel('Ancho (píxeles)')
    axes[1, 0].set_ylabel('Alto (píxeles)')
    axes[1, 0].set_title('Relación Ancho vs Alto')
    axes[1, 0].plot([0, max(widths)], [0, max(widths)], 'r--', label='Cuadrado')
    axes[1, 0].legend()
    
    # Estadísticas
    stats_text = f"""
    Estadísticas de Tamaños:
    
    Ancho:
    - Media: {np.mean(widths):.2f} px
    - Mediana: {np.median(widths):.2f} px
    - Std: {np.std(widths):.2f} px
    - Min: {np.min(widths)} px
    - Max: {np.max(widths)} px
    
    Alto:
    - Media: {np.mean(heights):.2f} px
    - Mediana: {np.median(heights):.2f} px
    - Std: {np.std(heights):.2f} px
    - Min: {np.min(heights)} px
    - Max: {np.max(heights)} px
    
    Total ROIs: {len(sizes)}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nHistograma guardado en: {output_path}")

if __name__ == "__main__":
    # Definir rutas base
    # Obtener el directorio donde está el script
    SCRIPT_DIR = Path(__file__).parent
    # El directorio BasesDeDatos es el padre del directorio Scripts
    BASE_DIR = SCRIPT_DIR.parent
    MASAS_DIR = BASE_DIR / "Masas" / "DDSM"
    
    print("=" * 60)
    print("PREPROCESAMIENTO DDSM")
    print("=" * 60)
    print(f"Directorio base: {BASE_DIR}")
    print(f"Directorio masas: {MASAS_DIR}")
    
    # Procesar DDSM
    roi_sizes = process_ddsm(BASE_DIR)
    
    # Crear carpeta de análisis
    analysis_path = BASE_DIR / 'Masas' / 'Análisis'
    analysis_path.mkdir(parents=True, exist_ok=True)
    
    # Crear histograma
    histogram_path = analysis_path / 'histogram_DDSM.png'
    create_histogram(roi_sizes, histogram_path, 'DDSM')
    
    print("\n" + "=" * 60)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Total de ROIs procesados: {len(roi_sizes)}")
    print(f"Análisis guardado en: {analysis_path}")