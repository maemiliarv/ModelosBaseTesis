import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

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

def parse_mias_info(info_file):
    """
    Parsea el archivo Info.txt de MIAS
    Retorna una lista con información de cada imagen
    
    Formato esperado:
    REFNUM BG_TISSUE CLASS SEVERITY X Y RADIUS
    Ejemplo: mdb144 F MISC B 233 994 29
    
    Maneja casos donde un mismo REFNUM tiene múltiples lesiones
    """
    # Leer el archivo
    with open(info_file, 'r') as f:
        lines = f.readlines()
    
    # Encontrar la línea de encabezados
    header_line = None
    header_parts = []
    for i, line in enumerate(lines):
        if 'REFNUM' in line.upper():
            header_line = i
            header_parts = line.strip().split()
            break
    
    if header_line is None:
        raise ValueError("No se encontró la línea de encabezados en Info.txt")
    
    print(f"\nEncabezados encontrados: {header_parts}")
    print(f"Formato esperado: REFNUM BG_TISSUE CLASS SEVERITY X Y RADIUS")
    
    # Parsear los datos
    data = []
    refnum_count = {}  # Para contar lesiones por REFNUM
    
    for line_num, line in enumerate(lines[header_line + 1:], start=header_line + 2):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) < 7:
            continue
        
        try:
            refnum = parts[0]
            bg_tissue = parts[1]
            class_type = parts[2]
            severity = parts[3].upper()
            
            # Las últimas 3 columnas deben ser X, Y, RADIUS
            try:
                x = int(parts[4])
                y = int(parts[5])
                radius = int(parts[6])
            except (ValueError, IndexError):
                print(f"Error línea {line_num}: No se pudieron leer X, Y, RADIUS: {line}")
                continue
            
            # Validar SEVERITY
            if severity not in ['B', 'M']:
                print(f"Advertencia línea {line_num}: SEVERITY no reconocida '{severity}': {line}")
                severity = 'U'
            
            # Contar lesiones para este REFNUM
            if refnum not in refnum_count:
                refnum_count[refnum] = 0
            refnum_count[refnum] += 1
            lesion_number = refnum_count[refnum]
            
            data.append({
                'REFNUM': refnum,
                'BG_TISSUE': bg_tissue,
                'CLASS': class_type,
                'SEVERITY': severity,
                'X': x,
                'Y': y,
                'RADIUS': radius,
                'LESION_NUM': lesion_number,
                'LINE': line_num
            })
            
            # Debug para casos con múltiples lesiones
            if refnum_count[refnum] > 1:
                print(f"INFO: {refnum} tiene múltiples lesiones - Lesión #{lesion_number}: SEVERITY={severity}, Centro=({x},{y}), Radio={radius}")
                
        except Exception as e:
            print(f"Error parseando línea {line_num}: {line[:60]}... Error: {e}")
            continue
    
    # Resumen de casos con múltiples lesiones
    multiple_lesions = {k: v for k, v in refnum_count.items() if v > 1}
    if multiple_lesions:
        print(f"\n⚠ Casos con múltiples lesiones encontrados:")
        for refnum, count in multiple_lesions.items():
            print(f"  - {refnum}: {count} lesiones")
    
    return data

def extract_roi_from_circle(image, center_x, center_y, radius, padding=100):
    """
    Extrae ROI basándose en centro y radio con padding
    """
    # Calcular bounding box
    x_min = center_x - radius
    x_max = center_x + radius
    y_min = center_y - radius
    y_max = center_y + radius
    
    # Añadir padding
    x_start = max(0, x_min - padding)
    y_start = max(0, y_min - padding)
    x_end = min(image.shape[1], x_max + padding)
    y_end = min(image.shape[0], y_max + padding)
    
    # Extraer ROI
    roi_cropped = image[y_start:y_end, x_start:x_end]
    
    return roi_cropped, (x_start, y_start, x_end - x_start, y_end - y_start)

def process_mias(base_path):
    """
    Procesa la base de datos MIAS
    """
    base_path = Path(base_path)
    roi_sizes = []
    
    mias_path = base_path / 'Masas' / 'MIAS'
    
    # Buscar archivo Info.txt en la carpeta MIAS principal
    info_file = None
    possible_names = ['Info.txt', 'info.txt', 'INFO.txt', 'INFO.TXT']
    
    for name in possible_names:
        candidate = mias_path / name
        if candidate.exists():
            info_file = candidate
            break
    
    if info_file is None:
        print(f"Error: No se encontró archivo Info.txt en {mias_path}")
        print(f"Archivos en el directorio:")
        if mias_path.exists():
            for item in mias_path.iterdir():
                print(f"  - {item.name}")
        return roi_sizes
    
    print(f"\nUsando archivo: {info_file}")
    
    # Parsear Info.txt
    try:
        all_roi_info = parse_mias_info(info_file)
        print(f"Total de ROIs encontrados en Info.txt: {len(all_roi_info)}")
    except Exception as e:
        print(f"Error parseando Info.txt: {e}")
        return roi_sizes
    
    # Separar por severidad (B=benigna, M=maligna)
    roi_by_category = {
        'Benignas': [],
        'Malignas': []
    }
    
    unknown_severity = []
    
    for roi in all_roi_info:
        severity = roi.get('SEVERITY', 'U').upper()
        if severity == 'B':
            roi_by_category['Benignas'].append(roi)
        elif severity == 'M':
            roi_by_category['Malignas'].append(roi)
        else:
            unknown_severity.append(roi)
    
    print(f"\nClasificación por SEVERITY:")
    print(f"  - Benignas (B): {len(roi_by_category['Benignas'])}")
    print(f"  - Malignas (M): {len(roi_by_category['Malignas'])}")
    print(f"  - Sin clasificar (U): {len(unknown_severity)}")
    
    if unknown_severity:
        print(f"\nAdvertencia: {len(unknown_severity)} ROIs sin clasificar correctamente:")
        for roi in unknown_severity[:5]:  # Mostrar solo los primeros 5
            print(f"  - {roi['REFNUM']}: SEVERITY={roi.get('SEVERITY', 'N/A')}")
        if len(unknown_severity) > 5:
            print(f"  ... y {len(unknown_severity) - 5} más")
    
    # Procesar cada categoría
    for category in ['Benignas', 'Malignas']:
        category_path = mias_path / category
        cropped_path = category_path / 'Cropped'
        
        # Crear carpeta Cropped
        cropped_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Procesando {category}...")
        print(f"{'='*60}")
        
        # Obtener ROIs de esta categoría según SEVERITY del Info.txt
        category_rois = roi_by_category[category]
        
        print(f"ROIs clasificados como {category} en Info.txt: {len(category_rois)}")
        
        # Si hay ROIs clasificados, usarlos directamente
        if len(category_rois) > 0:
            # Verificar que los archivos existan en la carpeta correspondiente
            rois_to_process = []
            for roi in category_rois:
                refnum = roi['REFNUM']
                pgm_file = category_path / f"{refnum}.pgm"
                
                # Buscar sin distinguir mayúsculas/minúsculas
                if not pgm_file.exists():
                    found = False
                    for file in category_path.glob('*.pgm'):
                        if file.stem.lower() == refnum.lower():
                            pgm_file = file
                            found = True
                            break
                    
                    if not found:
                        print(f"⚠ Advertencia: {refnum} clasificado como {category} pero no encontrado en la carpeta")
                        continue
                
                rois_to_process.append(roi)
            
            category_rois = rois_to_process
        else:
            # Modo de respaldo: buscar archivos en la carpeta
            print(f"No hay ROIs clasificados como {category} en Info.txt")
            print(f"Buscando todos los archivos .pgm en {category_path}...")
            
            pgm_files = list(category_path.glob('*.pgm'))
            print(f"Archivos .pgm encontrados: {len(pgm_files)}")
            
            # Buscar coincidencias en all_roi_info
            for pgm_file in pgm_files:
                refnum = pgm_file.stem
                matching_roi = None
                for roi in all_roi_info:
                    if roi['REFNUM'].lower() == refnum.lower():
                        matching_roi = roi
                        break
                
                if matching_roi:
                    category_rois.append(matching_roi)
        
        print(f"ROIs a procesar en {category}: {len(category_rois)}")
        
        total_processed = 0
        
        # Procesar cada ROI
        for info in category_rois:
            refnum = info['REFNUM']
            x = info['X']
            y = info['Y']
            radius = info['RADIUS']
            
            # Buscar archivo .pgm correspondiente
            pgm_file = category_path / f"{refnum}.pgm"
            
            # Si no existe, buscar sin distinguir mayúsculas/minúsculas
            if not pgm_file.exists():
                found = False
                for file in category_path.glob('*.pgm'):
                    if file.stem.lower() == refnum.lower():
                        pgm_file = file
                        found = True
                        break
                
                if not found:
                    print(f"Advertencia: No se encontró {refnum}.pgm en {category}")
                    continue
            
            try:
                # Leer imagen PGM
                image = cv2.imread(str(pgm_file), cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Error: No se pudo leer {pgm_file.name}")
                    continue
                
                # Convertir a float32
                image = image.astype(np.float32)
                
                # Extraer ROI con padding
                roi_cropped, bbox = extract_roi_from_circle(image, x, y, radius, padding=100)
                
                # Aplicar filtros
                roi_filtered = apply_filters(roi_cropped)
                
                # Guardar tamaño
                roi_sizes.append(roi_filtered.shape)
                
                # Guardar imagen procesada
                output_name = f"{refnum}_ROI.png"
                output_path = cropped_path / output_name
                cv2.imwrite(str(output_path), roi_filtered)
                
                total_processed += 1
                print(f"✓ Procesado: {output_name} - Tamaño: {roi_filtered.shape} - Centro: ({x},{y}) Radio: {radius}")
                
            except Exception as e:
                print(f"✗ Error procesando {refnum}: {str(e)}")
        
        print(f"\nTotal ROIs procesados en {category}: {total_processed}")
    
    return roi_sizes

def create_histogram(sizes, output_path, database_name):
    """Crea histograma de tamaños de ROIs"""
    if len(sizes) == 0:
        print(f"\nAdvertencia: No hay ROIs para crear histograma de {database_name}")
        return
    
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
    MASAS_DIR = BASE_DIR / "Masas" / "MIAS"
    
    print("=" * 60)
    print("PREPROCESAMIENTO MIAS")
    print("=" * 60)
    print(f"Directorio base: {BASE_DIR}")
    print(f"Directorio masas: {MASAS_DIR}")
    
    # Procesar MIAS
    roi_sizes = process_mias(BASE_DIR)
    
    if len(roi_sizes) == 0:
        print("\n" + "=" * 60)
        print("ADVERTENCIA: No se procesaron ROIs")
        print("=" * 60)
        print("Por favor verifica:")
        print("1. Que exista el archivo Info.txt en las carpetas Benignas y Malignas")
        print("2. Que los archivos .pgm estén en las carpetas correspondientes")
        print("3. Que el formato del archivo Info.txt sea correcto")
       
    
    # Crear carpeta de análisis
    analysis_path = BASE_DIR / 'Masas' / 'Análisis'
    analysis_path.mkdir(parents=True, exist_ok=True)
    
    # Crear histograma
    histogram_path = analysis_path / 'histogram_MIAS.png'
    create_histogram(roi_sizes, histogram_path, 'MIAS')
    
    print("\n" + "=" * 60)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Total de ROIs procesados: {len(roi_sizes)}")
    print(f"Análisis guardado en: {analysis_path}")