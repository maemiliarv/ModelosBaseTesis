import os
import pydicom
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

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

def parse_inbreast_xml(xml_file):
    """
    Parsea archivo XML de INbreast (formato plist) para extraer coordenadas de ROIs
    Retorna lista de ROIs (cada uno con sus puntos)
    
    Formato: plist con estructura Images > ROIs > Name="Mass" > Point_px
    """
    import plistlib
    
    try:
        # Leer el archivo plist
        with open(xml_file, 'rb') as f:
            plist_data = plistlib.load(f)
        
        rois = []
        
        # Navegar por la estructura del plist
        if 'Images' not in plist_data:
            return rois
        
        for image in plist_data['Images']:
            if 'ROIs' not in image:
                continue
            
            for roi in image['ROIs']:
                # Verificar que sea una masa
                if 'Name' not in roi or roi['Name'].lower() != 'mass':
                    continue
                
                # Extraer puntos en píxeles
                if 'Point_px' not in roi:
                    continue
                
                points = []
                for point_str in roi['Point_px']:
                    # Formato: "(x, y)" o "(x.xxx, y.yyy)"
                    # Limpiar paréntesis y separar
                    point_str = point_str.strip('()')
                    coords = point_str.split(',')
                    
                    if len(coords) >= 2:
                        try:
                            x = float(coords[0].strip())
                            y = float(coords[1].strip())
                            points.append([int(x), int(y)])
                        except ValueError:
                            continue
                
                if len(points) > 0:
                    rois.append(np.array(points))
        
        return rois
        
    except Exception as e:
        print(f"Error parseando XML con plistlib: {e}")
        # Intentar método alternativo con xml.etree
        return parse_inbreast_xml_alternative(xml_file)

def parse_inbreast_xml_alternative(xml_file):
    """
    Método alternativo para parsear XML de INbreast usando xml.etree
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        rois = []
        
        # Buscar todos los elementos 'dict' que contienen ROIs
        for array_elem in root.iter('array'):
            parent = array_elem.find('..')
            if parent is not None:
                # Buscar si hay una key "ROIs" antes de este array
                keys = parent.findall('key')
                for i, key in enumerate(keys):
                    if key.text == 'ROIs':
                        # Este es el array de ROIs
                        for dict_elem in array_elem.findall('dict'):
                            # Buscar Name="Mass"
                            name_found = False
                            for j, key in enumerate(dict_elem.findall('key')):
                                if key.text == 'Name':
                                    string_elem = dict_elem.findall('string')[j]
                                    if string_elem.text and 'mass' in string_elem.text.lower():
                                        name_found = True
                                        break
                            
                            if not name_found:
                                continue
                            
                            # Buscar Point_px
                            for j, key in enumerate(dict_elem.findall('key')):
                                if key.text == 'Point_px':
                                    # Encontrar el array correspondiente
                                    arrays = dict_elem.findall('array')
                                    if len(arrays) > j:
                                        point_array = arrays[j]
                                        points = []
                                        
                                        for string_elem in point_array.findall('string'):
                                            point_str = string_elem.text
                                            if point_str:
                                                # Formato: "(x, y)"
                                                point_str = point_str.strip('()')
                                                coords = point_str.split(',')
                                                
                                                if len(coords) >= 2:
                                                    try:
                                                        x = float(coords[0].strip())
                                                        y = float(coords[1].strip())
                                                        points.append([int(x), int(y)])
                                                    except ValueError:
                                                        continue
                                        
                                        if len(points) > 0:
                                            rois.append(np.array(points))
        
        return rois
        
    except Exception as e:
        print(f"Error con método alternativo: {e}")
        return []

def extract_roi_from_contour(image, contour_points, padding=100):
    """
    Extrae ROI basándose en puntos de contorno con padding
    """
    # Obtener bounding box del contorno
    x_coords = contour_points[:, 0]
    y_coords = contour_points[:, 1]
    
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    # Añadir padding
    x_start = max(0, x_min - padding)
    y_start = max(0, y_min - padding)
    x_end = min(image.shape[1], x_max + padding)
    y_end = min(image.shape[0], y_max + padding)
    
    # Extraer ROI
    roi_cropped = image[y_start:y_end, x_start:x_end]
    
    return roi_cropped, (x_start, y_start, x_end - x_start, y_end - y_start)

def get_patient_id_from_dicom(dicom_filename):
    """
    Extrae el ID del paciente del nombre del archivo DICOM
    Formato: 20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm
    Retorna: 20586908
    """
    return dicom_filename.split('_')[0]

def process_inbreast(base_path):
    """
    Procesa la base de datos INbreast
    """
    base_path = Path(base_path)
    roi_sizes = []
    
    for category in ['Benignas', 'Malignas']:
        category_path = base_path / 'Masas' / 'INbreast' / category
        dicoms_path = category_path / 'DICOMS'
        xml_path = category_path / 'XML'
        cropped_path = category_path / 'Cropped'
        
        # Crear carpeta Cropped
        cropped_path.mkdir(parents=True, exist_ok=True)
        
        # Obtener todos los archivos DICOM
        dicom_files = list(dicoms_path.glob('*.dcm'))
        
        print(f"\nProcesando {category}...")
        print(f"Total de mamografías encontradas: {len(dicom_files)}")
        
        total_rois = 0
        
        for dicom_file in dicom_files:
            try:
                # Extraer ID del paciente
                patient_id = get_patient_id_from_dicom(dicom_file.name)
                
                # Buscar archivo XML correspondiente
                xml_file = xml_path / f"{patient_id}.xml"
                
                if not xml_file.exists():
                    print(f"Advertencia: No se encontró XML para {patient_id}")
                    continue
                
                # Leer archivo DICOM
                dcm = pydicom.dcmread(dicom_file)
                full_image = dcm.pixel_array.astype(np.float32)
                
                # Parsear XML para obtener ROIs
                rois = parse_inbreast_xml(xml_file)
                
                if len(rois) == 0:
                    print(f"Advertencia: No se encontraron masas en {xml_file.name}")
                    continue
                
                print(f"  {dicom_file.name}: {len(rois)} masa(s) encontrada(s)")
                
                # Procesar cada ROI encontrado
                for idx, roi_points in enumerate(rois, start=1):
                    # Extraer ROI con padding
                    roi_cropped, bbox = extract_roi_from_contour(full_image, roi_points, padding=100)
                    
                    # Aplicar filtros
                    roi_filtered = apply_filters(roi_cropped)
                    
                    # Guardar tamaño
                    roi_sizes.append(roi_filtered.shape)
                    
                    # Guardar imagen procesada
                    # Mantener información del archivo original
                    base_name = dicom_file.stem
                    if len(rois) > 1:
                        # Si hay múltiples masas, agregar número de lesión
                        output_name = f"{base_name}_lesion{idx}_ROI.png"
                    else:
                        output_name = f"{base_name}_ROI.png"
                    
                    output_path = cropped_path / output_name
                    cv2.imwrite(str(output_path), roi_filtered)
                    
                    total_rois += 1
                    
                    if len(rois) > 1:
                        print(f"    ✓ Lesión {idx}/{len(rois)}: {output_name} - Tamaño: {roi_filtered.shape}")
                    else:
                        print(f"    ✓ {output_name} - Tamaño: {roi_filtered.shape}")
                
            except Exception as e:
                print(f"Error procesando {dicom_file.name}: {str(e)}")
        
        print(f"Total ROIs extraídos de {category}: {total_rois}")
    
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
    MASAS_DIR = BASE_DIR / "Masas" / "INbreast"
    
    print("=" * 60)
    print("PREPROCESAMIENTO INBREAST")
    print("=" * 60)
    print(f"Directorio base: {BASE_DIR}")
    print(f"Directorio masas: {MASAS_DIR}")
    
    # Procesar INbreast
    roi_sizes = process_inbreast(BASE_DIR)
    
    # Crear carpeta de análisis
    analysis_path = BASE_DIR / 'Masas' / 'Análisis'
    analysis_path.mkdir(parents=True, exist_ok=True)
    
    # Crear histograma
    histogram_path = analysis_path / 'histogram_INbreast.png'
    create_histogram(roi_sizes, histogram_path, 'INbreast')
    
    print("\n" + "=" * 60)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Total de ROIs procesados: {len(roi_sizes)}")
    print(f"Análisis guardado en: {analysis_path}")