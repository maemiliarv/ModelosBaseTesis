import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def smart_resize(image, target_size=512):
    """
    Resize inteligente que preserva aspect ratio cuando es necesario
    
    Si la imagen es casi cuadrada (ratio < 1.2): resize directo
    Si no: pad primero para hacerla cuadrada, luego resize
    """
    h, w = image.shape[:2]
    
    # Calcular aspect ratio
    aspect_ratio = max(h, w) / min(h, w)
    
    # Si es casi cuadrada, resize directo
    if aspect_ratio < 1.2:
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return resized, 'direct'
    
    # Si no, pad primero para hacerla cuadrada
    max_dim = max(h, w)
    
    # Crear imagen cuadrada con padding negro
    padded = np.zeros((max_dim, max_dim), dtype=image.dtype)
    
    # Calcular offsets para centrar la imagen
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    
    # Colocar imagen en el centro
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    # Resize la imagen paddeada
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized, 'padded'

def resize_database(base_path, database_name, target_size=512):
    """
    Redimensiona todas las imágenes de una base de datos
    """
    base_path = Path(base_path)
    db_path = base_path / 'Masas' / database_name
    
    stats = {
        'direct_resize': 0,
        'padded_resize': 0,
        'total': 0,
        'errors': []
    }
    
    print(f"\n{'='*60}")
    print(f"Procesando {database_name}")
    print(f"{'='*60}")
    
    for category in ['Benignas', 'Malignas']:
        cropped_path = db_path / category / 'Cropped'
        resized_path = db_path / category / f'Resized_{target_size}'
        
        # Crear carpeta de salida
        resized_path.mkdir(parents=True, exist_ok=True)
        
        # Obtener todas las imágenes
        image_files = list(cropped_path.glob('*.png'))
        
        if len(image_files) == 0:
            print(f"⚠️ No se encontraron imágenes en {cropped_path}")
            continue
        
        print(f"\n{category}: {len(image_files)} imágenes")
        
        # Procesar cada imagen con barra de progreso
        for img_file in tqdm(image_files, desc=f"  Resize {category}"):
            try:
                # Leer imagen
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    stats['errors'].append(f"No se pudo leer {img_file.name}")
                    continue
                
                # Resize inteligente
                resized_img, method = smart_resize(img, target_size)
                
                # Actualizar estadísticas
                stats['total'] += 1
                if method == 'direct':
                    stats['direct_resize'] += 1
                else:
                    stats['padded_resize'] += 1
                
                # Guardar imagen redimensionada
                output_path = resized_path / img_file.name
                cv2.imwrite(str(output_path), resized_img)
                
            except Exception as e:
                error_msg = f"Error en {img_file.name}: {str(e)}"
                stats['errors'].append(error_msg)
                print(f"\n✗ {error_msg}")
    
    return stats

def create_resize_analysis(all_stats, output_path, target_size):
    """
    Crea visualización del análisis de resize
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Análisis de Resize a {target_size}×{target_size}', fontsize=16, fontweight='bold')
    
    # Preparar datos
    databases = list(all_stats.keys())
    direct = [all_stats[db]['direct_resize'] for db in databases]
    padded = [all_stats[db]['padded_resize'] for db in databases]
    totals = [all_stats[db]['total'] for db in databases]
    errors = [len(all_stats[db]['errors']) for db in databases]
    
    # 1. Gráfico de barras: Métodos de resize
    x = np.arange(len(databases))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, direct, width, label='Resize Directo', color='#4CAF50', alpha=0.8)
    axes[0, 0].bar(x + width/2, padded, width, label='Con Padding', color='#2196F3', alpha=0.8)
    axes[0, 0].set_xlabel('Base de Datos', fontweight='bold')
    axes[0, 0].set_ylabel('Número de Imágenes', fontweight='bold')
    axes[0, 0].set_title('Métodos de Resize Utilizados')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(databases)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Gráfico circular: Proporción de métodos (total)
    total_direct = sum(direct)
    total_padded = sum(padded)
    
    colors = ['#4CAF50', '#2196F3']
    axes[0, 1].pie([total_direct, total_padded], 
                    labels=['Resize Directo', 'Con Padding'],
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90)
    axes[0, 1].set_title('Distribución Global de Métodos')
    
    # 3. Tabla de resumen
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    
    table_data = []
    for db in databases:
        stats = all_stats[db]
        table_data.append([
            db,
            stats['total'],
            stats['direct_resize'],
            stats['padded_resize'],
            len(stats['errors'])
        ])
    
    table = axes[1, 0].table(cellText=table_data,
                              colLabels=['Base de Datos', 'Total', 'Directo', 'Padding', 'Errores'],
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo de encabezados
    for i in range(5):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 0].set_title('Resumen Detallado', fontweight='bold', pad=20)
    
    # 4. Información de compatibilidad con modelos
    axes[1, 1].axis('off')
    
    info_text = f"""
    ✓ Tamaño de salida: {target_size}×{target_size}
    
    Compatibilidad con Modelos:
    
    Vision Transformer (ViT):
    • Patch 16×16 → 1,024 tokens
    • Patch 32×32 → 256 tokens
    • ✓ Completamente compatible
    
    Compact Vision Transformer (CvT):
    • Hierarchical structure
    • Convolutional token embedding
    • ✓ Óptimo para este tamaño
    
    Otros Modelos Compatibles:
    • ResNet, EfficientNet, DenseNet
    • Swin Transformer
    • ConvNeXt
    
    Total Imágenes Procesadas: {sum(totals):,}
    """
    
    axes[1, 1].text(0.1, 0.5, info_text,
                    transform=axes[1, 1].transAxes,
                    fontsize=11,
                    verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Análisis guardado en: {output_path}")

def main():
    """
    Función principal para resize de todas las bases de datos
    """
    # Definir rutas base
    SCRIPT_DIR = Path(__file__).parent
    BASE_DIR = SCRIPT_DIR.parent
    TARGET_SIZE = 512
    
    print("="*60)
    print(f"RESIZE A {TARGET_SIZE}×{TARGET_SIZE} - TODAS LAS BASES DE DATOS")
    print("="*60)
    print(f"Directorio base: {BASE_DIR}")
    
    # Procesar cada base de datos
    all_stats = {}
    
    for database in ['MIAS', 'INbreast', 'DDSM']:
        stats = resize_database(BASE_DIR, database, TARGET_SIZE)
        all_stats[database] = stats
        
        # Mostrar resumen
        print(f"\n✓ {database} completado:")
        print(f"  - Total procesadas: {stats['total']}")
        print(f"  - Resize directo: {stats['direct_resize']}")
        print(f"  - Con padding: {stats['padded_resize']}")
        if stats['errors']:
            print(f"  - Errores: {len(stats['errors'])}")
            for error in stats['errors'][:3]:  # Mostrar solo los primeros 3
                print(f"    • {error}")
    
    # Crear análisis visual
    analysis_path = BASE_DIR / 'Masas' / 'Análisis'
    analysis_path.mkdir(parents=True, exist_ok=True)
    
    output_plot = analysis_path / f'resize_analysis_{TARGET_SIZE}.png'
    create_resize_analysis(all_stats, output_plot, TARGET_SIZE)
    
    # Resumen final
    total_images = sum(stats['total'] for stats in all_stats.values())
    total_errors = sum(len(stats['errors']) for stats in all_stats.values())
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"✓ Total de imágenes procesadas: {total_images:,}")
    print(f"✓ Tamaño final: {TARGET_SIZE}×{TARGET_SIZE}")
    print(f"✓ Errores: {total_errors}")
    print(f"✓ Análisis guardado en: {output_plot}")
    print("\n" + "="*60)
    print("CARPETAS CREADAS:")
    print("="*60)
    for db in ['MIAS', 'INbreast', 'DDSM']:
        for cat in ['Benignas', 'Malignas']:
            path = BASE_DIR / 'Masas' / db / cat / f'Resized_{TARGET_SIZE}'
            if path.exists():
                num_files = len(list(path.glob('*.png')))
                print(f"✓ {db}/{cat}/Resized_{TARGET_SIZE}/ → {num_files} imágenes")
    
    print("\n¡Listo para entrenar ViT y CvT!")

if __name__ == "__main__":
    main()