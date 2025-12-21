"""
Visualisation de l'architecture CNN-LSTM pour présentation.
Génère des diagrammes et statistiques du modèle.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from hit_n_bounce.cnn_lstm_detector import build_cnn_lstm_model, FocalLoss
    _HAS_TF = True
except:
    _HAS_TF = False
    print("⚠️ TensorFlow non disponible")


def visualize_architecture():
    """Visualise l'architecture du modèle."""
    
    if not _HAS_TF:
        print("❌ TensorFlow requis pour visualiser l'architecture")
        return
    
    print("="*70)
    print("📐 VISUALISATION DE L'ARCHITECTURE CNN-LSTM")
    print("="*70)
    
    # Construire le modèle
    print("\n🏗️  Construction du modèle...")
    model = build_cnn_lstm_model(window_size=31, n_features=9, n_classes=3)
    
    # Summary détaillé
    print("\n📊 Architecture Complète:")
    print("-" * 70)
    model.summary()
    
    # Statistiques
    print("\n📈 Statistiques:")
    print("-" * 70)
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"  Paramètres totaux: {total_params:,}")
    print(f"  Paramètres entraînables: {trainable_params:,}")
    print(f"  Paramètres non-entraînables: {non_trainable_params:,}")
    
    # Taille du modèle
    model.save('/tmp/temp_model.keras')
    model_size_mb = Path('/tmp/temp_model.keras').stat().st_size / (1024 * 1024)
    print(f"  Taille du modèle: {model_size_mb:.2f} MB")
    
    # Analyse par couche
    print("\n🔍 Analyse par Type de Couche:")
    print("-" * 70)
    
    layer_types = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type not in layer_types:
            layer_types[layer_type] = {'count': 0, 'params': 0}
        layer_types[layer_type]['count'] += 1
        layer_types[layer_type]['params'] += layer.count_params()
    
    for layer_type, stats in sorted(layer_types.items(), key=lambda x: x[1]['params'], reverse=True):
        print(f"  {layer_type:25s}: {stats['count']:2d} couches, {stats['params']:8,} params")
    
    # Visualisation graphique
    print("\n🎨 Génération des visualisations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution des paramètres par couche
    ax = axes[0, 0]
    layer_names = []
    layer_params = []
    
    for layer in model.layers:
        if layer.count_params() > 0:
            layer_names.append(layer.name[:15])
            layer_params.append(layer.count_params())
    
    y_pos = np.arange(len(layer_names))
    ax.barh(y_pos, layer_params, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_xlabel('Nombre de Paramètres')
    ax.set_title('Distribution des Paramètres par Couche')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Répartition par type
    ax = axes[0, 1]
    types = list(layer_types.keys())
    params = [layer_types[t]['params'] for t in types]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    wedges, texts, autotexts = ax.pie(
        params, 
        labels=types, 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax.set_title('Répartition des Paramètres par Type')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 3. Flux de données (dimensions)
    ax = axes[1, 0]
    
    # Simuler un forward pass pour obtenir les shapes
    input_shape = (1, 31, 9)
    x = tf.random.normal(input_shape)
    
    layer_outputs = []
    layer_names_flow = []
    
    temp_model = keras.Model(
        inputs=model.input,
        outputs=[layer.output for layer in model.layers if 'input' not in layer.name]
    )
    
    outputs = temp_model(x)
    
    for i, (layer, output) in enumerate(zip(model.layers[1:], outputs)):
        if output.shape.ndims > 1:
            layer_names_flow.append(f"{i+1}. {layer.name[:12]}")
            # Calculer le nombre d'éléments (batch size exclu)
            n_elements = np.prod(output.shape[1:])
            layer_outputs.append(n_elements)
    
    ax.plot(range(len(layer_outputs)), layer_outputs, 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax.set_xticks(range(len(layer_names_flow)))
    ax.set_xticklabels(layer_names_flow, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Nombre d\'Éléments (par sample)')
    ax.set_title('Flux de Dimensions à Travers le Réseau')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Architecture en blocs
    ax = axes[1, 1]
    ax.axis('off')
    
    blocks = [
        {'name': 'INPUT', 'size': (31, 9), 'color': 'lightblue'},
        {'name': 'Conv1D Block\n(3 layers)', 'size': (15, 256), 'color': 'lightcoral'},
        {'name': 'Bi-LSTM Block\n(2 layers)', 'size': (15, 128), 'color': 'lightgreen'},
        {'name': 'Dense Head\n(3 layers)', 'size': (64,), 'color': 'lightyellow'},
        {'name': 'OUTPUT', 'size': (3,), 'color': 'lightgray'}
    ]
    
    y_pos = 0.9
    box_height = 0.15
    
    for block in blocks:
        # Rectangle
        rect = plt.Rectangle((0.1, y_pos - box_height), 0.8, box_height, 
                            facecolor=block['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Texte
        ax.text(0.5, y_pos - box_height/2, block['name'],
               ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Dimensions
        size_str = str(block['size'])
        ax.text(0.92, y_pos - box_height/2, size_str,
               ha='left', va='center', fontsize=9, style='italic')
        
        # Flèche
        if y_pos > box_height + 0.05:
            ax.annotate('', xy=(0.5, y_pos - box_height - 0.02), 
                       xytext=(0.5, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        y_pos -= (box_height + 0.05)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Architecture en Blocs', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'architecture_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Sauvegardé: {output_path}")
    
    plt.show()
    
    # Diagramme Focal Loss
    print("\n📊 Génération du diagramme Focal Loss...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Focal Loss vs CE
    ax = axes[0]
    p_t = np.linspace(0.01, 0.99, 100)
    
    ce = -np.log(p_t)
    
    gammas = [0, 0.5, 1, 2, 5]
    for gamma in gammas:
        fl = -(1 - p_t)**gamma * np.log(p_t)
        label = 'CE' if gamma == 0 else f'FL (γ={gamma})'
        ax.plot(p_t, fl, label=label, linewidth=2)
    
    ax.set_xlabel('Probabilité de la vraie classe (p_t)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Focal Loss vs Cross-Entropy', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)
    
    # Effet du facteur de modulation
    ax = axes[1]
    modulation = (1 - p_t)**2
    
    ax.fill_between(p_t, 0, modulation, alpha=0.3, color='red', label='Poids augmenté')
    ax.plot(p_t, modulation, 'r-', linewidth=2, label='(1 - p_t)²')
    ax.axhline(1.0, color='blue', linestyle='--', linewidth=2, label='CE standard')
    
    ax.set_xlabel('Probabilité de la vraie classe (p_t)', fontsize=11)
    ax.set_ylabel('Facteur de Modulation', fontsize=11)
    ax.set_title('Effet du Facteur de Modulation (γ=2)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotations
    ax.annotate('Exemples difficiles\n(fort poids)', 
                xy=(0.2, 0.8), xytext=(0.3, 1.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    ax.annotate('Exemples faciles\n(faible poids)', 
                xy=(0.8, 0.05), xytext=(0.6, 0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    output_path_fl = output_dir / 'focal_loss_visualization.png'
    plt.savefig(output_path_fl, dpi=150, bbox_inches='tight')
    print(f"  ✓ Sauvegardé: {output_path_fl}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("✅ VISUALISATION TERMINÉE")
    print("="*70)
    print(f"\nFichiers générés:")
    print(f"  - {output_path}")
    print(f"  - {output_path_fl}")


def compare_architectures():
    """Compare différentes architectures."""
    
    if not _HAS_TF:
        print("❌ TensorFlow requis")
        return
    
    print("\n📊 COMPARAISON D'ARCHITECTURES")
    print("="*70)
    
    configs = [
        {'name': 'CNN-LSTM (proposée)', 'window': 31, 'features': 9},
        {'name': 'CNN-LSTM (large)', 'window': 51, 'features': 9},
        {'name': 'CNN-LSTM (petit)', 'window': 15, 'features': 9},
    ]
    
    results = []
    
    for config in configs:
        model = build_cnn_lstm_model(
            window_size=config['window'],
            n_features=config['features'],
            n_classes=3
        )
        
        params = model.count_params()
        
        # Estimer FLOPS (approximatif)
        # Conv1D: 2 * input_size * output_size * kernel_size
        # LSTM: 4 * hidden_size * (hidden_size + input_size)
        
        flops_cnn = 0
        flops_cnn += 2 * 31 * 9 * 64 * 5  # Conv1D(64, k=5)
        flops_cnn += 2 * 31 * 64 * 128 * 3  # Conv1D(128, k=3)
        flops_cnn += 2 * 15 * 128 * 256 * 3  # Conv1D(256, k=3)
        
        flops_lstm = 0
        flops_lstm += 4 * 128 * (128 + 256) * 15  # Bi-LSTM(128)
        flops_lstm += 4 * 64 * (64 + 256) * 15  # Bi-LSTM(64)
        
        total_flops = (flops_cnn + flops_lstm) / 1e6  # MFLOPs
        
        results.append({
            'name': config['name'],
            'window': config['window'],
            'params': params,
            'flops': total_flops
        })
        
        print(f"\n{config['name']}:")
        print(f"  Window: {config['window']} frames")
        print(f"  Paramètres: {params:,}")
        print(f"  FLOPs estimés: {total_flops:.2f} MFLOPs")
    
    # Graphique comparatif
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [r['name'] for r in results]
    params = [r['params'] for r in results]
    flops = [r['flops'] for r in results]
    
    # Paramètres
    ax = axes[0]
    bars = ax.bar(range(len(names)), params, color=['steelblue', 'coral', 'lightgreen'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Nombre de Paramètres')
    ax.set_title('Comparaison des Paramètres')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, param) in enumerate(zip(bars, params)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{param/1e6:.2f}M',
               ha='center', va='bottom', fontweight='bold')
    
    # FLOPs
    ax = axes[1]
    bars = ax.bar(range(len(names)), flops, color=['steelblue', 'coral', 'lightgreen'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('FLOPs (Millions)')
    ax.set_title('Comparaison de la Complexité')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, flop) in enumerate(zip(bars, flops)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{flop:.1f}M',
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path('models') / 'architecture_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Sauvegardé: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    visualize_architecture()
    compare_architectures()
