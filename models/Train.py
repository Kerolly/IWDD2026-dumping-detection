"""
Script de bază pentru antrenament YOLOv8
Modifică parametrii după nevoile tale
"""

from ultralytics import YOLO
import torch

def main():
    # Verifică disponibilitatea GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Folosesc device-ul: {device}")
    
    # Încarcă modelul pre-antrenat
    # Opțiuni: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO('yolov8n.pt')
    
    print("📊 Încep antrenamentul...")
    
    # Configurare și antrenament
    results = model.train(
        # Dataset
        data='data.yaml',           # MODIFICĂ: calea către fișierul data.yaml
        
        # Parametri de bază
        epochs=50,                 # număr de epoci
        imgsz=640,                  # dimensiune imagine
        batch=16,                   # mărime batch (scade dacă ai out of memory)
        
        # Salvare
        name='my_yolo_model',       # numele experimentului
        project='runs/detect',      # folder pentru rezultate
        save=True,
        save_period=10,             # salvează checkpoint la fiecare 10 epoci
        
        # Optimizare
        optimizer='Adam',           # SGD, Adam, AdamW
        lr0=0.01,                   # learning rate inițial
        weight_decay=0.0005,
        patience=50,                # early stopping
        
        # Data Augmentation
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,                 # flip orizontal
        
        # Hardware
        device=device,
        workers=8,
        
        # Validare
        val=True,
        plots=True,
        verbose=True
    )
    
    print(f"\n✅ Antrenament finalizat!")
    print(f"📁 Rezultate salvate în: {results.save_dir}")
    print(f"🎯 Best model: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()