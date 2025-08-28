# extract_2d_hu.py
# Script pour extraire les descripteurs Hu Moments à partir des vues générées

import os
import cv2
import numpy as np
import pandas as pd

# Fonction pour calculer Hu log-signés
def hu_moments(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    m = cv2.moments(img)
    hu = cv2.HuMoments(m).flatten()

    # Transformation log-signée
    for i in range(len(hu)):
        hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-12)
    return hu

def extract_features(input_root, output_csv):
    rows = []
    for classe in os.listdir(input_root):
        class_path = os.path.join(input_root, classe)

        if not os.path.isdir(class_path):
            continue

        for obj in os.listdir(class_path):
            obj_path = os.path.join(class_path, obj)
            if not os.path.isdir(obj_path):
                continue

            vues = [os.path.join(obj_path, v) for v in os.listdir(obj_path) if v.endswith(".png")]
            vues.sort()  # avoir view_01, ....,view_12 . toujours dans l'ordre

            if len(vues) == 0:
                print(f"Aucun PNG trouvé pour {obj_path}")
                continue

            hu_all = []
            for vue in vues:
                hu = hu_moments(vue)
                if hu is not None:
                    hu_all.append(hu)

            if len(hu_all) == 0:
                print(f"Impossible de calculer Hu pour {obj_path}")
                continue

            hu_all = np.array(hu_all)
            features = np.concatenate([hu_all.mean(axis=0), hu_all.std(axis=0)])  
            rows.append([obj, classe] + features.tolist())

    cols = ['id', 'label'] + [f"hu{i+1}" for i in range(14)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Features sauvegardées dans {output_csv} ({len(rows)} objets)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Dossier contenant les vues générées")
    parser.add_argument("--output_csv", required=True, help="Fichier CSV de sortie")
    args = parser.parse_args()
    extract_features(args.input_root, args.output_csv)
