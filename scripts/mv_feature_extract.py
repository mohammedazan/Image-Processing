# mv_feature_extract.py
# Extraction de descripteurs multi-vues (agrégation Hu moments)

import os
import cv2
import numpy as np
import pandas as pd

def hu_moments(image_path):
    # Calcule les moments Hu log-signés pour image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    m = cv2.moments(img)
    hu = cv2.HuMoments(m).flatten()
    for i in range(len(hu)):
        hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-12)
    return hu

def extract_mv_features(input_root, output_csv):
    # input_root : dossier contenant les vues générées (par render_views.py)
    # output_csv : fichier de sortie (CSV)
    
    rows = []
    for classe in os.listdir(input_root):
        class_path = os.path.join(input_root, classe)
        if not os.path.isdir(class_path):
            continue

        for obj in os.listdir(class_path):
            obj_path = os.path.join(class_path, obj)
            if not os.path.isdir(obj_path):
                continue

            vues = []
            for v in os.listdir(obj_path):
                if v.endswith(".png"):
                    path = os.path.join(obj_path, v)  # make full path
                    vues.append(path)

            vues.sort()

            if len(vues) == 0:
                print(f"Pas de vues pour {obj_path}")
                continue

            hu_all = []
            for vue in vues:
                hu = hu_moments(vue)
                if hu is not None:
                    hu_all.append(hu)

            if len(hu_all) == 0:
                print(f"Hu non calculable pour {obj_path}")
                continue

            hu_all = np.array(hu_all)

            # Agrégation => moyenne + écart-type des 7 Hu-moments
            features = np.concatenate([hu_all.mean(axis=0), hu_all.std(axis=0)])

            rows.append([obj, classe] + features.tolist())

    cols = ['id', 'label']

    # ajouter hu1_mean ,..., hu7_mean
    for i in range(7):
        cols.append("hu" + str(i+1) + "_mean")

    # ajouter hu1_std, ..., hu7_std
    for i in range(7):
        cols.append("hu" + str(i+1) + "_std")

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"Multi-view features sauvegardées dans {output_csv} ({len(rows)} objets)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Dossier contenant les vues générés")
    parser.add_argument("--output_csv", required=True, help="Fichier CSV de sorti")
    args = parser.parse_args()
    extract_mv_features(args.input_root, args.output_csv)
