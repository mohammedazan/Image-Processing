# render_views.py
# Générer des vues 2D 

import os
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')  # Pour tourner sans interface graphique (ex: Colab)
import matplotlib.pyplot as plt

# Fonction pour lire un fichier point cloud (.ply, .pcd, .xyz, .txt)

def read_point_cloud(fichier):
    ext = os.path.splitext(fichier)[1].lower()

    if ext in ['.ply', '.pcd', '.xyz']:
        return o3d.io.read_point_cloud(fichier)

    elif ext == '.txt':
        try:
            pts = np.loadtxt(fichier)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.shape[1] > 3:
                pts = pts[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            return pcd
        except:
            return None
    else:
        return None

def render_views(input_dir, output_dir, nb_views=12):
    os.makedirs(output_dir, exist_ok=True)

    for classe in os.listdir(input_dir):
        class_path = os.path.join(input_dir, classe)

        if not os.path.isdir(class_path):
            continue

        for fichier in os.listdir(class_path):
            if not fichier.lower().endswith(('.ply', '.pcd', '.xyz', '.txt')):
                continue

            id_objet = os.path.splitext(fichier)[0]
            out_dir_obj = os.path.join(output_dir, classe, id_objet)

            # déjà traité => on saute
            if os.path.exists(out_dir_obj):  
                continue

            in_path = os.path.join(class_path, fichier)
            pcd = read_point_cloud(in_path)
            if pcd is None:
                continue

            pts = np.asarray(pcd.points)
            if pts.size == 0:
                continue

            os.makedirs(out_dir_obj, exist_ok=True)

            # Création du figure
            figure = plt.figure(figsize=(2.24, 2.24))
            ax = figure.add_subplot(111, projection='3d')
            ax.axis('off')

            # Génération des vues (nb_views angles)
            for i , angle in enumerate(np.linspace (0, 360, nb_views, endpoint=False)):
                ax.view_init(elev=30, azim=angle)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5)
                plt.savefig(os.path.join(out_dir_obj, f"view_{i+1:02d}.png"),
                            dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
                ax.cla()
            plt.close(figure)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--views", type=int, default=12)
    args = parser.parse_args()
    render_views(args.input_dir, args.output_dir, args.views)
