import torch, smplx
import numpy as np
import open3d as o3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "./data_inputs/body_models/smpl/SMPL_MALE.pkl"
model = smplx.create(MODEL_DIR, model_type="smpl", gender="male", use_pca=False).to(device)

# === Set shape & pose ===
batch_size = 1
betas = torch.zeros(batch_size, 10, device=device)              # shape
global_orient = torch.zeros(batch_size, 3, device=device)       # axis-angle
body_pose = torch.zeros(batch_size, 69, device=device)          # 23*3 axis-angle
transl = torch.zeros(batch_size, 3, device=device)

with torch.no_grad():
    out = model(betas=betas, body_pose=body_pose,
                   global_orient=global_orient, transl=transl)
    V = out.vertices[0].cpu().numpy()
    F = model.faces.astype(np.int32)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(V)
mesh.triangles = o3d.utility.Vector3iVector(F)
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])