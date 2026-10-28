"""Continuous backbone measurements not used by token-search acceptance.

Reference-relative errors are diagnostics, not universal physical validity scores.
Planarity accepts cis and trans alike and does not assess Ramachandran regions.
"""
import numpy as np


def angle(a,b,c):
    u,v=a-b,c-b
    norm=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
    if np.any(norm<1e-12):raise ValueError("Undefined angle")
    return np.degrees(np.arccos(np.clip(np.sum(u*v,axis=-1)/norm,-1.,1.)))


def omega(backbone):
    a,b,c,d=backbone[:-1,1],backbone[:-1,2],backbone[1:,0],backbone[1:,1]
    axis=c-b;length=np.linalg.norm(axis,axis=-1,keepdims=True)
    if np.any(length<1e-12):raise ValueError("Undefined peptide axis")
    axis=axis/length
    u=a-b;v=d-c
    u=u-np.sum(u*axis,axis=-1,keepdims=True)*axis
    v=v-np.sum(v*axis,axis=-1,keepdims=True)*axis
    if np.any(np.linalg.norm(u,axis=-1)<1e-12) or np.any(np.linalg.norm(v,axis=-1)<1e-12):raise ValueError("Undefined peptide plane")
    return np.degrees(np.arctan2(np.sum(np.cross(axis,u)*v,axis=-1),np.sum(u*v,axis=-1)))


def measurements(backbone,reference):
    x,y=np.asarray(backbone,float),np.asarray(reference,float)
    if x.shape!=y.shape or x.ndim!=3 or x.shape[1:]!=(3,3) or len(x)<2:raise ValueError("Expected matching N,3,3 arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("Nonfinite backbone")
    wx,wy=omega(x),omega(y)
    planarity=np.minimum(np.abs(wx),180-np.abs(wx))
    result=dict(peptide_planarity_deviation_degrees=float(planarity.mean()),
                reference_omega_circular_mae_degrees=float(np.abs((wx-wy+180)%360-180).mean()),
                reference_ca_c_n_angle_mae_degrees=float(np.abs(angle(x[:-1,1],x[:-1,2],x[1:,0])-angle(y[:-1,1],y[:-1,2],y[1:,0])).mean()),
                reference_c_n_ca_angle_mae_degrees=float(np.abs(angle(x[:-1,2],x[1:,0],x[1:,1])-angle(y[:-1,2],y[1:,0],y[1:,1])).mean()))
    # Rigid intra-residue lengths can be enforced by the decoder: report as controls.
    for name,i,j in [("n_ca",0,1),("ca_c",1,2)]:
        result["control_reference_"+name+"_bond_mae_angstrom"]=float(np.abs(np.linalg.norm(x[:,i]-x[:,j],axis=-1)-np.linalg.norm(y[:,i]-y[:,j],axis=-1)).mean())
    return result
