"""Check normalized-input realizability of specific donor-factor combinations.

This is linear algebra on the LayerNorm domain, not a natural-prefix intervention.
No downstream forward is executed.
"""
from pathlib import Path
import json
import numpy as np

R=Path(__file__).resolve().parent
z=np.load(R.parent/"results/mlp1_replay.npz")
labels=z["labels"].tolist()
gate=np.memmap(R.parent/"tensors/gate_proj.f32",mode="r",dtype="<f4",shape=(8192,2048))
up=np.memmap(R.parent/"tensors/up_proj.f32",mode="r",dtype="<f4",shape=(8192,2048))
down=np.memmap(R.parent/"tensors/down_proj.f32",mode="r",dtype="<f4",shape=(2048,8192))
neurons=(1710,3950,4786)
A=np.array([row[j] for j in neurons for row in (gate,up)],dtype=np.float64)
A-=A.mean(axis=1,keepdims=True)
G=A@A.T
Q=np.linalg.qr(A[:2].T)[0]
axis_weights=np.array([(down[1764,j]-down[:,j].mean(dtype=np.float64))/np.sqrt(1-1/2048) for j in neurons])


def factors(h):
    f=A@h
    writes=f[::2]*np.exp(-np.logaddexp(0,-f[::2]))*f[1::2]*axis_weights
    return dict(gate_up=f.tolist(),primary_axis=float(writes[0]),pair_axis=float(writes[1:].sum()))


rows=[];constructed={};projection=[]
for panel in ("water_cycle","library","bread","programming"):
    hr=z["normalized_pre_mlp1"][labels.index([panel,"newline_R"])].astype(np.float64)
    he=z["normalized_pre_mlp1"][labels.index([panel,"newline_R_edge_H14"])].astype(np.float64)
    hr-=hr.mean();he-=he.mean()
    fr,fe=A@hr,A@he
    radius2=float(he@he)
    delta=he-hr;projected=Q@(Q.T@delta)
    projection.append(dict(panel=panel,primary_feature_span_delta_energy=float(projected@projected/(delta@delta)),
        backup_gate_change_share_in_primary_span=(A[[2,4]]@projected/(A[[2,4]]@delta)).tolist(),
        backup_up_change_share_in_primary_span=(A[[3,5]]@projected/(A[[3,5]]@delta)).tolist()))
    for primary_donor,backup_donor in (("R","E"),("E","R")):
        f=np.r_[fr[:2] if primary_donor=="R" else fe[:2],
                fr[2:] if backup_donor=="R" else fe[2:]]
        coeff=np.linalg.solve(G,f)
        hmin=A.T@coeff
        minimum_norm2=float(f@coeff)
        residual=hr-A.T@np.linalg.solve(G,A@hr)
        feasible=minimum_norm2<radius2
        row=dict(panel=panel,primary_donor=primary_donor,pair_donor=backup_donor,
                 minimum_squared_norm=minimum_norm2,layernorm_squared_norm_ceiling=2048,
                 chosen_squared_norm=radius2,feasible_at_chosen_radius=feasible,
                 impossible_any_ideal_layernorm_input=minimum_norm2>=2048)
        if feasible:
            h=hmin+residual*np.sqrt((radius2-minimum_norm2)/(residual@residual))
            error=float(np.max(abs(A@h-f)))
            assert error<1e-10 and abs(h.mean())<1e-12
            assert abs(h@h-radius2)<1e-9
            # Every mean-zero h below the ceiling is produced by some real
            # non-affine LayerNorm preimage with this fixed positive epsilon.
            pre=h*np.sqrt(1e-5/(1-radius2/2048))
            back=pre/np.sqrt(np.mean(pre*pre)+1e-5)
            assert np.max(abs(back-h))<1e-10
            row.update(max_feature_error=error,**factors(h))
            constructed[panel+"_primaryR_pairE"]=h
        rows.append(row)
    # Match only the primary's E factors, retain R orientation in their nullspace,
    # and match E's total norm. This makes the pair positive, not native-R-like.
    fixed=Q@(Q.T@he);null=hr-Q@(Q.T@hr)
    h=fixed+null*np.sqrt((radius2-fixed@fixed)/(null@null))
    assert np.max(abs(A[:2]@(h-he)))<1e-10
    constructed[panel+"_primaryE_nullR"]=h
    rows.append(dict(panel=panel,primary_donor="E",pair_donor="unconstrained_R_nullspace",
        chosen_squared_norm=radius2,primary_feature_error=float(np.max(abs(A[:2]@(h-he)))),
        **factors(h)))

result=dict(scope="Real-arithmetic LayerNorm-domain feasibility, not reachability by any token prefix or downstream causal test.",
    theorem="For centered row matrix A of full row rank, min ||h||^2 subject to A h=b is b^T(AA^T)^(-1)b. LayerNorm inputs satisfy ||h||^2<2048 for positive epsilon.",
    row_order=[f"{kind}_{j}" for j in neurons for kind in ("gate","up")],
    gram_condition_number=float(np.linalg.cond(G)),
    singular_values=np.linalg.svd(A,compute_uv=False).tolist(),
    row_cosines=(G/np.sqrt(np.outer(np.diag(G),np.diag(G)))).tolist(),
    cases=rows,delta_decompositions=projection)
(R/"results/feature_feasibility.json").write_text(json.dumps(result,indent=2))
np.savez_compressed(R/"results/feature_feasibility_inputs.npz",rows=A,**constructed)
print(json.dumps(result,indent=2))
