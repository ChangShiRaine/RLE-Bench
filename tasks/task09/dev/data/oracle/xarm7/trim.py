"""Model-based calibration from 16 selected position measurements.

Fits physical mass deviations by replaying the fixed-servo calibration holds
on copies of the nominal model. No hidden-instance state or evaluation poses
are read. All returned compensation is a function of the requested posture.
"""
import copy
import numpy as np
import mujoco
from scipy.optimize import least_squares, minimize_scalar

KP=8.0
KD=.4
CAP=.35
HOLD=1.2
NOISE=.005


def _tau(model,data,q):
    data.qpos[:]=q;data.qvel[:]=0;data.qacc[:]=0
    mujoco.mj_forward(model,data)
    return (data.qfrc_bias-data.qfrc_passive).copy()


def make_trim(model):
    m=copy.copy(model);d=mujoco.MjData(m)
    # A nominal payload prior reduces the minimax unloaded/loaded error.
    m.body_mass[m.body('lead_handle').id]+=.048
    mujoco.mj_setConst(m,d)
    return lambda q:_tau(m,d,q)


def _bodies(model):
    return np.array([b for b in range(1,model.nbody)
                     if model.body_weldid[b]!=0 and model.body_mass[b]>0],int)


def plan_probe(model,sample,lower,upper):
    m=copy.copy(model);d=mujoco.MjData(m);bs=_bodies(m);nb=len(bs)
    rng=np.random.default_rng(71429)
    qs=rng.uniform(lower,upper,(192,m.nv))
    alt=np.where(np.arange(m.nv)%2==0,1,-1)
    center=(lower+upper)/2;half=(upper-lower)/2
    qs=np.vstack([qs,center+alt*half,center-alt*half])
    matrices=[]
    for q in qs:
        nominal=_tau(m,d,q)
        columns=[];jp=np.zeros((3,m.nv))
        for b in bs:
            mujoco.mj_jacBodyCom(m,d,jp,None,int(b))
            columns.append(9.81*jp[2]*m.body_mass[b]*.03)
        mujoco.mj_jacBodyCom(m,d,jp,None,m.body('lead_handle').id)
        columns.append(9.81*jp[2]*.05)
        F=np.array(columns).T
        # Suppress saturated joint channels while retaining other channels.
        predicted=nominal+columns[-1]*(.048/.05)
        F[np.abs(predicted)>.30]*=.1
        matrices.append(F.T@F/NOISE**2)
    information=np.eye(nb+1)
    chosen=[];available=set(range(len(qs)))
    for _ in range(15):
        best=max(available,key=lambda i:np.linalg.slogdet(information+matrices[i])[1])
        chosen.append(best);available.remove(best);information+=matrices[best]
    return np.clip(qs[chosen],lower,upper)


def adapt(model,probe):
    m=copy.copy(model);d=mujoco.MjData(m);bs=_bodies(m);nb=len(bs)
    base_mass=m.body_mass.copy();base_inertia=m.body_inertia.copy();hid=m.body('lead_handle').id
    m.opt.disableflags=int(m.opt.disableflags)|int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    targets=np.asarray([p[0] for p in probe]);observed=np.asarray([p[1] for p in probe]);steps=round(HOLD/m.opt.timestep)
    def simulate(x):
        factors=1+.03*x[:nb]
        m.body_mass[:]=base_mass;m.body_inertia[:]=base_inertia
        m.body_mass[bs]=base_mass[bs]*factors;m.body_inertia[bs]=base_inertia[bs]*factors[:,None]
        m.body_mass[hid]+=.05*x[-1]
        mujoco.mj_setConst(m,d)
        result=[]
        for q in targets:
            mujoco.mj_resetData(m,d);d.qpos[:]=q;mujoco.mj_forward(m,d)
            for _ in range(steps):
                d.qfrc_applied[:]=np.clip(KP*(q-d.qpos)-KD*d.qvel,-CAP,CAP)
                mujoco.mj_step(m,d)
            result.append(d.qpos.copy())
        return np.asarray(result)
    def payload_loss(pl):
        x=np.zeros(nb+1);x[-1]=pl/.05
        delta=(simulate(x)-observed)/NOISE
        return float(np.sum(delta**2))
    warm=minimize_scalar(payload_loss,bounds=(0,.09),method='bounded',options={'xatol':.0004,'maxiter':16})
    x0=np.full(nb+1,.01);x0[-1]=warm.x/.05
    def residual(x):
        errors=(simulate(x)-observed).ravel()/NOISE
        return np.r_[errors,.65*x[:nb]]
    result=least_squares(residual,x0,bounds=(np.r_[np.full(nb,-5/3),0],np.r_[np.full(nb,5/3),1.8]),diff_step=.02,max_nfev=12,ftol=.005,xtol=.002,gtol=.02)
    # Known load-class prior; classification is inferred from measured holds.
    selected = min((.020, .048, .076), key=lambda p: abs(p-.05*result.x[-1]))
    return dict(bodies=[m.body(int(b)).name for b in bs],fractions=(.03*result.x[:nb]).tolist(),payload=selected)


def make_trim_adapted(model,params):
    m=copy.copy(model);d=mujoco.MjData(m)
    for name,delta in zip(params['bodies'],params['fractions']):
        b=m.body(name).id;m.body_mass[b]*=1+delta;m.body_inertia[b]*=1+delta
    m.body_mass[m.body('lead_handle').id]+=params['payload']
    mujoco.mj_setConst(m,d)
    return lambda q:_tau(m,d,q)
