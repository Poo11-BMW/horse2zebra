"""
Quick-results demo — identical architecture to cycle_gan.py (all 4 levels),
100 training steps so results appear in ~5 minutes on Apple Metal GPU.
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')          # no display needed
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE    = tf.data.AUTOTUNE
OUTPUT_DIR  = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
print("[1/7] Loading horse2zebra dataset …")
dataset, _ = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)
train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses,  test_zebras  = dataset['testA'],  dataset['testB']

BUFFER_SIZE, BATCH_SIZE = 1000, 1
IMG_H = IMG_W = 256

def normalize(image):
    return (tf.cast(image, tf.float32) / 127.5) - 1.0

def random_jitter(image):
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, [IMG_H, IMG_W, 3])
    return tf.image.random_flip_left_right(image)

def preprocess_train(img, _): return random_jitter(normalize(img))
def preprocess_test(img, _):  return normalize(img)

def make_ds(ds, train):
    fn = preprocess_train if train else preprocess_test
    ds = ds.map(fn, num_parallel_calls=AUTOTUNE)
    if train: ds = ds.cache().shuffle(BUFFER_SIZE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_horses_ds = make_ds(train_horses, True)
train_zebras_ds = make_ds(train_zebras, True)
test_horses_ds  = make_ds(test_horses,  False)
test_zebras_ds  = make_ds(test_zebras,  False)

sample_horse = next(iter(test_horses_ds))

# ── Custom layers ─────────────────────────────────────────────────────────────
class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self, **kw): super().__init__(**kw); self.epsilon = 1e-5
    def build(self, s):
        C = s[-1]
        self.scale  = self.add_weight(name='scale',  shape=(C,), initializer='ones')
        self.offset = self.add_weight(name='offset', shape=(C,), initializer='zeros')
    def call(self, x):
        m, v = tf.nn.moments(x, [1,2], keepdims=True)
        return self.scale * (x-m)/tf.sqrt(v+self.epsilon) + self.offset

class SelfAttention(tf.keras.layers.Layer):
    def build(self, s):
        C = s[-1]
        self.q  = tf.keras.layers.Conv2D(C//8,1,use_bias=False)
        self.k  = tf.keras.layers.Conv2D(C//8,1,use_bias=False)
        self.v  = tf.keras.layers.Conv2D(C,   1,use_bias=False)
        self.op = tf.keras.layers.Conv2D(C,   1,use_bias=False)
        self.g  = self.add_weight(name='gamma', shape=(), initializer='zeros')
    def call(self, x):
        B,H,W,C = (tf.shape(x)[i] for i in range(4)); Ck=C//8
        q=tf.reshape(self.q(x),[B,H*W,Ck]); k=tf.reshape(self.k(x),[B,H*W,Ck])
        v=tf.reshape(self.v(x),[B,H*W,C])
        a=tf.nn.softmax(tf.matmul(q,k,transpose_b=True)*tf.cast(Ck,tf.float32)**-.5)
        return self.g*self.op(tf.reshape(tf.matmul(a,v),[B,H,W,C]))+x

_SN = tf.keras.layers.SpectralNormalization

# ── Generator ─────────────────────────────────────────────────────────────────
def res_block(x, f):
    s=x
    x=InstanceNorm()(tf.keras.layers.ReLU()(InstanceNorm()(tf.keras.layers.Conv2D(f,3,padding='same')(x))))
    x=InstanceNorm()(tf.keras.layers.Conv2D(f,3,padding='same')(x))
    return tf.keras.layers.Add()([s,x])

def build_generator():
    inp = tf.keras.layers.Input((256,256,3))
    x=InstanceNorm()(tf.keras.layers.Conv2D(64,7,padding='same')(inp)); x=tf.keras.layers.ReLU()(x); f0=x
    x=tf.keras.layers.ReLU()(InstanceNorm()(tf.keras.layers.Conv2D(128,3,strides=2,padding='same')(x))); f1=x
    x=tf.keras.layers.ReLU()(InstanceNorm()(tf.keras.layers.Conv2D(256,3,strides=2,padding='same')(x))); f2=x
    for _ in range(6): x=res_block(x,256)
    x=SelfAttention()(x)
    x=tf.keras.layers.ReLU()(InstanceNorm()(tf.keras.layers.Conv2DTranspose(128,3,strides=2,padding='same')(x)))
    x=tf.keras.layers.ReLU()(InstanceNorm()(tf.keras.layers.Conv2DTranspose(64, 3,strides=2,padding='same')(x)))
    out=tf.keras.layers.Conv2DTranspose(3,7,strides=1,padding='same',activation='tanh')(x)
    return tf.keras.Model(inp,out), tf.keras.Model(inp,[f0,f1,f2])

# ── Discriminator (multi-scale + spectral norm) ───────────────────────────────
def make_disc():
    inp=tf.keras.layers.Input((None,None,3))
    x=tf.keras.layers.LeakyReLU(.2)(_SN(tf.keras.layers.Conv2D(64, 4,strides=2,padding='same'))(inp))
    x=tf.keras.layers.LeakyReLU(.2)(InstanceNorm()(_SN(tf.keras.layers.Conv2D(128,4,strides=2,padding='same'))(x)))
    x=tf.keras.layers.LeakyReLU(.2)(InstanceNorm()(_SN(tf.keras.layers.Conv2D(256,4,strides=2,padding='same'))(x)))
    x=tf.keras.layers.LeakyReLU(.2)(InstanceNorm()(_SN(tf.keras.layers.Conv2D(128,4,strides=2,padding='same'))(x)))
    out=_SN(tf.keras.layers.Conv2D(1,4,padding='same'))(x)
    return tf.keras.Model(inp,out)

class MultiScaleDisc(tf.keras.Model):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.disc_scales=[make_disc() for _ in range(2)]
        self.pool=tf.keras.layers.AveragePooling2D(2,2,padding='same')
    def call(self,x,training=False):
        out=[]
        for i,d in enumerate(self.disc_scales):
            if i: x=self.pool(x)
            out.append(d(x,training=training))
        return out

# ── PatchMLP (NCE) ────────────────────────────────────────────────────────────
class PatchMLP(tf.keras.Model):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.fc1=tf.keras.layers.Dense(256); self.relu=tf.keras.layers.ReLU(); self.fc2=tf.keras.layers.Dense(256)
    def call(self,x,training=False):
        return tf.math.l2_normalize(self.fc2(self.relu(self.fc1(x))),axis=-1)

# ── VGG perceptual extractor ──────────────────────────────────────────────────
print("[2/7] Loading VGG-16 …")
_vgg=tf.keras.applications.VGG16(include_top=False,weights='imagenet'); _vgg.trainable=False
perc_ext=tf.keras.Model(_vgg.input,[
    _vgg.get_layer('block1_conv2').output,
    _vgg.get_layer('block2_conv2').output,
    _vgg.get_layer('block3_conv3').output])

# ── Models ────────────────────────────────────────────────────────────────────
generator_G, encoder_G = build_generator()   # Horse→Zebra
generator_F, encoder_F = build_generator()   # Zebra→Horse
disc_X = MultiScaleDisc(); disc_Y = MultiScaleDisc()
nce_G  = [PatchMLP() for _ in range(3)]; nce_F = [PatchMLP() for _ in range(3)]

# Pre-build PatchMLP Dense layers (they are built lazily) so their variables
# exist BEFORE @tf.function traces gen_step — otherwise the optimizer locks
# on generator-only variables and rejects the NCE vars on the next trace.
for i, (mg, mf) in enumerate(zip(nce_G, nce_F)):
    _d = tf.zeros([1, [64, 128, 256][i]])
    mg(_d); mf(_d)

opt_gG = tf.keras.optimizers.Adam(2e-4, beta_1=.5)
opt_gF = tf.keras.optimizers.Adam(2e-4, beta_1=.5)
opt_nG = tf.keras.optimizers.Adam(2e-4, beta_1=.5)   # separate for NCE-G vars
opt_nF = tf.keras.optimizers.Adam(2e-4, beta_1=.5)   # separate for NCE-F vars
opt_dX = tf.keras.optimizers.Adam(2e-4, beta_1=.5)
opt_dY = tf.keras.optimizers.Adam(2e-4, beta_1=.5)

# ── Losses ────────────────────────────────────────────────────────────────────
LAMBDA=10; LAMBDA_NCE=1.0; LAMBDA_FREQ=0.5; LAMBDA_PERC=0.1
mse=tf.keras.losses.MeanSquaredError()
def to_list(x): return x if isinstance(x,list) else [x]
def disc_loss(r,g):
    ps=list(zip(to_list(r),to_list(g)))
    return sum((mse(tf.ones_like(ri),ri)+mse(tf.zeros_like(gi),gi))*.5 for ri,gi in ps)/len(ps)
def gen_loss(g):
    outs=to_list(g); return sum(mse(tf.ones_like(o),o) for o in outs)/len(outs)
def cyc_loss(r,c): return LAMBDA*tf.reduce_mean(tf.abs(r-c))
def id_loss(r,s):  return LAMBDA*.5*tf.reduce_mean(tf.abs(r-s))
def freq_loss(r,g):
    rg=tf.reduce_mean(r,axis=-1); gg=tf.reduce_mean(g,axis=-1)
    return tf.reduce_mean(tf.abs(tf.abs(tf.signal.rfft2d(rg))-tf.abs(tf.signal.rfft2d(gg))))
def perc_loss(r,g):
    prep=tf.keras.applications.vgg16.preprocess_input
    rf=perc_ext(prep((r+1)*127.5),training=False); gf=perc_ext(prep((g+1)*127.5),training=False)
    return sum(tf.reduce_mean(tf.abs(a-b)) for a,b in zip(rf,gf))/3.
def nce_loss(enc,rx,fy,mlps,training=False):
    sf=enc(rx,training=training); gf=enc(fy,training=training)
    tot=tf.constant(0.0)
    for s,g,m in zip(sf,gf,mlps):
        B,H,W,C=(tf.shape(s)[i] for i in range(4)); N=tf.minimum(256,H*W)
        idx=tf.random.shuffle(tf.range(H*W))[:N]
        sp=tf.reshape(tf.gather(tf.reshape(s,[B,H*W,C]),idx,axis=1),[-1,C])
        gp=tf.reshape(tf.gather(tf.reshape(g,[B,H*W,C]),idx,axis=1),[-1,C])
        se=tf.reshape(m(sp,training=training),[B,N,-1]); ge=tf.reshape(m(gp,training=training),[B,N,-1])
        logits=tf.matmul(ge,se,transpose_b=True)/.07; labels=tf.eye(N,batch_shape=[B])
        tot=tot+tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    return tot/tf.cast(len(mlps),tf.float32)

# ── ImagePool ─────────────────────────────────────────────────────────────────
class ImagePool:
    def __init__(self): self.pool=[]
    def query(self,img):
        if len(self.pool)<50: self.pool.append(img); return img
        if np.random.rand()>.5:
            i=np.random.randint(len(self.pool)); s=self.pool[i]; self.pool[i]=img; return s
        return img
pool_x=ImagePool(); pool_y=ImagePool()

# ── Training steps ────────────────────────────────────────────────────────────
@tf.function
def gen_step(rx, ry):
    gvG = generator_G.trainable_variables
    gvF = generator_F.trainable_variables
    nvG = [v for m in nce_G for v in m.trainable_variables]
    nvF = [v for m in nce_F for v in m.trainable_variables]
    with tf.GradientTape(persistent=True) as tape:
        fy=generator_G(rx,training=True); cx=generator_F(fy,training=True)
        fx=generator_F(ry,training=True); cy=generator_G(fx,training=True)
        sy=generator_G(ry,training=True); sx=generator_F(rx,training=True)
        dfx=disc_X(fx,training=False); dfy=disc_Y(fy,training=False)
        cyc=cyc_loss(rx,cx)+cyc_loss(ry,cy)
        perc=(perc_loss(rx,cx)+perc_loss(ry,cy))*.5
        freq=(freq_loss(rx,cx)+freq_loss(ry,cy))*.5
        nG=nce_loss(encoder_G,rx,fy,nce_G,True); nF=nce_loss(encoder_F,ry,fx,nce_F,True)
        lG=gen_loss(dfy)+cyc+id_loss(ry,sy)+LAMBDA_NCE*nG+LAMBDA_FREQ*freq+LAMBDA_PERC*perc
        lF=gen_loss(dfx)+cyc+id_loss(rx,sx)+LAMBDA_NCE*nF+LAMBDA_FREQ*freq+LAMBDA_PERC*perc
    opt_gG.apply_gradients(zip(tape.gradient(lG, gvG), gvG))
    opt_nG.apply_gradients(zip(tape.gradient(lG, nvG), nvG))
    opt_gF.apply_gradients(zip(tape.gradient(lF, gvF), gvF))
    opt_nF.apply_gradients(zip(tape.gradient(lF, nvF), nvF))
    return fx,fy,lG,lF,nG,nF,freq,perc

@tf.function
def disc_step(rx,ry,bfx,bfy):
    with tf.GradientTape(persistent=True) as tape:
        dlX=disc_loss(disc_X(rx,training=True),disc_X(bfx,training=True))
        dlY=disc_loss(disc_Y(ry,training=True),disc_Y(bfy,training=True))
    opt_dX.apply_gradients(zip(tape.gradient(dlX,disc_X.trainable_variables),disc_X.trainable_variables))
    opt_dY.apply_gradients(zip(tape.gradient(dlY,disc_Y.trainable_variables),disc_Y.trainable_variables))
    return dlX,dlY

# ── Train (100 steps to get real numbers fast) ────────────────────────────────
STEPS   = 100
history = {k:[] for k in ['gG','gF','dX','dY','nce','freq','perc']}

print(f"[3/7] Training ({STEPS} steps on Metal GPU) …")
t0=time.time(); step=0
for rx,ry in tf.data.Dataset.zip((train_horses_ds,train_zebras_ds)):
    if step>=STEPS: break
    fx,fy,lG,lF,nG,nF,fr,pe=gen_step(rx,ry)
    dX,dY=disc_step(rx,ry,pool_x.query(fx),pool_y.query(fy))
    for k,v in zip(['gG','gF','dX','dY','nce','freq','perc'],
                   [lG,lF,dX,dY,(nG+nF)*.5,fr,pe]):
        history[k].append(float(v))
    step+=1
    if step%10==0:
        print(f"  step {step:>3}/{STEPS} | "
              f"G_G:{np.mean(history['gG'][-10:]):.3f}  "
              f"G_F:{np.mean(history['gF'][-10:]):.3f}  "
              f"D_X:{np.mean(history['dX'][-10:]):.3f}  "
              f"NCE:{np.mean(history['nce'][-10:]):.3f}  "
              f"Freq:{np.mean(history['freq'][-10:]):.2f}  "
              f"Perc:{np.mean(history['perc'][-10:]):.2f}")

elapsed=time.time()-t0
print(f"  Training done in {elapsed:.1f}s  ({elapsed/STEPS:.2f}s/step)")

# ── Save translation sample ───────────────────────────────────────────────────
pred=generator_G(sample_horse,training=False)
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(sample_horse[0].numpy()*.5+.5); ax[0].set_title('Input horse'); ax[0].axis('off')
ax[1].imshow(pred[0].numpy()*.5+.5);         ax[1].set_title('Generated zebra'); ax[1].axis('off')
fig.savefig(os.path.join(OUTPUT_DIR,'translation_demo.png'),bbox_inches='tight',dpi=120)
plt.close(fig); print("  Saved translation_demo.png")

# ── Loss curves ───────────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,3,figsize=(15,4))
axes[0].plot(history['gG'],label='Gen G'); axes[0].plot(history['gF'],label='Gen F')
axes[0].plot(history['dX'],label='Disc X'); axes[0].plot(history['dY'],label='Disc Y')
axes[0].set_title('Adversarial'); axes[0].legend(); axes[0].set_xlabel('step')
axes[1].plot(history['nce']); axes[1].set_title('PatchNCE loss'); axes[1].set_xlabel('step')
axes[2].plot(history['freq'],label='Frequency'); axes[2].plot(history['perc'],label='Perceptual (VGG)')
axes[2].set_title('Freq + Perc'); axes[2].legend(); axes[2].set_xlabel('step')
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR,'loss_curves.png'),bbox_inches='tight')
plt.close(fig); print("  Saved loss_curves.png")

# ── Evaluation ────────────────────────────────────────────────────────────────
print("[4/7] Collecting test translations …")
gen_zeb=[]; real_zeb=[]; real_hor=[]; cyc_hor=[]
for hb in test_horses_ds:
    fz=generator_G(hb,training=False); ch=generator_F(fz,training=False)
    gen_zeb.append(fz.numpy()); real_hor.append(hb.numpy()); cyc_hor.append(ch.numpy())
for zb in test_zebras_ds:
    real_zeb.append(zb.numpy())
gen_zeb=np.concatenate(gen_zeb); real_zeb=np.concatenate(real_zeb)
real_hor=np.concatenate(real_hor); cyc_hor=np.concatenate(cyc_hor)
print(f"  gen_zebras:{len(gen_zeb)}  real_zebras:{len(real_zeb)}")

print("[5/7] Loading Inception-v3 for FID/KID/IS …")
_inc_base=tf.keras.applications.InceptionV3(include_top=True,weights='imagenet'); _inc_base.trainable=False
inc_feats=tf.keras.Model(_inc_base.input,_inc_base.get_layer('avg_pool').output)
inc_probs=tf.keras.Model(_inc_base.input,_inc_base.output)

def inc_prep(imgs):
    imgs=tf.image.resize(imgs,[299,299])
    return tf.keras.applications.inception_v3.preprocess_input((imgs+1.)*127.5)

def extract(imgs_np,bs=32):
    F,P=[],[]
    for i in range(0,len(imgs_np),bs):
        b=inc_prep(tf.constant(imgs_np[i:i+bs],tf.float32))
        F.append(inc_feats(b,training=False).numpy())
        P.append(inc_probs(b,training=False).numpy())
    return np.concatenate(F),np.concatenate(P)

print("[6/7] Extracting Inception features …")
rF,_   = extract(real_zeb)
gF,gP  = extract(gen_zeb)

# FID
def sym_sqrt(A):
    v,V=np.linalg.eigh(A); return V@np.diag(np.sqrt(np.maximum(v,0)))@V.T
def compute_fid(rf,gf):
    eps=1e-6*np.eye(rf.shape[1])
    mr,sr=rf.mean(0),np.cov(rf,rowvar=False)+eps
    mg,sg=gf.mean(0),np.cov(gf,rowvar=False)+eps
    sqsr=sym_sqrt(sr); A=sqsr@sg@sqsr; sqA=sym_sqrt(A)
    return float((mr-mg)@(mr-mg)+np.trace(sr+sg-2*sqA))

# KID
def compute_kid(rf,gf,S=100,K=50):
    K=min(K,len(rf),len(gf)); d=rf.shape[1]; kids=[]
    for _ in range(S):
        r=rf[np.random.choice(len(rf),K,replace=False)]
        g=gf[np.random.choice(len(gf),K,replace=False)]
        rr=(r@r.T/d+1)**3; rg=(r@g.T/d+1)**3; gg=(g@g.T/d+1)**3
        np.fill_diagonal(rr,0); np.fill_diagonal(gg,0)
        kids.append(rr.sum()/(K*(K-1))-2*rg.mean()+gg.sum()/(K*(K-1)))
    return float(np.mean(kids)),float(np.std(kids))

# IS
def compute_is(probs,splits=5):
    n=len(probs); ns=max(1,n//splits); sc=[]
    for i in range(splits):
        p=probs[i*ns:(i+1)*ns]
        if not len(p): continue
        py=p.mean(0,keepdims=True)
        sc.append(float(np.exp((p*(np.log(p+1e-10)-np.log(py+1e-10))).sum(1).mean())))
    return float(np.mean(sc)),float(np.std(sc))

# LPIPS (VGG)
def compute_lpips(r_np,c_np,bs=16):
    prep=tf.keras.applications.vgg16.preprocess_input; ds=[]
    for i in range(0,len(r_np),bs):
        r=tf.constant(r_np[i:i+bs],tf.float32); c=tf.constant(c_np[i:i+bs],tf.float32)
        rf=perc_ext(prep((r+1)*127.5),training=False); cf=perc_ext(prep((c+1)*127.5),training=False)
        ds.append((sum(tf.reduce_mean(tf.abs(a-b),axis=[1,2,3]) for a,b in zip(rf,cf))/3).numpy())
    return float(np.concatenate(ds).mean())

print("[7/7] Computing metrics …")
fid          = compute_fid(rF,gF)
kid_m,kid_s  = compute_kid(rF,gF)
is_m,is_s    = compute_is(gP,splits=max(2,len(gF)//10))
lpips        = compute_lpips(real_hor,cyc_hor)

results={
    'training_steps': STEPS,
    'training_time_sec': round(elapsed,1),
    'FID':       round(fid,2),
    'KID_mean':  round(kid_m,6),
    'KID_std':   round(kid_s,6),
    'IS_mean':   round(is_m,4),
    'IS_std':    round(is_s,4),
    'LPIPS_VGG': round(lpips,4),
}
with open(os.path.join(OUTPUT_DIR,'metrics.json'),'w') as f:
    json.dump(results,f,indent=2)

W=52
print('\n'+'='*W)
print('  RESULTS  (100-step demo, Metal GPU)')
print('='*W)
print(f"  Training time      {elapsed:>14.1f} s")
print(f"  Sec / step         {elapsed/STEPS:>14.2f} s")
print('─'*W)
print(f"  {'Metric':<18} {'Value':>14}  Direction")
print('─'*W)
print(f"  {'FID [J]':<18} {fid:>14.2f}  ↓ lower is better")
print(f"  {'KID [K]':<18} {kid_m:>8.5f}±{kid_s:.5f}  ↓ lower is better")
print(f"  {'IS [L]':<18} {is_m:>9.4f}±{is_s:.4f}  ↑ higher is better")
print(f"  {'LPIPS VGG [M]':<18} {lpips:>14.4f}  ↓ lower is better")
print('─'*W)
print(f"  Saved → {OUTPUT_DIR}/metrics.json")
print(f"  Saved → {OUTPUT_DIR}/translation_demo.png")
print('='*W)
