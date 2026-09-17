import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import io

st.set_page_config(page_title="CRAFT Educativo", page_icon="🏭", layout="wide")

def centroides(grid, deps, cell):
    out={}
    for d in deps:
        p=np.argwhere(grid==d)
        out[d]=((p[:,1]+.5).mean()*cell,(p[:,0]+.5).mean()*cell)
    return out

def distancias(c, deps, metodo):
    n=len(deps); D=np.zeros((n,n))
    for i,a in enumerate(deps):
        for j,b in enumerate(deps):
            xa,ya=c[a]; xb,yb=c[b]
            D[i,j]=abs(xa-xb)+abs(ya-yb) if metodo=="Rectilínea" else np.hypot(xa-xb,ya-yb)
    return D

def costo(grid,deps,F,C,cell,metodo):
    cen=centroides(grid,deps,cell)
    D=distancias(cen,deps,metodo)
    return float(np.sum(F*C*D)),cen,D

def conexo(mask):
    pts=np.argwhere(mask)
    if len(pts)<2: return True
    inicio=tuple(pts[0]); vistos={inicio}; q=deque([inicio])
    R,K=mask.shape
    while q:
        r,c=q.popleft()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<R and 0<=cc<K and mask[rr,cc] and (rr,cc) not in vistos:
                vistos.add((rr,cc)); q.append((rr,cc))
    return len(vistos)==len(pts)

def todos_conexos(grid,deps):
    return all(conexo(grid==d) for d in deps)

def layout_inicial(rows,cols,deps,q):
    if sum(q)>rows*cols: return None
    path=[]
    for r in range(rows):
        cs=range(cols) if r%2==0 else range(cols-1,-1,-1)
        path += [(r,c) for c in cs]
    g=np.full((rows,cols),"LIBRE",dtype=object); k=0
    for d,need in zip(deps,q):
        for _ in range(need):
            r,c=path[k]; g[r,c]=d; k+=1
    return g

def vecinos(grid,deps,q,ampliado):
    V=[]
    # Intercambio completo para áreas iguales
    for i in range(len(deps)):
        for j in range(i+1,len(deps)):
            if q[i]==q[j]:
                a,b=deps[i],deps[j]; g=grid.copy()
                ma,mb=grid==a,grid==b
                g[ma]=b; g[mb]=a
                if todos_conexos(g,deps): V.append((f"{a} ↔ {b}",g))
    # Intercambio local de frontera para áreas diferentes/iguales
    if ampliado:
        R,K=grid.shape
        for r in range(R):
            for c in range(K):
                a=grid[r,c]
                if a not in deps: continue
                for dr,dc in ((1,0),(0,1)):
                    rr,cc=r+dr,c+dc
                    if rr>=R or cc>=K: continue
                    b=grid[rr,cc]
                    if b in deps and b!=a:
                        g=grid.copy(); g[r,c],g[rr,cc]=b,a
                        if todos_conexos(g,deps):
                            V.append((f"Frontera {a} ↔ {b}",g))
    return V

def optimizar(grid,deps,q,F,C,cell,metodo,ampliado,maxiter):
    actual=grid.copy(); z,_,_=costo(actual,deps,F,C,cell,metodo)
    hist=[{"Iteración":0,"Movimiento":"Layout inicial","Costo":z,"Ahorro":0.0}]
    for k in range(1,maxiter+1):
        mejor=None; mejorz=z
        for nombre,g in vecinos(actual,deps,q,ampliado):
            zz,_,_=costo(g,deps,F,C,cell,metodo)
            if zz<mejorz-1e-9: mejorz=zz; mejor=(nombre,g)
        if mejor is None: break
        hist.append({"Iteración":k,"Movimiento":mejor[0],"Costo":mejorz,"Ahorro":z-mejorz})
        actual=mejor[1]; z=mejorz
    return actual,pd.DataFrame(hist)

def grafico(g,titulo):
    labs=[x for x in np.unique(g) if x!="LIBRE"]
    A=np.full(g.shape,np.nan)
    for i,d in enumerate(labs): A[g==d]=i
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.imshow(A,interpolation="nearest",aspect="equal")
    R,K=g.shape
    ax.set_xticks(np.arange(-.5,K,1),minor=True); ax.set_yticks(np.arange(-.5,R,1),minor=True)
    ax.grid(which="minor",linewidth=.35)
    ax.tick_params(which="both",bottom=False,left=False,labelbottom=False,labelleft=False)
    for d in labs:
        p=np.argwhere(g==d); r,c=p.mean(axis=0)
        ax.text(c,r,d,ha="center",va="center",fontweight="bold",
                bbox=dict(boxstyle="round",fc="white",alpha=.75))
    ax.set_title(titulo)
    return fig

st.title("🏭 CRAFT Educativo")
st.caption("Aplicación didáctica para análisis y mejoramiento de distribución de planta.")

with st.sidebar:
    st.header("Configuración")
    proyecto=st.text_input("Proyecto","Ejemplo CRAFT")
    L=st.number_input("Largo (m)",1.0,value=20.0)
    W=st.number_input("Ancho (m)",1.0,value=10.0)
    cell=st.number_input("Lado de celda (m)",0.25,value=1.0,step=.25)
    n=int(st.number_input("Departamentos",2,20,4))
    metodo=st.selectbox("Distancia",["Rectilínea","Euclidiana"])
    ampliado=st.checkbox("Permitir movimientos para áreas desiguales",True)
    maxiter=int(st.number_input("Máx. iteraciones",1,500,100))

rows=int(round(W/cell)); cols=int(round(L/cell))
if abs(rows*cell-W)>1e-8 or abs(cols*cell-L)>1e-8:
    st.error("Largo y ancho deben ser múltiplos del lado de celda."); st.stop()
deps=[f"D{i+1}" for i in range(n)]

st.header("1. Departamentos")
areas0=[60,40,80,20] if n==4 and L==20 and W==10 and cell==1 else [L*W/n]*n
df=pd.DataFrame({"Departamento":deps,"Nombre":[f"Departamento {i+1}" for i in range(n)],"Área (m²)":areas0})
df=st.data_editor(df,disabled=["Departamento"],hide_index=True,use_container_width=True)
areas=pd.to_numeric(df["Área (m²)"],errors="coerce").fillna(0).to_numpy()
qf=areas/(cell*cell)
if np.any(abs(qf-np.round(qf))>1e-8):
    st.error("Las áreas deben poder expresarse exactamente con la celda seleccionada."); st.stop()
q=np.round(qf).astype(int)
if areas.sum()>L*W:
    st.error("Las áreas exceden el área de planta."); st.stop()
a,b,c=st.columns(3)
a.metric("Área planta",f"{L*W:.2f} m²"); b.metric("Área requerida",f"{areas.sum():.2f} m²"); c.metric("Área libre",f"{L*W-areas.sum():.2f} m²")

st.header("2. Matriz From-To")
F0=np.zeros((n,n))
if n==4: F0=np.array([[0,2,7,4],[3,0,5,5],[6,7,0,33],[8,2,3,0]],float)
Fdf=st.data_editor(pd.DataFrame(F0,index=deps,columns=deps),use_container_width=True)
F=Fdf.to_numpy(float); np.fill_diagonal(F,0)

st.header("3. Costos unitarios")
C0=np.ones((n,n)); np.fill_diagonal(C0,0)
Cdf=st.data_editor(pd.DataFrame(C0,index=deps,columns=deps),use_container_width=True)
C=Cdf.to_numpy(float); np.fill_diagonal(C,0)

signature=(rows,cols,tuple(q),tuple(deps))
if st.session_state.get("signature")!=signature:
    st.session_state.grid=layout_inicial(rows,cols,deps,q); st.session_state.signature=signature
    st.session_state.pop("final",None)

st.header("4. Layout inicial")
if st.button("Regenerar layout inicial"):
    st.session_state.grid=layout_inicial(rows,cols,deps,q); st.session_state.pop("final",None)
g=st.session_state.grid
z0,cen,D=costo(g,deps,F,C,cell,metodo)
x,y=st.columns([1.2,1])
with x: st.pyplot(grafico(g,"Distribución inicial"))
with y:
    st.metric("Costo inicial",f"{z0:,.2f}")
    st.dataframe(pd.DataFrame([{"Depto.":d,"X":cen[d][0],"Y":cen[d][1]} for d in deps]),hide_index=True)
    with st.expander("Matriz de distancias"): st.dataframe(pd.DataFrame(D,index=deps,columns=deps))

st.header("5. Optimización CRAFT")
st.latex(r"Z=\sum_i\sum_{j\ne i} f_{ij}c_{ij}d_{ij}")
if st.button("▶ Optimizar con CRAFT",type="primary"):
    st.session_state.final,st.session_state.hist=optimizar(g,deps,q,F,C,cell,metodo,ampliado,maxiter)

if "final" in st.session_state:
    final=st.session_state.final; hist=st.session_state.hist
    zf=float(hist.iloc[-1]["Costo"]); ahorro=z0-zf; pct=100*ahorro/z0 if z0 else 0
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Costo inicial",f"{z0:,.2f}"); m2.metric("Mejor costo",f"{zf:,.2f}")
    m3.metric("Ahorro",f"{ahorro:,.2f}"); m4.metric("Reducción",f"{pct:.2f}%")
    p1,p2=st.columns(2)
    with p1: st.pyplot(grafico(g,"Antes"))
    with p2: st.pyplot(grafico(final,"Después"))
    st.subheader("Procedimiento")
    st.dataframe(hist,hide_index=True,use_container_width=True)

    out=io.BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        df.to_excel(w,"Departamentos",index=False)
        Fdf.to_excel(w,"From-To"); Cdf.to_excel(w,"Costos")
        pd.DataFrame(g).to_excel(w,"Layout inicial",index=False,header=False)
        pd.DataFrame(final).to_excel(w,"Layout final",index=False,header=False)
        hist.to_excel(w,"Iteraciones",index=False)
    st.download_button("Exportar resultados a Excel",out.getvalue(),"resultado_CRAFT.xlsx")
    st.warning("CRAFT es heurístico: se reporta la mejor solución encontrada, no un óptimo global.")
