import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import io

st.set_page_config(page_title="CRAFT Educativo V2", page_icon="🏭", layout="wide")

# ==========================================================
# MOTOR MATEMÁTICO
# ==========================================================
def centroides(grid, deps, cell):
    out = {}
    for d in deps:
        p = np.argwhere(grid == d)
        if len(p) == 0:
            out[d] = (np.nan, np.nan)
        else:
            out[d] = (float((p[:,1] + .5).mean() * cell),
                      float((p[:,0] + .5).mean() * cell))
    return out

def matriz_distancias(cen, deps, metodo):
    n = len(deps)
    D = np.zeros((n,n), dtype=float)
    for i,a in enumerate(deps):
        xa,ya = cen[a]
        for j,b in enumerate(deps):
            xb,yb = cen[b]
            if metodo == "Rectilínea":
                D[i,j] = abs(xa-xb) + abs(ya-yb)
            else:
                D[i,j] = np.hypot(xa-xb, ya-yb)
    return D

def costo_layout(grid, deps, F, C, cell, metodo):
    cen = centroides(grid, deps, cell)
    D = matriz_distancias(cen, deps, metodo)
    Z = float(np.sum(F * C * D))
    return Z, cen, D

def conectado(mask):
    pts = np.argwhere(mask)
    if len(pts) <= 1:
        return True
    start = tuple(pts[0])
    vistos = {start}
    q = deque([start])
    R,K = mask.shape
    while q:
        r,c = q.popleft()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr,cc = r+dr,c+dc
            if 0 <= rr < R and 0 <= cc < K and mask[rr,cc] and (rr,cc) not in vistos:
                vistos.add((rr,cc)); q.append((rr,cc))
    return len(vistos) == len(pts)

def todos_conectados(grid, deps):
    return all(conectado(grid == d) for d in deps)

def layout_serpentina(rows, cols, deps, cantidades):
    if sum(cantidades) > rows*cols:
        return None
    path=[]
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols-1,-1,-1)
        path.extend((r,c) for c in cs)
    g = np.full((rows,cols), "LIBRE", dtype=object)
    k=0
    for d,q in zip(deps,cantidades):
        for _ in range(int(q)):
            r,c=path[k]; g[r,c]=d; k+=1
    return g

def vecinos_bloque(grid, deps, cantidades):
    """Intercambio completo de ubicaciones para departamentos de igual área."""
    out=[]
    for i in range(len(deps)):
        for j in range(i+1,len(deps)):
            if cantidades[i] != cantidades[j]:
                continue
            a,b=deps[i],deps[j]
            g=grid.copy()
            ma=(grid==a); mb=(grid==b)
            g[ma]=b; g[mb]=a
            if todos_conectados(g,deps):
                out.append((f"Intercambio completo {a} ↔ {b}",g))
    return out

def vecinos_frontera(grid, deps):
    """
    Extensión para áreas desiguales.
    Intercambia segmentos contiguos de frontera de longitud 1..máx.
    preservando exactamente el área de cada departamento y su conectividad.
    """
    R,K=grid.shape
    out=[]
    firmas=set()
    # Pares adyacentes por frontera
    pares=set()
    for r in range(R):
        for c in range(K):
            a=grid[r,c]
            if a not in deps: continue
            for dr,dc in ((1,0),(0,1)):
                rr,cc=r+dr,c+dc
                if rr<R and cc<K:
                    b=grid[rr,cc]
                    if b in deps and b!=a:
                        pares.add(tuple(sorted((a,b))))
    # Para cada par, prueba intercambio de subconjuntos fronterizos pequeños
    for a,b in pares:
        ca=[]
        cb=[]
        for r in range(R):
            for c in range(K):
                if grid[r,c]==a:
                    if any(0<=r+dr<R and 0<=c+dc<K and grid[r+dr,c+dc]==b
                           for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))):
                        ca.append((r,c))
                elif grid[r,c]==b:
                    if any(0<=r+dr<R and 0<=c+dc<K and grid[r+dr,c+dc]==a
                           for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))):
                        cb.append((r,c))
        # Desplazamientos de bandas: intercambia k celdas de frontera de cada lado.
        # Se limita para mantener velocidad didáctica.
        maxk=min(8,len(ca),len(cb))
        for k in range(1,maxk+1):
            # ventanas ordenadas espacialmente, no combinatoria exhaustiva
            for la in (sorted(ca), sorted(ca,key=lambda x:(x[1],x[0]))):
                for lb in (sorted(cb), sorted(cb,key=lambda x:(x[1],x[0]))):
                    for ia in range(0,max(1,len(la)-k+1)):
                        A=la[ia:ia+k]
                        if len(A)<k: continue
                        for ib in range(0,max(1,len(lb)-k+1)):
                            B=lb[ib:ib+k]
                            if len(B)<k: continue
                            firma=(a,b,tuple(A),tuple(B))
                            if firma in firmas: continue
                            firmas.add(firma)
                            g=grid.copy()
                            for p in A: g[p]=b
                            for p in B: g[p]=a
                            if todos_conectados(g,deps):
                                out.append((f"Intercambio de frontera {a} ↔ {b} ({k} celda{'s' if k>1 else ''})",g))
    return out

def optimizar(grid, deps, cantidades, F, C, cell, metodo, ampliado=True, maxiter=50):
    actual=grid.copy()
    z,_,_=costo_layout(actual,deps,F,C,cell,metodo)
    hist=[{"Iteración":0,"Movimiento":"Layout inicial","Candidatos evaluados":0,
           "Factibles":0,"Costo":z,"Ahorro":0.0}]
    auditoria=[]
    for it in range(1,maxiter+1):
        candidatos=vecinos_bloque(actual,deps,cantidades)
        if ampliado:
            candidatos += vecinos_frontera(actual,deps)
        mejor=None; mejorz=z
        for nombre,g in candidatos:
            zz,_,_=costo_layout(g,deps,F,C,cell,metodo)
            auditoria.append({"Iteración":it,"Movimiento":nombre,"Costo candidato":zz,
                              "Mejora":z-zz})
            if zz < mejorz - 1e-9:
                mejorz=zz; mejor=(nombre,g)
        if mejor is None:
            hist.append({"Iteración":it,"Movimiento":"Sin mejora adicional",
                         "Candidatos evaluados":len(candidatos),"Factibles":len(candidatos),
                         "Costo":z,"Ahorro":0.0})
            break
        ahorro=z-mejorz
        actual=mejor[1]
        hist.append({"Iteración":it,"Movimiento":mejor[0],
                     "Candidatos evaluados":len(candidatos),"Factibles":len(candidatos),
                     "Costo":mejorz,"Ahorro":ahorro})
        z=mejorz
    return actual,pd.DataFrame(hist),pd.DataFrame(auditoria)

def grafico(g,titulo):
    labs=[x for x in np.unique(g) if x!="LIBRE"]
    A=np.full(g.shape,np.nan)
    for i,d in enumerate(labs): A[g==d]=i
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.imshow(A,interpolation="nearest",aspect="equal")
    R,K=g.shape
    ax.set_xticks(np.arange(-.5,K,1),minor=True)
    ax.set_yticks(np.arange(-.5,R,1),minor=True)
    ax.grid(which="minor",linewidth=.35)
    ax.tick_params(which="both",bottom=False,left=False,labelbottom=False,labelleft=False)
    for d in labs:
        p=np.argwhere(g==d)
        if len(p):
            r,c=p.mean(axis=0)
            ax.text(c,r,d,ha="center",va="center",fontweight="bold",
                    bbox=dict(boxstyle="round,pad=.2",fc="white",alpha=.8))
    ax.set_title(titulo)
    return fig

# ==========================================================
# INTERFAZ
# ==========================================================
st.title("🏭 CRAFT Educativo — V2")
st.caption("Herramienta didáctica para analizar y mejorar distribuciones de planta.")

with st.sidebar:
    st.header("Configuración")
    proyecto=st.text_input("Proyecto","Ejemplo CRAFT")
    L=st.number_input("Largo (m)",min_value=1.0,value=20.0,step=1.0)
    W=st.number_input("Ancho (m)",min_value=1.0,value=10.0,step=1.0)
    cell=st.number_input("Lado de celda (m)",min_value=.25,value=1.0,step=.25)
    n=int(st.number_input("Departamentos",min_value=2,max_value=20,value=4))
    metodo=st.selectbox("Distancia",["Rectilínea","Euclidiana"],index=1)
    ampliado=st.checkbox("Permitir movimientos para áreas desiguales",True)
    maxiter=int(st.number_input("Máx. iteraciones",min_value=1,max_value=100,value=30))

rows=int(round(W/cell)); cols=int(round(L/cell))
if abs(rows*cell-W)>1e-8 or abs(cols*cell-L)>1e-8:
    st.error("El largo y el ancho deben ser múltiplos exactos del lado de la celda.")
    st.stop()
deps=[f"D{i+1}" for i in range(n)]

st.header("1. Departamentos")
areas0=[60.,40.,80.,20.] if n==4 and L==20 and W==10 and cell==1 else [L*W/n]*n
df0=pd.DataFrame({"Departamento":deps,
                  "Nombre":[f"Departamento {i+1}" for i in range(n)],
                  "Área (m²)":areas0})
df=st.data_editor(df0,disabled=["Departamento"],hide_index=True,use_container_width=True,key="deps")
areas=pd.to_numeric(df["Área (m²)"],errors="coerce").fillna(0).to_numpy(dtype=float).copy()
qf=areas/(cell*cell)
if np.any(np.abs(qf-np.round(qf))>1e-8):
    st.error("Alguna área no puede representarse exactamente con la celda seleccionada.")
    st.stop()
q=np.round(qf).astype(int)
if areas.sum()>L*W+1e-9:
    st.error("La suma de áreas requeridas excede el área de la planta.")
    st.stop()

a,b,c=st.columns(3)
a.metric("Área planta",f"{L*W:.2f} m²")
b.metric("Área requerida",f"{areas.sum():.2f} m²")
c.metric("Área libre",f"{L*W-areas.sum():.2f} m²")

st.header("2. Matriz From-To")
F0=np.zeros((n,n),dtype=float)
if n==4:
    F0=np.array([[0,2,7,4],[3,0,5,5],[6,7,0,33],[8,2,3,0]],dtype=float)
Fdf=st.data_editor(pd.DataFrame(F0,index=deps,columns=deps),use_container_width=True,key="F")
F=Fdf.to_numpy(dtype=float).copy()
np.fill_diagonal(F,0)

st.header("3. Matriz de costos unitarios")
C0=np.ones((n,n),dtype=float); np.fill_diagonal(C0,0)
Cdf=st.data_editor(pd.DataFrame(C0,index=deps,columns=deps),use_container_width=True,key="C")
C=Cdf.to_numpy(dtype=float).copy()
np.fill_diagonal(C,0)

firma=(rows,cols,tuple(q),tuple(deps))
if st.session_state.get("firma")!=firma:
    st.session_state.grid=layout_serpentina(rows,cols,deps,q)
    st.session_state.firma=firma
    for k in ("final","hist","audit"):
        st.session_state.pop(k,None)

st.header("4. Layout inicial")
if st.button("Regenerar layout inicial"):
    st.session_state.grid=layout_serpentina(rows,cols,deps,q)
    for k in ("final","hist","audit"): st.session_state.pop(k,None)

g=st.session_state.grid
if g is None:
    st.error("No fue posible generar el layout inicial."); st.stop()

z0,cen,D=costo_layout(g,deps,F,C,cell,metodo)
izq,der=st.columns([1.2,1])
with izq:
    st.pyplot(grafico(g,"Distribución inicial"))
with der:
    st.metric("Costo inicial",f"{z0:,.2f}")
    st.dataframe(pd.DataFrame([{"Depto.":d,"X":cen[d][0],"Y":cen[d][1]} for d in deps]),
                 hide_index=True,use_container_width=True)
    with st.expander("Matriz de distancias"):
        st.dataframe(pd.DataFrame(D,index=deps,columns=deps),use_container_width=True)

st.header("5. Optimización CRAFT")
st.latex(r"Z=\sum_i\sum_{j\ne i} f_{ij}\,c_{ij}\,d_{ij}")
st.caption("Los intercambios completos corresponden a departamentos de igual área. "
           "Para áreas desiguales, V2 usa una extensión por segmentos de frontera que "
           "preserva exactamente área y conectividad; se reporta separadamente para fines didácticos.")

if st.button("▶ Optimizar",type="primary"):
    final,hist,audit=optimizar(g,deps,q,F,C,cell,metodo,ampliado,maxiter)
    st.session_state.final=final
    st.session_state.hist=hist
    st.session_state.audit=audit

if "final" in st.session_state:
    final=st.session_state.final
    hist=st.session_state.hist
    audit=st.session_state.audit
    zf=float(hist.iloc[-1]["Costo"])
    ahorro=z0-zf
    pct=100*ahorro/z0 if z0 else 0
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Costo inicial",f"{z0:,.2f}")
    m2.metric("Mejor costo encontrado",f"{zf:,.2f}")
    m3.metric("Ahorro",f"{ahorro:,.2f}")
    m4.metric("Reducción",f"{pct:.2f}%")
    p1,p2=st.columns(2)
    with p1: st.pyplot(grafico(g,"Antes"))
    with p2: st.pyplot(grafico(final,"Después"))
    st.subheader("Iteraciones aceptadas")
    st.dataframe(hist,hide_index=True,use_container_width=True)
    with st.expander("Auditoría de alternativas evaluadas"):
        if audit.empty:
            st.info("No se generaron alternativas factibles.")
        else:
            st.dataframe(audit,hide_index=True,use_container_width=True)

    out=io.BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        df.to_excel(w,sheet_name="Departamentos",index=False)
        Fdf.to_excel(w,sheet_name="From-To",index=True)
        Cdf.to_excel(w,sheet_name="Costos",index=True)
        pd.DataFrame(g).to_excel(w,sheet_name="Layout inicial",index=False,header=False)
        pd.DataFrame(final).to_excel(w,sheet_name="Layout final",index=False,header=False)
        hist.to_excel(w,sheet_name="Iteraciones",index=False)
        audit.to_excel(w,sheet_name="Auditoria",index=False)
    st.download_button("⬇ Exportar resultados a Excel",data=out.getvalue(),
                       file_name="resultado_CRAFT_V2.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.warning("CRAFT es una heurística de mejora. El resultado es la mejor solución encontrada "
               "por el vecindario evaluado; no se afirma optimalidad global.")
