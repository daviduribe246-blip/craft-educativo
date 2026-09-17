import streamlit as st
import pandas as pd
import numpy as np
import io, json, math

st.set_page_config(page_title="CRAFT Educativo V4", page_icon="🏭", layout="wide")

# ---------------- MOTOR MATEMÁTICO ----------------
def distancia(a,b,metodo):
    dx=abs(a[0]-b[0]); dy=abs(a[1]-b[1])
    return dx+dy if metodo=="Rectilínea" else float(np.hypot(dx,dy))

def centroides_grid(grid,n,cell):
    cen=[]; areas=[]
    for d in range(1,n+1):
        rr,cc=np.where(grid==d)
        areas.append(len(rr)*cell*cell)
        if len(rr)==0: cen.append((np.nan,np.nan))
        else:
            cen.append(((cc.mean()+0.5)*cell,(rr.mean()+0.5)*cell))
    return cen,areas

def costo_layout(grid,n,cell,F,C,metodo):
    cen,_=centroides_grid(grid,n,cell)
    if any(np.isnan(x) for p in cen for x in p): return np.inf
    z=0.0
    for i in range(n):
        for j in range(n):
            if i!=j:
                z += F[i,j]*C[i,j]*distancia(cen[i],cen[j],metodo)
    return float(z)

def conectado(grid,d):
    pts=list(zip(*np.where(grid==d)))
    if not pts: return False
    S=set(pts); seen={pts[0]}; stack=[pts[0]]
    while stack:
        r,c=stack.pop()
        for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if q in S and q not in seen:
                seen.add(q); stack.append(q)
    return len(seen)==len(S)

def adyacentes(grid,a,b):
    R,Cc=grid.shape
    for r,c in zip(*np.where(grid==a)):
        for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=rr<R and 0<=cc<Cc and grid[rr,cc]==b: return True
    return False

def validar(grid,n,cell,areas_obj,tol_cells=0):
    _,areas=centroides_grid(grid,n,cell)
    errores=[]
    acell=cell*cell
    for i in range(n):
        obj=areas_obj[i]
        if abs(areas[i]-obj)>max(1e-8,tol_cells*acell):
            errores.append(f"D{i+1}: área dibujada {areas[i]:.2f} m²; requerida {obj:.2f} m².")
        if areas[i]>0 and not conectado(grid,i+1):
            errores.append(f"D{i+1}: está dividido en partes no conectadas.")
    return errores

# CRAFT didáctico: intercambio de etiquetas/ubicaciones.
# Igual área: cualquier par. Área desigual: solo adyacentes.
def craft(grid,n,cell,areas,F,C,metodo,maxit):
    actual=grid.copy()
    z=costo_layout(actual,n,cell,F,C,metodo)
    hist=[{"Iteración":0,"Movimiento":"Inicial","Costo":z,"Ahorro":0.0}]
    audit=[]
    for k in range(1,maxit+1):
        candidatos=[]
        for a in range(1,n+1):
            for b in range(a+1,n+1):
                igual=abs(areas[a-1]-areas[b-1])<1e-9
                ady=adyacentes(actual,a,b)
                if not (igual or ady): continue
                # Evaluación CRAFT por intercambio de centroides/identidades:
                q=actual.copy()
                ma=actual==a; mb=actual==b
                q[ma]=b; q[mb]=a
                z1=costo_layout(q,n,cell,F,C,metodo)
                ah=z-z1
                audit.append({"Iteración":k,"Par":f"D{a} ↔ D{b}",
                              "Áreas iguales":igual,"Adyacentes":ady,
                              "Costo estimado":z1,"Ahorro estimado":ah})
                candidatos.append((ah,z1,a,b,q))
        mejoras=[x for x in candidatos if x[0]>1e-9]
        if not mejoras:
            hist.append({"Iteración":k,"Movimiento":"Sin ahorro positivo","Costo":z,"Ahorro":0.0})
            break
        ah,z1,a,b,q=max(mejoras,key=lambda x:x[0])
        actual=q; z=z1
        hist.append({"Iteración":k,"Movimiento":f"D{a} ↔ D{b}","Costo":z,"Ahorro":ah})
    return actual,pd.DataFrame(hist),pd.DataFrame(audit)

def svg_layout(grid,cell,titulo,centros=None):
    R,Cc=grid.shape
    scale=min(50,max(18,700/max(R,Cc)))
    W=Cc*scale; H=R*scale
    palette=["#e8f1ff","#fff1d6","#e5f7e8","#f7e5f2","#eee8ff","#ffe8e8",
             "#e4f6f7","#f3f0d9","#e8e8e8","#dff0ff","#fce7d8","#e4ead8"]
    s=[f'<div style="font-weight:700;text-align:center;margin:5px">{titulo}</div>',
       f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-height:560px;border:1px solid #888;background:white">']
    for r in range(R):
        for c in range(Cc):
            d=int(grid[r,c]); fill="white" if d==0 else palette[(d-1)%len(palette)]
            s.append(f'<rect x="{c*scale}" y="{r*scale}" width="{scale}" height="{scale}" fill="{fill}" stroke="#bbb" stroke-width="0.7"/>')
    # contornos fuertes entre departamentos
    for r in range(R):
        for c in range(Cc):
            d=int(grid[r,c])
            if d==0: continue
            x=c*scale;y=r*scale
            if r==0 or grid[r-1,c]!=d: s.append(f'<line x1="{x}" y1="{y}" x2="{x+scale}" y2="{y}" stroke="#222" stroke-width="2"/>')
            if r==R-1 or grid[r+1,c]!=d: s.append(f'<line x1="{x}" y1="{y+scale}" x2="{x+scale}" y2="{y+scale}" stroke="#222" stroke-width="2"/>')
            if c==0 or grid[r,c-1]!=d: s.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+scale}" stroke="#222" stroke-width="2"/>')
            if c==Cc-1 or grid[r,c+1]!=d: s.append(f'<line x1="{x+scale}" y1="{y}" x2="{x+scale}" y2="{y+scale}" stroke="#222" stroke-width="2"/>')
    if centros:
        for i,(x,y) in enumerate(centros):
            if not np.isnan(x):
                sx=x/cell*scale; sy=y/cell*scale
                s.append(f'<circle cx="{sx}" cy="{sy}" r="13" fill="white" stroke="#111" stroke-width="1.5"/>')
                s.append(f'<text x="{sx}" y="{sy}" text-anchor="middle" dominant-baseline="middle" font-size="13" font-weight="700">D{i+1}</text>')
    s.append("</svg>")
    return "".join(s)

def franjas_iniciales(rows,cols,n,areas,cell):
    g=np.zeros((rows,cols),dtype=int)
    counts=[int(round(a/(cell*cell))) for a in areas]
    pos=0
    for d,cnt in enumerate(counts,1):
        for _ in range(cnt):
            if pos>=rows*cols: break
            r=pos//cols; c=pos%cols
            g[r,c]=d; pos+=1
    return g

# ---------------- INTERFAZ ----------------
st.title("🏭 CRAFT Educativo — V4")
st.caption("Editor 2D de planta + cálculo de centroides y costo + optimización CRAFT.")

with st.sidebar:
    st.header("Planta")
    L=float(st.number_input("Largo (m)",1.0,value=20.0,step=1.0))
    W=float(st.number_input("Ancho (m)",1.0,value=10.0,step=1.0))
    cell=float(st.selectbox("Tamaño de celda (m)",[0.5,1.0,2.0],index=1))
    n=int(st.number_input("Departamentos",2,12,4))
    metodo=st.selectbox("Distancia",["Rectilínea","Euclidiana"],index=1)
    maxit=int(st.number_input("Máx. iteraciones CRAFT",1,100,30))

rows=int(round(W/cell)); cols=int(round(L/cell))
if abs(rows*cell-W)>1e-8 or abs(cols*cell-L)>1e-8:
    st.error("Largo y ancho deben ser múltiplos del tamaño de celda."); st.stop()

deps=[f"D{i+1}" for i in range(n)]
areas0=[60.,40.,80.,20.] if n==4 and L==20 and W==10 else [round(L*W/n,2)]*n

st.header("1. Datos de los departamentos")
df=st.data_editor(pd.DataFrame({"Departamento":deps,"Área requerida (m²)":areas0}),
                  disabled=["Departamento"],hide_index=True,use_container_width=True,key="areas")
areas=pd.to_numeric(df["Área requerida (m²)"],errors="coerce").fillna(0).to_numpy(float).copy()
if areas.sum()>L*W+1e-9:
    st.error("La suma de áreas excede el área disponible de la planta."); st.stop()
for a in areas:
    if abs(a/(cell*cell)-round(a/(cell*cell)))>1e-8:
        st.error("Cada área debe poder expresarse como un número entero de celdas. Cambia el tamaño de celda o el área."); st.stop()

st.header("2. Flujo y costos")
F0=np.zeros((n,n))
if n==4: F0=np.array([[0,2,7,4],[3,0,5,5],[6,7,0,33],[8,2,3,0]],float)
c1,c2=st.columns(2)
with c1:
    st.subheader("Matriz From-To")
    Fdf=st.data_editor(pd.DataFrame(F0,index=deps,columns=deps),use_container_width=True,key="F")
with c2:
    st.subheader("Costos unitarios")
    C0=np.ones((n,n)); np.fill_diagonal(C0,0)
    Cdf=st.data_editor(pd.DataFrame(C0,index=deps,columns=deps),use_container_width=True,key="C")
F=Fdf.to_numpy(float).copy(); C=Cdf.to_numpy(float).copy()
np.fill_diagonal(F,0); np.fill_diagonal(C,0)

# Inicializar editor
sig=(rows,cols,n,tuple(np.round(areas,8)))
if st.session_state.get("grid_sig")!=sig:
    st.session_state.grid=franjas_iniciales(rows,cols,n,areas,cell)
    st.session_state.grid_sig=sig
    st.session_state.pop("craft_result",None)

st.header("3. Editor 2D del layout inicial")
st.write("Cada celda representa **%.2f m × %.2f m**. Escribe **0** para espacio libre y **1, 2, 3...** para D1, D2, D3..."%(cell,cell))

left,right=st.columns([1.25,1])
with left:
    grid_df=pd.DataFrame(st.session_state.grid,
                         index=[f"F{r+1}" for r in range(rows)],
                         columns=[f"C{c+1}" for c in range(cols)])
    edited=st.data_editor(grid_df,use_container_width=True,height=min(650,35*(rows+1)),
                          key="grid_editor")
    try:
        g=np.rint(edited.to_numpy(float)).astype(int)
    except:
        st.error("La cuadrícula solo admite números."); st.stop()
    if np.any(g<0) or np.any(g>n):
        st.error(f"Solo puedes usar valores entre 0 y {n}."); st.stop()
    st.session_state.grid=g.copy()

with right:
    cen,areas_dib=centroides_grid(g,n,cell)
    st.markdown(svg_layout(g,cell,"Vista 2D",cen),unsafe_allow_html=True)
    resumen=pd.DataFrame({
        "Depto":deps,
        "Área req.":areas,
        "Área dibujada":np.round(areas_dib,2),
        "Centro X":[None if np.isnan(x) else round(x,2) for x,y in cen],
        "Centro Y":[None if np.isnan(y) else round(y,2) for x,y in cen]
    })
    st.dataframe(resumen,hide_index=True,use_container_width=True)

b1,b2,b3=st.columns(3)
with b1:
    if st.button("↩ Restablecer layout"):
        st.session_state.grid=franjas_iniciales(rows,cols,n,areas,cell)
        st.session_state.pop("craft_result",None)
        st.rerun()
with b2:
    errores=validar(g,n,cell,areas)
    if not errores: st.success("✓ Layout válido: áreas correctas y departamentos conectados.")
    else: st.warning("El layout aún no es válido.")
with b3:
    z0=costo_layout(g,n,cell,F,C,metodo)
    st.metric("Costo del layout dibujado", "—" if not np.isfinite(z0) else f"{z0:,.2f}")

if errores:
    with st.expander("Ver qué debo corregir",expanded=True):
        for e in errores: st.write("• "+e)

st.header("4. Experimentación manual")
st.write("Modifica la cuadrícula y observa cómo cambian los **centroides** y el **costo**. "
         "Así puedes intentar mejorar el layout antes de ejecutar CRAFT.")

st.header("5. Optimización CRAFT")
st.latex(r"Z=\sum_i\sum_{j\ne i} f_{ij}c_{ij}d_{ij}")
st.write("En esta versión se evalúan intercambios por pares. Para áreas iguales se permiten pares; "
         "para áreas diferentes se restringen a departamentos adyacentes.")

if st.button("▶ Optimizar con CRAFT",type="primary",disabled=bool(errores)):
    final,hist,audit=craft(g,n,cell,areas,F,C,metodo,maxit)
    st.session_state.craft_result=(g.copy(),final,hist,audit,z0)

if "craft_result" in st.session_state:
    inicial,final,hist,audit,zbase=st.session_state.craft_result
    zf=costo_layout(final,n,cell,F,C,metodo)
    a,b,c,d=st.columns(4)
    a.metric("Costo inicial",f"{zbase:,.2f}")
    b.metric("Costo final estimado",f"{zf:,.2f}")
    c.metric("Ahorro",f"{zbase-zf:,.2f}")
    d.metric("Reducción",f"{100*(zbase-zf)/zbase:.2f}%" if zbase else "0%")
    q1,q2=st.columns(2)
    ci,_=centroides_grid(inicial,n,cell); cf,_=centroides_grid(final,n,cell)
    with q1: st.markdown(svg_layout(inicial,cell,"ANTES",ci),unsafe_allow_html=True)
    with q2: st.markdown(svg_layout(final,cell,"DESPUÉS",cf),unsafe_allow_html=True)
    st.subheader("Iteraciones")
    st.dataframe(hist,hide_index=True,use_container_width=True)
    with st.expander("Auditoría de pares evaluados"):
        st.dataframe(audit,hide_index=True,use_container_width=True)

    out=io.BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        df.to_excel(w,sheet_name="Departamentos",index=False)
        Fdf.to_excel(w,sheet_name="From-To")
        Cdf.to_excel(w,sheet_name="Costos")
        pd.DataFrame(inicial).to_excel(w,sheet_name="Layout inicial",index=False,header=False)
        pd.DataFrame(final).to_excel(w,sheet_name="Layout final",index=False,header=False)
        hist.to_excel(w,sheet_name="Iteraciones",index=False)
        audit.to_excel(w,sheet_name="Auditoria",index=False)
    st.download_button("⬇ Exportar resultados a Excel",out.getvalue(),
                       "resultado_CRAFT_V4.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.info("Lectura didáctica: el editor 2D permite definir el layout inicial y calcular sus centroides reales. "
        "En departamentos de áreas desiguales, CRAFT usa una estimación basada en intercambios; la reconstrucción "
        "geométrica de formas irregulares es una limitación conocida del método.")
