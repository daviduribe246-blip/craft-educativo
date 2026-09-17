import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="CRAFT Educativo V3",page_icon="🏭",layout="wide")

def dist(a,b,metodo):
    dx=abs(a[0]-b[0]); dy=abs(a[1]-b[1])
    return dx+dy if metodo=="Rectilínea" else float(np.hypot(dx,dy))

def costo(asig,centros,F,C,metodo):
    n=len(asig); z=0.0
    for i in range(n):
        for j in range(n):
            if i!=j:
                z += F[i,j]*C[i,j]*dist(centros[asig[i]],centros[asig[j]],metodo)
    return float(z)

def adyacentes_slots(a,b,centros,dimensiones,tol=1e-9):
    xa,ya=centros[a]; xb,yb=centros[b]
    wa,ha=dimensiones[a]; wb,hb=dimensiones[b]
    horizontal=abs(abs(xa-xb)-(wa+wb)/2)<=tol and abs(ya-yb)<=(ha+hb)/2+tol
    vertical=abs(abs(ya-yb)-(ha+hb)/2)<=tol and abs(xa-xb)<=(wa+wb)/2+tol
    return horizontal or vertical

def candidatos(asig,areas,centros,dimensiones,F,C,metodo):
    n=len(asig); z0=costo(asig,centros,F,C,metodo); rows=[]
    for i in range(n):
        for j in range(i+1,n):
            si,sj=asig[i],asig[j]
            igual=abs(areas[i]-areas[j])<1e-9
            ady=adyacentes_slots(si,sj,centros,dimensiones)
            permitido=igual or ady
            if permitido:
                q=asig.copy(); q[i],q[j]=q[j],q[i]
                z1=costo(q,centros,F,C,metodo)
                rows.append((i,j,z1,z0-z1,igual,ady,q))
    return rows

def craft(areas,centros,dimensiones,F,C,metodo,maxit=50):
    n=len(areas); asig=list(range(n)); z=costo(asig,centros,F,C,metodo)
    hist=[{"Iteración":0,"Intercambio":"Inicial","Costo":z,"Ahorro":0.0}]
    audit=[]
    for k in range(1,maxit+1):
        cand=candidatos(asig,areas,centros,dimensiones,F,C,metodo)
        for i,j,z1,ah,ig,ad,q in cand:
            audit.append({"Iteración":k,"Par":f"D{i+1} ↔ D{j+1}",
                          "Áreas iguales":ig,"Adyacentes":ad,
                          "Costo estimado":z1,"Ahorro estimado":ah})
        mejoras=[x for x in cand if x[3]>1e-9]
        if not mejoras:
            hist.append({"Iteración":k,"Intercambio":"Sin ahorro positivo","Costo":z,"Ahorro":0.0})
            break
        best=max(mejoras,key=lambda x:x[3])
        i,j,z1,ah,ig,ad,q=best
        asig=q; z=z1
        hist.append({"Iteración":k,"Intercambio":f"D{i+1} ↔ D{j+1}",
                     "Costo":z,"Ahorro":ah})
    return asig,pd.DataFrame(hist),pd.DataFrame(audit)

def layout_franjas(L,W,areas):
    total=sum(areas)
    # franjas horizontales proporcionales; si queda área libre, se conserva al final
    y=0.; centros=[]; dims=[]
    for a in areas:
        h=a/L
        centros.append((L/2,y+h/2)); dims.append((L,h)); y+=h
    return centros,dims,y

def grafico(L,W,areas,asig,centros,dims,titulo):
    fig,ax=plt.subplots(figsize=(9,4.6))
    # slot s contiene el departamento cuyo asig[d]=s
    dep_en_slot={slot:d for d,slot in enumerate(asig)}
    for s,(x,y) in enumerate(centros):
        w,h=dims[s]
        d=dep_en_slot.get(s)
        rect=plt.Rectangle((x-w/2,y-h/2),w,h,fill=False,linewidth=1.5)
        ax.add_patch(rect)
        if d is not None:
            ax.text(x,y,f"D{d+1}",ha="center",va="center",fontweight="bold")
    ax.set_xlim(0,L); ax.set_ylim(W,0); ax.set_aspect("equal")
    ax.set_title(titulo); ax.set_xlabel("m"); ax.set_ylabel("m")
    return fig

st.title("🏭 CRAFT Educativo — V3")
st.caption("Implementación didáctica rápida basada en intercambio de centroides.")

with st.sidebar:
    st.header("Configuración")
    L=st.number_input("Largo (m)",1.0,value=20.0)
    W=st.number_input("Ancho (m)",1.0,value=10.0)
    n=int(st.number_input("Departamentos",2,20,4))
    metodo=st.selectbox("Distancia",["Rectilínea","Euclidiana"],index=1)
    maxit=int(st.number_input("Máx. iteraciones",1,100,30))

deps=[f"D{i+1}" for i in range(n)]
areas0=[60.,40.,80.,20.] if n==4 and L==20 and W==10 else [L*W/n]*n
st.header("1. Departamentos")
df=st.data_editor(pd.DataFrame({"Departamento":deps,"Área (m²)":areas0}),
                  disabled=["Departamento"],hide_index=True,use_container_width=True)
areas=pd.to_numeric(df["Área (m²)"],errors="coerce").fillna(0).to_numpy(float).copy()
if areas.sum()>L*W+1e-9:
    st.error("Las áreas exceden el área de planta."); st.stop()

st.header("2. Matriz From-To")
F0=np.zeros((n,n))
if n==4: F0=np.array([[0,2,7,4],[3,0,5,5],[6,7,0,33],[8,2,3,0]],float)
Fdf=st.data_editor(pd.DataFrame(F0,index=deps,columns=deps),use_container_width=True)
F=Fdf.to_numpy(float).copy(); np.fill_diagonal(F,0)

st.header("3. Costos unitarios")
C0=np.ones((n,n)); np.fill_diagonal(C0,0)
Cdf=st.data_editor(pd.DataFrame(C0,index=deps,columns=deps),use_container_width=True)
C=Cdf.to_numpy(float).copy(); np.fill_diagonal(C,0)

centros,dims,ocupado=layout_franjas(L,W,areas)
if ocupado>W+1e-9:
    st.error("Con este generador de franjas, las áreas no caben."); st.stop()

inicial=list(range(n))
z0=costo(inicial,centros,F,C,metodo)
st.header("4. Layout inicial")
a,b=st.columns([1.3,1])
with a: st.pyplot(grafico(L,W,areas,inicial,centros,dims,"Layout inicial"))
with b:
    st.metric("Costo inicial",f"{z0:,.2f}")
    st.dataframe(pd.DataFrame([{"Departamento":deps[i],"X":centros[i][0],"Y":centros[i][1]} for i in range(n)]),
                 hide_index=True,use_container_width=True)

st.header("5. CRAFT")
st.latex(r"Z=\sum_i\sum_{j\ne i} f_{ij}c_{ij}d_{ij}")
st.info("Regla implementada: en cada iteración se evalúan pares de departamentos. "
        "Los de igual área pueden intercambiarse; los de áreas diferentes solo se consideran "
        "si sus ubicaciones son adyacentes. El ahorro se estima intercambiando centroides.")

if st.button("▶ Optimizar con CRAFT",type="primary"):
    asig,hist,audit=craft(areas,centros,dims,F,C,metodo,maxit)
    st.session_state["res"]=(asig,hist,audit)

if "res" in st.session_state:
    asig,hist,audit=st.session_state["res"]
    zf=float(hist.iloc[-1]["Costo"])
    ahorro=z0-zf; pct=100*ahorro/z0 if z0 else 0
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Costo inicial",f"{z0:,.2f}")
    c2.metric("Mejor costo estimado",f"{zf:,.2f}")
    c3.metric("Ahorro",f"{ahorro:,.2f}")
    c4.metric("Reducción",f"{pct:.2f}%")
    x,y=st.columns(2)
    with x: st.pyplot(grafico(L,W,areas,inicial,centros,dims,"Antes"))
    with y: st.pyplot(grafico(L,W,areas,asig,centros,dims,"Después (asignación de centroides)"))
    st.subheader("Iteraciones")
    st.dataframe(hist,hide_index=True,use_container_width=True)
    with st.expander("Auditoría de pares evaluados"):
        st.dataframe(audit,hide_index=True,use_container_width=True)

    out=io.BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        df.to_excel(w,sheet_name="Departamentos",index=False)
        Fdf.to_excel(w,sheet_name="From-To")
        Cdf.to_excel(w,sheet_name="Costos")
        hist.to_excel(w,sheet_name="Iteraciones",index=False)
        audit.to_excel(w,sheet_name="Auditoria",index=False)
    st.download_button("⬇ Exportar Excel",out.getvalue(),"resultado_CRAFT_V3.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.warning("Nota metodológica: para departamentos de áreas desiguales, el intercambio de centroides "
           "es una estimación CRAFT y puede producir formas físicas irregulares al reconstruir el layout. "
           "Por ello esta V3 separa la evaluación de centroides de una futura reconstrucción geométrica.")
