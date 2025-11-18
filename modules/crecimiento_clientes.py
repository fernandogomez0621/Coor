import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os

# Importar Prophet solo cuando sea necesario
def importar_prophet():
    """Importa Prophet de forma segura"""
    try:
        from prophet import Prophet
        return Prophet
    except ImportError:
        st.error("❌ Prophet no está instalado. Ejecuta: pip install prophet")
        return None

def verificar_modelos_clientes():
    """Verifica si existen modelos entrenados para clientes"""
    carpeta_modelos = Path('modelos_clientes_prophet/modelos')
    if not carpeta_modelos.exists():
        return False
    
    modelos = list(carpeta_modelos.glob('*.pkl'))
    return len(modelos) > 0

def entrenar_modelo_cliente(df_cliente, cliente, Prophet, min_observaciones=90):
    """
    Entrena un modelo Prophet para un cliente específico
    """
    if len(df_cliente) < min_observaciones:
        return None, None, None, None
    
    # Agrupar por fecha y SUMAR unidades
    df_prophet = df_cliente.groupby('Fecha_Entrega').agg({
        'Unidades': 'sum',
        'Peso': 'sum'
    }).reset_index()
    
    df_prophet.columns = ['ds', 'y', 'peso']
    df_prophet = df_prophet.sort_values('ds')
    
    # Verificar que tenga al menos 120 días únicos después de agrupar
    if len(df_prophet) < 120:
        return None, None, None, None
    
    # FASE 1: VALIDACIÓN
    fecha_corte = df_prophet['ds'].max() - pd.Timedelta(days=42)
    train = df_prophet[df_prophet['ds'] <= fecha_corte].copy()
    test = df_prophet[df_prophet['ds'] > fecha_corte].copy()
    
    if len(test) < 5:  # Necesitamos al menos algunos días de test
        return None, None, None, None
    
    m_validacion = Prophet(
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.80
    )
    m_validacion.add_country_holidays(country_name='CO')
    
    try:
        m_validacion.fit(train)
        forecast_validacion = m_validacion.predict(test[['ds']])
        
        test_merged = test.merge(
            forecast_validacion[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
            on='ds', 
            how='left'
        )
        
        mae_test = np.mean(np.abs(test_merged['y'] - test_merged['yhat']))
        rmse_test = np.sqrt(np.mean((test_merged['y'] - test_merged['yhat'])**2))
        
        # FASE 2: MODELO PRODUCTIVO
        m_final = Prophet(
            seasonality_mode='additive',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.80
        )
        m_final.add_country_holidays(country_name='CO')
        m_final.fit(df_prophet)
        
        future = m_final.make_future_dataframe(periods=42)
        forecast_final = m_final.predict(future)
        
        fecha_max_historico = df_prophet['ds'].max()
        forecast_futuro = forecast_final[forecast_final['ds'] > fecha_max_historico].copy()
        
        metricas = {
            'Cliente': cliente,
            'Observaciones_Total': len(df_prophet),
            'Observaciones_Train': len(train),
            'Observaciones_Test': len(test),
            'Fecha_Inicio': df_prophet['ds'].min(),
            'Fecha_Fin': df_prophet['ds'].max(),
            'Fecha_Corte_Validacion': fecha_corte,
            'MAE_Test': round(mae_test, 2),
            'RMSE_Test': round(rmse_test, 2),
            'Unidades_Promedio_Historico': round(df_prophet['y'].mean(), 2),
            'Unidades_Std_Historico': round(df_prophet['y'].std(), 2),
            'Unidades_Promedio_Predichas_6sem': round(forecast_futuro['yhat'].mean(), 2),
            'Unidades_Total_Predichas_6sem': round(forecast_futuro['yhat'].sum(), 2),
            'Crecimiento_Porcentual': round(
                ((forecast_futuro['yhat'].mean() - df_prophet['y'].mean()) / df_prophet['y'].mean()) * 100, 2
            ) if df_prophet['y'].mean() > 0 else 0
        }
        
        return m_final, forecast_final, forecast_validacion, metricas
        
    except Exception as e:
        return None, None, None, None

def entrenar_todos_modelos_clientes(df):
    """Entrena modelos para el TOP 100 clientes con suficientes datos"""
    
    Prophet = importar_prophet()
    if Prophet is None:
        return False
    
    # Crear estructura de carpetas
    carpeta_base = 'modelos_clientes_prophet'
    os.makedirs(carpeta_base, exist_ok=True)
    
    carpeta_modelos = os.path.join(carpeta_base, 'modelos')
    carpeta_predicciones = os.path.join(carpeta_base, 'predicciones')
    carpeta_metricas = os.path.join(carpeta_base, 'metricas')
    
    for carpeta in [carpeta_modelos, carpeta_predicciones, carpeta_metricas]:
        os.makedirs(carpeta, exist_ok=True)
    
    # Obtener clientes únicos con todas las métricas necesarias
    clientes_info = df.groupby('Cliente').agg({
        'Fecha_Entrega': lambda x: x.nunique(),  # Días únicos
        'ID': 'count',  # Total registros
        'Unidades': 'sum'  # Total unidades (para ranking)
    }).reset_index()
    clientes_info.columns = ['Cliente', 'dias_unicos', 'total_registros', 'total_unidades']
    
    # Filtrar: ≥90 registros Y ≥120 días únicos
    clientes_validos = clientes_info[
        (clientes_info['total_registros'] >= 90) & 
        (clientes_info['dias_unicos'] >= 120)
    ].copy()
    
    # Ordenar por total de unidades (descendente) y tomar TOP 100
    clientes_validos = clientes_validos.sort_values('total_unidades', ascending=False).head(100)
    
    st.info(f"🏆 Se entrenarán modelos para el TOP 100 clientes (por volumen de unidades)")
    
    if len(clientes_validos) == 0:
        st.error("❌ No hay clientes que cumplan los criterios mínimos")
        return False
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    modelos_exitosos = []
    todas_metricas = []
    
    total_clientes = len(clientes_validos)
    
    for idx, row in enumerate(clientes_validos.itertuples(), 1):
        cliente = row.Cliente
        
        progreso = min(idx / total_clientes, 1.0)
        progress_bar.progress(progreso)
        status_text.text(f"Entrenando {idx}/{total_clientes}: {cliente}")
        
        df_filtrado = df[df['Cliente'] == cliente].copy()
        
        m_final, forecast_final, forecast_validacion, metricas = entrenar_modelo_cliente(
            df_filtrado, cliente, Prophet
        )
        
        if m_final is not None:
            nombre_archivo = f"{cliente}".replace(' ', '_').replace('/', '-').replace('.', '')
            
            # Guardar modelo
            archivo_modelo = os.path.join(carpeta_modelos, f'{nombre_archivo}.pkl')
            with open(archivo_modelo, 'wb') as f:
                pickle.dump(m_final, f)
            
            # Guardar predicciones futuras
            fecha_max = df_filtrado['Fecha_Entrega'].max()
            predicciones_futuras = forecast_final[forecast_final['ds'] > fecha_max].copy()
            archivo_predicciones = os.path.join(carpeta_predicciones, f'{nombre_archivo}_6semanas.csv')
            predicciones_futuras.to_csv(archivo_predicciones, index=False)
            
            todas_metricas.append(metricas)
            modelos_exitosos.append(cliente)
    
    progress_bar.empty()
    status_text.empty()
    
    # Guardar métricas
    if todas_metricas:
        df_metricas = pd.DataFrame(todas_metricas)
        archivo_metricas = os.path.join(carpeta_metricas, 'resumen_metricas_clientes.csv')
        df_metricas.to_csv(archivo_metricas, index=False)
        
        st.success(f"✅ {len(modelos_exitosos)} modelos entrenados exitosamente")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 MAE Promedio", f"{df_metricas['MAE_Test'].mean():.2f}")
        with col2:
            st.metric("📈 RMSE Promedio", f"{df_metricas['RMSE_Test'].mean():.2f}")
        with col3:
            crecimiento_promedio = df_metricas['Crecimiento_Porcentual'].mean()
            st.metric("📈 Crecimiento Promedio", f"{crecimiento_promedio:.1f}%")
        
        return True
    
    return False

def run():
    """Función principal del módulo de crecimiento por cliente"""
    
    st.markdown("## 👥 Crecimiento Proyectado por Cliente")
    st.markdown("---")
    
    # Cargar datos
    df = cargar_datos()
    if df is None:
        return
    
    # Verificar si hay modelos entrenados
    modelos_existen = verificar_modelos_clientes()
    
    if not modelos_existen:
        st.warning("⚠️ No se detectaron modelos de clientes entrenados")
        st.info("🤖 Es necesario entrenar los modelos Prophet para clientes antes de continuar")
        
        # Mostrar estadísticas de clientes disponibles
        clientes_info = df.groupby('Cliente').agg({
            'Fecha_Entrega': lambda x: x.nunique(),
            'ID': 'count'
        }).reset_index()
        clientes_info.columns = ['Cliente', 'Dias_Unicos', 'Total_Registros']
        
        clientes_validos = clientes_info[
            (clientes_info['Total_Registros'] >= 90) & 
            (clientes_info['Dias_Unicos'] >= 120)
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Total Clientes", len(clientes_info))
        with col2:
            st.metric("✅ Clientes Válidos", len(clientes_validos))
            st.caption("≥90 registros Y ≥120 días")
        with col3:
            st.metric("📦 Total Registros", f"{len(df):,}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Entrenar Modelos de Clientes", type="primary", use_container_width=True):
                with st.spinner("Entrenando modelos... Esto puede tomar varios minutos..."):
                    exito = entrenar_todos_modelos_clientes(df)
                    if exito:
                        st.balloons()
                        st.success("🎉 ¡Modelos de clientes entrenados exitosamente!")
                        st.info("🔄 Recarga la página para usar los pronósticos")
        return
    
    # Si hay modelos, mostrar tabs normales
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Análisis por Cliente", 
        "📊 Dashboard General",
        "🏆 Top Clientes",
        "📉 Métricas de Modelos"
    ])
    
    # TAB 1: ANÁLISIS POR CLIENTE
    with tab1:
        st.markdown("### 🔍 Análisis Individual de Cliente")
        
        # Cargar métricas para obtener lista de clientes disponibles
        metricas = cargar_metricas_clientes()
        if metricas is not None:
            clientes = sorted(metricas['Cliente'].unique())
            
            # CONTENEDOR CENTRADO PARA LA SELECCIÓN DE CLIENTE
            st.markdown("""
            <div style='display: flex; justify-content: center; margin: 20px 0;'>
                <div style='width: 80%; max-width: 600px; text-align: center;'>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                cliente_sel = st.selectbox("👤 Selecciona Cliente:", clientes, key="cliente_select")
                if st.button("📊 Cargar Análisis", type="primary", use_container_width=True):
                    with st.spinner("Cargando modelo y predicciones..."):
                        mostrar_analisis_cliente(df, cliente_sel)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.error("❌ No se encontraron métricas de clientes")
    
    # TAB 2: DASHBOARD GENERAL
    with tab2:
        st.markdown("### 📊 Dashboard General de Clientes")
        mostrar_dashboard_clientes(df)
    
    # TAB 3: TOP CLIENTES
    with tab3:
        st.markdown("### 🏆 Top Clientes por Crecimiento")
        mostrar_top_clientes()
    
    # TAB 4: MÉTRICAS DE MODELOS
    with tab4:
        st.markdown("### 📉 Métricas de Validación de Modelos")
        mostrar_metricas_modelos_clientes()
    
    # Botón para re-entrenar
    st.markdown("---")
    with st.expander("⚙️ Opciones Avanzadas"):
        st.warning("⚠️ Re-entrenar eliminará los modelos actuales y creará nuevos")
        if st.button("🔄 Re-entrenar Todos los Modelos", type="secondary"):
            with st.spinner("Re-entrenando modelos de clientes..."):
                exito = entrenar_todos_modelos_clientes(df)
                if exito:
                    st.success("✅ Modelos re-entrenados exitosamente")
                    st.info("🔄 Recarga la página para ver los cambios")


def cargar_datos():
    """Carga el dataset principal"""
    try:
        df = pd.read_csv('DataSet_Entregas.csv')
        df['Fecha_Entrega'] = pd.to_datetime(df['Fecha_Entrega'])
        df['Fecha_Recogida'] = pd.to_datetime(df['Fecha_Recogida'])
        st.success(f"✅ Dataset cargado: {len(df):,} registros")
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'DataSet_Entregas.csv'")
        st.info("📁 Asegúrate de que el archivo CSV esté en la misma carpeta que main.py")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")
        return None


def cargar_modelo_cliente(cliente):
    """Carga un modelo Prophet de cliente previamente entrenado"""
    nombre_archivo = f"{cliente}".replace(' ', '_').replace('/', '-').replace('.', '')
    ruta_modelo = Path(f'modelos_clientes_prophet/modelos/{nombre_archivo}.pkl')
    
    if ruta_modelo.exists():
        with open(ruta_modelo, 'rb') as f:
            return pickle.load(f)
    return None


def cargar_predicciones_cliente(cliente):
    """Carga las predicciones guardadas de 6 semanas para un cliente"""
    nombre_archivo = f"{cliente}".replace(' ', '_').replace('/', '-').replace('.', '')
    ruta_pred = Path(f'modelos_clientes_prophet/predicciones/{nombre_archivo}_6semanas.csv')
    
    if ruta_pred.exists():
        df = pd.read_csv(ruta_pred)
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    return None


def cargar_metricas_clientes():
    """Carga el resumen de métricas de todos los modelos de clientes"""
    ruta_metricas = Path('modelos_clientes_prophet/metricas/resumen_metricas_clientes.csv')
    
    if ruta_metricas.exists():
        df = pd.read_csv(ruta_metricas)
        if 'Fecha_Inicio' in df.columns:
            df['Fecha_Inicio'] = pd.to_datetime(df['Fecha_Inicio'])
        if 'Fecha_Fin' in df.columns:
            df['Fecha_Fin'] = pd.to_datetime(df['Fecha_Fin'])
        return df
    return None


def mostrar_analisis_cliente(df, cliente):
    """Muestra análisis completo de un cliente CON ORDEN CORREGIDO"""
    
    modelo = cargar_modelo_cliente(cliente)
    predicciones = cargar_predicciones_cliente(cliente)
    
    if modelo is None:
        st.error(f"❌ No se encontró modelo entrenado para el cliente {cliente}")
        st.info("💡 El modelo puede no haberse entrenado por datos insuficientes")
        return
    
    if predicciones is None:
        st.warning("⚠️ No se encontraron predicciones guardadas")
        return
    
    df_hist = df[df['Cliente'] == cliente].copy()
    
    # Cargar métricas específicas
    metricas_df = cargar_metricas_clientes()
    metricas_cliente = None
    if metricas_df is not None:
        metricas_cliente = metricas_df[metricas_df['Cliente'] == cliente]
        if len(metricas_cliente) > 0:
            metricas_cliente = metricas_cliente.iloc[0]
    
    st.success(f"✅ Modelo cargado exitosamente para {cliente}")
    
    # CONTENEDOR PRINCIPAL CENTRADO
    st.markdown("""
    <div style='display: flex; justify-content: center;'>
        <div style='width: 95%; max-width: 1200px;'>
    """, unsafe_allow_html=True)
    
    # 1. MÉTRICAS PRINCIPALES (4 columnas)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Registros Históricos", f"{len(df_hist):,}")
    
    with col2:
        unidades_promedio = predicciones['yhat'].mean()
        st.metric("📊 Unidades Promedio/Día", f"{unidades_promedio:.1f}", 
                 delta="próximas 6 semanas")
    
    with col3:
        unidades_total = predicciones['yhat'].sum()
        st.metric("📈 Total Unidades 6 sem", f"{unidades_total:.0f}")
    
    with col4:
        if metricas_cliente is not None:
            crecimiento = metricas_cliente['Crecimiento_Porcentual']
            delta_color = "normal" if crecimiento >= 0 else "inverse"
            st.metric("📈 Crecimiento", f"{crecimiento:.1f}%",
                     delta=f"vs histórico", delta_color=delta_color)
        else:
            st.metric("📈 Crecimiento", "N/A")
    
    st.markdown("---")
    
    # 2. GRÁFICO PRINCIPAL - PRONÓSTICO COMPLETO
    st.markdown("### 📊 Pronóstico de Unidades - 6 Semanas Futuras")
    
    try:
        future = modelo.make_future_dataframe(periods=42)
        forecast_completo = modelo.predict(future)
        
        fig_principal = crear_grafico_cliente_completo(df_hist, forecast_completo, cliente)
        st.plotly_chart(fig_principal, use_container_width=True)
    except:
        fig_principal = crear_grafico_cliente_basico(df_hist, predicciones, cliente)
        st.plotly_chart(fig_principal, use_container_width=True)
    
    st.markdown("---")
    
    # 3. COMPONENTES DEL MODELO (ancho completo)
    st.markdown("### 🔧 Componentes del Modelo (Tendencia y Estacionalidades)")
    try:
        fig_componentes = crear_grafico_componentes_cliente(forecast_completo)
        st.plotly_chart(fig_componentes, use_container_width=True)
    except:
        st.info("ℹ️ Componentes del modelo no disponibles")
    
    st.markdown("---")
    
    # 4. FILA: VALIDACIÓN + PRODUCTOS
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 Validación en Test Set")
        if metricas_cliente is not None:
            try:
                fig_validacion = crear_grafico_validacion_cliente(df_hist, forecast_completo, metricas_cliente)
                st.plotly_chart(fig_validacion, use_container_width=True)
            except:
                st.info("ℹ️ Métricas de validación no disponibles")
        else:
            st.info("ℹ️ Métricas de validación no disponibles")
    
    with col2:
        st.markdown("### 📦 Productos Más Demandados")
        fig_productos = crear_grafico_productos_cliente(df_hist)
        st.plotly_chart(fig_productos, use_container_width=True)
    
    st.markdown("---")
    
    # 5. FILA: RUTAS + MÉTRICAS
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🛣️ Rutas Principales")
        fig_rutas = crear_grafico_rutas_cliente(df_hist)
        st.plotly_chart(fig_rutas, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Resumen de Métricas")
        mostrar_panel_metricas_cliente(metricas_cliente, predicciones, df_hist)
    
    st.markdown("---")
    
    # 6. TABLA DE PREDICCIONES
    st.markdown("### 📋 Tabla de Predicciones Detalladas")
    df_tabla = predicciones[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    df_tabla.columns = ['Fecha', 'Unidades Predichas', 'Límite Inferior (80%)', 'Límite Superior (80%)']
    df_tabla['Fecha'] = df_tabla['Fecha'].dt.strftime('%Y-%m-%d')
    df_tabla['Unidades Predichas'] = df_tabla['Unidades Predichas'].round(1)
    df_tabla['Límite Inferior (80%)'] = df_tabla['Límite Inferior (80%)'].round(1)
    df_tabla['Límite Superior (80%)'] = df_tabla['Límite Superior (80%)'].round(1)
    
    st.dataframe(df_tabla, use_container_width=True, height=400)
    
    csv = df_tabla.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Predicciones (CSV)",
        data=csv,
        file_name=f"predicciones_cliente_{cliente}.csv",
        mime="text/csv"
    )
    
    # CERRAR CONTENEDOR CENTRADO
    st.markdown("</div></div>", unsafe_allow_html=True)


def mostrar_dashboard_clientes(df):
    """Muestra dashboard con resumen de todos los clientes"""
    
    metricas = cargar_metricas_clientes()
    
    if metricas is None:
        st.warning("⚠️ No se encontraron métricas de clientes")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Clientes con Modelo", len(metricas))
    
    with col2:
        unidades_total = metricas['Unidades_Total_Predichas_6sem'].sum()
        st.metric("📦 Total Unidades 6 sem", f"{unidades_total:,.0f}")
    
    with col3:
        crecimiento_promedio = metricas['Crecimiento_Porcentual'].mean()
        st.metric("📈 Crecimiento Promedio", f"{crecimiento_promedio:.1f}%")
    
    with col4:
        mae_promedio = metricas['MAE_Test'].mean()
        st.metric("📊 MAE Promedio", f"{mae_promedio:.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Top 15 Clientes por Demanda Predicha")
        fig = crear_grafico_top_demanda_clientes(metricas)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Distribución de Crecimiento")
        fig = crear_grafico_distribucion_crecimiento(metricas)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Clientes con Mayor Crecimiento")
        top_crecimiento = metricas.nlargest(10, 'Crecimiento_Porcentual')[
            ['Cliente', 'Crecimiento_Porcentual', 'Unidades_Total_Predichas_6sem']
        ].copy()
        top_crecimiento.columns = ['Cliente', 'Crecimiento %', 'Unidades 6 sem']
        st.dataframe(top_crecimiento, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Clientes con Decrecimiento")
        decrecimiento = metricas[metricas['Crecimiento_Porcentual'] < 0].nsmallest(10, 'Crecimiento_Porcentual')[
            ['Cliente', 'Crecimiento_Porcentual', 'Unidades_Total_Predichas_6sem']
        ].copy()
        if len(decrecimiento) > 0:
            decrecimiento.columns = ['Cliente', 'Crecimiento %', 'Unidades 6 sem']
            st.dataframe(decrecimiento, use_container_width=True)
        else:
            st.info("✅ No hay clientes con decrecimiento proyectado")


def mostrar_top_clientes():
    """Muestra análisis de top clientes"""
    
    metricas = cargar_metricas_clientes()
    
    if metricas is None:
        st.warning("⚠️ No se encontraron métricas de clientes")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top 20 por Demanda Total Predicha")
        top_demanda = metricas.nlargest(20, 'Unidades_Total_Predichas_6sem')[
            ['Cliente', 'Unidades_Total_Predichas_6sem', 'Crecimiento_Porcentual', 'MAE_Test']
        ].copy()
        top_demanda.columns = ['Cliente', 'Unidades 6 sem', 'Crecimiento %', 'MAE']
        top_demanda['Unidades 6 sem'] = top_demanda['Unidades 6 sem'].round(0)
        top_demanda['Crecimiento %'] = top_demanda['Crecimiento %'].round(1)
        top_demanda['MAE'] = top_demanda['MAE'].round(2)
        st.dataframe(top_demanda, use_container_width=True, height=600)
    
    with col2:
        st.markdown("#### 📈 Top 20 por Crecimiento Porcentual")
        top_crecimiento = metricas.nlargest(20, 'Crecimiento_Porcentual')[
            ['Cliente', 'Crecimiento_Porcentual', 'Unidades_Total_Predichas_6sem', 'Unidades_Promedio_Historico']
        ].copy()
        top_crecimiento.columns = ['Cliente', 'Crecimiento %', 'Unidades 6 sem', 'Prom. Histórico']
        top_crecimiento['Crecimiento %'] = top_crecimiento['Crecimiento %'].round(1)
        top_crecimiento['Unidades 6 sem'] = top_crecimiento['Unidades 6 sem'].round(0)
        top_crecimiento['Prom. Histórico'] = top_crecimiento['Prom. Histórico'].round(1)
        st.dataframe(top_crecimiento, use_container_width=True, height=600)


def mostrar_metricas_modelos_clientes():
    """Muestra tabla completa de métricas de todos los clientes"""
    
    metricas = cargar_metricas_clientes()
    
    if metricas is None:
        st.warning("⚠️ No se encontraron métricas de clientes")
        return
    
    st.markdown("#### 📋 Métricas Completas de Validación por Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        orden = st.selectbox("Ordenar por:", [
            'Cliente',
            'Unidades_Total_Predichas_6sem',
            'Crecimiento_Porcentual',
            'MAE_Test',
            'Observaciones_Total'
        ])
    
    with col2:
        direccion = st.radio("Dirección:", ['Descendente', 'Ascendente'], horizontal=True)
    
    df_mostrar = metricas.copy()
    ascending = True if direccion == 'Ascendente' else False
    df_mostrar = df_mostrar.sort_values(orden, ascending=ascending)
    
    st.info(f"📊 Mostrando {len(df_mostrar)} clientes")
    
    columnas_mostrar = ['Cliente', 'MAE_Test', 'RMSE_Test', 
                       'Observaciones_Total', 'Unidades_Promedio_Historico',
                       'Unidades_Total_Predichas_6sem', 'Crecimiento_Porcentual']
    
    df_display = df_mostrar[columnas_mostrar].copy()
    st.dataframe(df_display, use_container_width=True, height=600)
    
    csv = df_mostrar.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Métricas Completas (CSV)",
        data=csv,
        file_name="metricas_clientes_completas.csv",
        mime="text/csv"
    )


def mostrar_panel_metricas_cliente(metricas, predicciones, df_hist):
    """Muestra panel de métricas del cliente"""
    
    if metricas is None:
        st.info("ℹ️ Métricas no disponibles")
        return
    
    # Panel de Métricas de Validación
    st.markdown("""
    <div style='background-color: #E8F4F8; padding: 15px; border-radius: 10px; border: 1px solid #00B4D8;'>
        <h4 style='margin-top: 0;'>📊 MÉTRICAS DE VALIDACIÓN</h4>
        <hr style='margin: 10px 0;'>
        <ul>
            <li><b>MAE:</b> {:.2f} unidades</li>
            <li><b>RMSE:</b> {:.2f} unidades</li>
        </ul>
    </div>
    """.format(metricas['MAE_Test'], metricas['RMSE_Test']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Panel de Pronóstico
    st.markdown("""
    <div style='background-color: #FFF4E6; padding: 15px; border-radius: 10px; border: 1px solid #FF6B35;'>
        <h4 style='margin-top: 0;'>📈 PRONÓSTICO 6 SEMANAS</h4>
        <hr style='margin: 10px 0;'>
        <ul>
            <li><b>Unidades promedio/día:</b> {:.1f}</li>
            <li><b>Total unidades 6 sem:</b> {:.0f}</li>
            <li><b>Crecimiento:</b> {:.1f}%</li>
        </ul>
    </div>
    """.format(
        metricas['Unidades_Promedio_Predichas_6sem'],
        metricas['Unidades_Total_Predichas_6sem'],
        metricas['Crecimiento_Porcentual']
    ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Panel de Datos Históricos
    st.markdown("""
    <div style='background-color: #F0F8FF; padding: 15px; border-radius: 10px; border: 1px solid #2E86AB;'>
        <h4 style='margin-top: 0;'>📦 DATOS HISTÓRICOS</h4>
        <hr style='margin: 10px 0;'>
        <ul>
            <li><b>Período:</b> {} a {}</li>
            <li><b>Total observaciones:</b> {}</li>
            <li><b>Unidades prom. histórico:</b> {:.1f}</li>
            <li><b>Desviación estándar:</b> {:.1f}</li>
        </ul>
    </div>
    """.format(
        metricas['Fecha_Inicio'].strftime('%Y-%m-%d'),
        metricas['Fecha_Fin'].strftime('%Y-%m-%d'),
        metricas['Observaciones_Total'],
        metricas['Unidades_Promedio_Historico'],
        metricas['Unidades_Std_Historico']
    ), unsafe_allow_html=True)


def crear_grafico_cliente_completo(df_hist, forecast, cliente):
    """Crea el gráfico principal con histórico completo y predicciones"""
    
    df_grouped = df_hist.groupby('Fecha_Entrega')['Unidades'].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    
    fecha_max = df_grouped['ds'].max()
    forecast_hist = forecast[forecast['ds'] <= fecha_max].copy()
    forecast_future = forecast[forecast['ds'] > fecha_max].copy()
    
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=df_grouped['ds'],
        y=df_grouped['y'],
        mode='markers',
        name='Datos históricos',
        marker=dict(size=4, color='#2E86AB', opacity=0.7),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Unidades:</b> %{y:.0f}<extra></extra>'
    ))
    
    # Ajuste del modelo
    fig.add_trace(go.Scatter(
        x=forecast_hist['ds'],
        y=forecast_hist['yhat'],
        mode='lines',
        name='Ajuste modelo',
        line=dict(color='#06A77D', width=2),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Ajuste:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Predicciones futuras
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines+markers',
        name='Predicción 6 semanas',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=5),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Predicción:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Intervalo de confianza
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_lower'],
        mode='lines',
        name='IC 80%',
        fill='tonexty',
        fillcolor='rgba(255, 107, 53, 0.2)',
        line=dict(width=0),
        hovertemplate='<b>IC:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Línea vertical
    fig.add_shape(
        type="line",
        x0=fecha_max, x1=fecha_max,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=fecha_max, y=1, yref="paper",
        text="Inicio predicciones",
        showarrow=False, yshift=10,
        font=dict(size=10, color="red"),
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
    
    fig.update_layout(
        title=f"Pronóstico de Unidades - Cliente: {cliente}",
        xaxis_title="Fecha",
        yaxis_title="Unidades",
        hovermode='x unified',
        height=550,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def crear_grafico_cliente_basico(df_hist, predicciones, cliente):
    """Crea gráfico básico cuando no hay forecast completo"""
    
    df_grouped = df_hist.groupby('Fecha_Entrega')['Unidades'].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    
    fecha_max = df_grouped['ds'].max()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_grouped['ds'],
        y=df_grouped['y'],
        mode='markers',
        name='Datos Históricos',
        marker=dict(size=6, color='#2E86AB', opacity=0.7),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Unidades:</b> %{y:.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=predicciones['ds'],
        y=predicciones['yhat'],
        mode='lines+markers',
        name='Predicción',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Predicción:</b> %{y:.1f}<extra></extra>'
    ))
    
    fig.add_shape(
        type="line",
        x0=fecha_max, x1=fecha_max,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.update_layout(
        title=f"Pronóstico - Cliente: {cliente}",
        xaxis_title="Fecha",
        yaxis_title="Unidades",
        hovermode='x unified',
        height=550,
        template="plotly_white"
    )
    
    return fig


def crear_grafico_validacion_cliente(df_hist, forecast, metricas):
    """Crea gráfico de validación en test set"""
    
    df_grouped = df_hist.groupby('Fecha_Entrega')['Unidades'].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    df_grouped = df_grouped.sort_values('ds')
    
    fecha_corte = metricas['Fecha_Corte_Validacion']
    train_data = df_grouped[df_grouped['ds'] <= fecha_corte]
    test_data = df_grouped[df_grouped['ds'] > fecha_corte]
    
    forecast_test = forecast[forecast['ds'] > fecha_corte].copy()
    forecast_test = forecast_test[forecast_test['ds'] <= df_grouped['ds'].max()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train_data['ds'],
        y=train_data['y'],
        mode='markers',
        name='Train',
        marker=dict(size=3, color='#06A77D', opacity=0.5),
        hovertemplate='<b>Train:</b> %{y:.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=test_data['ds'],
        y=test_data['y'],
        mode='markers',
        name='Test (real)',
        marker=dict(size=6, color='#2E86AB', opacity=0.8),
        hovertemplate='<b>Real:</b> %{y:.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_test['ds'],
        y=forecast_test['yhat'],
        mode='lines+markers',
        name='Predicción Test',
        line=dict(color='#FF6B35', width=2),
        marker=dict(size=5),
        hovertemplate='<b>Predicción:</b> %{y:.1f}<extra></extra>'
    ))
    
    fig.add_shape(
        type="line",
        x0=fecha_corte, x1=fecha_corte,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        text=f"MAE: {metricas['MAE_Test']:.2f}<br>RMSE: {metricas['RMSE_Test']:.2f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=11)
    )
    
    fig.update_layout(
        title="Validación en Test Set (últimos 42 días)",
        xaxis_title="Fecha",
        yaxis_title="Unidades",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )
    
    return fig


def crear_grafico_componentes_cliente(forecast):
    """Crea gráfico de componentes del modelo"""
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Tendencia', 'Estacionalidad Anual', 'Estacionalidad Semanal'),
        vertical_spacing=0.1,
        row_heights=[0.33, 0.33, 0.33]
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            name='Tendencia',
            line=dict(color='#06A77D', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yearly'],
            mode='lines',
            name='Estac. Anual',
            line=dict(color='#D62828', width=1.5)
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5, opacity=0.3, row=2, col=1)
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['weekly'],
            mode='lines',
            name='Estac. Semanal',
            line=dict(color='#9B59B6', width=1.5)
        ),
        row=3, col=1
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5, opacity=0.3, row=3, col=1)
    
    fig.update_xaxes(title_text="Fecha", row=3, col=1)
    fig.update_yaxes(title_text="Tendencia", row=1, col=1)
    fig.update_yaxes(title_text="Estac. Anual", row=2, col=1)
    fig.update_yaxes(title_text="Estac. Semanal", row=3, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_white",
        title_text="Componentes del Modelo"
    )
    
    return fig


def crear_grafico_productos_cliente(df_hist):
    """Crea gráfico de productos más demandados por el cliente"""
    
    productos = df_hist.groupby('Producto')['Unidades'].sum().reset_index()
    productos = productos.sort_values('Unidades', ascending=True).tail(10)
    
    fig = go.Figure(go.Bar(
        x=productos['Unidades'],
        y=productos['Producto'],
        orientation='h',
        marker_color='#06A77D',
        text=productos['Unidades'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Top 10 Productos",
        xaxis_title="Unidades Totales",
        yaxis_title="",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def crear_grafico_rutas_cliente(df_hist):
    """Crea gráfico de rutas principales del cliente"""
    
    df_hist['Ruta'] = df_hist['Terminal_Origen'] + ' → ' + df_hist['Terminal_Destino']
    rutas = df_hist.groupby('Ruta')['Unidades'].sum().reset_index()
    rutas = rutas.sort_values('Unidades', ascending=False).head(8)
    
    fig = px.pie(
        rutas,
        values='Unidades',
        names='Ruta',
        title='Distribución por Ruta',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400, template="plotly_white")
    
    return fig


def crear_grafico_top_demanda_clientes(metricas):
    """Crea gráfico de top clientes por demanda predicha"""
    
    top15 = metricas.nlargest(15, 'Unidades_Total_Predichas_6sem')
    
    fig = go.Figure(go.Bar(
        x=top15['Unidades_Total_Predichas_6sem'],
        y=top15['Cliente'],
        orientation='h',
        marker=dict(
            color=top15['Crecimiento_Porcentual'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Crecimiento %")
        ),
        text=top15['Unidades_Total_Predichas_6sem'].round(0),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Unidades: %{x:.0f}<br>Crecimiento: %{marker.color:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Unidades Predichas (6 semanas)",
        yaxis_title="",
        height=500,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def crear_grafico_distribucion_crecimiento(metricas):
    """Crea histograma de distribución de crecimiento"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=metricas['Crecimiento_Porcentual'],
        nbinsx=30,
        marker_color='#06A77D',
        opacity=0.7,
        name='Distribución'
    ))
    
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Sin crecimiento",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=metricas['Crecimiento_Porcentual'].mean(),
        line_dash="dot",
        line_color="blue",
        line_width=2,
        annotation_text=f"Promedio: {metricas['Crecimiento_Porcentual'].mean():.1f}%",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Distribución de Crecimiento por Cliente",
        xaxis_title="Crecimiento Porcentual (%)",
        yaxis_title="Número de Clientes",
        height=500,
        template="plotly_white",
        showlegend=False
    )
    
    return fig