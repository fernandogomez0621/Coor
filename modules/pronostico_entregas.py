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

def verificar_modelos_entrenados():
    """Verifica si existen modelos entrenados"""
    carpeta_modelos = Path('modelos_prophet_validados/modelos')
    if not carpeta_modelos.exists():
        return False
    
    modelos = list(carpeta_modelos.glob('*.pkl'))
    return len(modelos) > 0

def entrenar_modelo_prophet_validado(df_grupo, terminal, producto, Prophet, min_observaciones=100):
    """
    Entrena un modelo Prophet con validación correcta
    """
    if len(df_grupo) < min_observaciones:
        return None, None, None, None
    
    # Agrupar por fecha y SUMAR
    df_prophet = df_grupo.groupby('Fecha_Entrega').agg({
        'Dias_Transcurridos': 'sum',
        'Unidades': 'sum',
        'Peso': 'sum'
    }).reset_index()
    
    df_prophet.columns = ['ds', 'y', 'unidades', 'peso']
    df_prophet = df_prophet.sort_values('ds')
    
    if len(df_prophet) < 90:
        return None, None, None, None
    
    # FASE 1: VALIDACIÓN
    fecha_corte = df_prophet['ds'].max() - pd.Timedelta(days=42)
    train = df_prophet[df_prophet['ds'] <= fecha_corte].copy()
    test = df_prophet[df_prophet['ds'] > fecha_corte].copy()
    
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
            'Terminal': terminal,
            'Producto': producto,
            'Observaciones_Total': len(df_prophet),
            'Observaciones_Train': len(train),
            'Observaciones_Test': len(test),
            'Fecha_Inicio': df_prophet['ds'].min(),
            'Fecha_Fin': df_prophet['ds'].max(),
            'Fecha_Corte_Validacion': fecha_corte,
            'MAE_Test': round(mae_test, 2),
            'RMSE_Test': round(rmse_test, 2),
            'Demanda_Promedio_Historica': round(df_prophet['y'].mean(), 2),
            'Desviacion_Std_Historica': round(df_prophet['y'].std(), 2),
            'Demanda_Promedio_Predicha_6sem': round(forecast_futuro['yhat'].mean(), 2),
            'Demanda_Total_Predicha_6sem': round(forecast_futuro['yhat'].sum(), 2)
        }
        
        return m_final, forecast_final, forecast_validacion, metricas
        
    except Exception as e:
        return None, None, None, None

def entrenar_todos_modelos(df):
    """Entrena todos los modelos con barra de progreso"""
    
    Prophet = importar_prophet()
    if Prophet is None:
        return False
    
    # Crear estructura de carpetas
    carpeta_base = 'modelos_prophet_validados'
    os.makedirs(carpeta_base, exist_ok=True)
    
    carpeta_modelos = os.path.join(carpeta_base, 'modelos')
    carpeta_predicciones = os.path.join(carpeta_base, 'predicciones')
    carpeta_metricas = os.path.join(carpeta_base, 'metricas')
    
    for carpeta in [carpeta_modelos, carpeta_predicciones, carpeta_metricas]:
        os.makedirs(carpeta, exist_ok=True)
    
    # Obtener combinaciones
    combinaciones = df.groupby(['Terminal_Destino', 'Producto']).size().reset_index(name='count')
    
    st.info(f"📊 Se entrenarán {len(combinaciones)} modelos")
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    modelos_exitosos = []
    todas_metricas = []
    
    for idx, row in combinaciones.iterrows():
        terminal = row['Terminal_Destino']
        producto = row['Producto']
        
        progreso = (idx + 1) / len(combinaciones)
        progress_bar.progress(progreso)
        status_text.text(f"Entrenando {idx+1}/{len(combinaciones)}: {terminal} - {producto}")
        
        df_filtrado = df[
            (df['Terminal_Destino'] == terminal) & 
            (df['Producto'] == producto)
        ].copy()
        
        m_final, forecast_final, forecast_validacion, metricas = entrenar_modelo_prophet_validado(
            df_filtrado, terminal, producto, Prophet
        )
        
        if m_final is not None:
            nombre_archivo = f"{terminal}_{producto}".replace(' ', '_').replace('/', '-').replace('.', '')
            
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
            modelos_exitosos.append(f"{terminal} - {producto}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Guardar métricas
    if todas_metricas:
        df_metricas = pd.DataFrame(todas_metricas)
        archivo_metricas = os.path.join(carpeta_metricas, 'resumen_metricas_validadas.csv')
        df_metricas.to_csv(archivo_metricas, index=False)
        
        st.success(f"✅ {len(modelos_exitosos)} modelos entrenados exitosamente")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 MAE Promedio", f"{df_metricas['MAE_Test'].mean():.2f}")
        with col2:
            st.metric("📈 RMSE Promedio", f"{df_metricas['RMSE_Test'].mean():.2f}")
        
        return True
    
    return False

def run():
    """Función principal del módulo de pronóstico"""
    
    st.markdown("## 📈 Pronóstico de Entregas")
    st.markdown("---")
    
    # Cargar datos
    df = cargar_datos()
    if df is None:
        return
    
    # Verificar si hay modelos entrenados
    modelos_existen = verificar_modelos_entrenados()
    
    if not modelos_existen:
        st.warning("⚠️ No se detectaron modelos entrenados")
        st.info("🤖 Es necesario entrenar los modelos Prophet antes de continuar")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Entrenar Modelos Ahora", type="primary", use_container_width=True):
                with st.spinner("Entrenando modelos... Esto puede tomar varios minutos..."):
                    exito = entrenar_todos_modelos(df)
                    if exito:
                        st.balloons()
                        st.success("🎉 ¡Modelos entrenados exitosamente!")
                        st.info("🔄 Recarga la página para usar los pronósticos")
        return
    
    # Si hay modelos, mostrar tabs normales
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Explorar Modelos", 
        "🎯 Hacer Predicción",
        "📊 Dashboard General",
        "📉 Métricas de Modelos"
    ])
    
    # TAB 1: EXPLORAR MODELOS
    with tab1:
        st.markdown("### 🔍 Explorar Modelos Entrenados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            terminales = sorted(df['Terminal_Destino'].unique())
            terminal_sel = st.selectbox("🏢 Selecciona Terminal:", terminales)
        
        with col2:
            df_filtrado = df[df['Terminal_Destino'] == terminal_sel]
            productos = sorted(df_filtrado['Producto'].unique())
            producto_sel = st.selectbox("📦 Selecciona Producto:", productos)
        
        if st.button("📊 Cargar Análisis", type="primary", use_container_width=True):
            with st.spinner("Cargando modelo y predicciones..."):
                mostrar_analisis_completo(df, terminal_sel, producto_sel)
    
    # TAB 2: HACER PREDICCIÓN
    with tab2:
        st.markdown("### 🎯 Generar Nueva Predicción")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            terminales = sorted(df['Terminal_Destino'].unique())
            terminal_pred = st.selectbox("🏢 Terminal:", terminales, key="pred_terminal")
        
        with col2:
            df_filtrado = df[df['Terminal_Destino'] == terminal_pred]
            productos = sorted(df_filtrado['Producto'].unique())
            producto_pred = st.selectbox("📦 Producto:", productos, key="pred_producto")
        
        with col3:
            dias_pred = st.number_input(
                "📅 Días a predecir:",
                min_value=1,
                max_value=90,
                value=42,
                step=1
            )
        
        if st.button("🚀 Generar Predicción", type="primary", use_container_width=True):
            with st.spinner("Generando predicción..."):
                generar_prediccion_personalizada(df, terminal_pred, producto_pred, dias_pred)
    
    # TAB 3: DASHBOARD GENERAL
    with tab3:
        st.markdown("### 📊 Dashboard General de Pronósticos")
        mostrar_dashboard_general(df)
    
    # TAB 4: MÉTRICAS DE MODELOS
    with tab4:
        st.markdown("### 📉 Métricas de Validación de Modelos")
        mostrar_metricas_modelos()
    
    # Botón para re-entrenar
    st.markdown("---")
    with st.expander("⚙️ Opciones Avanzadas"):
        st.warning("⚠️ Re-entrenar eliminará los modelos actuales y creará nuevos")
        if st.button("🔄 Re-entrenar Todos los Modelos", type="secondary"):
            with st.spinner("Re-entrenando modelos..."):
                exito = entrenar_todos_modelos(df)
                if exito:
                    st.success("✅ Modelos re-entrenados exitosamente")
                    st.info("🔄 Recarga la página para ver los cambios")


def cargar_datos():
    """Carga el dataset principal"""
    try:
        df = pd.read_csv('DataSet_Entregas.csv')
        df['Fecha_Entrega'] = pd.to_datetime(df['Fecha_Entrega'])
        st.success(f"✅ Dataset cargado: {len(df):,} registros")
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'DataSet_Entregas.csv'")
        st.info("📁 Asegúrate de que el archivo CSV esté en la misma carpeta que main.py")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")
        return None


def cargar_modelo(terminal, producto):
    """Carga un modelo Prophet previamente entrenado"""
    nombre_archivo = f"{terminal}_{producto}".replace(' ', '_').replace('/', '-').replace('.', '')
    ruta_modelo = Path(f'modelos_prophet_validados/modelos/{nombre_archivo}.pkl')
    
    if ruta_modelo.exists():
        with open(ruta_modelo, 'rb') as f:
            return pickle.load(f)
    return None


def cargar_predicciones(terminal, producto):
    """Carga las predicciones guardadas de 6 semanas"""
    nombre_archivo = f"{terminal}_{producto}".replace(' ', '_').replace('/', '-').replace('.', '')
    ruta_pred = Path(f'modelos_prophet_validados/predicciones/{nombre_archivo}_6semanas.csv')
    
    if ruta_pred.exists():
        df = pd.read_csv(ruta_pred)
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    return None


def cargar_metricas():
    """Carga el resumen de métricas de todos los modelos"""
    ruta_metricas = Path('modelos_prophet_validados/metricas/resumen_metricas_validadas.csv')
    
    if ruta_metricas.exists():
        df = pd.read_csv(ruta_metricas)
        if 'Fecha_Inicio' in df.columns:
            df['Fecha_Inicio'] = pd.to_datetime(df['Fecha_Inicio'])
        if 'Fecha_Fin' in df.columns:
            df['Fecha_Fin'] = pd.to_datetime(df['Fecha_Fin'])
        return df
    return None


def mostrar_analisis_completo(df, terminal, producto):
    """Muestra análisis completo de un modelo"""
    
    modelo = cargar_modelo(terminal, producto)
    predicciones = cargar_predicciones(terminal, producto)
    
    if modelo is None:
        st.error(f"❌ No se encontró modelo entrenado para {terminal} - {producto}")
        st.info("💡 El modelo puede no haberse entrenado por datos insuficientes")
        return
    
    if predicciones is None:
        st.warning("⚠️ No se encontraron predicciones guardadas")
        return
    
    df_hist = df[(df['Terminal_Destino'] == terminal) & (df['Producto'] == producto)].copy()
    
    # Cargar métricas específicas
    metricas_df = cargar_metricas()
    metricas_modelo = None
    if metricas_df is not None:
        metricas_modelo = metricas_df[
            (metricas_df['Terminal'] == terminal) & 
            (metricas_df['Producto'] == producto)
        ]
        if len(metricas_modelo) > 0:
            metricas_modelo = metricas_modelo.iloc[0]
    
    st.success(f"✅ Modelo cargado exitosamente")
    
    # Métricas principales en cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Registros Históricos", f"{len(df_hist):,}")
    
    with col2:
        demanda_promedio = predicciones['yhat'].mean()
        st.metric("📊 Demanda Promedio Predicha", f"{demanda_promedio:.1f}", 
                 delta="próximas 6 semanas")
    
    with col3:
        demanda_total = predicciones['yhat'].sum()
        st.metric("📈 Demanda Total Predicha", f"{demanda_total:.0f}", 
                 delta="6 semanas")
    
    with col4:
        fecha_inicio = df_hist['Fecha_Entrega'].min()
        fecha_fin = df_hist['Fecha_Entrega'].max()
        dias_historico = (fecha_fin - fecha_inicio).days
        st.metric("📅 Días de Histórico", f"{dias_historico}",
                 delta=f"{fecha_inicio.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    # Gráfico principal - GRANDE
    st.markdown("### 📊 Pronóstico de Demanda - 6 Semanas Futuras")
    
    # Necesitamos generar el forecast completo para mostrar histórico + futuro
    try:
        future = modelo.make_future_dataframe(periods=42)
        forecast_completo = modelo.predict(future)
        
        fig_principal = crear_grafico_principal_completo(df_hist, forecast_completo, terminal, producto)
        st.plotly_chart(fig_principal, use_container_width=True)
    except:
        # Si falla, usar solo las predicciones
        fig_principal = crear_grafico_historico_futuro(df_hist, predicciones, terminal, producto)
        st.plotly_chart(fig_principal, use_container_width=True)
    
    st.markdown("---")
    
    # Segunda fila: Validación y Análisis de Residuos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 Validación en Test Set (últimos 42 días)")
        if metricas_modelo is not None:
            try:
                fig_validacion = crear_grafico_validacion(df_hist, forecast_completo, metricas_modelo)
                st.plotly_chart(fig_validacion, use_container_width=True)
            except:
                st.info("ℹ️ Métricas de validación no disponibles")
        else:
            st.info("ℹ️ Métricas de validación no disponibles")
    
    with col2:
        st.markdown("### 🎯 Análisis de Residuos - Test Set")
        if metricas_modelo is not None:
            try:
                fig_residuos = crear_grafico_residuos(df_hist, forecast_completo, metricas_modelo)
                st.plotly_chart(fig_residuos, use_container_width=True)
            except:
                st.info("ℹ️ Análisis de residuos no disponible")
        else:
            st.info("ℹ️ Análisis de residuos no disponible")
    
    st.markdown("---")
    
    # Tercera fila: Componentes del modelo
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔧 Componentes del Modelo")
        fig_componentes = crear_grafico_componentes(forecast_completo)
        st.plotly_chart(fig_componentes, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Métricas y Configuración")
        mostrar_panel_metricas(metricas_modelo, predicciones, df_hist)
    
    st.markdown("---")
    
    # Gráficos adicionales en fila
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Distribución de Predicciones")
        fig_dist = crear_grafico_distribucion(predicciones)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Intervalos de Confianza")
        fig_ic = crear_grafico_intervalos(predicciones)
        st.plotly_chart(fig_ic, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla de predicciones
    st.markdown("### 📋 Tabla de Predicciones Detalladas")
    df_tabla = predicciones[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    df_tabla.columns = ['Fecha', 'Predicción', 'Límite Inferior (80%)', 'Límite Superior (80%)']
    df_tabla['Fecha'] = df_tabla['Fecha'].dt.strftime('%Y-%m-%d')
    df_tabla['Predicción'] = df_tabla['Predicción'].round(1)
    df_tabla['Límite Inferior (80%)'] = df_tabla['Límite Inferior (80%)'].round(1)
    df_tabla['Límite Superior (80%)'] = df_tabla['Límite Superior (80%)'].round(1)
    
    st.dataframe(df_tabla, use_container_width=True, height=400)
    
    # Botón de descarga
    csv = df_tabla.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Predicciones (CSV)",
        data=csv,
        file_name=f"predicciones_{terminal}_{producto}.csv",
        mime="text/csv"
    )


def generar_prediccion_personalizada(df, terminal, producto, dias):
    """Genera una predicción personalizada con días específicos"""
    
    modelo = cargar_modelo(terminal, producto)
    
    if modelo is None:
        st.error(f"❌ No se encontró modelo entrenado para {terminal} - {producto}")
        return
    
    try:
        future = modelo.make_future_dataframe(periods=dias)
        forecast = modelo.predict(future)
        
        df_hist = df[(df['Terminal_Destino'] == terminal) & (df['Producto'] == producto)].copy()
        fecha_max = df_hist['Fecha_Entrega'].max()
        forecast_futuro = forecast[forecast['ds'] > fecha_max].copy()
        
        st.success(f"✅ Predicción generada para los próximos {dias} días")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Demanda Promedio", f"{forecast_futuro['yhat'].mean():.1f}")
        
        with col2:
            st.metric("📈 Demanda Total", f"{forecast_futuro['yhat'].sum():.0f}")
        
        with col3:
            st.metric("📅 Período", f"{dias} días")
        
        fig = crear_grafico_historico_futuro(df_hist, forecast, terminal, producto)
        st.plotly_chart(fig, use_container_width=True)
        
        df_tabla = forecast_futuro[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        df_tabla.columns = ['Fecha', 'Predicción', 'Límite Inferior', 'Límite Superior']
        df_tabla['Fecha'] = df_tabla['Fecha'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(df_tabla, use_container_width=True, height=400)
        
        csv = df_tabla.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Predicción",
            data=csv,
            file_name=f"prediccion_{terminal}_{producto}_{dias}dias.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error al generar predicción: {e}")


def mostrar_dashboard_general(df):
    """Muestra dashboard con resumen de todos los modelos"""
    
    metricas = cargar_metricas()
    
    if metricas is None:
        st.warning("⚠️ No se encontraron métricas de modelos")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Modelos Entrenados", len(metricas))
    
    with col2:
        mae_promedio = metricas['MAE_Test'].mean()
        st.metric("📊 MAE Promedio", f"{mae_promedio:.2f}")
    
    with col3:
        rmse_promedio = metricas['RMSE_Test'].mean()
        st.metric("📈 RMSE Promedio", f"{rmse_promedio:.2f}")
    
    with col4:
        demanda_total = metricas['Demanda_Total_Predicha_6sem'].sum()
        st.metric("📦 Demanda Total 6 sem", f"{demanda_total:,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Demanda Predicha por Terminal")
        fig = crear_grafico_demanda_terminal(metricas)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Distribución de MAE por Producto")
        fig = crear_grafico_mae_producto(metricas)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🏆 Top 10 Mejores Modelos (Menor MAE)")
    top10 = metricas.nsmallest(10, 'MAE_Test')[['Terminal', 'Producto', 'MAE_Test', 'RMSE_Test', 'Demanda_Total_Predicha_6sem']]
    top10_display = top10.copy()
    top10_display['MAE_Test'] = top10_display['MAE_Test'].round(2)
    top10_display['RMSE_Test'] = top10_display['RMSE_Test'].round(2)
    top10_display['Demanda_Total_Predicha_6sem'] = top10_display['Demanda_Total_Predicha_6sem'].round(0)
    st.dataframe(top10_display, use_container_width=True)


def mostrar_metricas_modelos():
    """Muestra tabla completa de métricas de todos los modelos"""
    
    metricas = cargar_metricas()
    
    if metricas is None:
        st.warning("⚠️ No se encontraron métricas de modelos")
        return
    
    st.markdown("#### 📋 Métricas Completas de Validación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        terminales = ['Todos'] + sorted(metricas['Terminal'].unique().tolist())
        terminal_filtro = st.selectbox("Filtrar por Terminal:", terminales)
    
    with col2:
        productos = ['Todos'] + sorted(metricas['Producto'].unique().tolist())
        producto_filtro = st.selectbox("Filtrar por Producto:", productos)
    
    df_filtrado = metricas.copy()
    if terminal_filtro != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Terminal'] == terminal_filtro]
    if producto_filtro != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Producto'] == producto_filtro]
    
    st.info(f"📊 Mostrando {len(df_filtrado)} modelos")
    
    columnas_mostrar = ['Terminal', 'Producto', 'MAE_Test', 'RMSE_Test', 
                       'Observaciones_Total', 'Demanda_Promedio_Historica',
                       'Demanda_Total_Predicha_6sem']
    
    df_display = df_filtrado[columnas_mostrar].copy()
    st.dataframe(df_display, use_container_width=True, height=600)
    
    csv = df_filtrado.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Métricas Completas (CSV)",
        data=csv,
        file_name="metricas_modelos_completas.csv",
        mime="text/csv"
    )


def crear_grafico_principal_completo(df_hist, forecast, terminal, producto):
    """Crea el gráfico principal con histórico completo y predicciones"""
    
    df_grouped = df_hist.groupby('Fecha_Entrega')['Dias_Transcurridos'].sum().reset_index()
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
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Demanda:</b> %{y:.0f}<extra></extra>'
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
        title=f"Pronóstico de Demanda - 6 Semanas Futuras",
        xaxis_title="Fecha",
        yaxis_title="Demanda (Días Transcurridos)",
        hovermode='x unified',
        height=550,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def crear_grafico_validacion(df_hist, forecast, metricas):
    """Crea gráfico de validación en test set CON predicciones e intervalo"""
    
    df_grouped = df_hist.groupby('Fecha_Entrega')['Dias_Transcurridos'].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    df_grouped = df_grouped.sort_values('ds')
    
    fecha_corte = metricas['Fecha_Corte_Validacion']
    train_data = df_grouped[df_grouped['ds'] <= fecha_corte]
    test_data = df_grouped[df_grouped['ds'] > fecha_corte]
    
    # Obtener predicciones del test set
    forecast_test = forecast[forecast['ds'] > fecha_corte].copy()
    forecast_test = forecast_test[forecast['ds'] <= df_grouped['ds'].max()]
    
    fig = go.Figure()
    
    # Train
    fig.add_trace(go.Scatter(
        x=train_data['ds'],
        y=train_data['y'],
        mode='markers',
        name='Train',
        marker=dict(size=3, color='#06A77D', opacity=0.5),
        hovertemplate='<b>Train:</b> %{y:.0f}<extra></extra>'
    ))
    
    # Test (real)
    fig.add_trace(go.Scatter(
        x=test_data['ds'],
        y=test_data['y'],
        mode='markers',
        name='Test (real)',
        marker=dict(size=6, color='#2E86AB', opacity=0.8),
        hovertemplate='<b>Real:</b> %{y:.0f}<extra></extra>'
    ))
    
    # Predicción en test
    fig.add_trace(go.Scatter(
        x=forecast_test['ds'],
        y=forecast_test['yhat'],
        mode='lines+markers',
        name='Predicción Test',
        line=dict(color='#FF6B35', width=2),
        marker=dict(size=5),
        hovertemplate='<b>Predicción:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Intervalo de confianza superior
    fig.add_trace(go.Scatter(
        x=forecast_test['ds'],
        y=forecast_test['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Intervalo de confianza inferior (con relleno)
    fig.add_trace(go.Scatter(
        x=forecast_test['ds'],
        y=forecast_test['yhat_lower'],
        mode='lines',
        name='IC 80%',
        fill='tonexty',
        fillcolor='rgba(255, 107, 53, 0.15)',
        line=dict(width=0),
        hovertemplate='<b>IC:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Línea de corte
    fig.add_shape(
        type="line",
        x0=fecha_corte, x1=fecha_corte,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Agregar texto con métricas
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
        yaxis_title="Demanda",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )
    
    return fig


def crear_grafico_residuos(df_hist, forecast, metricas):
    """Crea gráfico de análisis de residuos"""
    
    df_grouped = df_hist.groupby('Fecha_Entrega')['Dias_Transcurridos'].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    
    fecha_corte = metricas['Fecha_Corte_Validacion']
    test_data = df_grouped[df_grouped['ds'] > fecha_corte].copy()
    
    # Merge con predicciones
    forecast_test = forecast[forecast['ds'].isin(test_data['ds'])][['ds', 'yhat']]
    test_merged = test_data.merge(forecast_test, on='ds', how='left')
    test_merged['residuo'] = test_merged['y'] - test_merged['yhat']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_merged['yhat'],
        y=test_merged['residuo'],
        mode='markers',
        marker=dict(size=8, color='#00B4D8', opacity=0.6, line=dict(color='white', width=1)),
        hovertemplate='<b>Predicho:</b> %{x:.1f}<br><b>Residuo:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Línea en y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
    
    fig.update_layout(
        title="Análisis de Residuos - Test Set",
        xaxis_title="Valores Predichos (Test)",
        yaxis_title="Residuos",
        height=400,
        template="plotly_white"
    )
    
    return fig


def crear_grafico_componentes(forecast):
    """Crea gráfico de componentes del modelo"""
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Tendencia', 'Estacionalidad Anual', 'Estacionalidad Semanal'),
        vertical_spacing=0.1,
        row_heights=[0.33, 0.33, 0.33]
    )
    
    # Tendencia
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
    
    # Estacionalidad anual
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
    
    # Estacionalidad semanal
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


def mostrar_panel_metricas(metricas, predicciones, df_hist):
    """Muestra panel de métricas en formato similar al original"""
    
    if metricas is None:
        st.info("ℹ️ Métricas no disponibles")
        return
    
    # Panel de Métricas de Validación
    st.markdown("""
    <div style='background-color: #E8F4F8; padding: 15px; border-radius: 10px; border: 1px solid #00B4D8;'>
        <h4 style='margin-top: 0;'>📊 MÉTRICAS DE VALIDACIÓN (Test Set)</h4>
        <hr style='margin: 10px 0;'>
        <ul>
            <li><b>MAE:</b> {:.2f} unidades</li>
            <li><b>RMSE:</b> {:.2f} unidades</li>
        </ul>
        <p style='font-size: 12px; color: #555;'>ℹ️ Estas métricas muestran el error promedio al predecir 6 semanas</p>
    </div>
    """.format(metricas['MAE_Test'], metricas['RMSE_Test']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Panel de Pronóstico
    st.markdown("""
    <div style='background-color: #FFF4E6; padding: 15px; border-radius: 10px; border: 1px solid #FF6B35;'>
        <h4 style='margin-top: 0;'>📈 PRONÓSTICO 6 SEMANAS</h4>
        <hr style='margin: 10px 0;'>
        <ul>
            <li><b>Demanda promedio semanal:</b> {:.1f}</li>
            <li><b>Demanda total 6 semanas:</b> {:.1f}</li>
        </ul>
    </div>
    """.format(metricas['Demanda_Promedio_Predicha_6sem'], metricas['Demanda_Total_Predicha_6sem']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Panel de Datos Históricos
    st.markdown("""
    <div style='background-color: #F0F8FF; padding: 15px; border-radius: 10px; border: 1px solid #2E86AB;'>
        <h4 style='margin-top: 0;'>📦 DATOS HISTÓRICOS</h4>
        <hr style='margin: 10px 0;'>
        <ul>
            <li><b>Período:</b> {} a {}</li>
            <li><b>Total días:</b> {}</li>
            <li><b>Train:</b> {} días | <b>Test:</b> {} días</li>
            <li><b>Demanda promedio histórica:</b> {:.1f}</li>
            <li><b>Desviación estándar:</b> {:.1f}</li>
        </ul>
    </div>
    """.format(
        metricas['Fecha_Inicio'].strftime('%Y-%m-%d'),
        metricas['Fecha_Fin'].strftime('%Y-%m-%d'),
        metricas['Observaciones_Total'],
        metricas['Observaciones_Train'],
        metricas['Observaciones_Test'],
        metricas['Demanda_Promedio_Historica'],
        metricas['Desviacion_Std_Historica']
    ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Panel de Configuración Prophet
    st.markdown("""
    <div style='background-color: #F5F0FF; padding: 15px; border-radius: 10px; border: 1px solid #9B59B6;'>
        <h4 style='margin-top: 0;'>⚙️ CONFIGURACIÓN PROPHET</h4>
        <hr style='margin: 10px 0;'>
        <ul>
            <li>✅ Estacionalidad anual</li>
            <li>✅ Estacionalidad semanal</li>
            <li>✅ Días festivos Colombia</li>
            <li>✅ Intervalos confianza: 80%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def crear_grafico_historico_futuro(df_hist, forecast, terminal, producto):
    """Crea gráfico interactivo con histórico y pronóstico"""
    
    df_grouped = df_hist.groupby('Fecha_Entrega')['Dias_Transcurridos'].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    
    fecha_max = df_grouped['ds'].max()
    
    # Verificar si forecast tiene las predicciones futuras o es solo el dataframe de predicciones
    if 'yhat' in forecast.columns:
        forecast_hist = forecast[forecast['ds'] <= fecha_max].copy()
        forecast_future = forecast[forecast['ds'] > fecha_max].copy()
    else:
        # Si es directamente las predicciones futuras
        forecast_hist = pd.DataFrame()
        forecast_future = forecast
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_grouped['ds'],
        y=df_grouped['y'],
        mode='markers',
        name='Datos Históricos',
        marker=dict(size=6, color='#2E86AB', opacity=0.7),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Demanda:</b> %{y:.0f}<extra></extra>'
    ))
    
    if len(forecast_hist) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_hist['ds'],
            y=forecast_hist['yhat'],
            mode='lines',
            name='Ajuste del Modelo',
            line=dict(color='#06A77D', width=2),
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Ajuste:</b> %{y:.1f}<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines+markers',
        name='Predicción',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Predicción:</b> %{y:.1f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_upper'],
        mode='lines',
        name='IC Superior (80%)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_lower'],
        mode='lines',
        name='Intervalo Confianza 80%',
        fill='tonexty',
        fillcolor='rgba(255, 107, 53, 0.2)',
        line=dict(width=0),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>IC Inferior:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Agregar línea vertical usando add_shape (más compatible con fechas)
    fig.add_shape(
        type="line",
        x0=fecha_max,
        x1=fecha_max,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Agregar anotación
    fig.add_annotation(
        x=fecha_max,
        y=1,
        yref="paper",
        text="Inicio Predicción",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="red"),
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
    
    fig.update_layout(
        title=f"Pronóstico - {terminal} | {producto}",
        xaxis_title="Fecha",
        yaxis_title="Demanda (Días Transcurridos)",
        hovermode='x unified',
        height=500,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def crear_grafico_distribucion(predicciones):
    """Crea histograma de distribución de predicciones"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=predicciones['yhat'],
        nbinsx=20,
        name='Distribución',
        marker_color='#06A77D',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribución de Predicciones",
        xaxis_title="Demanda Predicha",
        yaxis_title="Frecuencia",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def crear_grafico_intervalos(predicciones):
    """Crea gráfico de barras con intervalos de confianza"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=predicciones['ds'],
        y=predicciones['yhat'],
        name='Predicción',
        marker_color='#FF6B35',
        error_y=dict(
            type='data',
            symmetric=False,
            array=predicciones['yhat_upper'] - predicciones['yhat'],
            arrayminus=predicciones['yhat'] - predicciones['yhat_lower']
        )
    ))
    
    fig.update_layout(
        title="Predicciones con Intervalos de Confianza",
        xaxis_title="Fecha",
        yaxis_title="Demanda",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def crear_grafico_demanda_terminal(metricas):
    """Crea gráfico de demanda total por terminal"""
    
    demanda_terminal = metricas.groupby('Terminal')['Demanda_Total_Predicha_6sem'].sum().reset_index()
    demanda_terminal = demanda_terminal.sort_values('Demanda_Total_Predicha_6sem', ascending=False)
    
    fig = px.bar(
        demanda_terminal,
        x='Terminal',
        y='Demanda_Total_Predicha_6sem',
        title='Demanda Total Predicha por Terminal (6 semanas)',
        color='Demanda_Total_Predicha_6sem',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Terminal",
        yaxis_title="Demanda Total",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def crear_grafico_mae_producto(metricas):
    """Crea boxplot de MAE por producto"""
    
    fig = px.box(
        metricas,
        y='Producto',
        x='MAE_Test',
        orientation='h',
        title='Distribución de MAE por Producto',
        color='Producto'
    )
    
    fig.update_layout(
        xaxis_title="MAE (Error Absoluto Medio)",
        yaxis_title="Producto",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig