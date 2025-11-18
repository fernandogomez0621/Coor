import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def cargar_datos():
    """Carga los datasets de entregas y móviles"""
    try:
        # Cargar dataset de entregas
        df_entregas = pd.read_csv('DataSet_Entregas.csv')
        df_entregas['Fecha_Recogida'] = pd.to_datetime(df_entregas['Fecha_Recogida'])
        df_entregas['Fecha_Entrega'] = pd.to_datetime(df_entregas['Fecha_Entrega'])
        
        # Cargar dataset de móviles
        df_moviles = pd.read_csv('Data_Set_Moviles.csv')
        
        return df_entregas, df_moviles, True
    
    except FileNotFoundError as e:
        st.error(f"❌ No se encontró el archivo: {e}")
        st.info("📁 Asegúrate de que los archivos 'DataSet_Entregas.csv' y 'Data_Set_Moviles.csv' estén en la carpeta raíz")
        return None, None, False
    
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")
        return None, None, False


def calcular_demanda_terminal(df_entregas, terminal, fecha_inicio, fecha_fin):
    """
    Calcula la demanda de un terminal en un período específico
    Agrupa por día y suma las unidades
    """
    df_filtrado = df_entregas[
        (df_entregas['Terminal_Origen'] == terminal) &
        (df_entregas['Fecha_Recogida'] >= fecha_inicio) &
        (df_entregas['Fecha_Recogida'] <= fecha_fin)
    ].copy()
    
    if len(df_filtrado) == 0:
        return None
    
    # Agrupar por día
    demanda_diaria = df_filtrado.groupby('Fecha_Recogida').agg({
        'Unidades': 'sum',
        'ID': 'count'
    }).reset_index()
    
    demanda_diaria.columns = ['Fecha', 'Unidades', 'Num_Envios']
    
    return demanda_diaria


def obtener_flota_terminal(df_moviles, terminal):
    """Obtiene información de la flota de un terminal"""
    df_flota = df_moviles[df_moviles['Terminal'] == terminal].copy()
    
    if len(df_flota) == 0:
        return None
    
    resumen_flota = {
        'Total_Vehiculos': len(df_flota),
        'Capacidad_Total_MT3': df_flota['Capacidad'].sum(),
        'Capacidad_Promedio_MT3': df_flota['Capacidad'].mean(),
        'Capacidad_Min_MT3': df_flota['Capacidad'].min(),
        'Capacidad_Max_MT3': df_flota['Capacidad'].max(),
        'Detalles': df_flota
    }
    
    return resumen_flota


def calcular_vehiculos_adicionales(demanda_maxima_unidades, flota_info, incremento_porcentaje):
    """
    Calcula cuántos vehículos adicionales se necesitan basado en:
    1. Día de máxima demanda en UNIDADES
    2. Asume que ese día se usó 100% de la capacidad
    3. Calcula MT3 por unidad
    4. Proyecta incremento del 20%
    """
    # Capacidad actual de la flota
    capacidad_total_mt3 = flota_info['Capacidad_Total_MT3']
    capacidad_promedio_mt3 = flota_info['Capacidad_Promedio_MT3']
    vehiculos_actuales = flota_info['Total_Vehiculos']
    
    # PASO 1: Calcular MT3 por unidad
    # Asumimos que el día de máxima demanda se usó el 100% de la capacidad
    mt3_por_unidad = capacidad_total_mt3 / demanda_maxima_unidades
    
    # PASO 2: Calcular demanda proyectada con incremento
    demanda_proyectada_unidades = demanda_maxima_unidades * (1 + incremento_porcentaje / 100)
    incremento_unidades = demanda_proyectada_unidades - demanda_maxima_unidades
    
    # PASO 3: Calcular MT3 necesarios para la demanda proyectada
    mt3_necesarios = demanda_proyectada_unidades * mt3_por_unidad
    
    # PASO 4: Calcular MT3 adicionales necesarios
    mt3_adicionales = mt3_necesarios - capacidad_total_mt3
    
    # PASO 5: Calcular vehículos adicionales necesarios
    if mt3_adicionales <= 0:
        vehiculos_adicionales = 0
        capacidad_adicional_real = 0
    else:
        vehiculos_adicionales = np.ceil(mt3_adicionales / capacidad_promedio_mt3)
        capacidad_adicional_real = vehiculos_adicionales * capacidad_promedio_mt3
    
    # PASO 6: Calcular distribución optimizada
    vehiculos_optimizados = calcular_distribucion_optimizada(
        mt3_adicionales, 
        flota_info['Detalles']
    )
    
    resultados = {
        # Datos de entrada
        'Demanda_Maxima_Unidades': demanda_maxima_unidades,
        'Incremento_Porcentaje': incremento_porcentaje,
        
        # Capacidad actual
        'Capacidad_Total_MT3': capacidad_total_mt3,
        'Capacidad_Promedio_MT3': capacidad_promedio_mt3,
        'Vehiculos_Actuales': vehiculos_actuales,
        
        # Cálculos intermedios
        'MT3_por_Unidad': mt3_por_unidad,
        'Unidades_por_MT3': 1 / mt3_por_unidad if mt3_por_unidad > 0 else 0,
        
        # Proyección
        'Demanda_Proyectada_Unidades': demanda_proyectada_unidades,
        'Incremento_Unidades': incremento_unidades,
        'MT3_Necesarios': mt3_necesarios,
        'MT3_Adicionales': max(0, mt3_adicionales),
        
        # Vehículos necesarios
        'Vehiculos_Adicionales': int(vehiculos_adicionales),
        'Capacidad_Adicional_Real_MT3': capacidad_adicional_real,
        'Total_Vehiculos_Requeridos': int(vehiculos_actuales + vehiculos_adicionales),
        
        # Utilización
        'Utilizacion_Actual_Porcentaje': 100.0,  # Por definición, el día máximo usa 100%
        'Utilizacion_Proyectada_Porcentaje': (mt3_necesarios / (capacidad_total_mt3 + capacidad_adicional_real) * 100) if (capacidad_total_mt3 + capacidad_adicional_real) > 0 else 0,
        
        # Opción optimizada
        'Vehiculos_Optimizados': vehiculos_optimizados
    }
    
    return resultados


def calcular_distribucion_optimizada(mt3_adicionales, df_flota):
    """
    Calcula la distribución optimizada de vehículos adicionales
    """
    if mt3_adicionales <= 0:
        return []
    
    # Obtener capacidades únicas ordenadas de mayor a menor
    capacidades_unicas = sorted(df_flota['Capacidad'].unique(), reverse=True)
    
    mt3_restante = mt3_adicionales
    vehiculos_necesarios = []
    
    for cap in capacidades_unicas:
        if mt3_restante > 0:
            num_vehiculos = int(mt3_restante // cap)
            if num_vehiculos > 0:
                vehiculos_necesarios.append({
                    'Capacidad_MT3': cap,
                    'Cantidad': num_vehiculos,
                    'Total_MT3': cap * num_vehiculos
                })
                mt3_restante -= cap * num_vehiculos
    
    # Si aún falta capacidad, agregar un vehículo más
    if mt3_restante > 0:
        capacidad_mas_comun = df_flota['Capacidad'].mode()[0]
        # Verificar si ya existe en la lista
        encontrado = False
        for vehiculo in vehiculos_necesarios:
            if vehiculo['Capacidad_MT3'] == capacidad_mas_comun:
                vehiculo['Cantidad'] += 1
                vehiculo['Total_MT3'] += capacidad_mas_comun
                encontrado = True
                break
        
        if not encontrado:
            vehiculos_necesarios.append({
                'Capacidad_MT3': capacidad_mas_comun,
                'Cantidad': 1,
                'Total_MT3': capacidad_mas_comun
            })
    
    return vehiculos_necesarios


def crear_grafico_demanda_diaria(demanda_diaria, terminal):
    """Crea gráfico de demanda diaria con línea de máximo"""
    
    fig = go.Figure()
    
    # Línea de demanda diaria
    fig.add_trace(go.Scatter(
        x=demanda_diaria['Fecha'],
        y=demanda_diaria['Unidades'],
        mode='lines+markers',
        name='Demanda Diaria',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)',
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Unidades:</b> %{y:.0f}<extra></extra>'
    ))
    
    # Encontrar máximo
    max_unidades = demanda_diaria['Unidades'].max()
    fecha_max = demanda_diaria.loc[demanda_diaria['Unidades'].idxmax(), 'Fecha']
    promedio_unidades = demanda_diaria['Unidades'].mean()
    
    # Línea del máximo
    fig.add_hline(
        y=max_unidades,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Máximo: {max_unidades:.0f} unidades",
        annotation_position="right"
    )
    
    # Línea del promedio
    fig.add_hline(
        y=promedio_unidades,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text=f"Promedio: {promedio_unidades:.0f}",
        annotation_position="left"
    )
    
    # Marcar el punto máximo
    fig.add_trace(go.Scatter(
        x=[fecha_max],
        y=[max_unidades],
        mode='markers+text',
        name='Día Máximo',
        marker=dict(size=15, color='red', symbol='star'),
        text=[f'{max_unidades:.0f}'],
        textposition='top center',
        hovertemplate=f'<b>Día de mayor demanda</b><br>Fecha: {fecha_max.strftime("%Y-%m-%d")}<br>Unidades: {max_unidades:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Demanda Diaria de Unidades - {terminal}",
        xaxis_title="Fecha",
        yaxis_title="Unidades",
        height=450,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def crear_grafico_comparacion_capacidad(resultados):
    """Crea gráfico comparando capacidad actual vs proyectada"""
    
    fig = go.Figure()
    
    # Escenario Actual
    fig.add_trace(go.Bar(
        name='Unidades',
        x=['Actual', f'Proyectado (+{resultados["Incremento_Porcentaje"]}%)'],
        y=[resultados['Demanda_Maxima_Unidades'], resultados['Demanda_Proyectada_Unidades']],
        marker_color='#FF6B35',
        text=[f"{resultados['Demanda_Maxima_Unidades']:.0f}", f"{resultados['Demanda_Proyectada_Unidades']:.0f}"],
        textposition='outside',
        yaxis='y',
        hovertemplate='<b>Unidades:</b> %{y:.0f}<extra></extra>'
    ))
    
    # Capacidad
    fig.add_trace(go.Bar(
        name='Vehículos',
        x=['Actual', f'Proyectado (+{resultados["Incremento_Porcentaje"]}%)'],
        y=[resultados['Vehiculos_Actuales'], resultados['Total_Vehiculos_Requeridos']],
        marker_color='#06A77D',
        text=[f"{resultados['Vehiculos_Actuales']}", f"{resultados['Total_Vehiculos_Requeridos']}"],
        textposition='outside',
        yaxis='y2',
        hovertemplate='<b>Vehículos:</b> %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Comparación: Demanda y Flota",
        yaxis=dict(title='Unidades', side='left'),
        yaxis2=dict(title='Vehículos', overlaying='y', side='right'),
        barmode='group',
        height=400,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def crear_grafico_capacidad_mt3(resultados):
    """Crea gráfico de capacidad en MT3"""
    
    fig = go.Figure()
    
    # Capacidad actual
    fig.add_trace(go.Bar(
        name='Capacidad Actual',
        x=['Capacidad'],
        y=[resultados['Capacidad_Total_MT3']],
        marker_color='#2ca02c',
        text=[f"{resultados['Capacidad_Total_MT3']:.2f} MT3"],
        textposition='inside',
        hovertemplate='<b>Capacidad Actual:</b> %{y:.2f} MT3<extra></extra>'
    ))
    
    # Capacidad adicional
    if resultados['MT3_Adicionales'] > 0:
        fig.add_trace(go.Bar(
            name='Capacidad Adicional',
            x=['Capacidad'],
            y=[resultados['MT3_Adicionales']],
            marker_color='#d62728',
            text=[f"+{resultados['MT3_Adicionales']:.2f} MT3"],
            textposition='inside',
            hovertemplate='<b>Capacidad Adicional:</b> %{y:.2f} MT3<extra></extra>'
        ))
    
    fig.update_layout(
        title='Capacidad Total en MT3',
        yaxis_title='Capacidad (MT3)',
        barmode='stack',
        height=350,
        template="plotly_white",
        showlegend=True
    )
    
    return fig


def crear_grafico_distribucion_flota(flota_info):
    """Crea gráfico de distribución de capacidades de la flota"""
    
    df_flota = flota_info['Detalles']
    
    fig = px.histogram(
        df_flota,
        x='Capacidad',
        nbins=20,
        title='Distribución de Capacidades de la Flota Actual',
        labels={'Capacidad': 'Capacidad (MT3)', 'count': 'Cantidad de Vehículos'},
        color_discrete_sequence=['#2E86AB']
    )
    
    fig.update_layout(
        height=350,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def run():
    """Función principal del módulo de dimensionamiento de flota"""
    
    st.markdown("## 🚛 Dimensionamiento de Flota por Terminal")
    st.markdown("---")
    
    # Cargar datos
    df_entregas, df_moviles, success = cargar_datos()
    
    if not success:
        return
    
    st.success(f"✅ Datos cargados: {len(df_entregas):,} entregas | {len(df_moviles)} vehículos")
    
    st.markdown("---")
    
    # ==========================================
    # SECCIÓN 1: PARÁMETROS
    # ==========================================
    st.markdown("### ⚙️ Configuración de Análisis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Obtener terminales disponibles
        terminales_disponibles = sorted(df_entregas['Terminal_Origen'].dropna().unique())
        terminal_seleccionado = st.selectbox(
            "🏢 Selecciona Terminal:",
            options=terminales_disponibles,
            index=terminales_disponibles.index('9. Turbaco') if '9. Turbaco' in terminales_disponibles else 0
        )
    
    with col2:
        # Seleccionar mes de análisis
        fechas_disponibles = df_entregas['Fecha_Recogida'].dt.to_period('M').unique()
        fechas_disponibles = sorted([str(f) for f in fechas_disponibles], reverse=True)
        
        mes_seleccionado = st.selectbox(
            "📅 Mes de Análisis:",
            options=fechas_disponibles,
            index=0
        )
    
    with col3:
        # Porcentaje de incremento
        incremento_porcentaje = st.number_input(
            "📈 Incremento Proyectado (%):",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=5.0,
            help="Ingresa el porcentaje de incremento esperado en la demanda"
        )
    
    # Parsear mes seleccionado
    año_mes = pd.Period(mes_seleccionado)
    fecha_inicio = año_mes.to_timestamp()
    fecha_fin = año_mes.to_timestamp() + pd.offsets.MonthEnd(0)
    
    st.info(f"""
    📊 **Análisis configurado:**
    - **Terminal:** {terminal_seleccionado}
    - **Período:** {fecha_inicio.strftime('%Y-%m-%d')} a {fecha_fin.strftime('%Y-%m-%d')}
    - **Incremento proyectado:** {incremento_porcentaje}%
    
    💡 **Metodología:** Se identificará el día de máxima demanda en el período y se asumirá que ese día 
    la flota operó al 100% de su capacidad para calcular los MT3 por unidad.
    """)
    
    # Botón de análisis
    if st.button("🚀 Realizar Análisis", type="primary", use_container_width=True):
        
        with st.spinner("Calculando dimensionamiento de flota..."):
            
            # ==========================================
            # PASO 1: Calcular demanda del terminal
            # ==========================================
            demanda_diaria = calcular_demanda_terminal(df_entregas, terminal_seleccionado, fecha_inicio, fecha_fin)
            
            if demanda_diaria is None or len(demanda_diaria) == 0:
                st.error(f"❌ No se encontraron datos de entregas para {terminal_seleccionado} en el período seleccionado")
                return
            
            # Obtener día de máxima demanda
            demanda_maxima = demanda_diaria['Unidades'].max()
            fecha_maxima = demanda_diaria.loc[demanda_diaria['Unidades'].idxmax(), 'Fecha']
            demanda_promedio = demanda_diaria['Unidades'].mean()
            
            # ==========================================
            # PASO 2: Obtener información de flota
            # ==========================================
            flota_info = obtener_flota_terminal(df_moviles, terminal_seleccionado)
            
            if flota_info is None:
                st.error(f"❌ No se encontraron vehículos para el terminal {terminal_seleccionado}")
                return
            
            # ==========================================
            # PASO 3: Calcular vehículos adicionales
            # ==========================================
            resultados = calcular_vehiculos_adicionales(demanda_maxima, flota_info, incremento_porcentaje)
            
            # ==========================================
            # MOSTRAR RESULTADOS
            # ==========================================
            st.markdown("---")
            st.markdown("## 📊 Resultados del Análisis")
            
            # Supuesto clave
            st.info(f"""
            **🔑 Supuesto Clave del Análisis:**
            
            El día **{fecha_maxima.strftime('%Y-%m-%d')}** se registró la máxima demanda de **{demanda_maxima:,.0f} unidades**.
            Se asume que ese día la flota de **{resultados['Vehiculos_Actuales']} vehículos** operó al **100% de su capacidad** 
            ({resultados['Capacidad_Total_MT3']:.2f} MT3).
            
            Por lo tanto: **1 unidad = {resultados['MT3_por_Unidad']:.6f} MT3**
            """)
            
            # Métricas principales
            st.markdown("### 📈 Métricas Clave")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📦 Demanda Máxima",
                    f"{resultados['Demanda_Maxima_Unidades']:,.0f}",
                    delta=f"{fecha_maxima.strftime('%Y-%m-%d')}",
                    help="Día de mayor demanda en el período"
                )
            
            with col2:
                st.metric(
                    "📈 Demanda Proyectada",
                    f"{resultados['Demanda_Proyectada_Unidades']:,.0f}",
                    delta=f"+{resultados['Incremento_Unidades']:,.0f} ({incremento_porcentaje}%)",
                    delta_color="normal"
                )
            
            with col3:
                st.metric(
                    "🧮 MT3 por Unidad",
                    f"{resultados['MT3_por_Unidad']:.6f}",
                    delta=f"{resultados['Unidades_por_MT3']:.2f} unidades/MT3"
                )
            
            with col4:
                st.metric(
                    "🚛 Flota Actual",
                    f"{resultados['Vehiculos_Actuales']}",
                    delta=f"{resultados['Capacidad_Total_MT3']:.0f} MT3"
                )
            
            st.markdown("---")
            
            # Resultado principal - DESTACADO
            st.markdown("### 🎯 Resultado Principal")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if resultados['Vehiculos_Adicionales'] > 0:
                    st.success(f"""
                    ## ✅ Se necesitan **{resultados['Vehiculos_Adicionales']} vehículos adicionales**
                    
                    **Desglose del cálculo:**
                    - Unidades proyectadas: **{resultados['Demanda_Proyectada_Unidades']:,.0f}**
                    - MT3 necesarios: **{resultados['MT3_Necesarios']:.2f}**
                    - Capacidad actual: **{resultados['Capacidad_Total_MT3']:.2f} MT3**
                    - **Déficit: {resultados['MT3_Adicionales']:.2f} MT3**
                    
                    ---
                    
                    **Flota requerida:**
                    - Vehículos adicionales: **+{resultados['Vehiculos_Adicionales']}**
                    - Total vehículos: **{resultados['Total_Vehiculos_Requeridos']}**
                    - Nueva capacidad: **{resultados['Capacidad_Total_MT3'] + resultados['Capacidad_Adicional_Real_MT3']:.2f} MT3**
                    - Utilización proyectada: **{resultados['Utilizacion_Proyectada_Porcentaje']:.1f}%**
                    """)
                else:
                    st.info(f"""
                    ## ℹ️ No se necesitan vehículos adicionales
                    
                    La capacidad actual de **{resultados['Capacidad_Total_MT3']:.2f} MT3** 
                    ({resultados['Vehiculos_Actuales']} vehículos) es suficiente para soportar 
                    el incremento proyectado del **{incremento_porcentaje}%**.
                    
                    **Utilización proyectada:** {resultados['Utilizacion_Proyectada_Porcentaje']:.1f}%
                    """)
            
            st.markdown("---")
            
            # Gráficos de análisis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Demanda Diaria del Período")
                fig_demanda = crear_grafico_demanda_diaria(demanda_diaria, terminal_seleccionado)
                st.plotly_chart(fig_demanda, use_container_width=True)
                
                # Estadísticas del período
                with st.expander("📈 Ver estadísticas del período"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Días analizados", len(demanda_diaria))
                        st.metric("Máximo", f"{demanda_maxima:,.0f}")
                    with col_b:
                        st.metric("Promedio", f"{demanda_promedio:,.0f}")
                        st.metric("Mínimo", f"{demanda_diaria['Unidades'].min():,.0f}")
                    with col_c:
                        st.metric("Total", f"{demanda_diaria['Unidades'].sum():,.0f}")
                        st.metric("Desv. Std", f"{demanda_diaria['Unidades'].std():,.0f}")
            
            with col2:
                st.markdown("### 📈 Comparación de Escenarios")
                fig_comparacion = crear_grafico_comparacion_capacidad(resultados)
                st.plotly_chart(fig_comparacion, use_container_width=True)
                
                st.markdown("### 📦 Capacidad en MT3")
                fig_mt3 = crear_grafico_capacidad_mt3(resultados)
                st.plotly_chart(fig_mt3, use_container_width=True)
            
            # Opciones de dimensionamiento
            if resultados['Vehiculos_Adicionales'] > 0:
                st.markdown("---")
                st.markdown("### 🚛 Opciones de Dimensionamiento")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📌 Opción 1: Capacidad Promedio")
                    st.info("Agregar vehículos con la capacidad promedio de la flota actual")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Vehículos", f"+{resultados['Vehiculos_Adicionales']}")
                        st.metric("Capacidad/Vehículo", f"{resultados['Capacidad_Promedio_MT3']:.2f} MT3")
                    with col_b:
                        st.metric("Capacidad Total", f"{resultados['Capacidad_Adicional_Real_MT3']:.2f} MT3")
                        exceso1 = resultados['Capacidad_Adicional_Real_MT3'] - resultados['MT3_Adicionales']
                        st.metric("Exceso", f"{exceso1:.2f} MT3")
                
                with col2:
                    st.markdown("#### 📌 Opción 2: Distribución Optimizada")
                    st.info("Combinación óptima de vehículos de diferentes capacidades")
                    
                    if resultados['Vehiculos_Optimizados']:
                        df_opt = pd.DataFrame(resultados['Vehiculos_Optimizados'])
                        total_veh_opt = df_opt['Cantidad'].sum()
                        total_cap_opt = df_opt['Total_MT3'].sum()
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Vehículos", f"+{int(total_veh_opt)}")
                            st.metric("Capacidad Total", f"{total_cap_opt:.2f} MT3")
                        with col_b:
                            exceso2 = total_cap_opt - resultados['MT3_Adicionales']
                            st.metric("Exceso", f"{exceso2:.2f} MT3")
                            eficiencia = (1 - exceso2/resultados['MT3_Adicionales']) * 100 if resultados['MT3_Adicionales'] > 0 else 100
                            st.metric("Eficiencia", f"{eficiencia:.1f}%")
                        
                        st.markdown("**Desglose:**")
                        st.dataframe(
                            df_opt.style.format({
                                'Capacidad_MT3': '{:.2f}',
                                'Cantidad': '{:.0f}',
                                'Total_MT3': '{:.2f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Recomendación
                        if total_veh_opt < resultados['Vehiculos_Adicionales']:
                            st.success(f"✅ **Recomendado:** Opción 2 ahorra {int(resultados['Vehiculos_Adicionales'] - total_veh_opt)} vehículo(s)")
                        else:
                            st.info("ℹ️ Ambas opciones son equivalentes")
            
            # Distribución de flota actual
            st.markdown("---")
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### 🚛 Distribución de Flota Actual")
                fig_flota = crear_grafico_distribucion_flota(flota_info)
                st.plotly_chart(fig_flota, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 Resumen de Flota")
                st.markdown(f"""
                **Flota Actual:**
                - Total vehículos: **{flota_info['Total_Vehiculos']}**
                - Capacidad total: **{flota_info['Capacidad_Total_MT3']:.2f} MT3**
                - Capacidad promedio: **{flota_info['Capacidad_Promedio_MT3']:.2f} MT3**
                - Capacidad mínima: **{flota_info['Capacidad_Min_MT3']:.0f} MT3**
                - Capacidad máxima: **{flota_info['Capacidad_Max_MT3']:.0f} MT3**
                
                **Proyección (+{incremento_porcentaje}%):**
                - Vehículos adicionales: **+{resultados['Vehiculos_Adicionales']}**
                - Total requerido: **{resultados['Total_Vehiculos_Requeridos']}**
                - Nueva capacidad: **{resultados['Capacidad_Total_MT3'] + resultados['Capacidad_Adicional_Real_MT3']:.2f} MT3**
                - Incremento flota: **{(resultados['Vehiculos_Adicionales']/resultados['Vehiculos_Actuales']*100):.1f}%**
                """)
            
            st.markdown("---")
            
            # Análisis de Sensibilidad
            st.markdown("### 📊 Análisis de Sensibilidad")
            
            st.info("¿Cómo varía la necesidad de vehículos según diferentes niveles de incremento?")
            
            # Calcular para diferentes incrementos
            incrementos = list(range(5, 51, 5))
            datos_sensibilidad = []
            
            for inc in incrementos:
                unidades_proy = demanda_maxima * (1 + inc/100)
                mt3_nec = unidades_proy * resultados['MT3_por_Unidad']
                mt3_adic = mt3_nec - resultados['Capacidad_Total_MT3']
                veh_adic = np.ceil(mt3_adic / resultados['Capacidad_Promedio_MT3']) if mt3_adic > 0 else 0
                
                datos_sensibilidad.append({
                    'Incremento (%)': inc,
                    'Unidades': int(unidades_proy),
                    'MT3 Adicionales': round(max(0, mt3_adic), 2),
                    'Vehículos Adicionales': int(veh_adic)
                })
            
            df_sensibilidad = pd.DataFrame(datos_sensibilidad)
            
            # Gráfico de sensibilidad
            fig_sens = go.Figure()
            
            fig_sens.add_trace(go.Scatter(
                x=df_sensibilidad['Incremento (%)'],
                y=df_sensibilidad['Vehículos Adicionales'],
                mode='lines+markers',
                name='Vehículos Adicionales',
                line=dict(color='#FF6B35', width=3),
                marker=dict(size=10),
                hovertemplate='<b>Incremento:</b> %{x}%<br><b>Vehículos:</b> %{y}<extra></extra>'
            ))
            
            # Marcar el punto actual
            fig_sens.add_trace(go.Scatter(
                x=[incremento_porcentaje],
                y=[resultados['Vehiculos_Adicionales']],
                mode='markers',
                name='Escenario Actual',
                marker=dict(size=20, color='red', symbol='star'),
                hovertemplate=f'<b>Escenario Seleccionado</b><br>Incremento: {incremento_porcentaje}%<br>Vehículos: {resultados["Vehiculos_Adicionales"]}<extra></extra>'
            ))
            
            fig_sens.update_layout(
                title='Vehículos Adicionales vs Incremento de Demanda',
                xaxis_title='Incremento en Demanda (%)',
                yaxis_title='Vehículos Adicionales Necesarios',
                height=400,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_sens, use_container_width=True)
            
            # Tabla de sensibilidad
            with st.expander("📋 Ver tabla completa de sensibilidad"):
                st.dataframe(
                    df_sensibilidad.style.format({
                        'Incremento (%)': '{}%',
                        'Unidades': '{:,.0f}',
                        'MT3 Adicionales': '{:.2f}',
                        'Vehículos Adicionales': '{:.0f}'
                    }),
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Resumen Ejecutivo
            st.markdown("### 📋 Resumen Ejecutivo")
            
            st.success(f"""
            ### Terminal: **{terminal_seleccionado}**
            
            #### 📊 Situación Actual (Período: {mes_seleccionado})
            - **Demanda máxima registrada:** {demanda_maxima:,.0f} unidades ({fecha_maxima.strftime('%Y-%m-%d')})
            - **Demanda promedio diaria:** {demanda_promedio:,.0f} unidades
            - **Flota disponible:** {resultados['Vehiculos_Actuales']} vehículos
            - **Capacidad total:** {resultados['Capacidad_Total_MT3']:.2f} MT3
            - **Capacidad por unidad:** {resultados['MT3_por_Unidad']:.6f} MT3/unidad
            
            #### 🚀 Proyección con +{incremento_porcentaje}% de Incremento
            - **Unidades proyectadas:** {resultados['Demanda_Proyectada_Unidades']:,.0f} (+{resultados['Incremento_Unidades']:,.0f})
            - **MT3 necesarios:** {resultados['MT3_Necesarios']:.2f}
            - **Déficit de capacidad:** {resultados['MT3_Adicionales']:.2f} MT3
            
            #### 🚛 Recomendación de Flota
            - **Vehículos adicionales necesarios:** **{resultados['Vehiculos_Adicionales']}**
            - **Total vehículos requeridos:** {resultados['Total_Vehiculos_Requeridos']}
            - **Incremento de flota:** {(resultados['Vehiculos_Adicionales']/resultados['Vehiculos_Actuales']*100):.1f}%
            - **Nueva capacidad total:** {resultados['Capacidad_Total_MT3'] + resultados['Capacidad_Adicional_Real_MT3']:.2f} MT3
            """)
            
            # Tabla comparativa detallada
            st.markdown("#### 📊 Tabla Comparativa Detallada")
            
            df_resumen = pd.DataFrame({
                'Métrica': [
                    'Vehículos',
                    'Capacidad Total (MT3)',
                    'Unidades Máximas',
                    'MT3 por Unidad',
                    'Utilización (%)',
                    'Unidades por Vehículo'
                ],
                'Actual': [
                    f"{resultados['Vehiculos_Actuales']}",
                    f"{resultados['Capacidad_Total_MT3']:.2f}",
                    f"{resultados['Demanda_Maxima_Unidades']:,.0f}",
                    f"{resultados['MT3_por_Unidad']:.6f}",
                    "100.0%",
                    f"{resultados['Demanda_Maxima_Unidades']/resultados['Vehiculos_Actuales']:.0f}"
                ],
                f'Proyectado (+{incremento_porcentaje}%)': [
                    f"{resultados['Total_Vehiculos_Requeridos']}",
                    f"{resultados['Capacidad_Total_MT3'] + resultados['Capacidad_Adicional_Real_MT3']:.2f}",
                    f"{resultados['Demanda_Proyectada_Unidades']:,.0f}",
                    f"{resultados['MT3_por_Unidad']:.6f}",
                    f"{resultados['Utilizacion_Proyectada_Porcentaje']:.1f}%",
                    f"{resultados['Demanda_Proyectada_Unidades']/resultados['Total_Vehiculos_Requeridos']:.0f}"
                ],
                'Diferencia': [
                    f"+{resultados['Vehiculos_Adicionales']}",
                    f"+{resultados['Capacidad_Adicional_Real_MT3']:.2f}",
                    f"+{resultados['Incremento_Unidades']:,.0f}",
                    "0",
                    f"{resultados['Utilizacion_Proyectada_Porcentaje'] - 100:.1f}%",
                    f"+{(resultados['Demanda_Proyectada_Unidades']/resultados['Total_Vehiculos_Requeridos']) - (resultados['Demanda_Maxima_Unidades']/resultados['Vehiculos_Actuales']):.0f}"
                ]
            })
            
            st.dataframe(df_resumen, use_container_width=True, height=260)
            
            st.markdown("---")
            
            
            st.markdown("---")
            
            # Datos detallados
            st.markdown("### 📁 Datos Detallados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("📋 Ver Demanda Diaria Completa"):
                    st.dataframe(
                        demanda_diaria.style.format({
                            'Unidades': '{:,.0f}',
                            'Num_Envios': '{:.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Botón de descarga
                    csv = demanda_diaria.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar CSV",
                        data=csv,
                        file_name=f"demanda_diaria_{terminal_seleccionado.replace('.', '').replace(' ', '_')}_{mes_seleccionado}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                with st.expander("🚛 Ver Detalle de Flota"):
                    st.dataframe(
                        flota_info['Detalles'].style.format({
                            'Capacidad': '{:.2f}',
                            'Movil': '{}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Botón de descarga
                    csv_flota = flota_info['Detalles'].to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar CSV",
                        data=csv_flota,
                        file_name=f"flota_{terminal_seleccionado.replace('.', '').replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
            
            
            # Notas finales
            st.markdown("---")
            st.markdown("### 📌 Notas Metodológicas")
            
            st.info("""
            **Supuestos del Análisis:**
            
            1. **Capacidad al 100%:** Se asume que el día de máxima demanda histórica, la flota operó al 100% de su capacidad.
            
            2. **Distribución uniforme:** El cálculo de MT3 por unidad asume que todas las unidades requieren espacio similar.
            
            3. **Proyección lineal:** Se asume crecimiento lineal en la demanda según el porcentaje especificado.
            
            4. **Capacidad promedio:** Los cálculos usan la capacidad promedio de la flota para estimar vehículos adicionales.
            
            5. **Sin estacionalidad:** No se consideran variaciones estacionales en el cálculo base (revisar datos históricos para ajustar).
            
            **Recomendaciones para validar:**
            - Verificar utilización real de la flota en el día de máxima demanda
            - Considerar variaciones estacionales y días especiales
            - Validar el supuesto de MT3 por unidad con datos operativos
            - Evaluar factores externos que puedan afectar la demanda
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><b>Módulo de Dimensionamiento de Flota</b> | Análisis basado en demanda histórica</p>
            <p style='font-size: 0.9em;'>Metodología: Día de máxima demanda = 100% utilización de capacidad</p>
        </div>
    """, unsafe_allow_html=True)