import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def run():
    st.markdown("## 📊 Análisis Descriptivo de Operaciones")
    st.markdown("Visualización integral de flujos logísticos, productos y desempeño")
    
    # Cargar datos
    csv_path = Path("DataSet_Entregas.csv")
    
    if not csv_path.exists():
        st.error("❌ No se encontró el archivo DataSet_Entregas.csv")
        return
    
    try:
        df = pd.read_csv(csv_path)
        
        # Limpieza básica
        df['Fecha_Recogida'] = pd.to_datetime(df['Fecha_Recogida'], errors='coerce')
        df['Fecha_Entrega'] = pd.to_datetime(df['Fecha_Entrega'], errors='coerce')
        
        # Clasificación de puntualidad
        df['Estado_Entrega'] = df.apply(
            lambda x: 'A Tiempo' if x['Dias_Transcurridos'] <= x['Dias_Ofrecidos'] else 'Tardía',
            axis=1
        )
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entregas", f"{len(df):,}")
        with col2:
            st.metric("Terminales Origen", df['Terminal_Origen'].nunique())
        with col3:
            st.metric("Terminales Destino", df['Terminal_Destino'].nunique())
        with col4:
            puntualidad = (df['Estado_Entrega'] == 'A Tiempo').mean() * 100
            st.metric("% Puntualidad", f"{puntualidad:.1f}%")
        
        st.markdown("---")
        
        # Tabs para diferentes análisis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔄 Flujo Origen-Destino",
            "📦 Análisis de Productos",
            "🎯 Desempeño por Terminal",
            "📈 Análisis de Volumen",
            "⏱️ Análisis de Puntualidad"
        ])
        
        # ==========================================
        # TAB 1: FLUJO ORIGEN-DESTINO (SANKEY)
        # ==========================================
        with tab1:
            st.markdown("### 🔄 Diagrama Sankey: Terminal Origen → Terminal Destino")
            
            # Opciones de filtro
            col1, col2 = st.columns(2)
            with col1:
                top_n = st.slider("Top N rutas más frecuentes", 10, 100, 30, 5)
            with col2:
                metric_sankey = st.selectbox(
                    "Métrica para grosor",
                    ["Cantidad de Entregas", "Total Unidades", "Total Peso"]
                )
            
            # Agrupar datos
            if metric_sankey == "Cantidad de Entregas":
                flujo = df.groupby(['Terminal_Origen', 'Terminal_Destino']).size().reset_index(name='Valor')
            elif metric_sankey == "Total Unidades":
                flujo = df.groupby(['Terminal_Origen', 'Terminal_Destino'])['Unidades'].sum().reset_index(name='Valor')
            else:
                flujo = df.groupby(['Terminal_Origen', 'Terminal_Destino'])['Peso'].sum().reset_index(name='Valor')
            
            flujo = flujo.nlargest(top_n, 'Valor')
            
            # Crear listas separadas para origen y destino
            terminales_origen = sorted(flujo['Terminal_Origen'].unique())
            terminales_destino = sorted(flujo['Terminal_Destino'].unique())
            
            # Crear etiquetas con prefijo para diferenciar
            labels_origen = [f"📤 {term}" for term in terminales_origen]
            labels_destino = [f"📥 {term}" for term in terminales_destino]
            
            # Combinar todas las etiquetas
            all_labels = labels_origen + labels_destino
            
            # Crear diccionario de mapeo
            node_dict = {}
            for idx, terminal in enumerate(terminales_origen):
                node_dict[('origen', terminal)] = idx
            for idx, terminal in enumerate(terminales_destino):
                node_dict[('destino', terminal)] = idx + len(terminales_origen)
            
            # Mapear source y target
            source_indices = []
            target_indices = []
            values = []
            
            for _, row in flujo.iterrows():
                source_idx = node_dict[('origen', row['Terminal_Origen'])]
                target_idx = node_dict[('destino', row['Terminal_Destino'])]
                source_indices.append(source_idx)
                target_indices.append(target_idx)
                values.append(row['Valor'])
            
            # Colores diferenciados
            node_colors = ['rgba(100, 149, 237, 0.8)'] * len(terminales_origen) + \
                         ['rgba(255, 140, 0, 0.8)'] * len(terminales_destino)
            
            # Crear Sankey
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=25,
                    line=dict(color="white", width=2),
                    label=all_labels,
                    color=node_colors,
                    x=[0.01] * len(terminales_origen) + [0.99] * len(terminales_destino),
                    y=[i/(len(terminales_origen)-1) if len(terminales_origen) > 1 else 0.5 
                       for i in range(len(terminales_origen))] + 
                      [i/(len(terminales_destino)-1) if len(terminales_destino) > 1 else 0.5 
                       for i in range(len(terminales_destino))]
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color='rgba(150, 150, 150, 0.3)'
                )
            )])
            
            fig_sankey.update_layout(
                title=f"Flujo Logístico por {metric_sankey}<br><sub>Azul: Origen | Naranja: Destino</sub>",
                font=dict(size=11),
                height=800,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            # Estadísticas del flujo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Terminales Origen", len(terminales_origen))
            with col2:
                st.metric("Terminales Destino", len(terminales_destino))
            with col3:
                st.metric("Rutas Mostradas", len(flujo))
            
            # Tabla de detalle
            st.markdown("#### 📋 Detalle de Rutas Principales")
            flujo_display = flujo[['Terminal_Origen', 'Terminal_Destino', 'Valor']].copy()
            flujo_display.columns = ['Terminal Origen', 'Terminal Destino', metric_sankey]
            flujo_display = flujo_display.sort_values(metric_sankey, ascending=False).reset_index(drop=True)
            flujo_display.index = flujo_display.index + 1
            st.dataframe(flujo_display, use_container_width=True, height=400)
        
        # ==========================================
        # TAB 2: ANÁLISIS DE PRODUCTOS
        # ==========================================
        with tab2:
            st.markdown("### 📦 Análisis de Productos")
            
            col1, col2 = st.columns(2)
            
            # Gráfico de porcentaje por producto
            with col1:
                productos = df['Producto'].value_counts()
                fig_productos_pct = px.pie(
                    values=productos.values,
                    names=productos.index,
                    title="Distribución Porcentual de Productos",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_productos_pct.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_productos_pct, use_container_width=True)
            
            # Gráfico de cantidad por producto
            with col2:
                fig_productos_cant = px.bar(
                    x=productos.index,
                    y=productos.values,
                    title="Cantidad de Entregas por Producto",
                    labels={'x': 'Producto', 'y': 'Cantidad de Entregas'},
                    color=productos.values,
                    color_continuous_scale='Blues'
                )
                fig_productos_cant.update_layout(showlegend=False)
                st.plotly_chart(fig_productos_cant, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🔄 Sankey: Productos → Terminales de Destino")
            
            # Control para el número de combinaciones
            top_combinaciones = st.slider("Top N combinaciones Producto-Terminal", 15, 60, 40, 5)
            
            # Sankey Producto -> Terminal Destino
            producto_terminal = df.groupby(['Producto', 'Terminal_Destino']).size().reset_index(name='Cantidad')
            producto_terminal = producto_terminal.nlargest(top_combinaciones, 'Cantidad')
            
            # Listas separadas
            productos_unicos = sorted(producto_terminal['Producto'].unique())
            terminales_dest_prod = sorted(producto_terminal['Terminal_Destino'].unique())
            
            # Etiquetas
            labels_productos = [f"📦 {prod}" for prod in productos_unicos]
            labels_term_dest = [f"📥 {term}" for term in terminales_dest_prod]
            
            nodos_pt = labels_productos + labels_term_dest
            
            # Diccionario de mapeo
            node_dict_pt = {}
            for idx, prod in enumerate(productos_unicos):
                node_dict_pt[('producto', prod)] = idx
            for idx, term in enumerate(terminales_dest_prod):
                node_dict_pt[('terminal', term)] = idx + len(productos_unicos)
            
            # Mapear
            source_pt = []
            target_pt = []
            values_pt = []
            
            for _, row in producto_terminal.iterrows():
                source_pt.append(node_dict_pt[('producto', row['Producto'])])
                target_pt.append(node_dict_pt[('terminal', row['Terminal_Destino'])])
                values_pt.append(row['Cantidad'])
            
            # Colores
            colores_nodos_pt = ['rgba(255, 99, 71, 0.8)'] * len(productos_unicos) + \
                              ['rgba(100, 149, 237, 0.8)'] * len(terminales_dest_prod)
            
            fig_sankey_prod = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=25,
                    line=dict(color="white", width=2),
                    label=nodos_pt,
                    color=colores_nodos_pt,
                    x=[0.01] * len(productos_unicos) + [0.99] * len(terminales_dest_prod),
                    y=[i/(len(productos_unicos)-1) if len(productos_unicos) > 1 else 0.5 
                       for i in range(len(productos_unicos))] + 
                      [i/(len(terminales_dest_prod)-1) if len(terminales_dest_prod) > 1 else 0.5 
                       for i in range(len(terminales_dest_prod))]
                ),
                link=dict(
                    source=source_pt,
                    target=target_pt,
                    value=values_pt,
                    color='rgba(200, 150, 150, 0.3)'
                )
            )])
            
            fig_sankey_prod.update_layout(
                title="Flujo: Productos → Terminales de Destino<br><sub>Rojo: Productos | Azul: Terminales</sub>",
                font=dict(size=11),
                height=700,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_sankey_prod, use_container_width=True)
            
            # Tabla resumen productos-terminales
            st.markdown("#### 📊 Top Combinaciones Producto-Terminal")
            top_comb = df.groupby(['Producto', 'Terminal_Destino']).agg({
                'ID': 'count',
                'Unidades': 'sum',
                'Peso': 'sum'
            }).reset_index()
            top_comb.columns = ['Producto', 'Terminal Destino', 'Entregas', 'Unidades', 'Peso Total']
            top_comb = top_comb.sort_values('Entregas', ascending=False).reset_index(drop=True)
            top_comb.index = top_comb.index + 1
            st.dataframe(top_comb.head(20), use_container_width=True, height=400)
        
        # ==========================================
        # TAB 3: DESEMPEÑO POR TERMINAL
        # ==========================================
        with tab3:
            st.markdown("### 🎯 Análisis de Desempeño por Terminal")
            
            col1, col2 = st.columns(2)
            
            # TOP terminales de destino con mejores llegadas a tiempo
            with col1:
                st.markdown("#### ✅ TOP 10: Mejor Puntualidad (Destino)")
                
                # Filtrar terminales con mínimo de entregas
                min_entregas = st.slider("Mínimo de entregas para considerar", 5, 50, 10, 5, key='min_punt')
                
                terminal_counts = df.groupby('Terminal_Destino').size()
                terminales_validas = terminal_counts[terminal_counts >= min_entregas].index
                
                df_filtered = df[df['Terminal_Destino'].isin(terminales_validas)]
                
                terminal_puntualidad = df_filtered.groupby('Terminal_Destino').agg({
                    'Estado_Entrega': lambda x: (x == 'A Tiempo').mean() * 100,
                    'ID': 'count'
                }).reset_index()
                terminal_puntualidad.columns = ['Terminal', 'Puntualidad_Pct', 'Total_Entregas']
                terminal_puntualidad = terminal_puntualidad.sort_values('Puntualidad_Pct', ascending=False).head(10)
                
                fig_top_puntual = px.bar(
                    terminal_puntualidad,
                    x='Puntualidad_Pct',
                    y='Terminal',
                    orientation='h',
                    title=f"Terminales con Mayor % de Entregas a Tiempo (≥{min_entregas} entregas)",
                    color='Puntualidad_Pct',
                    color_continuous_scale='Greens',
                    labels={'Puntualidad_Pct': '% Puntualidad'},
                    hover_data=['Total_Entregas']
                )
                fig_top_puntual.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_top_puntual, use_container_width=True)
                
                # Tabla
                st.dataframe(terminal_puntualidad.reset_index(drop=True), use_container_width=True)
            
            # TOP terminales de destino con más retrasos
            with col2:
                st.markdown("#### ❌ TOP 10: Mayor Retraso (Destino)")
                
                terminal_retraso = df_filtered.groupby('Terminal_Destino').agg({
                    'Estado_Entrega': lambda x: (x == 'Tardía').mean() * 100,
                    'ID': 'count'
                }).reset_index()
                terminal_retraso.columns = ['Terminal', 'Retraso_Pct', 'Total_Entregas']
                terminal_retraso = terminal_retraso.sort_values('Retraso_Pct', ascending=False).head(10)
                
                fig_top_retraso = px.bar(
                    terminal_retraso,
                    x='Retraso_Pct',
                    y='Terminal',
                    orientation='h',
                    title=f"Terminales con Mayor % de Entregas Tardías (≥{min_entregas} entregas)",
                    color='Retraso_Pct',
                    color_continuous_scale='Reds',
                    labels={'Retraso_Pct': '% Entregas Tardías'},
                    hover_data=['Total_Entregas']
                )
                fig_top_retraso.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_top_retraso, use_container_width=True)
                
                # Tabla
                st.dataframe(terminal_retraso.reset_index(drop=True), use_container_width=True)
            
            st.markdown("---")
            
            # Desempeño por terminal de origen
            st.markdown("#### 📤 Desempeño por Terminal de Origen")
            
            origen_stats = df.groupby('Terminal_Origen').agg({
                'ID': 'count',
                'Estado_Entrega': lambda x: (x == 'A Tiempo').mean() * 100,
                'Unidades': 'sum',
                'Peso': 'sum'
            }).reset_index()
            origen_stats.columns = ['Terminal', 'Total_Entregas', 'Puntualidad_%', 'Total_Unidades', 'Peso_Total']
            origen_stats = origen_stats.sort_values('Total_Entregas', ascending=False)
            
            fig_origen = px.scatter(
                origen_stats,
                x='Total_Entregas',
                y='Puntualidad_%',
                size='Total_Unidades',
                color='Puntualidad_%',
                hover_name='Terminal',
                title="Volumen vs Puntualidad por Terminal de Origen",
                labels={'Total_Entregas': 'Total de Entregas', 'Puntualidad_%': '% Puntualidad'},
                color_continuous_scale='RdYlGn',
                size_max=60
            )
            fig_origen.update_layout(height=500)
            st.plotly_chart(fig_origen, use_container_width=True)
            
            st.dataframe(origen_stats.reset_index(drop=True), use_container_width=True, height=400)
        
        # ==========================================
        # TAB 4: ANÁLISIS DE VOLUMEN
        # ==========================================
        with tab4:
            st.markdown("### 📈 Análisis de Volumen y Capacidad")
            
            col1, col2 = st.columns(2)
            
            # Unidades por terminal destino
            with col1:
                unidades_terminal = df.groupby('Terminal_Destino')['Unidades'].sum().sort_values(ascending=False).head(15)
                fig_unidades = px.bar(
                    x=unidades_terminal.values,
                    y=unidades_terminal.index,
                    orientation='h',
                    title="TOP 15: Unidades Entregadas por Terminal",
                    labels={'x': 'Total Unidades', 'y': 'Terminal'},
                    color=unidades_terminal.values,
                    color_continuous_scale='Viridis'
                )
                fig_unidades.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                st.plotly_chart(fig_unidades, use_container_width=True)
            
            # Peso por terminal destino
            with col2:
                peso_terminal = df.groupby('Terminal_Destino')['Peso'].sum().sort_values(ascending=False).head(15)
                fig_peso = px.bar(
                    x=peso_terminal.values,
                    y=peso_terminal.index,
                    orientation='h',
                    title="TOP 15: Peso Total por Terminal (kg)",
                    labels={'x': 'Peso Total (kg)', 'y': 'Terminal'},
                    color=peso_terminal.values,
                    color_continuous_scale='Oranges'
                )
                fig_peso.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                st.plotly_chart(fig_peso, use_container_width=True)
            
            st.markdown("---")
            
            # Análisis temporal de volumen
            st.markdown("#### 📅 Evolución Temporal del Volumen")
            
            df_temporal = df.copy()
            df_temporal['Mes'] = df_temporal['Fecha_Recogida'].dt.to_period('M').astype(str)
            
            volumen_temporal = df_temporal.groupby('Mes').agg({
                'ID': 'count',
                'Unidades': 'sum',
                'Peso': 'sum'
            }).reset_index()
            volumen_temporal.columns = ['Mes', 'Entregas', 'Unidades', 'Peso']
            
            fig_temporal = go.Figure()
            fig_temporal.add_trace(go.Scatter(
                x=volumen_temporal['Mes'],
                y=volumen_temporal['Entregas'],
                name='Entregas',
                mode='lines+markers',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            fig_temporal.add_trace(go.Scatter(
                x=volumen_temporal['Mes'],
                y=volumen_temporal['Unidades'],
                name='Unidades',
                mode='lines+markers',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ))
            
            fig_temporal.update_layout(
                title="Evolución de Entregas y Unidades por Mes",
                xaxis=dict(title='Mes'),
                yaxis=dict(title='Número de Entregas', side='left'),
                yaxis2=dict(title='Unidades', overlaying='y', side='right'),
                hovermode='x unified',
                height=500,
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Tabla resumen mensual
            st.markdown("#### 📋 Resumen Mensual")
            volumen_temporal['Peso'] = volumen_temporal['Peso'].round(2)
            st.dataframe(volumen_temporal, use_container_width=True)
        
        # ==========================================
        # TAB 5: ANÁLISIS DE PUNTUALIDAD
        # ==========================================
        with tab5:
            st.markdown("### ⏱️ Análisis Detallado de Puntualidad")
            
            col1, col2 = st.columns(2)
            
            # Distribución general
            with col1:
                estado_counts = df['Estado_Entrega'].value_counts()
                fig_estado = px.pie(
                    values=estado_counts.values,
                    names=estado_counts.index,
                    title="Distribución Global de Puntualidad",
                    color=estado_counts.index,
                    color_discrete_map={'A Tiempo': '#28a745', 'Tardía': '#dc3545'},
                    hole=0.4
                )
                fig_estado.update_traces(textposition='inside', textinfo='percent+label+value')
                st.plotly_chart(fig_estado, use_container_width=True)
            
            # Días de retraso promedio
            with col2:
                df['Dias_Retraso'] = df['Dias_Transcurridos'] - df['Dias_Ofrecidos']
                df['Dias_Retraso'] = df['Dias_Retraso'].clip(lower=0)
                
                retraso_terminal = df[df['Estado_Entrega'] == 'Tardía'].groupby('Terminal_Destino')['Dias_Retraso'].mean().sort_values(ascending=False).head(10)
                
                fig_retraso = px.bar(
                    x=retraso_terminal.values,
                    y=retraso_terminal.index,
                    orientation='h',
                    title="TOP 10: Días Promedio de Retraso",
                    labels={'x': 'Días de Retraso Promedio', 'y': 'Terminal'},
                    color=retraso_terminal.values,
                    color_continuous_scale='Reds'
                )
                fig_retraso.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                st.plotly_chart(fig_retraso, use_container_width=True)
            
            st.markdown("---")
            
            # Puntualidad por producto
            st.markdown("#### 📦 Puntualidad por Tipo de Producto")
            
            puntualidad_producto = df.groupby(['Producto', 'Estado_Entrega']).size().unstack(fill_value=0)
            puntualidad_producto['Total'] = puntualidad_producto.sum(axis=1)
            puntualidad_producto['% A Tiempo'] = (puntualidad_producto.get('A Tiempo', 0) / puntualidad_producto['Total'] * 100).round(2)
            
            # Preparar datos para el gráfico
            productos_list = puntualidad_producto.index.tolist()
            a_tiempo = puntualidad_producto.get('A Tiempo', pd.Series(0, index=puntualidad_producto.index)).tolist()
            tardia = puntualidad_producto.get('Tardía', pd.Series(0, index=puntualidad_producto.index)).tolist()
            
            fig_prod_punt = go.Figure()
            fig_prod_punt.add_trace(go.Bar(
                name='A Tiempo',
                x=productos_list,
                y=a_tiempo,
                marker_color='#28a745'
            ))
            fig_prod_punt.add_trace(go.Bar(
                name='Tardía',
                x=productos_list,
                y=tardia,
                marker_color='#dc3545'
            ))
            
            fig_prod_punt.update_layout(
                title="Entregas A Tiempo vs Tardías por Producto",
                xaxis_title='Producto',
                yaxis_title='Cantidad',
                barmode='stack',
                height=500,
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_prod_punt, use_container_width=True)
            
            # Tabla resumen
            puntualidad_producto_display = puntualidad_producto.reset_index()
            if 'A Tiempo' in puntualidad_producto_display.columns and 'Tardía' in puntualidad_producto_display.columns:
                puntualidad_producto_display = puntualidad_producto_display[['Producto', 'A Tiempo', 'Tardía', 'Total', '% A Tiempo']]
            st.dataframe(puntualidad_producto_display, use_container_width=True)
            
            st.markdown("---")
            
            # Heatmap de puntualidad por ruta
            st.markdown("#### 🗺️ Mapa de Calor: Puntualidad por Ruta")
            
            min_entregas_ruta = st.slider("Mínimo de entregas por ruta", 5, 50, 10, 5, key='min_ruta')
            
            ruta_puntualidad = df.groupby(['Terminal_Origen', 'Terminal_Destino']).agg({
                'Estado_Entrega': lambda x: (x == 'A Tiempo').mean() * 100,
                'ID': 'count'
            }).reset_index()
            ruta_puntualidad.columns = ['Origen', 'Destino', 'Puntualidad_%', 'Total']
            ruta_puntualidad = ruta_puntualidad[ruta_puntualidad['Total'] >= min_entregas_ruta]
            
            # Crear matriz pivot
            if len(ruta_puntualidad) > 0:
                matriz_puntualidad = ruta_puntualidad.pivot(index='Origen', columns='Destino', values='Puntualidad_%')
                
                fig_heatmap = px.imshow(
                    matriz_puntualidad,
                    labels=dict(x="Terminal Destino", y="Terminal Origen", color="% Puntualidad"),
                    color_continuous_scale='RdYlGn',
                    title=f"Puntualidad % por Ruta (mínimo {min_entregas_ruta} entregas)",
                    aspect='auto',
                    zmin=0,
                    zmax=100
                )
                fig_heatmap.update_layout(height=700)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Estadísticas del heatmap
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rutas Analizadas", len(ruta_puntualidad))
                with col2:
                    st.metric("Puntualidad Promedio", f"{ruta_puntualidad['Puntualidad_%'].mean():.1f}%")
                with col3:
                    st.metric("Total Entregas", f"{ruta_puntualidad['Total'].sum():,}")
            else:
                st.warning(f"No hay rutas con al menos {min_entregas_ruta} entregas para mostrar en el mapa de calor.")
        
    except Exception as e:
        st.error(f"❌ Error al procesar los datos: {e}")
        import traceback
        st.code(traceback.format_exc())