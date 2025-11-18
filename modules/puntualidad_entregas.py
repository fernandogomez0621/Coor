import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def verificar_modelos_puntualidad():
    """Verifica si existen modelos de puntualidad entrenados"""
    carpeta_modelos = Path('modelos_puntualidad')
    if not carpeta_modelos.exists():
        return False
    
    rf_model = carpeta_modelos / "random_forest_puntualidad.pkl"
    xgb_model = carpeta_modelos / "xgboost_puntualidad.pkl"
    
    return rf_model.exists() and xgb_model.exists()


def preparar_datos_puntualidad(df):
    """
    Prepara los datos para el modelo de puntualidad
    - Crea la columna de clasificación (A_Tiempo / Tardia)
    - Limpia columnas innecesarias
    - Codifica variables categóricas
    """
    st.info("🔄 Preparando datos...")
    
    # Hacer copia
    df_prep = df.copy()
    
    # 1. CREAR COLUMNA DE CLASIFICACIÓN
    df_prep['Puntualidad'] = df_prep.apply(
        lambda row: 'A_Tiempo' if row['Dias_Transcurridos'] <= row['Dias_Ofrecidos'] else 'Tardia',
        axis=1
    )
    
    # 2. SELECCIONAR SOLO COLUMNAS DE INTERÉS
    columnas_necesarias = [
        'Terminal_Origen', 'Terminal_Destino', 'Producto',
        'Dias_Ofrecidos', 'Peso', 'Peso_Volumen', 'Unidades',
        'Puntualidad'
    ]
    
    # Filtrar solo columnas que existan en el dataframe
    columnas_existentes = [col for col in columnas_necesarias if col in df_prep.columns]
    df_prep = df_prep[columnas_existentes].copy()
    
    # Verificar que tengamos las columnas mínimas necesarias
    columnas_minimas = ['Terminal_Origen', 'Terminal_Destino', 'Producto', 'Puntualidad']
    if not all(col in df_prep.columns for col in columnas_minimas):
        st.error("❌ El dataset no tiene las columnas mínimas requeridas")
        return None, None, None
    
    # 3. ELIMINAR REGISTROS CON VALORES NULOS
    df_prep = df_prep.dropna()
    
    # 4. CODIFICAR VARIABLES CATEGÓRICAS
    label_encoders = {}
    columnas_categoricas = ['Terminal_Origen', 'Terminal_Destino', 'Producto']
    
    for col in columnas_categoricas:
        if col in df_prep.columns:
            le = LabelEncoder()
            df_prep[f'{col}_encoded'] = le.fit_transform(df_prep[col].astype(str))
            label_encoders[col] = le
    
    # 5. PREPARAR X e y
    columnas_features = []
    
    # Agregar categóricas codificadas
    for col in columnas_categoricas:
        if col in df_prep.columns:
            columnas_features.append(f'{col}_encoded')
    
    # Agregar numéricas
    columnas_numericas = ['Dias_Ofrecidos', 'Peso', 'Peso_Volumen', 'Unidades']
    for col in columnas_numericas:
        if col in df_prep.columns:
            columnas_features.append(col)
    
    X = df_prep[columnas_features]
    y = df_prep['Puntualidad']
    
    # Información de preparación
    st.success(f"✅ Datos preparados: {len(df_prep):,} registros")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total registros", f"{len(df_prep):,}")
    with col2:
        a_tiempo = (y == 'A_Tiempo').sum()
        st.metric("✅ A Tiempo", f"{a_tiempo:,} ({a_tiempo/len(y)*100:.1f}%)")
    with col3:
        tardias = (y == 'Tardia').sum()
        st.metric("⏰ Tardías", f"{tardias:,} ({tardias/len(y)*100:.1f}%)")
    
    return X, y, label_encoders


def entrenar_modelos_puntualidad(X, y):
    """
    Entrena Random Forest y XGBoost para clasificación de puntualidad
    """
    if not XGBOOST_AVAILABLE:
        st.warning("⚠️ XGBoost no está instalado. Solo se entrenará Random Forest.")
    
    st.info("🤖 Entrenando modelos...")
    
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    st.write(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    resultados = {}
    
    # ============================================
    # RANDOM FOREST
    # ============================================
    st.markdown("### 🌲 Entrenando Random Forest...")
    progress_bar = st.progress(0)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    progress_bar.progress(50)
    
    # Predicciones RF
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Métricas RF
    resultados['RF'] = {
        'model': rf_model,
        'y_test': y_test,
        'y_pred': y_pred_rf,
        'y_pred_proba': y_pred_proba_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, pos_label='Tardia'),
        'recall': recall_score(y_test, y_pred_rf, pos_label='Tardia'),
        'f1': f1_score(y_test, y_pred_rf, pos_label='Tardia'),
        'auc': roc_auc_score((y_test == 'Tardia').astype(int), y_pred_proba_rf),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf, labels=['A_Tiempo', 'Tardia'])
    }
    
    progress_bar.progress(100)
    st.success("✅ Random Forest entrenado")
    
    # ============================================
    # XGBOOST
    # ============================================
    if XGBOOST_AVAILABLE:
        st.markdown("### ⚡ Entrenando XGBoost...")
        progress_bar = st.progress(0)
        
        # Convertir labels a numérico para XGBoost
        y_train_numeric = (y_train == 'Tardia').astype(int)
        y_test_numeric = (y_test == 'Tardia').astype(int)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=len(y_train_numeric[y_train_numeric==0])/len(y_train_numeric[y_train_numeric==1])
        )
        
        xgb_model.fit(X_train, y_train_numeric)
        progress_bar.progress(50)
        
        # Predicciones XGBoost
        y_pred_xgb_numeric = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        # Convertir predicciones de vuelta a labels
        y_pred_xgb = ['Tardia' if p == 1 else 'A_Tiempo' for p in y_pred_xgb_numeric]
        
        # Métricas XGBoost
        resultados['XGB'] = {
            'model': xgb_model,
            'y_test': y_test,
            'y_pred': y_pred_xgb,
            'y_pred_proba': y_pred_proba_xgb,
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'precision': precision_score(y_test, y_pred_xgb, pos_label='Tardia'),
            'recall': recall_score(y_test, y_pred_xgb, pos_label='Tardia'),
            'f1': f1_score(y_test, y_pred_xgb, pos_label='Tardia'),
            'auc': roc_auc_score(y_test_numeric, y_pred_proba_xgb),
            'confusion_matrix': confusion_matrix(y_test, y_pred_xgb, labels=['A_Tiempo', 'Tardia'])
        }
        
        progress_bar.progress(100)
        st.success("✅ XGBoost entrenado")
    
    return resultados, X_test, y_test


def guardar_modelos_puntualidad(resultados, label_encoders, feature_names):
    """Guarda los modelos entrenados y sus metadatos"""
    
    carpeta_modelos = Path('modelos_puntualidad')
    carpeta_modelos.mkdir(exist_ok=True)
    
    # Guardar Random Forest
    with open(carpeta_modelos / 'random_forest_puntualidad.pkl', 'wb') as f:
        pickle.dump(resultados['RF']['model'], f)
    
    # Guardar XGBoost si existe
    if 'XGB' in resultados:
        with open(carpeta_modelos / 'xgboost_puntualidad.pkl', 'wb') as f:
            pickle.dump(resultados['XGB']['model'], f)
    
    # Guardar label encoders
    with open(carpeta_modelos / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Guardar feature names
    with open(carpeta_modelos / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Guardar métricas
    metricas_df = pd.DataFrame({
        'Modelo': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'AUC': []
    })
    
    for modelo_name, datos in resultados.items():
        metricas_df = pd.concat([metricas_df, pd.DataFrame({
            'Modelo': [modelo_name],
            'Accuracy': [datos['accuracy']],
            'Precision': [datos['precision']],
            'Recall': [datos['recall']],
            'F1-Score': [datos['f1']],
            'AUC': [datos['auc']]
        })], ignore_index=True)
    
    metricas_df.to_csv(carpeta_modelos / 'metricas_resumen.csv', index=False)
    
    st.success("✅ Modelos guardados en la carpeta 'modelos_puntualidad/'")


def cargar_modelos_puntualidad():
    """Carga los modelos de puntualidad guardados"""
    carpeta_modelos = Path('modelos_puntualidad')
    
    modelos = {}
    
    # Cargar Random Forest
    rf_path = carpeta_modelos / 'random_forest_puntualidad.pkl'
    if rf_path.exists():
        with open(rf_path, 'rb') as f:
            modelos['RF'] = pickle.load(f)
    
    # Cargar XGBoost
    xgb_path = carpeta_modelos / 'xgboost_puntualidad.pkl'
    if xgb_path.exists():
        with open(xgb_path, 'rb') as f:
            modelos['XGB'] = pickle.load(f)
    
    # Cargar label encoders
    le_path = carpeta_modelos / 'label_encoders.pkl'
    if le_path.exists():
        with open(le_path, 'rb') as f:
            modelos['label_encoders'] = pickle.load(f)
    
    # Cargar feature names
    fn_path = carpeta_modelos / 'feature_names.pkl'
    if fn_path.exists():
        with open(fn_path, 'rb') as f:
            modelos['feature_names'] = pickle.load(f)
    
    return modelos


def mostrar_metricas_detalladas(resultados):
    """Muestra las métricas de ambos modelos lado a lado"""
    
    st.markdown("## 📊 Comparación de Modelos")
    
    # Tabla comparativa
    metricas_comparacion = []
    
    for modelo_name in ['RF', 'XGB']:
        if modelo_name in resultados:
            datos = resultados[modelo_name]
            metricas_comparacion.append({
                'Modelo': 'Random Forest' if modelo_name == 'RF' else 'XGBoost',
                'Accuracy': f"{datos['accuracy']:.4f}",
                'Precision': f"{datos['precision']:.4f}",
                'Recall': f"{datos['recall']:.4f}",
                'F1-Score': f"{datos['f1']:.4f}",
                'AUC': f"{datos['auc']:.4f}"
            })
    
    df_comparacion = pd.DataFrame(metricas_comparacion)
    st.dataframe(df_comparacion, use_container_width=True)
    
    st.markdown("---")
    
    # Métricas detalladas por modelo
    cols = st.columns(len(resultados))
    
    for idx, (modelo_name, datos) in enumerate(resultados.items()):
        with cols[idx]:
            nombre_modelo = 'Random Forest' if modelo_name == 'RF' else 'XGBoost'
            st.markdown(f"### {nombre_modelo}")
            
            # Matriz de confusión
            st.markdown("#### 📋 Matriz de Confusión")
            fig_cm = crear_matriz_confusion(datos['confusion_matrix'], nombre_modelo)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Métricas en cards
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Accuracy", f"{datos['accuracy']:.4f}")
                st.metric("🎯 Precision", f"{datos['precision']:.4f}")
            with col2:
                st.metric("📊 Recall", f"{datos['recall']:.4f}")
                st.metric("⭐ F1-Score", f"{datos['f1']:.4f}")
            
            st.metric("📈 AUC-ROC", f"{datos['auc']:.4f}")
    
    st.markdown("---")
    
    # Curvas ROC juntas
    st.markdown("### 📈 Comparación de Curvas ROC")
    fig_roc = crear_curvas_roc_comparacion(resultados)
    st.plotly_chart(fig_roc, use_container_width=True)


def crear_matriz_confusion(cm, nombre_modelo):
    """Crea visualización de matriz de confusión"""
    
    labels = ['A_Tiempo', 'Tardia']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title=f"Matriz de Confusión - {nombre_modelo}",
        xaxis_title="Predicción",
        yaxis_title="Real",
        height=400,
        template="plotly_white"
    )
    
    return fig


def crear_curvas_roc_comparacion(resultados):
    """Crea gráfico comparativo de curvas ROC"""
    
    fig = go.Figure()
    
    colores = {'RF': '#06A77D', 'XGB': '#FF6B35'}
    nombres = {'RF': 'Random Forest', 'XGB': 'XGBoost'}
    
    for modelo_name, datos in resultados.items():
        y_test_numeric = (datos['y_test'] == 'Tardia').astype(int)
        fpr, tpr, _ = roc_curve(y_test_numeric, datos['y_pred_proba'])
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f"{nombres[modelo_name]} (AUC={datos['auc']:.4f})",
            line=dict(color=colores[modelo_name], width=3)
        ))
    
    # Línea diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Clasificador Aleatorio',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Curvas ROC - Comparación de Modelos",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        height=500,
        template="plotly_white",
        legend=dict(x=0.6, y=0.1)
    )
    
    return fig


def mostrar_importancia_features(resultados, feature_names):
    """Muestra la importancia de las features"""
    
    st.markdown("### 🔍 Importancia de Variables")
    
    cols = st.columns(len(resultados))
    
    for idx, (modelo_name, datos) in enumerate(resultados.items()):
        with cols[idx]:
            nombre_modelo = 'Random Forest' if modelo_name == 'RF' else 'XGBoost'
            
            # Obtener importancias
            if modelo_name == 'RF':
                importancias = datos['model'].feature_importances_
            else:
                importancias = datos['model'].feature_importances_
            
            # Crear dataframe
            df_importancia = pd.DataFrame({
                'Feature': feature_names,
                'Importancia': importancias
            }).sort_values('Importancia', ascending=False)
            
            # Gráfico
            fig = px.bar(
                df_importancia,
                x='Importancia',
                y='Feature',
                orientation='h',
                title=f"Importancia - {nombre_modelo}",
                color='Importancia',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                height=400,
                template="plotly_white",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)


def interfaz_prediccion(modelos):
    """Interfaz para hacer predicciones individuales"""
    
    st.markdown("### 🎯 Predicción Individual")
    
    st.info("Ingresa los datos de la entrega para predecir si llegará a tiempo o tarde")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📍 Ubicaciones")
        terminal_origen = st.selectbox(
            "Terminal Origen",
            options=list(modelos['label_encoders']['Terminal_Origen'].classes_)
        )
        terminal_destino = st.selectbox(
            "Terminal Destino",
            options=list(modelos['label_encoders']['Terminal_Destino'].classes_)
        )
        producto = st.selectbox(
            "Producto",
            options=list(modelos['label_encoders']['Producto'].classes_)
        )
    
    with col2:
        st.markdown("#### 📦 Detalles del Envío")
        dias_ofrecidos = st.number_input("Días Ofrecidos", min_value=1, max_value=14, value=3)
        peso = st.number_input("Peso (kg)", min_value=0.0, value=10.0, step=0.5)
        peso_volumen = st.number_input("Peso Volumétrico", min_value=0.0, value=5.0, step=0.1)
        unidades = st.number_input("Unidades", min_value=1, value=1, step=1)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        modelo_seleccionado = st.radio(
            "Selecciona el modelo:",
            options=['Random Forest', 'XGBoost'],
            horizontal=True
        )
    
    with col2:
        if st.button("🚀 Predecir", type="primary", use_container_width=True):
            hacer_prediccion(
                modelos,
                modelo_seleccionado,
                terminal_origen,
                terminal_destino,
                producto,
                dias_ofrecidos,
                peso,
                peso_volumen,
                unidades
            )


def hacer_prediccion(modelos, modelo_seleccionado, terminal_origen, terminal_destino, 
                    producto, dias_ofrecidos, peso, peso_volumen, unidades):
    """Realiza la predicción con el modelo seleccionado"""
    
    # Codificar variables categóricas
    terminal_origen_enc = modelos['label_encoders']['Terminal_Origen'].transform([terminal_origen])[0]
    terminal_destino_enc = modelos['label_encoders']['Terminal_Destino'].transform([terminal_destino])[0]
    producto_enc = modelos['label_encoders']['Producto'].transform([producto])[0]
    
    # Crear feature vector
    X_pred = pd.DataFrame({
        'Terminal_Origen_encoded': [terminal_origen_enc],
        'Terminal_Destino_encoded': [terminal_destino_enc],
        'Producto_encoded': [producto_enc],
        'Dias_Ofrecidos': [dias_ofrecidos],
        'Peso': [peso],
        'Peso_Volumen': [peso_volumen],
        'Unidades': [unidades]
    })
    
    # Asegurar orden correcto de columnas
    X_pred = X_pred[modelos['feature_names']]
    
    # Seleccionar modelo
    modelo_key = 'RF' if modelo_seleccionado == 'Random Forest' else 'XGB'
    modelo = modelos[modelo_key]
    
    # Hacer predicción
    if modelo_key == 'RF':
        prediccion = modelo.predict(X_pred)[0]
        probabilidad = modelo.predict_proba(X_pred)[0]
    else:
        prediccion_num = modelo.predict(X_pred)[0]
        prediccion = 'Tardia' if prediccion_num == 1 else 'A_Tiempo'
        probabilidad = modelo.predict_proba(X_pred)[0]
    
    # Mostrar resultado
    st.markdown("---")
    st.markdown("## 🎯 Resultado de la Predicción")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediccion == 'A_Tiempo':
            st.success("### ✅ La entrega llegará A TIEMPO")
            prob_a_tiempo = probabilidad[0] if modelo_key == 'RF' else probabilidad[0]
            st.metric("Probabilidad A Tiempo", f"{prob_a_tiempo*100:.1f}%")
        else:
            st.error("### ⏰ La entrega llegará TARDE")
            prob_tardia = probabilidad[1] if modelo_key == 'RF' else probabilidad[1]
            st.metric("Probabilidad Tardía", f"{prob_tardia*100:.1f}%")
        
        # Gráfico de probabilidades
        fig = go.Figure(go.Bar(
            x=['A Tiempo', 'Tardía'],
            y=[probabilidad[0], probabilidad[1]],
            marker_color=['#06A77D', '#FF6B35'],
            text=[f"{probabilidad[0]*100:.1f}%", f"{probabilidad[1]*100:.1f}%"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Distribución de Probabilidades",
            yaxis_title="Probabilidad",
            height=350,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


def run():
    """Función principal del módulo de puntualidad"""
    
    st.markdown("## ⏱️ Predicción de Puntualidad de Entregas")
    st.markdown("---")
    
    # Cargar datos
    try:
        df = pd.read_csv('DataSet_Entregas.csv')
        st.success(f"✅ Dataset cargado: {len(df):,} registros")
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'DataSet_Entregas.csv'")
        return
    
    # Verificar si hay modelos entrenados
    modelos_existen = verificar_modelos_puntualidad()
    
    if not modelos_existen:
        st.warning("⚠️ No se detectaron modelos de puntualidad entrenados")
        st.info("🤖 Es necesario entrenar los modelos antes de continuar")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Entrenar Modelos Ahora", type="primary", use_container_width=True):
                with st.spinner("Preparando datos y entrenando modelos..."):
                    
                    # Preparar datos
                    X, y, label_encoders = preparar_datos_puntualidad(df)
                    
                    if X is not None:
                        # Entrenar modelos
                        resultados, X_test, y_test = entrenar_modelos_puntualidad(X, y)
                        
                        # Guardar modelos
                        guardar_modelos_puntualidad(resultados, label_encoders, X.columns.tolist())
                        
                        # Mostrar resultados
                        st.markdown("---")
                        mostrar_metricas_detalladas(resultados)
                        mostrar_importancia_features(resultados, X.columns.tolist())
                        
                        st.balloons()
                        st.success("🎉 ¡Modelos entrenados exitosamente!")
                        st.info("🔄 Recarga la página para usar las predicciones")
        return
    
    # Si hay modelos, mostrar tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Hacer Predicción",
        "📊 Métricas de Modelos",
        "🔍 Importancia de Variables",
        "⚙️ Re-entrenar Modelos"
    ])
    
    # Cargar modelos
    modelos = cargar_modelos_puntualidad()
    
    # TAB 1: PREDICCIÓN
    with tab1:
        if 'RF' in modelos or 'XGB' in modelos:
            interfaz_prediccion(modelos)
        else:
            st.error("❌ No se pudieron cargar los modelos")
    
    # TAB 2: MÉTRICAS
    with tab2:
        st.markdown("### 📊 Métricas de los Modelos")
        
        metricas_path = Path('modelos_puntualidad/metricas_resumen.csv')
        if metricas_path.exists():
            df_metricas = pd.read_csv(metricas_path)
            
            # Mostrar tabla
            st.dataframe(df_metricas, use_container_width=True)
            
            # Gráfico comparativo
            df_melted = df_metricas.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
            fig = px.bar(
                df_melted,
                x='Métrica',
                y='Valor',
                color='Modelo',
                barmode='group',
                title='Comparación de Métricas entre Modelos'
            )
            
            fig.update_layout(
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ No se encontraron métricas guardadas")
    
    # TAB 3: IMPORTANCIA DE VARIABLES
    with tab3:
        st.markdown("### 🔍 Importancia de Variables")
        
        if 'RF' in modelos and 'XGB' in modelos:
            cols = st.columns(2)
            
            with cols[0]:
                st.markdown("#### 🌲 Random Forest")
                importancias_rf = modelos['RF'].feature_importances_
                df_imp_rf = pd.DataFrame({
                    'Variable': modelos['feature_names'],
                    'Importancia': importancias_rf
                }).sort_values('Importancia', ascending=False)
                
                fig_rf = px.bar(
                    df_imp_rf,
                    x='Importancia',
                    y='Variable',
                    orientation='h',
                    color='Importancia',
                    color_continuous_scale='Greens'
                )
                fig_rf.update_layout(height=400, template="plotly_white", showlegend=False)
                st.plotly_chart(fig_rf, use_container_width=True)
            
            with cols[1]:
                st.markdown("#### ⚡ XGBoost")
                importancias_xgb = modelos['XGB'].feature_importances_
                df_imp_xgb = pd.DataFrame({
                    'Variable': modelos['feature_names'],
                    'Importancia': importancias_xgb
                }).sort_values('Importancia', ascending=False)
                
                fig_xgb = px.bar(
                    df_imp_xgb,
                    x='Importancia',
                    y='Variable',
                    orientation='h',
                    color='Importancia',
                    color_continuous_scale='Oranges'
                )
                fig_xgb.update_layout(height=400, template="plotly_white", showlegend=False)
                st.plotly_chart(fig_xgb, use_container_width=True)
        
        elif 'RF' in modelos:
            st.markdown("#### 🌲 Random Forest")
            importancias_rf = modelos['RF'].feature_importances_
            df_imp_rf = pd.DataFrame({
                'Variable': modelos['feature_names'],
                'Importancia': importancias_rf
            }).sort_values('Importancia', ascending=False)
            
            fig_rf = px.bar(
                df_imp_rf,
                x='Importancia',
                y='Variable',
                orientation='h',
                color='Importancia',
                color_continuous_scale='Greens'
            )
            fig_rf.update_layout(height=500, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_rf, use_container_width=True)
        
        else:
            st.warning("⚠️ No se encontraron modelos cargados")
    
    # TAB 4: RE-ENTRENAR
    with tab4:
        st.markdown("### ⚙️ Re-entrenar Modelos")
        
        st.warning("⚠️ Re-entrenar eliminará los modelos actuales y creará nuevos")
        st.info("💡 Esto es útil si has actualizado el dataset o quieres cambiar parámetros")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔄 Re-entrenar Modelos", type="secondary", use_container_width=True):
                with st.spinner("Re-entrenando modelos..."):
                    
                    # Preparar datos
                    X, y, label_encoders = preparar_datos_puntualidad(df)
                    
                    if X is not None:
                        # Entrenar modelos
                        resultados, X_test, y_test = entrenar_modelos_puntualidad(X, y)
                        
                        # Guardar modelos
                        guardar_modelos_puntualidad(resultados, label_encoders, X.columns.tolist())
                        
                        # Mostrar resultados
                        st.markdown("---")
                        mostrar_metricas_detalladas(resultados)
                        mostrar_importancia_features(resultados, X.columns.tolist())
                        
                        st.success("✅ Modelos re-entrenados exitosamente")
                        st.info("🔄 Recarga la página para ver los cambios")
        
        st.markdown("---")
        
        # Información adicional
        with st.expander("ℹ️ Información sobre el Re-entrenamiento"):
            st.markdown("""
            ### ¿Cuándo re-entrenar los modelos?
            
            **Deberías re-entrenar cuando:**
            - 📊 Has agregado nuevos datos al dataset
            - 🔧 Quieres experimentar con diferentes parámetros
            - 📈 El rendimiento de los modelos ha disminuido
            - 🆕 Has realizado cambios en las columnas del dataset
            
            ### Configuración actual de modelos:
            
            **Random Forest:**
            - n_estimators: 100
            - max_depth: 10
            - min_samples_split: 10
            - min_samples_leaf: 5
            - class_weight: balanced
            
            **XGBoost:**
            - n_estimators: 100
            - max_depth: 6
            - learning_rate: 0.1
            - subsample: 0.8
            - colsample_bytree: 0.8
            - scale_pos_weight: automático (balanceo de clases)
            
            ### Variables utilizadas:
            - Terminal_Origen (categórica)
            - Terminal_Destino (categórica)
            - Producto (categórica)
            - Dias_Ofrecidos (numérica)
            - Peso (numérica)
            - Peso_Volumen (numérica)
            - Unidades (numérica)
            
            ### Variable objetivo:
            - **Puntualidad**: A_Tiempo (si Dias_Transcurridos ≤ Dias_Ofrecidos) o Tardia (en otro caso)
            """)
    
    # Footer con información
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><b>Módulo de Puntualidad de Entregas</b> | Random Forest & XGBoost</p>
            <p style='font-size: 0.9em;'>Predicción de entregas a tiempo vs tardías basada en Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)