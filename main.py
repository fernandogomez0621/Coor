import streamlit as st
import sys
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Análisis Logístico",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .coming-soon {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">🚚 Sistema de Análisis Logístico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Plataforma Integral de Análisis y Predicción</div>', unsafe_allow_html=True)

# Sidebar con navegación
st.sidebar.title("📊 Navegación")
st.sidebar.markdown("---")

# Menú de opciones
menu_option = st.sidebar.radio(
    "Seleccione un módulo:",
    [
        "🏠 Inicio",
        "📊 Análisis Descriptivo",
        "📈 Pronóstico de Entregas",
        "👥 Crecimiento por Cliente",
        "⏱️ Puntualidad de Entregas",
        "🚛 Dimensionamiento de Flota"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **Sistema de Análisis Logístico v1.0**
    
    Desarrollado con:
    - Python 3.x
    - Streamlit
    - Prophet
    - Scikit-learn
    - XGBoost
    - Pandas
    - Plotly
""")

# ==========================================
# PÁGINA DE INICIO
# ==========================================
if menu_option == "🏠 Inicio":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 👋 Bienvenido al Sistema de Análisis Logístico")
        st.markdown("""
        Esta plataforma proporciona herramientas avanzadas de análisis y predicción 
        para optimizar las operaciones logísticas de tu empresa.
        
        #### 🎯 Módulos Disponibles:
        
        **1. 📊 Análisis Descriptivo** ✅
        - Diagramas Sankey de flujos logísticos
        - Análisis de productos por terminal
        - Rankings de desempeño y puntualidad
        - Análisis de volumen y capacidad
        
        **2. 📈 Pronóstico de Entregas** ✅
        - Predicciones basadas en modelos Prophet
        - Análisis de demanda por terminal y producto
        - Intervalos de confianza y métricas de validación
        - Visualizaciones interactivas
        
        **3. 👥 Crecimiento por Cliente** ✅
        - Proyecciones de crecimiento a 6 semanas
        - Análisis de demanda por cliente
        - Identificación de clientes clave
        - Tendencias y estacionalidad
        
        **4. ⏱️ Puntualidad de Entregas** ✅
        - Predicción de entregas a tiempo vs tardías
        - Modelos Random Forest y XGBoost
        - Análisis de factores de riesgo
        - Métricas completas de clasificación
        
        **5. 🚛 Dimensionamiento de Flota** ✅
        - Análisis por terminal de origen
        - Cálculo de vehículos adicionales necesarios
        - Proyección de incrementos de demanda
        - Optimización de capacidad vs demanda
        """)
        
        st.success("✅ Sistema inicializado correctamente")
    
    with col2:
        st.markdown("### 📊 Estado del Sistema")
        
        # Verificar archivo CSV de entregas
        csv_path = Path("DataSet_Entregas.csv")
        if csv_path.exists():
            st.success("✅ Dataset Entregas cargado")
        else:
            st.error("❌ Dataset Entregas no encontrado")
        
        # Verificar archivo CSV de móviles
        csv_moviles_path = Path("Data_Set_Moviles.csv")
        if csv_moviles_path.exists():
            st.success("✅ Dataset Móviles cargado")
        else:
            st.error("❌ Dataset Móviles no encontrado")
        
        # Verificar carpeta de modelos de entregas
        models_path = Path("modelos_prophet_validados/modelos")
        if models_path.exists():
            pkl_files = list(models_path.glob("*.pkl"))
            st.success(f"✅ {len(pkl_files)} modelos de entregas")
        else:
            st.warning("⚠️ Modelos de entregas no encontrados")
        
        # Verificar carpeta de modelos de clientes
        models_clientes_path = Path("modelos_clientes_prophet/modelos")
        if models_clientes_path.exists():
            pkl_files_clientes = list(models_clientes_path.glob("*.pkl"))
            st.success(f"✅ {len(pkl_files_clientes)} modelos de clientes")
        else:
            st.warning("⚠️ Modelos de clientes no encontrados")
        
        # Verificar modelos de puntualidad
        models_puntualidad_path = Path("modelos_puntualidad")
        if models_puntualidad_path.exists():
            rf_model = models_puntualidad_path / "random_forest_puntualidad.pkl"
            xgb_model = models_puntualidad_path / "xgboost_puntualidad.pkl"
            if rf_model.exists() and xgb_model.exists():
                st.success("✅ Modelos de puntualidad")
            else:
                st.warning("⚠️ Modelos de puntualidad no encontrados")
        else:
            st.warning("⚠️ Modelos de puntualidad no encontrados")
        
        st.markdown("---")
        st.markdown("### 🚀 Inicio Rápido")
        st.markdown("""
        1. Selecciona un módulo del menú lateral
        2. Explora los análisis disponibles
        3. Genera predicciones personalizadas
        """)

# ==========================================
# MÓDULO 1: ANÁLISIS DESCRIPTIVO
# ==========================================
elif menu_option == "📊 Análisis Descriptivo":
    try:
        from modules import analisis_descriptivo
        analisis_descriptivo.run()
    except ImportError as e:
        st.error(f"❌ Error al cargar el módulo: {e}")
        st.info("Asegúrate de que el archivo 'modules/analisis_descriptivo.py' exista")

# ==========================================
# MÓDULO 2: PRONÓSTICO DE ENTREGAS
# ==========================================
elif menu_option == "📈 Pronóstico de Entregas":
    try:
        from modules import pronostico_entregas
        pronostico_entregas.run()
    except ImportError as e:
        st.error(f"❌ Error al cargar el módulo: {e}")
        st.info("Asegúrate de que el archivo 'modules/pronostico_entregas.py' exista")

# ==========================================
# MÓDULO 3: CRECIMIENTO POR CLIENTE
# ==========================================
elif menu_option == "👥 Crecimiento por Cliente":
    try:
        from modules import crecimiento_clientes
        crecimiento_clientes.run()
    except ImportError as e:
        st.error(f"❌ Error al cargar el módulo: {e}")
        st.info("Asegúrate de que el archivo 'modules/crecimiento_clientes.py' exista")

# ==========================================
# MÓDULO 4: PUNTUALIDAD DE ENTREGAS
# ==========================================
elif menu_option == "⏱️ Puntualidad de Entregas":
    try:
        from modules import puntualidad_entregas
        puntualidad_entregas.run()
    except ImportError as e:
        st.error(f"❌ Error al cargar el módulo: {e}")
        st.info("Asegúrate de que el archivo 'modules/puntualidad_entregas.py' exista")

# ==========================================
# MÓDULO 5: DIMENSIONAMIENTO DE FLOTA
# ==========================================
elif menu_option == "🚛 Dimensionamiento de Flota":
    try:
        from modules import dimensionamiento_flota
        dimensionamiento_flota.run()
    except ImportError as e:
        st.error(f"❌ Error al cargar el módulo: {e}")
        st.info("Asegúrate de que el archivo 'modules/dimensionamiento_flota.py' exista")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Sistema de Análisis Logístico | Powered by Streamlit, Prophet & ML</p>
        <p>© 2025 - Todos los derechos reservados</p>
    </div>
""", unsafe_allow_html=True)