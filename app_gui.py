# ======================================================================
# APLICACIÓN DE PRODUCCIÓN MoW (M9D^X) v3.0 (Demo + Clonar + Fixes)
# ======================================================================
# Autor: Gemini (Basado en la co-creación con el usuario)
# Stack v3.0:
# - GUI: Python, Tkinter, ttkbootstrap
# - Base de Datos: Driver intercambiable (SQLite / MySQL)
# - Modelos: Numpy, Pandas (para AHP, M9D)
# - ML: Scikit-learn (para MoW)
# - Grafos: NetworkX (para análisis de conexiones)
# - IA Cualitativa: Conexión Ollama (list, show, generate)
# - E/S: Exportación (PDF/CSV/JSON), Importación (CSV, JSON)
# - Producción: Threading y Queue (GUI no bloqueante)
# ----------------------------------------------------------------------
# FIX v3.0:
# 1. (Tu sugerencia) Añadido botón "Cargar Set de Proyectos Demo"
#    para resolver el problema de "arranque en frío" (cold start).
# 2. Corregido 'SyntaxError: unterminated string literal' en T_LABELS_SHORT.
# 3. Corregido 'AttributeError: prompt' usando simpledialog.
# 4. Corregido 'TclError: can't add ... as slave' en GUI.
# 5. Añadida la clase 'IAGenerator' (antes faltaba).
# 6. (Tu sugerencia) Añadido botón "Importar Proyecto (M9D JSON)".
# 7. (Tu sugerencia) Añadido botón "Clonar Proyecto Seleccionado".
# ======================================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Toplevel, Text, END, Scrollbar, Canvas, Frame
from tkinter import simpledialog # FIX v2.6
import ttkbootstrap as btk
from ttkbootstrap.scrolled import ScrolledFrame, ScrolledText
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import csv
import os
import threading
import queue
import requests  # Para Ollama
from typing import List, Dict, Tuple, Any

# --- Librerías de Base de Datos ---
import sqlalchemy as db
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, ForeignKey, Text as DBText

# --- Librerías de ML ---
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import networkx as nx

# --- Librerías de Exportación ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# ======================================================================
# SECCIÓN 1: CONSTANTES Y CONFIGURACIÓN
# ======================================================================

# --- CONFIGURACIÓN DE BASE DE DATOS ---
USE_DB_TYPE = 'sqlite'  # Cambia a 'mysql' para usar MySQL
DB_CONFIG = {
    'sqlite': 'sqlite:///mow_portfolio_v2.db',
    'mysql': 'mysql+pymysql://USUARIO:PASSWORD@HOST:PUERTO/NOMBRE_DB'
}

# --- CONFIGURACIÓN DE OLLAMA ---
OLLAMA_API_URL = "http://localhost:11434"

# --- Etiquetas del Modelo ---
D_LABELS = [
    "D1: Propósito", "D2: Procesos", "D3: Tecnología", "D4: Comunidad",
    "D5: Solución", "D6: Territorio", "D7: Academia", "D8: S. Privado", "D9: S. Público"
]
# FIX v2.7: Corregida la coma ('",')
T_LABELS_SHORT = ["T1(P+)", "T2(P-)", "T3(PN)", "T4(R+)", "T5(R-)", "T6(RN)", "T7(F+)", "T8(F-)", "T9(FN)"]
VME_LABELS = ['Herencia (IH)', 'Situacional (IS)', 'Prospectiva (IP)']
RI_SAATY = { 3: 0.58, 9: 1.45 }
AHP_GROUPS = ['dimensions', 'past', 'present', 'future']
AHP_SCALE_MAP = {
    0: (1/9, "1/9 Extr."), 1: (1/7, "1/7 Muy F."), 2: (1/5, "1/5 Fuerte"), 3: (1/3, "1/3 Ligero"),
    4: (1, "1 Igual"),
    5: (3, "3 Ligero"), 6: (5, "5 Fuerte"), 7: (7, "7 Muy F."), 8: (9, "9 Extr.")
}
AHP_VALUE_TO_SLIDER = {v[0]: k for k, v in AHP_SCALE_MAP.items()}


# ======================================================================
# SECCIÓN 2: LÓGICA DEL MODELO (models.py)
# ======================================================================

class AHPValidator:
    """Calcula los pesos AHP y el Ratio de Consistencia (CR)."""
    def __init__(self, pairwise_matrix: np.ndarray):
        self.matrix = np.array(pairwise_matrix)
        self.n = self.matrix.shape[0]
        self.weights = np.full(self.n, 1/self.n)
        self.consistency_ratio = 0.0
        if self.n not in RI_SAATY:
            raise ValueError(f"Tamaño de matriz AHP no soportado: {self.n}")

    def calculate(self) -> Tuple[np.ndarray, float]:
        col_sums = self.matrix.sum(axis=0)
        norm_matrix = self.matrix / col_sums
        self.weights = norm_matrix.mean(axis=1)
        self.weights /= np.sum(self.weights)
        
        weighted_sum_vector = np.dot(self.matrix, self.weights)
        lambda_vector = weighted_sum_vector / self.weights
        lambda_max = np.mean(lambda_vector)
        
        if (self.n - 1) == 0:
            ci = 0
        else:
            ci = (lambda_max - self.n) / (self.n - 1)
            
        ri = RI_SAATY.get(self.n)
        
        if ri == 0:
            self.consistency_ratio = 0
        else:
            self.consistency_ratio = ci / ri
        
        return self.weights, self.consistency_ratio

class M9DModel:
    """Modela un solo proyecto, su estrategia y sus momentos A-B-C."""
    def __init__(self, project_id: int, project_name: str, strategy_weights: Dict):
        self.project_id = project_id
        self.project_name = project_name
        self.strategy_weights = strategy_weights
        self.w_i = strategy_weights['w_i']
        self.v_j = strategy_weights['v_j']
        
        self.scores: Dict[str, np.ndarray] = {}
        self.vme_results: Dict[str, np.ndarray] = {}
        self.pbt_results: Dict[str, np.ndarray] = {}

    def set_reality_scores(self, scores_matrix: np.ndarray, moment: str):
        self.scores[moment] = scores_matrix

    def calculate_vme(self, moment: str) -> Tuple[np.ndarray, np.ndarray]:
        if moment not in self.scores:
            raise ValueError(f"No se encontraron puntuaciones para el momento '{moment}'")
        
        scores_matrix = self.scores[moment]
        s_past = scores_matrix[:, 0:3]; s_present = scores_matrix[:, 3:6]; s_future = scores_matrix[:, 6:9]
        v_j = self.strategy_weights['v_j']
        
        pbt_past = np.dot(s_past, v_j['past'])
        pbt_present = np.dot(s_present, v_j['present'])
        pbt_future = np.dot(s_future, v_j['future'])
        
        pbt_matrix = np.stack([pbt_past, pbt_present, pbt_future], axis=1)
        vme_vector = np.dot(pbt_matrix.T, self.w_i)
        
        self.vme_results[moment] = vme_vector
        self.pbt_results[moment] = pbt_matrix
        return vme_vector, pbt_matrix
    
    def get_vme_vector(self, moment='t0') -> np.ndarray:
        return self.vme_results.get(moment, np.zeros(3))
    def get_scores_vector(self, moment='t0') -> np.ndarray:
        return self.scores.get(moment, np.zeros((9,9))).flatten()
    def get_strategy_vector(self) -> np.ndarray:
        return np.concatenate([
            self.w_i, self.v_j['past'], self.v_j['present'], self.v_j['future']
        ])
    def get_full_data_package(self, moment='t0') -> Dict:
        """Paquete de datos para Ollama y exportación."""
        if moment not in self.scores:
             return {
                "project_name": self.project_name,
                "project_id": self.project_id,
                "error": f"Datos no encontrados para el momento {moment}"
             }
        # Asegurarse de que los resultados VME/PBT para ese momento estén calculados
        if moment not in self.vme_results:
            self.calculate_vme(moment)
            
        return {
            "project_name": self.project_name,
            "project_id": self.project_id,
            "strategy": {
                "w_i": self.w_i.tolist(),
                "v_j": {
                    "past": self.v_j['past'].tolist(),
                    "present": self.v_j['present'].tolist(),
                    "future": self.v_j['future'].tolist()
                }
            },
            "reality": {
                "moment": moment,
                "scores_S_ij": self.scores[moment].tolist(),
                "pbt_matrix": self.pbt_results[moment].tolist(),
                "vme_result": self.vme_results[moment].tolist()
            }
        }

# FIX v2.8: Añadida la clase IAGenerator
class IAGenerator:
    """Genera juicios y puntuaciones para simular a un equipo de expertos."""
    
    @staticmethod
    def get_ahp_matrix(labels: List[str], importance_profile: Dict[str, float]) -> np.ndarray:
        n = len(labels)
        matrix = np.ones((n, n))
        default_importance = importance_profile.get("default", 1.0)
        imp_values = np.array([importance_profile.get(label, default_importance) for label in labels])
        
        for i in range(n):
            for j in range(i + 1, n):
                ratio = imp_values[i] / imp_values[j] * np.random.uniform(0.95, 1.05) # Ruido
                matrix[i, j] = ratio
                matrix[j, i] = 1.0 / ratio
        return matrix
    
    @staticmethod
    def generate_ahp_strategy(dim_profile: Dict, state_profile: Dict) -> Dict:
        """Genera un diccionario de estrategia completo y validado."""
        weights_to_save = {'v_j': {}}
        
        # 1. Validar Dimensiones (w_i)
        matrix_dims = IAGenerator.get_ahp_matrix(D_LABELS, dim_profile)
        validator_dims = AHPValidator(matrix_dims)
        weights_dims, cr_dims = validator_dims.calculate()
        
        if cr_dims > 0.10: # Re-intentar si es inconsistente
            return IAGenerator.generate_ahp_strategy(dim_profile, state_profile) 

        weights_to_save['w_i'] = weights_dims
        weights_to_save['cr_dims'] = cr_dims

        # 2. Validar Estados (v_j)
        for group in ['past', 'present', 'future']:
            labels = [l for l in T_LABELS_SHORT if group in l.lower()]
            matrix = IAGenerator.get_ahp_matrix(labels, state_profile[group])
            validator = AHPValidator(matrix)
            weights, cr = validator.calculate()
            if cr > 0.10: # Re-intentar si es inconsistente
                return IAGenerator.generate_ahp_strategy(dim_profile, state_profile)
            weights_to_save['v_j'][group] = weights

        return weights_to_save

    @staticmethod
    def get_scores_matrix(profile_name: str, base_scores: np.ndarray = None) -> np.ndarray:
        """Genera una matriz de puntuaciones 9x9."""
        PROFILES = {
            "Neutro": {"mean": 0.0, "std": 0.1},
            "Baseline Difícil": {"mean": -1.5, "std": 1.0},
            "Baseline Optimista": {"mean": 1.0, "std": 1.0},
            "Mejora Moderada": {"mean": 0.8, "std": 0.3},
            "Mejora Alta": {"mean": 1.5, "std": 0.5},
            "Crisis (D4,D9)": {"mean": -2.0, "std": 0.5, "dims": [3, 8]},
            "Oportunidad (D3,D7)": {"mean": 2.0, "std": 0.5, "dims": [2, 6]}
        }
        
        profile = PROFILES[profile_name]
        scores = np.random.normal(loc=0.0, scale=0.5, size=(9, 9))

        if base_scores is not None:
            scores = base_scores.copy()
            mejora = np.random.normal(loc=profile["mean"], scale=profile["std"], size=(9, 6))
            scores[:, 3:9] += mejora
        elif "dims" in profile:
            for dim_index in profile["dims"]:
                scores[dim_index, :] += np.random.normal(loc=profile["mean"], scale=profile["std"], size=9)
        else:
            scores = np.random.normal(loc=profile["mean"], scale=profile["std"], size=(9, 9))
            
        return np.clip(scores, -3, 3)

# ======================================================================
# SECCIÓN 3: CAPA DE BASE DE DATOS (database.py)
# ======================================================================
class DatabaseManager:
    """Gestiona la lógica de la base de datos (SQLite o MySQL)."""
    
    def __init__(self, db_type: str = 'sqlite', config: Dict = DB_CONFIG):
        try:
            self.engine = create_engine(config[db_type])
            self.metadata = MetaData()
            
            # Definir tablas
            self.tbl_strategies = Table('mow_strategies', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String(255), unique=True),
                Column('weights_json', DBText, nullable=False),
                Column('consistency_ratio', Float, nullable=False)
            )
            
            self.tbl_projects = Table('mow_projects', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String(255), unique=True),
                Column('strategy_id', Integer, ForeignKey('mow_strategies.id'))
            )
            
            self.tbl_realities = Table('mow_realities', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('project_id', Integer, ForeignKey('mow_projects.id')),
                Column('moment', String(50), nullable=False), # 't0', 't1'...
                Column('scores_json', DBText, nullable=False),
                db.UniqueConstraint('project_id', 'moment', name='uidx_proj_moment')
            )
            
            self.metadata.create_all(self.engine)
            self.conn = self.engine.connect()
        except Exception as e:
            print(f"ERROR CRÍTICO DE DB: No se pudo conectar a la base de datos: {e}")
            print("Verifique su driver (ej. PyMySQL) y sus credenciales de conexión.")
            raise e
        
    def save_strategy(self, name: str, weights: Dict, cr: float) -> int:
        weights_json = json.dumps({
            'w_i': weights['w_i'].tolist(),
            'v_j': {k: v.tolist() for k, v in weights['v_j'].items()}
        })
        # Upsert (actualizar si el nombre existe, sino insertar)
        try:
            ins = self.tbl_strategies.insert().values(
                name=name, weights_json=weights_json, consistency_ratio=cr
            )
            result = self.conn.execute(ins)
            self.conn.commit()
            return result.inserted_primary_key[0]
        except db.exc.IntegrityError: # El nombre ya existe
            upd = self.tbl_strategies.update().where(self.tbl_strategies.c.name == name).values(
                weights_json=weights_json, consistency_ratio=cr
            )
            self.conn.execute(upd)
            self.conn.commit()
            # Devolver el ID existente
            query = self.tbl_strategies.select().where(self.tbl_strategies.c.name == name)
            return self.conn.execute(query).fetchone()[0]


    def get_strategies_list(self) -> List[Dict]:
        query = self.tbl_strategies.select()
        results = self.conn.execute(query).fetchall()
        return [{"id": r[0], "name": r[1], "cr": r[3]} for r in results]

    def get_strategy_by_id(self, strategy_id: int) -> Dict:
        query = self.tbl_strategies.select().where(self.tbl_strategies.c.id == strategy_id)
        r = self.conn.execute(query).fetchone()
        if r:
            weights_dict = json.loads(r[2])
            return {
                'id': r[0], 'name': r[1],
                'w_i': np.array(weights_dict['w_i']),
                'v_j': {k: np.array(v) for k, v in weights_dict['v_j'].items()}
            }
        return None

    def save_project(self, name: str, strategy_id: int) -> int:
        # Upsert
        try:
            ins = self.tbl_projects.insert().values(name=name, strategy_id=strategy_id)
            result = self.conn.execute(ins)
            self.conn.commit()
            return result.inserted_primary_key[0]
        except db.exc.IntegrityError:
            upd = self.tbl_projects.update().where(self.tbl_projects.c.name == name).values(
                strategy_id=strategy_id
            )
            self.conn.execute(upd)
            self.conn.commit()
            query = self.tbl_projects.select().where(self.tbl_projects.c.name == name)
            return self.conn.execute(query).fetchone()[0]
        
    def save_reality(self, project_id: int, moment: str, scores: np.ndarray):
        scores_json = json.dumps(scores.tolist())
        
        # Borrar si ya existe (Upsert)
        delete_q = self.tbl_realities.delete().where(
            (self.tbl_realities.c.project_id == project_id) & (self.tbl_realities.c.moment == moment)
        )
        self.conn.execute(delete_q)
        
        ins = self.tbl_realities.insert().values(
            project_id=project_id, moment=moment, scores_json=scores_json
        )
        self.conn.execute(ins)
        self.conn.commit()

    def load_full_portfolio(self) -> Dict[int, M9DModel]:
        portfolio = {}
        proj_query = self.tbl_projects.select()
        projects = self.conn.execute(proj_query).fetchall()
        
        for p in projects:
            project_id, project_name, strategy_id = p
            strategy = self.get_strategy_by_id(strategy_id)
            if not strategy: continue
            model = M9DModel(project_id, project_name, strategy)
            reality_query = self.tbl_realities.select().where(self.tbl_realities.c.project_id == project_id)
            realities = self.conn.execute(reality_query).fetchall()
            
            for r in realities:
                moment = r[2]
                scores = np.array(json.loads(r[3]))
                model.set_reality_scores(scores, moment)
                model.calculate_vme(moment) # Pre-calcular
                
            portfolio[project_id] = model
            
        return portfolio

    def close(self):
        self.conn.close()

# ======================================================================
# SECCIÓN 4: CAPA DE ANÁLISIS (analysis.py)
# ======================================================================
class AnalysisService:
    """Servicio para ejecutar análisis pesados (ML y LLM) en hilos."""
    
    def __init__(self, platform: Dict[int, M9DModel] = None, golden_strategy_vector: np.ndarray = None):
        self.platform = platform
        self.golden_strategy_vector = golden_strategy_vector
        
    def run_mow_pipeline(self, n_clusters: int, threshold: float, moment: str) -> Dict:
        """Ejecuta el pipeline completo de 3 fases de MoW."""
        # --- FASE 1: Filtro de Similitud ---
        project_ids = [model.project_id for model in self.platform.values()]
        project_names = {model.project_id: model.project_name for model in self.platform.values()}
        strategy_vectors = [model.get_strategy_vector() for model in self.platform.values()]
        
        if not strategy_vectors:
             raise Exception("Portafolio vacío. No hay proyectos para analizar.")
        
        # Matriz de similitud NxN
        similarity_matrix_full = cosine_similarity(strategy_vectors)
        
        # Filtro contra Golden Standard
        similarities = cosine_similarity(strategy_vectors, [self.golden_strategy_vector])
        sim_df = pd.DataFrame({'ProjectID': project_ids, 'Similarity': similarities.flatten()})
        
        comparable_projects_ids = sim_df[sim_df['Similarity'] >= threshold]['ProjectID'].tolist()
        if not comparable_projects_ids:
            raise Exception(f"Ningún proyecto cumple el umbral de similitud > {threshold}")

        # --- FASE 2: Clustering K-Means ---
        vme_vectors = []
        valid_project_ids_for_cluster = []
        for pid in comparable_projects_ids:
            try:
                if moment in self.platform[pid].vme_results:
                    vme_vectors.append(self.platform[pid].get_vme_vector(moment))
                    valid_project_ids_for_cluster.append(pid)
            except Exception: pass
        
        if not vme_vectors:
            raise Exception(f"Ningún proyecto comparable tiene datos para el momento '{moment}'")
            
        vme_matrix = np.array(vme_vectors)
        n_clusters = min(n_clusters, len(vme_vectors))
        if n_clusters < 2:
             raise Exception("Se necesita al menos 2 proyectos comparables para agrupar.")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(vme_matrix)
        
        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(vme_matrix)
        
        cluster_df = pd.DataFrame({
            'ProjectID': valid_project_ids_for_cluster, 
            'ProjectName': [project_names[pid] for pid in valid_project_ids_for_cluster],
            'Cluster': clusters,
            'VME_IH': vme_matrix[:, 0], 'VME_IS': vme_matrix[:, 1], 'VME_IP': vme_matrix[:, 2],
            'PC1': components[:, 0], 'PC2': components[:, 1]
        })
        cluster_centers = kmeans.cluster_centers_

        # --- FASE 3: Análisis de Causa Raíz (Random Forest) ---
        X_scores = [self.platform[pid].get_scores_vector(moment) for pid in valid_project_ids_for_cluster]
        y_clusters = cluster_df['Cluster'].values
            
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scores, y_clusters)
        
        importances = rf.feature_importances_
        feature_labels = [f"D{i+1}-{T_LABELS_SHORT[j]}" for i in range(9) for j in range(9)]
        
        importance_df = pd.DataFrame({'Factor (S_i,j)': feature_labels, 'Importancia': importances})
        importance_df = importance_df.sort_values(by='Importancia', ascending=False).reset_index(drop=True)
        
        X_df = pd.DataFrame(X_scores, columns=feature_labels)
        X_df['Cluster'] = y_clusters
        mean_scores_by_cluster = X_df.groupby('Cluster').mean().T.rename(columns=lambda c: f'Mean_Score_C{c}')
        importance_df = importance_df.join(mean_scores_by_cluster, on='Factor (S_i,j)')
        
        return {
            "similarity_df": sim_df,
            "similarity_matrix_full": similarity_matrix_full,
            "clustering_df": cluster_df,
            "cluster_centers": cluster_centers,
            "importance_df": importance_df
        }

    def call_ollama(self, endpoint: str, payload: Dict) -> Dict:
        """Función genérica para llamar a la API de Ollama."""
        try:
            url = f"{OLLAMA_API_URL}/api/{endpoint}"
            if endpoint == 'tags': # 'ollama list'
                response = requests.get(url, timeout=5)
            else: # 'ollama show' o 'ollama generate'
                response = requests.post(url, json=payload, timeout=60)
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": f"Error de Conexión: No se pudo conectar a Ollama en {OLLAMA_API_URL}. ¿Está corriendo?"}
        except requests.exceptions.Timeout:
            return {"error": "Error: Timeout. La solicitud a Ollama tardó demasiado."}
        except requests.exceptions.RequestException as e:
            return {"error": f"Error de Ollama: {e}"}

# ======================================================================
# SECCIÓN 5: CAPA DE E/S (io_export.py)
# ======================================================================
class ExportService:
    """Maneja la exportación de datos a CSV, JSON y PDF."""
    
    @staticmethod
    def export_to_json(data: Dict, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
    @staticmethod
    def export_to_csv(dataframe: pd.DataFrame, filepath: str):
        dataframe.to_csv(filepath, index=False, encoding='utf-8')
        
    @staticmethod
    def export_to_pdf(project_data: Dict, filepath: str):
        p_name = project_data['project_name']
        vme = project_data['vme_result']
        
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(inch, height - inch, f"Reporte de Proyecto M9D: {p_name}")
        
        c.setFont("Helvetica", 12)
        c.drawString(inch, height - 1.5*inch, "Resultados del Vector de Momentum Estratégico (VME)")
        
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.5, 0, 0) # Rojo
        c.drawString(1.5*inch, height - 1.8*inch, f"Herencia (IH):    {vme[0]:.3f}")
        c.setFillColorRGB(0, 0, 0.5) # Azul
        c.drawString(1.5*inch, height - 2.0*inch, f"Situacional (IS): {vme[1]:.3f}")
        c.setFillColorRGB(0, 0.5, 0) # Verde
        c.drawString(1.5*inch, height - 2.2*inch, f"Prospectiva (IP): {vme[2]:.3f}")
        
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 10)
        c.drawString(inch, inch, f"Reporte generado por MOW v2.9 - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        c.save()

# ======================================================================
# SECCIÓN 6: INTERFAZ GRÁFICA (GUI)
# ======================================================================

class AHPSliderFrame(btk.Frame):
    """Un Frame que contiene N sliders AHP."""
    def __init__(self, parent, labels: List[str]):
        super().__init__(parent)
        self.labels = labels
        self.n = len(labels)
        self.slider_vars: Dict[Tuple[int, int], tk.IntVar] = {}
        self.label_vars: Dict[Tuple[int, int], tk.StringVar] = {}
        self.data_matrix = np.ones((self.n, self.n))
        self.create_sliders()

    def create_sliders(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pair_frame = btk.Frame(self)
                pair_frame.pack(fill="x", pady=2, padx=5)
                
                initial_val = self.data_matrix[i, j]
                slider_idx = AHP_VALUE_TO_SLIDER.get(initial_val, 4)
                
                var = tk.IntVar(value=slider_idx)
                label_var = tk.StringVar(value=f"{AHP_SCALE_MAP[slider_idx][1]}")
                self.slider_vars[(i, j)] = var
                self.label_vars[(i, j)] = label_var
                
                btk.Label(pair_frame, text=self.labels[i], width=12, anchor="e").pack(side="left")
                slider = btk.Scale(pair_frame, from_=0, to=8, orient="horizontal", variable=var, length=200,
                                   command=lambda v, i=i, j=j, l=label_var: self.on_slider_change(v, i, j, l))
                slider.pack(side="left", fill="x", expand=True, padx=5)
                btk.Label(pair_frame, text=self.labels[j], width=12, anchor="w").pack(side="left")
                btk.Label(pair_frame, textvariable=label_var, width=10, anchor="w").pack(side="left", padx=5)

    def on_slider_change(self, slider_value: str, i: int, j: int, label_var: tk.StringVar):
        idx = int(float(slider_value))
        val, label = AHP_SCALE_MAP[idx]
        self.data_matrix[i, j] = val
        self.data_matrix[j, i] = 1.0 / val
        label_var.set(label)

    def get_data(self) -> np.ndarray:
        return self.data_matrix

class AHPEditorWindow(Toplevel):
    """Ventana emergente para la edición rigurosa de las 4 matrices AHP."""
    def __init__(self, parent, db: DatabaseManager, app_callback):
        super().__init__(parent)
        self.title("Editor de Estrategia AHP Riguroso")
        self.geometry("900x600")
        self.transient(parent)
        self.grab_set()
        
        self.db = db
        self.app_callback = app_callback
        
        top_frame = btk.Frame(self, padding=10)
        top_frame.pack(fill="x")
        
        btk.Label(top_frame, text="Nombre de Estrategia:").pack(side="left", padx=5)
        self.entry_strategy_name = btk.Entry(top_frame, width=50)
        self.entry_strategy_name.pack(side="left", fill="x", expand=True, padx=5)
        
        self.notebook = btk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, pady=5, padx=5)
        
        self.ahp_frames: Dict[str, AHPSliderFrame] = {}
        
        # FIX v2.7: Contenedor para el ScrolledFrame
        tab_dims_container = btk.Frame(self.notebook, padding=0)
        frame_dims_scrolled = ScrolledFrame(tab_dims_container, autohide=True)
        frame_dims_scrolled.pack(fill="both", expand=True)
        self.ahp_frames['dimensions'] = AHPSliderFrame(frame_dims_scrolled, D_LABELS)
        self.ahp_frames['dimensions'].pack(fill="both", expand=True)
        self.notebook.add(tab_dims_container, text="Dimensiones (9x9)")

        groups = [('past', T_LABELS_SHORT[0:3]), 
                  ('present', T_LABELS_SHORT[3:6]), 
                  ('future', T_LABELS_SHORT[6:9])]
        
        for name, labels in groups:
            frame = btk.Frame(self.notebook, padding=20) 
            self.notebook.add(frame, text=f"{name.capitalize()} (3x3)")
            self.ahp_frames[name] = AHPSliderFrame(frame, labels)
            self.ahp_frames[name].pack(fill="x", expand=True, padx=20)
        
        bottom_frame = btk.Frame(self, padding=10)
        bottom_frame.pack(fill="x")
        
        self.btn_save = btk.Button(bottom_frame, text="Validar y Guardar Estrategia", 
                                   command=self.on_save_strategy, bootstyle="primary")
        self.btn_save.pack(side="right", padx=5)
        
    def on_save_strategy(self):
        strategy_name = self.entry_strategy_name.get()
        if not strategy_name:
            messagebox.showerror("Error de Validación", "El nombre de la estrategia no puede estar vacío.", parent=self)
            return
            
        weights_to_save = {'v_j': {}}
        all_consistent = True
        cr_text = ""
        
        try:
            for group_name, frame in self.ahp_frames.items():
                matrix = frame.get_data()
                validator = AHPValidator(matrix)
                weights, cr = validator.calculate()
                
                cr_text += f"CR {group_name}: {cr:.3f}\n"
                if cr > 0.10:
                    all_consistent = False
                
                if group_name == 'dimensions':
                    weights_to_save['w_i'] = weights
                    weights_to_save['cr_dims'] = cr
                else:
                    weights_to_save['v_j'][group_name] = weights
            
            if not all_consistent:
                messagebox.showwarning("Error de Consistencia", 
                                     f"¡Juicios inconsistentes! (CR > 0.10)\n\n{cr_text}\n"
                                     "Ajuste los sliders y vuelva a intentarlo. La estrategia NO se guardó.", parent=self)
                return
            
            strategy_id = self.db.save_strategy(strategy_name, weights_to_save, weights_to_save['cr_dims'])
            messagebox.showinfo("Éxito", 
                                f"Estrategia '{strategy_name}' (ID: {strategy_id}) guardada.\n\n{cr_text}", parent=self)
            
            self.app_callback()
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error de Cálculo", str(e), parent=self)

class MainApplication(btk.Window):

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(title="MoW (M9D^X) - Plataforma de Análisis de Portafolio v2.9", themename="cyborg", size=(1500, 950))
        self.db = db_manager
        
        self.portfolio: Dict[int, M9DModel] = {}
        self.golden_strategy: Dict = None
        self.analysis_queue = queue.Queue()
        self.current_mow_results: Dict = {}
        
        self.running = True
        self._after_job_id = None
        
        self.build_gui()
        self.load_portfolio_from_db()
        self.after(100, self.load_ollama_models_threaded)

    def build_gui(self):
        main_pane = btk.PanedWindow(self, orient="horizontal")
        main_pane.pack(fill="both", expand=True)
        
        # Frame contenedor para el panel de control
        control_container_frame = btk.Frame(main_pane)
        main_pane.add(control_container_frame, weight=2)
        
        self.control_panel = self.create_control_panel(control_container_frame)
        self.control_panel.pack(fill="both", expand=True)
        
        self.notebook = btk.Notebook(main_pane)
        main_pane.add(self.notebook, weight=5)
        
        # Crear pestañas
        self.tab_portfolio = self.create_portfolio_tab(self.notebook)
        self.tab_project = self.create_project_tab(self.notebook)
        self.tab_strategy = self.create_strategy_tab(self.notebook)
        self.tab_ollama = self.create_ollama_tab(self.notebook)
        
        self.notebook.add(self.tab_portfolio, text="  Portafolio (MoW) ")
        self.notebook.add(self.tab_project, text="  Proyecto (M9D) ")
        self.notebook.add(self.tab_strategy, text="  Estrategias (AHP) ")
        self.notebook.add(self.tab_ollama, text="  Análisis IA (Ollama) ")

    # --- Constructores de Paneles y Pestañas ---

    def create_control_panel(self, parent) -> btk.Frame:
        frame = ScrolledFrame(parent, autohide=True, padding=15)
        btk.Label(frame, text="PANEL DE CONTROL MoW", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        proj_frame = btk.Labelframe(frame, text="Gestión de Proyectos", padding=10)
        proj_frame.pack(fill="x", pady=5)
        btk.Button(proj_frame, text="Crear Nuevo Proyecto...", command=self.on_create_project, bootstyle="info").pack(fill="x", pady=5)
        btk.Button(proj_frame, text="Importar Proyecto (M9D JSON)...", command=self.on_import_project_json, bootstyle="info").pack(fill="x", pady=5)
        
        # --- NUEVO BOTÓN DEMO (v2.8) ---
        btk.Button(proj_frame, text="Cargar Set de Proyectos Demo", command=self.on_load_demo_data, bootstyle="warning-outline").pack(fill="x", pady=(10, 5))
        
        # --- NUEVO BOTÓN CLONAR (v2.9) ---
        btk.Button(proj_frame, text="Clonar Proyecto Seleccionado", command=self.on_clone_project, bootstyle="secondary").pack(fill="x", pady=5)
        
        btk.Button(proj_frame, text="Refrescar Portafolio de DB", command=self.load_portfolio_from_db, bootstyle="info-outline").pack(fill="x", pady=5)
        
        mow_frame = btk.Labelframe(frame, text="Análisis de Portafolio (MoW)", padding=10)
        mow_frame.pack(fill="x", pady=10)
        
        btk.Label(mow_frame, text="Estrategia Golden (Filtro 1):").pack(anchor="w")
        self.cb_golden_strategy = btk.Combobox(mow_frame, state="readonly")
        self.cb_golden_strategy.pack(fill="x", pady=2)
        
        btk.Label(mow_frame, text="Umbral de Similitud (0.1 - 1.0):").pack(anchor="w")
        self.scale_threshold = btk.Scale(mow_frame, from_=0.1, to=1.0, value=0.8, orient="horizontal")
        self.scale_threshold.pack(fill="x", pady=2)
        
        btk.Label(mow_frame, text="Número de Clústeres (2-5):").pack(anchor="w")
        self.spin_clusters = btk.Spinbox(mow_frame, from_=2, to=5, width=5)
        self.spin_clusters.set(3)
        self.spin_clusters.pack(fill="x", pady=2)
        
        btk.Label(mow_frame, text="Momento de Análisis:").pack(anchor="w")
        self.cb_mow_moment = btk.Combobox(mow_frame, state="readonly", values=['t0', 't1', 't2'])
        self.cb_mow_moment.set('t0')
        self.cb_mow_moment.pack(fill="x", pady=2)
        
        self.btn_run_mow = btk.Button(mow_frame, text="EJECUTAR ANÁLISIS MoW", command=self.on_run_mow, bootstyle="success")
        self.btn_run_mow.pack(fill="x", pady=10)
        
        status_frame = btk.Labelframe(frame, text="Estado del Sistema", padding=10)
        status_frame.pack(fill="both", pady=10, expand=True)
        self.status_bar_text = btk.StringVar(value="Listo.")
        btk.Label(status_frame, textvariable=self.status_bar_text, wraplength=350, justify="left").pack(fill="both", expand=True, anchor="nw")
        
        return frame

    def create_portfolio_tab(self, parent) -> btk.Frame:
        """Pestaña 1: Visualización del Portafolio (Clústeres, Factores)."""
        frame = btk.Frame(parent, padding=10)
        self.charts_portfolio = {}
        
        mow_notebook = btk.Notebook(frame)
        mow_notebook.pack(fill="both", expand=True)
        
        tab_cluster = btk.Frame(mow_notebook, padding=10)
        tab_cause = btk.Frame(mow_notebook, padding=10)
        
        mow_notebook.add(tab_cluster, text="  Análisis de Clúster  ")
        mow_notebook.add(tab_cause, text="  Análisis de Causa Raíz  ")
        
        self.charts_portfolio['cluster_frame'] = btk.Labelframe(tab_cluster, text="Gráfico MoW 1: Clústeres de Proyectos (PCA)", padding=5)
        self.charts_portfolio['cluster_frame'].pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.charts_portfolio['cluster_radar_frame'] = btk.Labelframe(tab_cluster, text="Gráfico MoW 2: Perfiles de Clúster (VME)", padding=5)
        self.charts_portfolio['cluster_radar_frame'].pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.charts_portfolio['importance_frame'] = btk.Labelframe(tab_cause, text="Gráfico MoW 3: Causa Raíz de Clúster (RF)", padding=5)
        self.charts_portfolio['importance_frame'].pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.charts_portfolio['network_frame'] = btk.Labelframe(tab_cause, text="Gráfico MoW 4: Red de Similitud de Proyectos (NX)", padding=5)
        self.charts_portfolio['network_frame'].pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        export_frame = btk.Labelframe(frame, text="Exportar Resultados Cuantitativos", padding=10)
        export_frame.pack(fill="x", pady=10)
        
        btk.Button(export_frame, text="Exportar Clústeres (CSV)", command=lambda: self.on_export('cluster_csv')).pack(side="left", padx=5)
        btk.Button(export_frame, text="Exportar Causa Raíz (CSV)", command=lambda: self.on_export('importance_csv')).pack(side="left", padx=5)
        
        return frame
        
    def create_project_tab(self, parent) -> btk.Frame:
        """Pestaña 2: Visualización de un proyecto M9D individual (A-B-C)."""
        frame = btk.Frame(parent, padding=10)
        
        selector_frame = btk.Frame(frame)
        selector_frame.pack(fill="x", pady=5)
        btk.Label(selector_frame, text="Seleccionar Proyecto:").pack(side="left", padx=5)
        self.cb_project_select = btk.Combobox(selector_frame, state="readonly", width=40)
        self.cb_project_select.pack(side="left", padx=5, fill="x", expand=True)
        self.cb_project_select.bind("<<ComboboxSelected>>", self.on_project_selected)

        import_frame = btk.Labelframe(frame, text="Gestión de Realidad (S_ij)", padding=10)
        import_frame.pack(fill="x", pady=10)
        btk.Label(import_frame, text="Momento:").pack(side="left", padx=5)
        self.cb_import_moment = btk.Combobox(import_frame, state="readonly", values=['t0', 't1', 't2'], width=5)
        self.cb_import_moment.set('t0')
        self.cb_import_moment.pack(side="left", padx=5)
        btk.Button(import_frame, text="Importar Realidad (CSV)...", command=self.on_import_reality_csv, bootstyle="info").pack(side="left", padx=10)

        self.charts_project = {}
        chart_frame = btk.Frame(frame)
        chart_frame.pack(fill="both", expand=True)

        self.charts_project['radar_frame'] = btk.Labelframe(chart_frame, text="Gráfico 2: Avance del Proyecto (VME)", padding=5)
        self.charts_project['radar_frame'].pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.charts_project['heatmap_frame'] = btk.Labelframe(chart_frame, text="Gráfico 3: Mapa de Calor (Acción - Momento t0)", padding=5)
        self.charts_project['heatmap_frame'].pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        export_frame = btk.Labelframe(frame, text="Exportar Reporte Cualitativo", padding=10)
        export_frame.pack(fill="x", pady=10)
        btk.Button(export_frame, text="Exportar Proyecto a PDF", command=lambda: self.on_export('project_pdf')).pack(side="left", padx=5)
        btk.Button(export_frame, text="Exportar Datos a JSON", command=lambda: self.on_export('project_json')).pack(side="left", padx=5)
        
        return frame

    def create_strategy_tab(self, parent) -> btk.Frame:
        """Pestaña 3: Creación y gestión de Estrategias AHP."""
        frame = btk.Frame(parent, padding=10)
        btk.Label(frame, text="Gestor de Estrategias (AHP)", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        btk.Button(frame, text="Crear Nueva Estrategia (Editor Riguroso)...", 
                   command=self.on_open_ahp_editor, 
                   bootstyle="primary").pack(fill="x", pady=10)
        
        btk.Label(frame, text="Estrategias Guardadas en la Base de Datos:").pack(anchor="w", pady=5)
        
        tree_frame = ScrolledFrame(frame, autohide=True)
        tree_frame.pack(fill="both", expand=True)
        
        self.tree_strategies = btk.Treeview(tree_frame, columns=("id", "name", "cr"), show="headings", height=10)
        self.tree_strategies.heading("id", text="ID")
        self.tree_strategies.heading("name", text="Nombre")
        self.tree_strategies.heading("cr", text="Ratio Consistencia (CR)")
        self.tree_strategies.column("id", width=50, anchor="center")
        self.tree_strategies.column("cr", width=150, anchor="e")
        self.tree_strategies.pack(fill="both", expand=True)
        
        return frame
        
    def create_ollama_tab(self, parent) -> btk.Frame:
        frame = btk.Frame(parent, padding=10)
        btk.Label(frame, text="Análisis Cualitativo con IA (Ollama)", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        f_model = btk.Labelframe(frame, text="Selector de Modelo Ollama", padding=10)
        f_model.pack(fill="x", pady=5)
        
        f_model_select = btk.Frame(f_model)
        f_model_select.pack(fill="x")
        btk.Label(f_model_select, text="Modelo Instalado:").pack(side="left", padx=5)
        self.cb_ollama_model = btk.Combobox(f_model_select, state="readonly", width=30)
        self.cb_ollama_model.pack(side="left", padx=5)
        self.cb_ollama_model.bind("<<ComboboxSelected>>", self.on_ollama_model_select)
        
        self.txt_ollama_info = ScrolledText(f_model, height=5, width=100, font=("Courier", 9), state="disabled")
        self.txt_ollama_info.pack(fill="x", pady=5, expand=True)

        f_select = btk.Frame(frame)
        f_select.pack(fill="x", pady=5)
        btk.Label(f_select, text="Comparar A:").pack(side="left", padx=5)
        self.cb_ollama_proj_a = btk.Combobox(f_select, state="readonly", width=30)
        self.cb_ollama_proj_a.pack(side="left", padx=5)
        
        btk.Label(f_select, text="con B:").pack(side="left", padx=20)
        self.cb_ollama_proj_b = btk.Combobox(f_select, state="readonly", width=30)
        self.cb_ollama_proj_b.pack(side="left", padx=5)
        
        f_prompt = btk.Labelframe(frame, text="Prompt", padding=5)
        f_prompt.pack(fill="x", pady=5)
        self_prompt_text = (
            "Eres 'Gemini-MoW', un analista experto en gestión de portafolios (MoW). Compara los dos proyectos M9D (Proyecto A y Proyecto B) "
            "que te proveeré como JSON. Enfócate en:\n"
            "1. ¿Por qué uno es más riesgoso que el otro (ver VME y PBT)?\n"
            "2. ¿Cuál es la 'causa raíz' de sus problemas (ver puntuaciones S_ij más bajas)?\n"
            "3. Dame una recomendación estratégica accionable para el Director del Portafolio.\n\n"
            "[DATOS PROYECTO A]:\n{data_a}\n\n"
            "[DATOS PROYECTO B]:\n{data_b}\n\n"
            "Tu análisis:"
        )
        self.txt_ollama_prompt = ScrolledText(f_prompt, height=10, width=100, font=("Helvetica", 10))
        self.txt_ollama_prompt.text.insert("1.0", self_prompt_text)
        self.txt_ollama_prompt.pack(fill="x", expand=True)
        
        self.btn_run_ollama = btk.Button(frame, text="Preguntar a IA (Ollama)", command=self.on_run_ollama, bootstyle="warning")
        self.btn_run_ollama.pack(fill="x", pady=10)

        f_response = btk.Labelframe(frame, text="Respuesta de la IA", padding=5)
        f_response.pack(fill="both", expand=True, pady=5)
        self.txt_ollama_response = ScrolledText(f_response, height=10, width=100, font=("Helvetica", 10), state="disabled", wrap="word")
        self.txt_ollama_response.pack(fill="both", expand=True)
        
        return frame

    # --- Lógica de la GUI (Controladores de Eventos) ---
    
    def on_create_project(self):
        """Abre un diálogo para crear un nuevo proyecto."""
        try:
            # FIX v2.6: Usar simpledialog.askstring
            name = simpledialog.askstring("Crear Proyecto", "Nombre del Nuevo Proyecto:", parent=self)
            if not name: return
            
            strategies = self.db.get_strategies_list()
            if not strategies:
                messagebox.showerror("Error", "No hay estrategias en la DB. Por favor, cree una en la Pestaña 3.")
                return
            
            choice_str = "\n".join([f"{s['id']}: {s['name']}" for s in strategies])
            # FIX v2.6: Usar simpledialog.askstring
            strategy_id_str = simpledialog.askstring("Asignar Estrategia", f"Elija un ID de Estrategia:\n{choice_str}", parent=self)
            
            if not strategy_id_str: return # El usuario canceló
            strategy_id = int(strategy_id_str)
            
            # Guardar en DB
            project_id = self.db.save_project(name, strategy_id)
            
            self.set_status(f"Proyecto '{name}' (ID {project_id}) creado. Ahora importe su realidad (CSV).")
            self.load_portfolio_from_db() # Refrescar toda la app
            
        except ValueError:
            messagebox.showerror("Error de Entrada", "El ID de la estrategia debe ser un número.")
        except Exception as e:
            messagebox.showerror("Error al Crear", str(e))
    
    def on_import_project_json(self):
        """(NUEVO v2.9) Importa un proyecto completo desde un archivo JSON M9D."""
        try:
            filepath = filedialog.askopenfilename(
                title="Importar Proyecto (Formato M9D JSON)",
                filetypes=[("M9D JSON Files", "*.json"), ("All Files", "*.*")]
            )
            if not filepath: return
            
            self.set_status(f"Importando proyecto M9D desde {filepath}...")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. Validar y guardar la Estrategia
            if 'strategy' not in data or 'project_name' not in data or 'reality' not in data:
                raise ValueError("Archivo JSON no es un formato M9D válido. Faltan claves.")
            
            strategy_data = data['strategy']
            temp_weights = {'w_i': np.array(strategy_data['w_i']), 'v_j': {}}
            for group, weights in strategy_data['v_j'].items():
                temp_weights['v_j'][group] = np.array(weights)
            
            # (Simplificación: asumimos CR=0, ya que no tenemos la matriz de juicio)
            strategy_name = f"Importada - {data['project_name']}"
            strategy_id = self.db.save_strategy(strategy_name, temp_weights, 0.0)

            # 2. Guardar el Proyecto
            project_id = self.db.save_project(data['project_name'], strategy_id)
            
            # 3. Guardar la Realidad
            reality_data = data['reality']
            moment = reality_data.get('moment', 't0')
            scores_matrix = np.array(reality_data['scores_S_ij'])
            
            self.db.save_reality(project_id, moment, scores_matrix)
            
            self.set_status(f"Proyecto '{data['project_name']}' importado exitosamente.")
            self.load_portfolio_from_db()

        except Exception as e:
            messagebox.showerror("Error de Importación JSON", str(e))
            self.set_status(f"Error al importar JSON: {e}")

    def on_load_demo_data(self):
        """(NUEVO v2.8) Carga un set de datos demo en la DB."""
        self.set_status("Cargando set de proyectos demo en segundo plano...")
        self.btn_run_mow.config(state="disabled")
        
        threading.Thread(target=self.run_demo_data_thread, daemon=True).start()
        self.after(100, self.check_analysis_queue)

    def run_demo_data_thread(self):
        """(NUEVO v2.8) (Función Hilo) Genera y guarda los datos demo."""
        try:
            # 1. Crear Estrategia Demo
            strategy_dims = {"D9: S. Público": 9.0, "D4: Comunidad": 7.0, "default": 1.0}
            strategy_states = {
                "past": {"T2(P-)": 7.0, "default": 1.0},
                "present": {"T5(R-)": 7.0, "default": 1.0},
                "future": {"T8(F-)": 9.0, "default": 1.0}
            }
            
            strategy_to_save = IAGenerator.generate_ahp_strategy(strategy_dims, strategy_states)
            
            strategy_id = self.db.save_strategy("Estrategia Demo (Riesgo)", strategy_to_save, strategy_to_save['cr_dims'])

            # 2. Crear Proyectos Demo
            project_profiles = {
                "Proyecto Crisis (Demo)": "Crisis (D4,D9)",
                "Proyecto Oportunidad (Demo)": "Oportunidad (D3,D7)",
                "Proyecto Estable (Demo)": "Baseline Optimista"
            }
            
            for name, profile in project_profiles.items():
                project_id = self.db.save_project(name, strategy_id)
                
                scores_t0 = IAGenerator.get_scores_matrix(profile)
                self.db.save_reality(project_id, 't0', scores_t0)
                
                scores_t1 = IAGenerator.get_scores_matrix("Mejora Moderada", base_scores=scores_t0)
                self.db.save_reality(project_id, 't1', scores_t1)
                
            self.analysis_queue.put({"type": "demo_success"})
        except Exception as e:
            self.analysis_queue.put({"type": "demo_error", "data": str(e)})

    def on_clone_project(self):
        """(NUEVO v2.9) Clona el proyecto seleccionado en la pestaña M9D."""
        try:
            pid_str = self.cb_project_select.get()
            if not pid_str:
                raise ValueError("No hay ningún proyecto seleccionado para clonar.")
            
            original_pid = int(pid_str.split(":")[0])
            original_model = self.portfolio.get(original_pid)
            if not original_model:
                raise ValueError("No se encontró el modelo original en el portafolio.")
            
            new_name = simpledialog.askstring("Clonar Proyecto", 
                                              f"Nombre para el clon de:\n'{original_model.project_name}'", 
                                              parent=self)
            if not new_name: return
            
            # 1. Guardar el nuevo proyecto con la misma estrategia
            new_pid = self.db.save_project(new_name, original_model.strategy_weights['id'])
            
            # 2. Copiar todas las realidades (t0, t1, etc.)
            for moment, scores in original_model.scores.items():
                self.db.save_reality(new_pid, moment, scores)
                
            self.set_status(f"Proyecto '{original_model.project_name}' clonado a '{new_name}' (ID: {new_pid}).")
            self.load_portfolio_from_db() # Refrescar todo
            
        except Exception as e:
            messagebox.showerror("Error al Clonar", str(e))
            self.set_status(f"Error al clonar: {e}")
            
    def on_import_reality_csv(self):
        """Importa una matriz 9x9 desde un CSV para el proyecto seleccionado."""
        try:
            pid_str = self.cb_project_select.get()
            if not pid_str:
                raise ValueError("No hay ningún proyecto seleccionado en el dropdown.")
                
            pid = int(pid_str.split(":")[0])
            moment = self.cb_import_moment.get()
            
            filepath = filedialog.askopenfilename(
                title=f"Seleccionar CSV para Proyecto {pid} (Momento {moment})",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if not filepath: return
            
            self.set_status(f"Importando CSV desde {filepath}...")
            
            df = pd.read_csv(filepath, header=None)
            if df.shape != (9, 9):
                raise ValueError(f"El CSV debe tener exactamente 9 filas y 9 columnas. El archivo tiene {df.shape}.")
            
            scores_matrix = df.to_numpy()
            
            if not (np.all(scores_matrix >= -3) and np.all(scores_matrix <= 3)):
                raise ValueError("Todas las puntuaciones en el CSV deben estar entre -3 y +3.")
                
            self.db.save_reality(pid, moment, scores_matrix)
            self.set_status(f"Realidad '{moment}' importada para el proyecto {pid}.")
            self.load_portfolio_from_db() # Recargar todo
            
        except Exception as e:
            messagebox.showerror("Error de Importación CSV", str(e))
            self.set_status(f"Error al importar CSV: {e}")

    def on_open_ahp_editor(self):
        """Abre la nueva ventana de edición AHP rigurosa."""
        AHPEditorWindow(self, self.db, self.update_strategy_dropdown)

    def on_run_mow(self):
        """Inicia el hilo de análisis de portafolio (MoW)."""
        self.set_status("Iniciando análisis MoW en segundo plano...")
        self.btn_run_mow.config(state="disabled")
        
        try:
            strategy_id_str = self.cb_golden_strategy.get()
            if not strategy_id_str:
                raise ValueError("No hay Estrategia Golden seleccionada.")
            
            strategy_id = int(strategy_id_str.split(":")[0])
            golden_strategy_full = self.db.get_strategy_by_id(strategy_id)
            if not golden_strategy_full:
                raise ValueError("Estrategia Golden no encontrada en la DB.")
            
            golden_vector = M9DModel(0, "golden", golden_strategy_full).get_strategy_vector()
            
            params = {
                "platform": self.portfolio,
                "golden_strategy_vector": golden_vector,
                "n_clusters": self.spin_clusters.get(),
                "threshold": self.scale_threshold.get(),
                "moment": self.cb_mow_moment.get()
            }
            
            threading.Thread(target=self.run_mow_thread, args=(params,), daemon=True).start()
            self.after(100, self.check_analysis_queue)
            
        except Exception as e:
            messagebox.showerror("Error de Parámetros", f"Error al iniciar el análisis: {e}")
            self.set_status("Error al iniciar. Revise los parámetros.")
            self.btn_run_mow.config(state="normal")

    def run_mow_thread(self, params: Dict):
        """(Función Hilo) Ejecuta el pipeline de ML."""
        try:
            service = AnalysisService(params['platform'], params['golden_strategy_vector'])
            results = service.run_mow_pipeline(
                int(params['n_clusters']),
                float(params['threshold']),
                params['moment']
            )
            self.analysis_queue.put({"type": "mow_success", "data": results})
        except Exception as e:
            self.analysis_queue.put({"type": "mow_error", "data": str(e)})

    def on_run_ollama(self):
        """Inicia el hilo de análisis cualitativo (Ollama)."""
        self.set_status("Contactando a la IA (Ollama)...")
        self.btn_run_ollama.config(state="disabled")
        
        try:
            pid_a_str = self.cb_ollama_proj_a.get()
            pid_b_str = self.cb_ollama_proj_b.get()
            model_name = self.cb_ollama_model.get()
            if not pid_a_str or not pid_b_str or not model_name:
                raise ValueError("Asegúrese de seleccionar Proyecto A, Proyecto B y un Modelo de Ollama.")

            pid_a = int(pid_a_str.split(":")[0])
            pid_b = int(pid_b_str.split(":")[0])
            model_a = self.portfolio.get(pid_a)
            model_b = self.portfolio.get(pid_b)
            
            moment = self.cb_mow_moment.get()
            data_a = model_a.get_full_data_package(moment)
            data_b = model_b.get_full_data_package(moment)
            
            prompt_template = self.txt_ollama_prompt.text.get("1.0", END)
            prompt = prompt_template.format(
                data_a=json.dumps(data_a, indent=2),
                data_b=json.dumps(data_b, indent=2)
            )
            
            payload = {"model": model_name, "prompt": prompt, "stream": False}
            
            threading.Thread(target=self.run_ollama_thread, args=(payload,), daemon=True).start()
            self.after(100, self.check_analysis_queue)

        except Exception as e:
            messagebox.showerror("Error de Parámetros", f"Error al preparar la consulta: {e}")
            self.set_status("Error. ¿Seleccionó ambos proyectos y un modelo?")
            self.btn_run_ollama.config(state="normal")

    def run_ollama_thread(self, payload: Dict):
        """(Función Hilo) Llama a la API de Ollama."""
        service = AnalysisService()
        response = service.call_ollama('generate', payload)
        self.analysis_queue.put({"type": "ollama_response", "data": response})

    def check_analysis_queue(self):
        """(Función GUI) Revisa la cola de resultados de hilos."""
        if not self.running: 
            return
            
        try:
            result = self.analysis_queue.get(block=False)
            
            if result['type'] == 'mow_success':
                self.current_mow_results = result['data']
                self.draw_portfolio_charts(result['data'])
                self.set_status("Análisis MoW completado exitosamente.")
                self.btn_run_mow.config(state="normal")
            
            elif result['type'] == 'demo_success':
                self.set_status("Set de proyectos Demo cargado exitosamente. Refrescando...")
                self.load_portfolio_from_db()
                self.btn_run_mow.config(state="normal")
                
            elif result['type'] == 'demo_error':
                messagebox.showerror("Error al Cargar Demo", result['data'])
                self.set_status(f"Error al cargar datos demo: {result['data']}")
                self.btn_run_mow.config(state="normal")
                
            elif result['type'] == 'mow_error':
                messagebox.showerror("Error de Análisis MoW", result['data'])
                self.set_status(f"Error en análisis MoW: {result['data']}")
                self.btn_run_mow.config(state="normal")
            
            elif result['type'] == 'ollama_response':
                data = result['data']
                self.txt_ollama_response.text.config(state="normal")
                self.txt_ollama_response.text.delete("1.0", END)
                if "error" in data:
                    self.txt_ollama_response.text.insert("1.0", data['error'])
                    self.set_status("Error de Ollama. Revisa la consola.")
                else:
                    self.txt_ollama_response.text.insert("1.0", data['response'])
                    self.set_status("Respuesta de IA (Ollama) recibida.")
                self.txt_ollama_response.text.config(state="disabled")
                self.btn_run_ollama.config(state="normal")
            
            elif result['type'] == 'ollama_list':
                data = result['data']
                if "error" in data:
                    self.set_status(f"Error de Ollama: {data['error']}")
                else:
                    models = [m['name'] for m in data.get('models', [])]
                    self.cb_ollama_model['values'] = models
                    if models:
                        self.cb_ollama_model.set(models[0])
                        self.on_ollama_model_select()
                    self.set_status("Modelos de Ollama cargados.")
            
            elif result['type'] == 'ollama_show':
                data = result['data']
                self.txt_ollama_info.text.config(state="normal")
                self.txt_ollama_info.text.delete("1.0", END) 
                if "error" in data:
                    self.txt_ollama_info.text.insert("1.0", data['error'])
                else:
                    details = (
                        f"Familia: {data.get('details', {}).get('family', 'N/A')}\n"
                        f"Parámetros: {data.get('details', {}).get('parameter_size', 'N/A')}\n"
                        f"Quantización: {data.get('details', {}).get('quantization_level', 'N/A')}\n"
                        f"Modificado: {data.get('modified_at', 'N/A').split('T')[0]}"
                    )
                    self.txt_ollama_info.text.insert("1.0", details)
                self.txt_ollama_info.text.config(state="disabled")

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error en check_analysis_queue: {e}")
            self.running = False
            
        if self.running:
            self._after_job_id = self.after(100, self.check_analysis_queue)

    def on_project_selected(self, event=None):
        """Actualiza la Pestaña 2 cuando se selecciona un proyecto."""
        try:
            pid_str = self.cb_project_select.get()
            if not pid_str:
                self.clear_project_charts()
                return
                
            pid = int(pid_str.split(":")[0])
            model = self.portfolio.get(pid)
            if not model: return
            
            vme_a, pbt_a = model.vme_results.get('t0'), model.pbt_results.get('t0')
            vme_b, pbt_b = model.vme_results.get('t1'), model.pbt_results.get('t1')
            
            if vme_a is None or pbt_a is None:
                self.set_status(f"Proyecto {pid} no tiene Momento 't0'. Importe un CSV.")
                self.clear_project_charts()
                return

            if vme_b is None:
                scores_t0 = model.scores['t0']
                scores_t1 = scores_t0.copy() 
                model.set_reality_scores(scores_t1, 't1')
                vme_b, pbt_b = model.calculate_vme('t1')
                self.set_status(f"Proyecto {pid} cargado. (Momento t1 no encontrado, mostrando t0 vs t0)")
                
            self.draw_project_charts(model.project_name, vme_a, vme_b, pbt_a)

        except Exception as e:
            messagebox.showerror("Error al Cargar Proyecto", str(e))
            self.clear_project_charts()
    
    def on_export(self, export_type: str):
        self.set_status(f"Preparando exportación: {export_type}...")
        try:
            path = None 
            if export_type == 'cluster_csv':
                if 'clustering_df' not in self.current_mow_results:
                    raise ValueError("No hay datos de clúster para exportar. Ejecute el análisis MoW.")
                path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
                if path: ExportService.export_to_csv(self.current_mow_results['clustering_df'], path)
                    
            elif export_type == 'importance_csv':
                if 'importance_df' not in self.current_mow_results:
                    raise ValueError("No hay datos de causa raíz. Ejecute el análisis MoW.")
                path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
                if path: ExportService.export_to_csv(self.current_mow_results['importance_df'], path)
            
            elif export_type == 'project_pdf' or export_type == 'project_json':
                pid_str = self.cb_project_select.get()
                if not pid_str: raise ValueError("No hay ningún proyecto seleccionado.")
                pid = int(pid_str.split(":")[0])
                model = self.portfolio.get(pid)
                if not model: raise ValueError("Proyecto no encontrado.")
                
                moment_to_export = self.cb_mow_moment.get()
                if moment_to_export not in model.scores:
                    raise ValueError(f"El proyecto seleccionado no tiene datos para el momento '{moment_to_export}'")
                
                data_pkg = model.get_full_data_package(moment_to_export)
                
                if export_type == 'project_pdf':
                    path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
                    if path: ExportService.export_to_pdf(data_pkg, path)
                else: # project_json
                    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("M9D JSON Files", "*.json")])
                    if path: ExportService.export_to_json(data_pkg, path)
            
            if path:
                self.set_status(f"Exportación exitosa a {path}")

        except Exception as e:
            messagebox.showerror("Error de Exportación", str(e))
            self.set_status(f"Error de exportación: {e}")

    # --- Funciones de Utilidad y Dibujo ---
    
    def draw_in_frame(self, frame: btk.Frame, fig: plt.Figure):
        """Limpia un frame y dibuja una figura de matplotlib."""
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw_portfolio_charts(self, mow_results: Dict):
        """Dibuja los 4 gráficos de la pestaña MoW."""
        try:
            # 1. Gráfico de Clústeres (PCA)
            fig1, ax1 = plt.subplots(figsize=(7, 6))
            sns.scatterplot(data=mow_results['clustering_df'], x='PC1', y='PC2', hue='Cluster',
                            palette='viridis', s=100, alpha=0.7, legend='full', ax=ax1)
            ax1.set_title('Gráfico MoW 1: Clústeres (PCA)')
            ax1.set_xlabel('Componente Principal 1'); ax1.set_ylabel('Componente Principal 2')
            ax1.grid(linestyle='--', alpha=0.5)
            fig1.tight_layout()
            self.draw_in_frame(self.charts_portfolio['cluster_frame'], fig1)
            
            # 2. Gráfico Radar de Clústeres (VME)
            fig2, ax2 = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
            centers = mow_results['cluster_centers']
            num_vars = len(VME_LABELS)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
            def close_loop(data): return np.concatenate((data, [data[0]]))
            
            colors = plt.cm.get_cmap('viridis', len(centers))
            for i, center in enumerate(centers):
                ax2.plot(angles, close_loop(center), 'o-', lw=2, label=f"Clúster {i}", color=colors(i/len(centers)))
                ax2.fill(angles, close_loop(center), alpha=0.2, color=colors(i/len(centers)))
                
            ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(VME_LABELS)
            ax2.set_title("Gráfico MoW 2: Perfiles de Clúster (VME)")
            ax2.set_ylim(-3, 3); ax2.grid(True)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
            fig2.tight_layout()
            self.draw_in_frame(self.charts_portfolio['cluster_radar_frame'], fig2)
            
            # 3. Gráfico de Causa Raíz (RF)
            fig3, ax3 = plt.subplots(figsize=(7, 6))
            importance_df = mow_results['importance_df'].head(15)
            sns.barplot(data=importance_df, x='Importancia', y='Factor (S_i,j)', orient='h', palette='rocket', ax=ax3)
            ax3.set_title('Gráfico MoW 3: Causa Raíz (Top 15 Factores)')
            ax3.set_xlabel('Importancia (Random Forest)')
            fig3.tight_layout()
            self.draw_in_frame(self.charts_portfolio['importance_frame'], fig3)
            
            # 4. Gráfico de Red (NetworkX)
            fig4, ax4 = plt.subplots(figsize=(7, 6))
            sim_matrix = mow_results['similarity_matrix_full']
            cluster_df = mow_results['clustering_df']
            
            all_pids_map = {pid: i for i, pid in enumerate(mow_results['similarity_df']['ProjectID'])}
            comparable_pids = cluster_df['ProjectID'].tolist()
            indices = [all_pids_map.get(pid) for pid in comparable_pids if pid in all_pids_map]
            
            if not indices:
                raise Exception("No se encontraron IDs comparables en el mapa de PIDs.")
                
            sim_matrix_filtered = sim_matrix[np.ix_(indices, indices)]
            
            sim_df_adj = pd.DataFrame(sim_matrix_filtered, index=comparable_pids, columns=comparable_pids)
            
            sim_df_adj[sim_df_adj < self.scale_threshold.get()] = 0
            
            G = nx.from_pandas_adjacency(sim_df_adj)
            G.remove_edges_from(nx.selfloop_edges(G))
            
            colors = plt.cm.get_cmap('viridis', self.spin_clusters.get())
            colors_map = cluster_df.set_index('ProjectID')['Cluster'].to_dict()
            node_colors = [colors(colors_map.get(node, -1) / (self.spin_clusters.get()-1)) for node in G.nodes()]
            
            nx.draw_kamada_kawai(G, ax=ax4, with_labels=True, node_color=node_colors, 
                                 font_size=8, alpha=0.8, node_size=500, labels={pid: name for pid, name in cluster_df[['ProjectID', 'ProjectName']].values})
            ax4.set_title("Gráfico MoW 4: Red de Similitud (NetworkX)")
            fig4.tight_layout()
            self.draw_in_frame(self.charts_portfolio['network_frame'], fig4)
            
        except Exception as e:
            self.set_status(f"Error al dibujar gráficos MoW: {e}")

    def draw_project_charts(self, project_name: str, vme_a: np.ndarray, vme_b: np.ndarray, pbt_a: np.ndarray):
        """Dibuja los 2 gráficos de la pestaña M9D."""
        try:
            # 1. Gráfico de Radar
            fig1, ax1 = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
            num_vars = len(VME_LABELS)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
            def close_loop(data): return np.concatenate((data, [data[0]]))
            
            ax1.plot(angles, close_loop(vme_a), 'o-', lw=2, label="Baseline (A)", color="gray", alpha=0.7)
            ax1.fill(angles, close_loop(vme_a), 'gray', alpha=0.25)
            ax1.plot(angles, close_loop(vme_b), 'o-', lw=2, label="Actual (B)", color="cyan")
            ax1.fill(angles, close_loop(vme_b), 'cyan', alpha=0.25)
            
            ax1.set_xticks(angles[:-1]); ax1.set_xticklabels(VME_LABELS)
            ax1.set_title(f"Avance VME (A vs B) - {project_name}")
            ax1.set_ylim(-3, 3); ax1.grid(True)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
            fig1.tight_layout()
            self.draw_in_frame(self.charts_project['radar_frame'], fig1)
            
            # 2. Mapa de Calor (solo del baseline t0)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(pbt_a, annot=True, fmt=".2f", cmap='RdYlGn', center=0, vmin=-3, vmax=3,
                        xticklabels=VME_LABELS, yticklabels=D_LABELS, ax=ax2)
            ax2.set_title(f"Mapa de Calor (Baseline 'A') - {project_name}")
            fig2.tight_layout()
            self.draw_in_frame(self.charts_project['heatmap_frame'], fig2)
        except Exception as e:
            self.set_status(f"Error al dibujar gráficos M9D: {e}")

    def clear_project_charts(self):
        for frame in self.charts_project.values():
            for widget in frame.winfo_children():
                widget.destroy()

    def load_portfolio_from_db(self):
        """Carga todos los proyectos de la DB y refresca la GUI."""
        self.set_status("Cargando portafolio desde la base de datos...")
        self.portfolio = self.db.load_full_portfolio()
        
        project_list = [f"{model.project_id}: {model.project_name}" for model in self.portfolio.values()]
        self.cb_project_select['values'] = project_list
        self.cb_ollama_proj_a['values'] = project_list
        self.cb_ollama_proj_b['values'] = project_list
        
        if project_list:
            self.cb_project_select.set(project_list[0])
            self.cb_ollama_proj_a.set(project_list[0])
            self.cb_ollama_proj_b.set(project_list[0])
            self.on_project_selected()
            
        self.update_strategy_dropdown()
        self.set_status(f"Portafolio cargado con {len(self.portfolio)} proyectos.")
        
    def update_strategy_dropdown(self):
        """Refresca el dropdown de estrategias desde la DB."""
        strategies = self.db.get_strategies_list()
        strategy_list = [f"{s['id']}: {s['name']} (CR: {s['cr']:.3f})" for s in strategies]
        self.cb_golden_strategy['values'] = strategy_list
        if strategy_list:
            self.cb_golden_strategy.set(strategy_list[-1])
        
        for i in self.tree_strategies.get_children():
            self.tree_strategies.delete(i)
        for s in strategies:
            self.tree_strategies.insert("", "end", values=(s['id'], s['name'], f"{s['cr']:.4f}"))

    def load_ollama_models_threaded(self):
        """Inicia un hilo para cargar la lista de modelos de Ollama."""
        self.set_status("Contactando a Ollama para listar modelos...")
        threading.Thread(target=self.run_ollama_list_thread, daemon=True).start()
        self._after_job_id = self.after(100, self.check_analysis_queue)
        
    def run_ollama_list_thread(self):
        """(Función Hilo) Llama a la API de Ollama 'tags'."""
        service = AnalysisService()
        response = service.call_ollama('tags', {})
        self.analysis_queue.put({"type": "ollama_list", "data": response})
        
    def on_ollama_model_select(self, event=None):
        """Muestra la info del modelo seleccionado."""
        model_name = self.cb_ollama_model.get()
        if not model_name: return
        
        self.set_status(f"Obteniendo info del modelo: {model_name}...")
        payload = {"name": model_name}
        threading.Thread(target=self.run_ollama_show_thread, args=(payload,), daemon=True).start()

    def run_ollama_show_thread(self, payload: Dict):
        """(Función Hilo) Llama a la API de Ollama 'show'."""
        service = AnalysisService()
        response = service.call_ollama('show', payload)
        self.analysis_queue.put({"type": "ollama_show", "data": response})

    def set_status(self, msg: str):
        """Actualiza la barra de estado."""
        print(f"STATUS: {msg}")
        if len(msg) > 200:
            msg = msg[:200] + "..."
        try:
            self.status_bar_text.set(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")
        except tk.TclError:
            pass
        
    def on_closing(self):
        """Limpia la conexión de la DB al cerrar."""
        if messagebox.askokcancel("Salir", "¿Seguro que quieres salir?"):
            self.running = False
            
            if self._after_job_id:
                self.after_cancel(self._after_job_id)
                self._after_job_id = None
                
            while not self.analysis_queue.empty():
                try: self.analysis_queue.get(block=False)
                except queue.Empty: break
            
            self.db.close()
            self.destroy()

# ======================================================================
# SECCIÓN 7: PUNTO DE ENTRADA PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    
    # --- Validar Conexión a Ollama (rápido) ---
    try:
        requests.get(OLLAMA_API_URL, timeout=1)
        print("Conexión inicial con Ollama establecida.")
    except requests.exceptions.ConnectionError:
        print("ADVERTENCIA: No se pudo conectar a Ollama.")
        print(f"La pestaña 'Análisis IA' fallará. Asegúrate de que Ollama esté corriendo en {OLLAMA_API_URL}")

    # --- Inicializar DB y App ---
    try:
        db_manager = DatabaseManager(USE_DB_TYPE, DB_CONFIG)
        app = MainApplication(db_manager)
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Error Crítico de Inicialización", 
                             f"No se pudo iniciar la aplicación.\n"
                             f"Verifique su configuración de DB (USE_DB_TYPE) o las dependencias.\n\n"
                             f"Error: {e}")
