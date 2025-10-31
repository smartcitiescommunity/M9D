# ======================================================================
# SCRIPT DE PRECARGA DEMO PARA MoW
# ======================================================================
# Propósito: Inyectar datos de demostración en la base de datos
#            para resolver el problema de "arranque en frío".
# Ejecución: python precarga_demo.py (SOLO UNA VEZ)
# ======================================================================

import numpy as np
import json
import configparser
import sqlalchemy as db
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, ForeignKey, Text as DBText

# --- Copia de Clases Lógicas Necesarias ---
# (Duplicamos las clases aquí para que el script sea 100% standalone)

class AHPValidator:
    RI_SAATY = { 3: 0.58, 9: 1.45 }
    def __init__(self, pairwise_matrix: np.ndarray):
        self.matrix = np.array(pairwise_matrix)
        self.n = self.matrix.shape[0]
        if self.n not in self.RI_SAATY:
            raise ValueError(f"Tamaño de matriz AHP no soportado: {self.n}")
    def calculate(self) -> Tuple[np.ndarray, float]:
        col_sums = self.matrix.sum(axis=0)
        norm_matrix = self.matrix / col_sums
        weights = norm_matrix.mean(axis=1)
        weights /= np.sum(weights)
        weighted_sum_vector = np.dot(self.matrix, weights)
        lambda_vector = weighted_sum_vector / weights
        lambda_max = np.mean(lambda_vector)
        ci = (lambda_max - self.n) / (self.n - 1) if (self.n - 1) > 0 else 0
        ri = self.RI_SAATY.get(self.n)
        consistency_ratio = ci / ri if ri != 0 else 0
        return weights, consistency_ratio

class IAGenerator:
    @staticmethod
    def get_ahp_matrix(labels: List[str], importance_profile: Dict[str, float]) -> np.ndarray:
        n = len(labels)
        matrix = np.ones((n, n))
        default_importance = importance_profile.get("default", 1.0)
        imp_values = np.array([importance_profile.get(label, default_importance) for label in labels])
        for i in range(n):
            for j in range(i + 1, n):
                ratio = imp_values[i] / imp_values[j] * np.random.uniform(0.95, 1.05)
                matrix[i, j] = ratio
                matrix[j, i] = 1.0 / ratio
        return matrix
    
    @staticmethod
    def generate_ahp_strategy(dim_profile: Dict, state_profile: Dict) -> Dict:
        weights_to_save = {'v_j': {}}
        matrix_dims = IAGenerator.get_ahp_matrix([f"D{i+1}" for i in range(9)], dim_profile)
        weights_dims, cr_dims = AHPValidator(matrix_dims).calculate()
        if cr_dims > 0.10:
            return IAGenerator.generate_ahp_strategy(dim_profile, state_profile)
        weights_to_save['w_i'] = weights_dims
        weights_to_save['cr_dims'] = cr_dims
        for group in ['past', 'present', 'future']:
            labels = [f"T{i+1}" for i in range(3)] # Etiquetas placeholder
            matrix = IAGenerator.get_ahp_matrix(labels, state_profile[group])
            weights, cr = AHPValidator(matrix).calculate()
            if cr > 0.10:
                return IAGenerator.generate_ahp_strategy(dim_profile, state_profile)
            weights_to_save['v_j'][group] = weights
        return weights_to_save

    @staticmethod
    def get_scores_matrix(profile_name: str, base_scores: np.ndarray = None) -> np.ndarray:
        PROFILES = {
            "Neutro": {"mean": 0.0, "std": 0.1},
            "Baseline Difícil": {"mean": -1.5, "std": 1.0},
            "Baseline Optimista": {"mean": 1.0, "std": 1.0},
            "Mejora Moderada": {"mean": 0.8, "std": 0.3},
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

class DatabaseManager:
    def __init__(self, db_type: str = 'sqlite', config: Dict = {}):
        if db_type == 'mysql':
            conn_str = f"mysql+pymysql://{config['mysql_user']}:{config['mysql_pass']}@{config['mysql_host']}:{config['mysql_port']}/{config['mysql_db_name']}"
        else: # sqlite por defecto
            conn_str = f"sqlite:///{config['sqlite_db_name']}"
            
        self.engine = create_engine(conn_str)
        self.metadata = MetaData()
        self.tbl_strategies = Table('mow_strategies', self.metadata,
            Column('id', Integer, primary_key=True), Column('name', String(255), unique=True),
            Column('weights_json', DBText, nullable=False), Column('consistency_ratio', Float, nullable=False))
        self.tbl_projects = Table('mow_projects', self.metadata,
            Column('id', Integer, primary_key=True), Column('name', String(255), unique=True),
            Column('strategy_id', Integer, ForeignKey('mow_strategies.id')))
        self.tbl_realities = Table('mow_realities', self.metadata,
            Column('id', Integer, primary_key=True), Column('project_id', Integer, ForeignKey('mow_projects.id')),
            Column('moment', String(50), nullable=False), Column('scores_json', DBText, nullable=False),
            db.UniqueConstraint('project_id', 'moment', name='uidx_proj_moment'))
        self.metadata.create_all(self.engine)
        self.conn = self.engine.connect()

    def save_strategy(self, name: str, weights: Dict, cr: float) -> int:
        weights_json = json.dumps({
            'w_i': weights['w_i'].tolist(),
            'v_j': {k: v.tolist() for k, v in weights['v_j'].items()}
        })
        try:
            ins = self.tbl_strategies.insert().values(name=name, weights_json=weights_json, consistency_ratio=cr)
            result = self.conn.execute(ins)
            self.conn.commit()
            return result.inserted_primary_key[0]
        except db.exc.IntegrityError:
            upd = self.tbl_strategies.update().where(self.tbl_strategies.c.name == name).values(weights_json=weights_json, consistency_ratio=cr)
            self.conn.execute(upd); self.conn.commit()
            query = self.tbl_strategies.select().where(self.tbl_strategies.c.name == name)
            return self.conn.execute(query).fetchone()[0]

    def save_project(self, name: str, strategy_id: int) -> int:
        try:
            ins = self.tbl_projects.insert().values(name=name, strategy_id=strategy_id)
            result = self.conn.execute(ins)
            self.conn.commit()
            return result.inserted_primary_key[0]
        except db.exc.IntegrityError:
            upd = self.tbl_projects.update().where(self.tbl_projects.c.name == name).values(strategy_id=strategy_id)
            self.conn.execute(upd); self.conn.commit()
            query = self.tbl_projects.select().where(self.tbl_projects.c.name == name)
            return self.conn.execute(query).fetchone()[0]
        
    def save_reality(self, project_id: int, moment: str, scores: np.ndarray):
        scores_json = json.dumps(scores.tolist())
        delete_q = self.tbl_realities.delete().where(
            (self.tbl_realities.c.project_id == project_id) & (self.tbl_realities.c.moment == moment))
        self.conn.execute(delete_q)
        ins = self.tbl_realities.insert().values(project_id=project_id, moment=moment, scores_json=scores_json)
        self.conn.execute(ins)
        self.conn.commit()

    def close(self):
        self.conn.close()

# --- SCRIPT PRINCIPAL DE PRECARGA ---
def precargar_demo():
    print("Iniciando precarga de datos demo...")
    
    # 1. Leer el archivo de configuración .ini
    config = configparser.ConfigParser()
    config.read('m9d.ini')
    
    db_type = config.get('Database', 'db_type', fallback='sqlite')
    db_config_dict = {}
    if db_type == 'mysql':
        db_config_dict['mysql_user'] = config.get('Database', 'mysql_user')
        db_config_dict['mysql_pass'] = config.get('Database', 'mysql_pass')
        db_config_dict['mysql_host'] = config.get('Database', 'mysql_host')
        db_config_dict['mysql_port'] = config.get('Database', 'mysql_port')
        db_config_dict['mysql_db_name'] = config.get('Database', 'mysql_db_name')
    else:
        db_config_dict['sqlite_db_name'] = config.get('Database', 'sqlite_db_name', fallback='mow_portfolio_v2.db')

    try:
        db_manager = DatabaseManager(db_type, db_config_dict)
    except Exception as e:
        print(f"Error fatal al conectar con la DB: {e}")
        return

    try:
        # 2. Crear Estrategia Demo
        print("Paso 1: Creando Estrategia Demo...")
        strategy_dims = {"D9": 9.0, "D4": 7.0, "default": 1.0}
        strategy_states = {
            "past": {"T2(P-)": 7.0, "default": 1.0},
            "present": {"T5(R-)": 7.0, "default": 1.0},
            "future": {"T8(F-)": 9.0, "default": 1.0}
        }
        
        strategy_to_save = IAGenerator.generate_ahp_strategy(strategy_dims, strategy_states)
        strategy_id = db_manager.save_strategy("Estrategia Demo (Riesgo)", strategy_to_save, strategy_to_save['cr_dims'])
        print(f"Estrategia 'Estrategia Demo (Riesgo)' guardada con ID: {strategy_id}")

        # 3. Crear Proyectos Demo
        print("Paso 2: Creando Proyectos Demo y sus Realidades (t0 y t1)...")
        project_profiles = {
            "Proyecto Crisis (Demo)": "Crisis (D4,D9)",
            "Proyecto Oportunidad (Demo)": "Oportunidad (D3,D7)",
            "Proyecto Estable (Demo)": "Baseline Optimista"
        }
        
        for name, profile in project_profiles.items():
            project_id = db_manager.save_project(name, strategy_id)
            
            # Generar y guardar t0
            scores_t0 = IAGenerator.get_scores_matrix(profile)
            db_manager.save_reality(project_id, 't0', scores_t0)
            
            # Generar y guardar t1
            scores_t1 = IAGenerator.get_scores_matrix("Mejora Moderada", base_scores=scores_t0)
            db_manager.save_reality(project_id, 't1', scores_t1)
            print(f"Proyecto '{name}' (ID: {project_id}) creado con momentos t0 y t1.")
        
        db_manager.close()
        print("\n¡ÉXITO! La base de datos ha sido precargada con datos de demostración.")
        print("Ya puedes ejecutar la aplicación principal 'mow_app_v3_0.py'.")
        
    except Exception as e:
        print(f"\nERROR DURANTE LA PRECARGA: {e}")
        db_manager.close()

if __name__ == "__main__":
    precargar_demo()
