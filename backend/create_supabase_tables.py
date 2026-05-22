"""
Script para crear las tablas necesarias en Supabase.
Ejecutar una sola vez: py -3 create_supabase_tables.py
"""
import httpx
import sys

SUPABASE_URL = "https://irlznyjhsbtirydyoebr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlybHpueWpoc2J0aXJ5ZHlvZWJyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkzOTA1MzQsImV4cCI6MjA5NDk2NjUzNH0.UtdBP28_-N02st3c9LcUGNV5_Vlq_pprJzjCAdu149Y"

SQL = """
-- Tabla de KPIs del pipeline
CREATE TABLE IF NOT EXISTS pipeline_kpis (
    id TEXT PRIMARY KEY DEFAULT 'latest',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    pipeline_status TEXT,
    total_socios INTEGER DEFAULT 0,
    tasa_mora_pct REAL DEFAULT 0,
    tasa_riesgo_alto REAL DEFAULT 0,
    socios_en_alerta INTEGER DEFAULT 0,
    socios_sobreendeudados INTEGER DEFAULT 0,
    menores_bloqueados INTEGER DEFAULT 0,
    menores_detectados INTEGER DEFAULT 0,
    alertas_desvio INTEGER DEFAULT 0,
    clusters_count INTEGER DEFAULT 0,
    avg_credit_score REAL DEFAULT 0,
    tasa_imputacion_ingresos_pct REAL DEFAULT 0,
    distribucion_riesgo JSONB,
    distribucion_alerta JSONB,
    distribucion_canal JSONB
);

-- Tabla de segmentos por producto
CREATE TABLE IF NOT EXISTS pipeline_segmentos (
    id TEXT PRIMARY KEY,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    prod_bancario TEXT,
    n_socios INTEGER DEFAULT 0,
    tasa_mora_pct REAL DEFAULT 0,
    score_riesgo REAL DEFAULT 0
);

-- Tabla de clusters
CREATE TABLE IF NOT EXISTS pipeline_clusters (
    id TEXT PRIMARY KEY,
    cluster_name TEXT,
    n_socios INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabla de resultados de scoring individual
CREATE TABLE IF NOT EXISTS scoring_results (
    id TEXT PRIMARY KEY,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    cliente_id REAL,
    riesgo_global REAL DEFAULT 0,
    elegibilidad_credito BOOLEAN DEFAULT FALSE,
    canal_cobranza TEXT,
    dia_pago_sugerido INTEGER,
    alertas_activas JSONB,
    acciones_priorizadas JSONB,
    bloqueos JSONB,
    agentes JSONB
);

-- Tabla de datos de socios
CREATE TABLE IF NOT EXISTS socios_data (
    id TEXT PRIMARY KEY,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    v_ah_cliente REAL,
    edad REAL,
    sexo TEXT,
    saldo_disponible REAL,
    ingresos REAL,
    egresos REAL,
    credito REAL,
    dias_sin_movimiento REAL,
    prod_bancario REAL,
    estado_cta TEXT,
    oficina_cta REAL,
    tipo_cuenta TEXT,
    desviacion_maxima REAL,
    fuente_alerta TEXT
);

-- Habilitar RLS pero permitir acceso anonimo para la API key
ALTER TABLE pipeline_kpis ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_segmentos ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE scoring_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE socios_data ENABLE ROW LEVEL SECURITY;

-- Politicas de acceso abierto (para desarrollo)
CREATE POLICY IF NOT EXISTS "Allow all on pipeline_kpis" ON pipeline_kpis FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY IF NOT EXISTS "Allow all on pipeline_segmentos" ON pipeline_segmentos FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY IF NOT EXISTS "Allow all on pipeline_clusters" ON pipeline_clusters FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY IF NOT EXISTS "Allow all on scoring_results" ON scoring_results FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY IF NOT EXISTS "Allow all on socios_data" ON socios_data FOR ALL USING (true) WITH CHECK (true);
"""

def main():
    print("Creando tablas en Supabase...")
    
    # Split SQL into individual statements
    statements = [s.strip() for s in SQL.split(';') if s.strip()]
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    
    # Use the SQL endpoint (rpc)
    url = f"{SUPABASE_URL}/rest/v1/rpc"
    
    # Try using the SQL API directly
    # Supabase doesn't have a direct SQL endpoint via REST, we need to use
    # the management API or the dashboard. Let's try a different approach.
    
    # Actually, for Supabase we can use the pg REST API's rpc endpoint
    # but we need to create a function first. Let's just print the SQL
    # and ask the user to run it in the Supabase dashboard.
    
    print("\n" + "=" * 60)
    print("INSTRUCCIONES:")
    print("=" * 60)
    print()
    print("1. Abre tu proyecto en Supabase Dashboard:")
    print(f"   {SUPABASE_URL.replace('.co', '.co').replace('https://', 'https://supabase.com/dashboard/project/')}")
    print()
    print("2. Ve a 'SQL Editor' en el menu lateral")
    print()
    print("3. Pega y ejecuta el siguiente SQL:")
    print()
    print("-" * 60)
    print(SQL)
    print("-" * 60)
    print()
    print("4. Haz clic en 'Run' para crear las tablas")
    print()
    
    # Also try to verify connection
    print("Verificando conexion a Supabase...")
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/",
            headers=headers,
            timeout=10.0,
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            print("  Conexion exitosa!")
        else:
            print(f"  Respuesta: {resp.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
