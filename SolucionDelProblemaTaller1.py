"""
Taller 1 - Métodos Cuantitativos

Integrante: Alejandro Cifuentes Arroyave
Parámetros del equipo: s = 1, d = 9
"""

import pulp

# ============================================================
# 1. PARÁMETROS
# ============================================================

s = 1
d = 9

productos = ["P1", "P2", "P3"]
materias = ["M1", "M2"]
periodos = [1, 2, 3, 4]

alpha = 0.01 + 0.005 * s  # tasa de incremento de precio por periodo

precio_base = {"P1": 600000, "P2": 550000, "P3": 700000}

# Precio_{i,t} = PrecioBase_i * (1+alpha)^(t-1)
precio = {
    (i, t): precio_base[i] * (1 + alpha) ** (t - 1)
    for i in productos for t in periodos
}

# Capacidad disponible por periodo (horas/semana -> horas/mes con 52/12 semanas/mes)
horas_por_unidad = {"P1": 3, "P2": 4, "P3": 2}

capacidad = {
    t: (180 + ((-1) ** (t - 1)) * d / 100) * (52 / 12)
    for t in periodos
}

# Uso de materia prima por unidad de producto
uso_mp = {
    ("P1", "M1"): 2, ("P1", "M2"): 1,
    ("P2", "M1"): 1, ("P2", "M2"): 3,
    ("P3", "M1"): 2, ("P3", "M2"): 2,
}

# Disponibilidad máxima de compra mensual
compra_max = {"M1": 600 - 12 * s, "M2": 480 + 8 * s}

# Inventario inicial de materias primas
im_inicial = {"M1": 40, "M2": 30}

# Costo de compra de materias primas
costo_mp = {"M1": 50000, "M2": 70000}

# Demanda por periodo y producto
demanda = {
    ("P1", 1): 120, ("P2", 1): 60, ("P3", 1): 72,
    ("P1", 2): 72, ("P2", 2): 80, ("P3", 2): 60,
    ("P1", 3): 100, ("P2", 3): 130, ("P3", 3): 80,
    ("P1", 4): 60, ("P2", 4): 62, ("P3", 4): 68,
}

# Inventario inicial de producto terminado
i_inicial = {"P1": 20, "P2": 24, "P3": 16}

# Costo de almacenamiento = s% del precio base (productos) / costo de compra (MP)
costo_almacen_pct = s / 100  # 0.01

# Descuento por entrega tardía = 0.1*d%
descuento_pct = 0.1 * d / 100  # 0.009

# Tamaño de lote
lote = {"P1": 5, "P2": 1, "P3": 7}

# Inventario mínimo de seguridad al final del horizonte
inv_min_final = 20

# ============================================================
# 2. MODELO
# ============================================================

modelo = pulp.LpProblem("JCR_Planeacion_Produccion", pulp.LpMaximize)

# --- Variables de decisión ---
L = pulp.LpVariable.dicts("L", (productos, periodos), lowBound=0, cat="Integer")
X = pulp.LpVariable.dicts("X", (productos, periodos), lowBound=0, cat="Continuous")
Cmp = pulp.LpVariable.dicts("Cmp", (materias, periodos), lowBound=0, cat="Continuous")
Im = pulp.LpVariable.dicts("Im", (materias, [0] + periodos), lowBound=0, cat="Continuous")
I = pulp.LpVariable.dicts("I", (productos, [0] + periodos), lowBound=0, cat="Continuous")
V = pulp.LpVariable.dicts("V", (productos, periodos), lowBound=0, cat="Continuous")
B = pulp.LpVariable.dicts("B", (productos, [0] + periodos), lowBound=0, cat="Continuous")

# --- Función objetivo ---
ingresos = pulp.lpSum(V[i][t] * precio[(i, t)] for i in productos for t in periodos)

costo_compras = pulp.lpSum(Cmp[m][t] * costo_mp[m] for m in materias for t in periodos)

costo_almacen_mp = pulp.lpSum(
    Im[m][t] * costo_almacen_pct * costo_mp[m] for m in materias for t in periodos
)

costo_almacen_pt = pulp.lpSum(
    I[i][t] * costo_almacen_pct * precio_base[i] for i in productos for t in periodos
)

costo_descuento_atraso = pulp.lpSum(
    B[i][t - 1] * descuento_pct * precio[(i, t)] for i in productos for t in periodos
)

modelo += (
    ingresos
    - costo_compras
    - costo_almacen_mp
    - costo_almacen_pt
    - costo_descuento_atraso
), "Utilidad_Total"

# --- Restricciones ---

# Condiciones iniciales
for m in materias:
    modelo += Im[m][0] == im_inicial[m], f"Im_inicial_{m}"
for i in productos:
    modelo += I[i][0] == i_inicial[i], f"I_inicial_{i}"
    modelo += B[i][0] == 0, f"B_inicial_{i}"

for t in periodos:
    for i in productos:
        # Definición y tamaño de lotes
        modelo += X[i][t] == lote[i] * L[i][t], f"Lote_{i}_{t}"

    # Capacidad de producción mensual escalada
    modelo += (
        pulp.lpSum(horas_por_unidad[i] * X[i][t] for i in productos) <= capacidad[t]
    ), f"Capacidad_{t}"

    # Disponibilidad máxima de compra
    for m in materias:
        modelo += Cmp[m][t] <= compra_max[m], f"CompraMax_{m}_{t}"

    # Balance de inventario de materias primas
    for m in materias:
        modelo += (
            Im[m][t] == Im[m][t - 1] + Cmp[m][t]
            - pulp.lpSum(uso_mp[(i, m)] * X[i][t] for i in productos)
        ), f"BalanceMP_{m}_{t}"

    # Balance de inventario de producto terminado
    for i in productos:
        modelo += (
            I[i][t] == I[i][t - 1] + X[i][t] - V[i][t]
        ), f"BalancePT_{i}_{t}"

    # Balance de demanda y faltantes (backorders)
    for i in productos:
        modelo += (
            demanda[(i, t)] == V[i][t] + B[i][t] - B[i][t - 1]
        ), f"BalanceDemanda_{i}_{t}"

# Condición de frontera: toda la demanda debe quedar satisfecha al final
for i in productos:
    modelo += B[i][4] == 0, f"B_final_{i}"

# Inventario mínimo de seguridad final
for i in productos:
    modelo += I[i][4] >= inv_min_final, f"InvMinFinal_{i}"

# ============================================================
# 3. RESOLVER
# ============================================================

solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=120, gapRel=0.003)
modelo.solve(solver)

print("=" * 60)
print("Estado de la solución:", pulp.LpStatus[modelo.status])
print("=" * 60)

# --- Verificación de factibilidad (chequeo manual de las restricciones clave) ---
tol = 1e-3
errores = []
for t in periodos:
    cap_usada = sum(horas_por_unidad[i] * X[i][t].value() for i in productos)
    if cap_usada > capacidad[t] + tol:
        errores.append(f"Capacidad excedida en t={t}: {cap_usada:.2f} > {capacidad[t]:.2f}")
    for m in materias:
        if Cmp[m][t].value() > compra_max[m] + tol:
            errores.append(f"Compra máxima excedida {m} en t={t}")
    for i in productos:
        prev_I = i_inicial[i] if t == 1 else I[i][t - 1].value()
        if abs((prev_I + X[i][t].value() - V[i][t].value()) - I[i][t].value()) > tol:
            errores.append(f"Balance PT violado {i} en t={t}")
        prev_B = 0 if t == 1 else B[i][t - 1].value()
        if abs((V[i][t].value() + B[i][t].value() - prev_B) - demanda[(i, t)]) > tol:
            errores.append(f"Balance demanda violado {i} en t={t}")
for i in productos:
    if I[i][4].value() < inv_min_final - tol:
        errores.append(f"Inventario mínimo final violado en {i}")
    if abs(B[i][4].value()) > tol:
        errores.append(f"Backorder final no es cero en {i}")

if errores:
    print("  ERRORES DE FACTIBILIDAD ENCONTRADOS:")
    for e in errores:
        print("  -", e)
else:
    print(" Verificación de restricciones: todas se cumplen correctamente.\n")

if pulp.LpStatus[modelo.status] == "Optimal":
    print(f"\nUtilidad total óptima: ${pulp.value(modelo.objective):,.2f}\n")

    for t in periodos:
        print(f"--- Periodo {t} ---")
        for i in productos:
            print(
                f"  {i}: Lotes={L[i][t].value():.0f}  "
                f"Producción={X[i][t].value():.1f}  "
                f"Ventas={V[i][t].value():.1f}  "
                f"Inventario={I[i][t].value():.1f}  "
                f"Backorder={B[i][t].value():.1f}"
            )
        for m in materias:
            print(
                f"  {m}: Compra={Cmp[m][t].value():.1f}  "
                f"Inventario={Im[m][t].value():.1f}"
            )
        print()
