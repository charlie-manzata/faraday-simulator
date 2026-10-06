# simulatore_faraday_streamlit.py
# -*- coding: utf-8 -*-

import ast
import math

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch


# ============================================================
# Configurazione pagina
# ============================================================

st.set_page_config(
    page_title="Simulatore di Faraday–Neumann–Lenz",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 Simulatore di Faraday–Neumann–Lenz")

st.caption(
    "Variazioni finite del flusso: ε = −ΔΦ/Δt. "
    "La derivata non viene utilizzata nel calcolo numerico."
)


# ============================================================
# Valutazione sicura di B(t)
# ============================================================

ALLOWED_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "abs": np.abs,
}

ALLOWED_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
}

ALLOWED_NAMES = (
    set(ALLOWED_FUNCTIONS)
    | set(ALLOWED_CONSTANTS)
    | {"t"}
)


def validate_expression(expr):
    """
    Controlla che l'espressione B(t) contenga soltanto
    operatori, funzioni e nomi autorizzati.
    """

    expr = expr.replace("^", "**")

    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):

        # ----------------------------------------------------
        # Chiamate a funzione, ad esempio sin(t)
        # ----------------------------------------------------

        if isinstance(node, ast.Call):

            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in ALLOWED_FUNCTIONS
            ):
                raise ValueError(
                    "Funzione non consentita."
                )

        # ----------------------------------------------------
        # Nomi, ad esempio t, sin, pi
        # ----------------------------------------------------

        elif isinstance(node, ast.Name):

            if node.id not in ALLOWED_NAMES:
                raise ValueError(
                    f"Nome non consentito: {node.id}"
                )

        # ----------------------------------------------------
        # Elementi sintattici consentiti
        # ----------------------------------------------------

        elif isinstance(
            node,
            (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.Mod,
                ast.USub,
                ast.UAdd,
                ast.Load,
                ast.Constant,
                ast.Call,
            ),
        ):
            pass

        else:
            raise ValueError(
                "L'espressione contiene un elemento non consentito."
            )

    return expr


def compile_expression(expr):
    """
    Compila l'espressione.

    IMPORTANTE:
    questa funzione NON usa st.cache_data.

    compile() restituisce un oggetto di tipo 'code', che non è
    serializzabile con pickle. Per questo non deve essere restituito
    da una funzione decorata con @st.cache_data.
    """

    safe_expr = validate_expression(expr)

    return compile(
        safe_expr,
        "<B(t)>",
        "eval",
    )


def evaluate_B(compiled, t):
    """
    Valuta l'espressione compilata.
    """

    env = {
        "t": t,
        **ALLOWED_FUNCTIONS,
        **ALLOWED_CONSTANTS,
    }

    return eval(
        compiled,
        {"__builtins__": {}},
        env,
    )


# ============================================================
# Disegno spira
# ============================================================

def draw_loop(area):

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    ax.set_aspect("equal")
    ax.axis("off")

    # --------------------------------------------------------
    # Spira vista in prospettiva
    # --------------------------------------------------------

    loop = Ellipse(
        (0, 0),
        width=5.4,
        height=2.8,
        fill=False,
        linewidth=2.5,
    )

    ax.add_patch(loop)

    # --------------------------------------------------------
    # Frecce per indicare il verso positivo sulla spira
    # --------------------------------------------------------

    for angle in [25, 205]:

        theta = math.radians(angle)

        x = 2.7 * math.cos(theta)
        y = 1.4 * math.sin(theta)

        dx = -0.45 * math.sin(theta)
        dy = 0.45 * math.cos(theta)

        ax.add_patch(
            FancyArrowPatch(
                (x - dx, y - dy),
                (x + dx, y + dy),
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.7,
            )
        )

    # --------------------------------------------------------
    # Vettore normale n
    # --------------------------------------------------------

    ax.add_patch(
        FancyArrowPatch(
            (0, 0),
            (0, 2.8),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.2,
        )
    )

    ax.text(
        0.15,
        3.0,
        r"$\vec n$",
        fontsize=16,
    )

    # --------------------------------------------------------
    # Campo magnetico parallelo a n
    # --------------------------------------------------------

    ax.add_patch(
        FancyArrowPatch(
            (0, -0.2),
            (0, -3.0),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.2,
        )
    )

    ax.text(
        0.15,
        -3.35,
        r"$\vec B \parallel \vec n$",
        fontsize=14,
    )

    # --------------------------------------------------------
    # Flusso
    # --------------------------------------------------------

    ax.text(
        0,
        0,
        r"$\Phi = BA$",
        fontsize=18,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.85,
        ),
    )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    ax.text(
        0,
        -3.85,
        f"$A = {area:g}$",
        ha="center",
        fontsize=11,
    )

    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-4.1, 4.1)

    ax.set_title(
        "Spira, normale e campo magnetico",
        fontsize=13,
    )

    return fig


# ============================================================
# Interfaccia
# ============================================================

with st.sidebar:

    st.header("Parametri")

    expression = st.text_input(
        "Campo magnetico B(t)",
        value="2*t^2 + 1",
        help=(
            "Esempi: 2*t^2+1, sin(t), 3*cos(2*t), "
            "exp(-t), 5*t"
        ),
    )

    t_min = st.number_input(
        "t iniziale",
        value=0.0,
    )

    t_max = st.number_input(
        "t finale",
        value=2.0,
    )

    area = st.number_input(
        "Area della spira A",
        min_value=0.000001,
        value=1.0,
    )

    N = st.number_input(
        "Numero di suddivisioni N",
        min_value=1,
        max_value=200,
        value=4,
        step=1,
    )

    st.divider()

    st.markdown("**Funzioni disponibili**")

    st.code(
        "sin  cos  tan\n"
        "exp  sqrt  log\n"
        "abs  sinh  cosh\n"
        "tanh  pi  e",
        language="text",
    )

    st.caption(
        "Le funzioni trigonometriche usano i radianti. "
        "Puoi usare ^ oppure ** per le potenze."
    )


# ============================================================
# Calcolo
# ============================================================

try:

    if t_max <= t_min:
        st.error(
            "Il tempo finale deve essere maggiore "
            "del tempo iniziale."
        )
        st.stop()

    if area <= 0:
        st.error(
            "L'area deve essere positiva."
        )
        st.stop()

    # --------------------------------------------------------
    # Compilazione dell'espressione
    # --------------------------------------------------------

    compiled = compile_expression(expression)

    # --------------------------------------------------------
    # Estremi degli intervalli
    # --------------------------------------------------------

    times = np.linspace(
        t_min,
        t_max,
        int(N) + 1,
    )

    # --------------------------------------------------------
    # Valori di B nei punti di suddivisione
    # --------------------------------------------------------

    B_values = np.asarray(
        evaluate_B(
            compiled,
            times,
        ),
        dtype=float,
    )

    # Se B(t) è costante, np.asarray restituisce uno scalare.
    # Lo trasformiamo in un array della stessa dimensione di times.

    if B_values.ndim == 0:
        B_values = np.full_like(
            times,
            float(B_values),
        )

    # --------------------------------------------------------
    # Controllo valori non numerici
    # --------------------------------------------------------

    if not np.all(np.isfinite(B_values)):
        raise ValueError(
            "B(t) non è definita in tutto l'intervallo scelto."
        )

except Exception as exc:

    st.error(
        f"Errore nell'espressione o nei parametri: {exc}"
    )

    st.stop()


# ============================================================
# Calcolo delle variazioni finite
# ============================================================

dt = np.diff(times)
dB = np.diff(B_values)

# ------------------------------------------------------------
# Flusso magnetico
#
# B parallelo a n:
#
# Φ = BA
# ------------------------------------------------------------

phi = area * B_values
dphi = np.diff(phi)

# ------------------------------------------------------------
# Legge di Faraday-Neumann-Lenz
#
# ε = - ΔΦ / Δt
# ------------------------------------------------------------

emf = -dphi / dt


# ============================================================
# Grafici
# ============================================================

col_graph, col_loop = st.columns(
    [1.35, 1]
)


# ============================================================
# Grafico B(t)
# ============================================================

with col_graph:

    st.subheader("Campo magnetico $B(t)$")

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    # --------------------------------------------------------
    # Curva esatta
    # --------------------------------------------------------

    t_dense = np.linspace(
        t_min,
        t_max,
        1200,
    )

    B_dense = np.asarray(
        evaluate_B(
            compiled,
            t_dense,
        ),
        dtype=float,
    )

    if B_dense.ndim == 0:
        B_dense = np.full_like(
            t_dense,
            float(B_dense),
        )

    ax.plot(
        t_dense,
        B_dense,
        linewidth=2.4,
        label="B(t) esatto",
    )

    # --------------------------------------------------------
    # Approssimazione lineare a tratti
    # --------------------------------------------------------

    ax.plot(
        times,
        B_values,
        marker="o",
        markersize=5,
        linewidth=1.7,
        label=(
            f"Approssimazione lineare "
            f"(N = {int(N)})"
        ),
    )

    # --------------------------------------------------------
    # Suddivisioni
    # --------------------------------------------------------

    for ti in times:

        ax.axvline(
            ti,
            linestyle=":",
            linewidth=0.7,
            alpha=0.35,
        )

    ax.set_xlabel("t")
    ax.set_ylabel("B")

    ax.set_title(
        f"B(t) e approssimazione lineare — "
        f"N = {int(N)}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    ax.margins(x=0.03)

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)

    st.info(
        "La curva continua è B(t). "
        "La spezzata collega i valori B(tᵢ): "
        "aumentando N, l'approssimazione lineare "
        "segue sempre meglio la curva."
    )


# ============================================================
# Disegno spira
# ============================================================

with col_loop:

    st.subheader("Spira")

    fig_loop = draw_loop(area)

    st.pyplot(
        fig_loop,
        use_container_width=True,
    )

    plt.close(fig_loop)

    st.latex(
        r"\Phi = BA \qquad "
        r"(\vec B \parallel \vec n)"
    )


# ============================================================
# Riepilogo
# ============================================================

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "N",
    int(N),
)

c2.metric(
    "Δt per intervallo",
    f"{dt[0]:.6g}",
)

c3.metric(
    "Area A",
    f"{area:.6g}",
)

c4.metric(
    "Intervalli",
    int(N),
)


# ============================================================
# Calcolo con variazioni finite
# ============================================================

st.subheader(
    "Calcolo con variazioni finite"
)

st.latex(
    r"\Delta B_i = B(t_{i+1}) - B(t_i)"
)

st.latex(
    r"\Delta\Phi_i = \Phi_{i+1} - \Phi_i"
    r" = A\,[B(t_{i+1}) - B(t_i)]"
)

st.latex(
    r"\boxed{\mathcal{E}_i = "
    r"-\frac{\Delta\Phi_i}{\Delta t_i}}"
)


# ============================================================
# Tabella
# ============================================================

rows = []

for i in range(int(N)):

    rows.append(
        {
            "i": i + 1,
            "tᵢ": times[i],
            "tᵢ₊₁": times[i + 1],
            "Bᵢ": B_values[i],
            "Bᵢ₊₁": B_values[i + 1],
            "ΔB": dB[i],
            "Δt": dt[i],
            "Φᵢ": phi[i],
            "Φᵢ₊₁": phi[i + 1],
            "ΔΦ": dphi[i],
            "ε = −ΔΦ/Δt": emf[i],
        }
    )


st.dataframe(
    rows,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# Spiegazione finale
# ============================================================

with st.expander(
    "Che cosa sta facendo l'applet?"
):

    st.markdown(
        """
Per ogni intervallo temporale l'applet **non calcola la derivata**.

Divide invece l'intervallo totale in N parti uguali e considera:

1. la variazione del campo:

   `ΔB = B(tᵢ₊₁) − B(tᵢ)`

2. la variazione del flusso:

   `ΔΦ = A · ΔB`

3. la forza elettromotrice media nell'intervallo:

   `ε = −ΔΦ / Δt`

Nel grafico, gli stessi estremi `tᵢ` e `tᵢ₊₁` sono utilizzati
per costruire la spezzata che approssima linearmente B(t).

Quindi modificando N si vede contemporaneamente:

- come cambia l'approssimazione lineare di B(t);
- come cambiano i singoli ΔB;
- come cambiano i singoli ΔΦ;
- come cambiano le f.e.m. medie nei vari intervalli.
        """
    )