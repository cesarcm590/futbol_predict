"""Heatmaps de jugador a partir de la ubicacion (x,y) de sus eventos.

v1: heatmap basado en ubicaciones de eventos (toques, pases, tiros, acciones
defensivas), no en tracking posicional continuo (StatsBomb Open Data no lo trae).
"""
import matplotlib.pyplot as plt
from mplsoccer import Pitch

EVENT_GROUPS = {
    "Todos los toques": None,
    "Pases": ["Pass"],
    "Tiros": ["Shot"],
    "Conducciones": ["Carry"],
    "Acciones defensivas": ["Pressure", "Interception", "Duel", "Clearance", "Block"],
}


def player_heatmap(events, player_name: str, event_group: str = "Todos los toques"):
    types = EVENT_GROUPS.get(event_group)
    pdf = events[events["player"] == player_name].dropna(subset=["location"])
    if types is not None:
        pdf = pdf[pdf["type"].isin(types)]

    xs = pdf["location"].apply(lambda l: l[0]).tolist()
    ys = pdf["location"].apply(lambda l: l[1]).tolist()

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0b1220", line_color="#5b6b82", line_zorder=2)
    fig, ax = pitch.draw(figsize=(8, 5.2))
    fig.set_facecolor("#0b1220")

    if len(xs) >= 3:
        pitch.kdeplot(
            xs, ys, ax=ax, cmap="magma", fill=True, levels=100,
            thresh=0.02, alpha=0.85, zorder=1,
        )
    pitch.scatter(xs, ys, ax=ax, s=12, color="white", edgecolors="black", linewidth=0.3, alpha=0.5, zorder=3)

    ax.set_title(f"{player_name} — {event_group} ({len(xs)} eventos)", color="white", fontsize=12)
    return fig
