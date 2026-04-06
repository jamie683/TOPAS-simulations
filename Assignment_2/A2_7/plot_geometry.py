"""
Section 2.7: Geometry schematic for report.

Draws a labelled cross-section of the collimator setup:
  beam source → brass collimator → water phantom
with correct positions, dimensions, and beam divergence cone.

Usage:
    python3 plot_geometry.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT = "output/geometry_schematic.png"


def main():
    fig, ax = plt.subplots(figsize=(14, 6))

    # ---- Coordinate system (Z horizontal, Y vertical, all in cm) ----
    # Beam source at Z = -100, phantom front at Z = 0, back at Z = 20
    # Collimator centred at Z = -50, thickness 5 cm

    # World bounds for drawing
    z_min, z_max = -110, 30
    y_min, y_max = -18, 18

    # ---- Water phantom (Z = 0 to 20, half-width 10 cm) ----
    phantom = patches.FancyBboxPatch(
        (0, -10), 20, 20, boxstyle="round,pad=0.2",
        facecolor="#aaddff", edgecolor="navy", linewidth=2)
    ax.add_patch(phantom)
    ax.text(10, 0, "Water\nPhantom\n20×20×20 cm",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="navy")

    # ---- Brass collimator (Z = -52.5 to -47.5) ----
    # Outer radius 12 cm, inner radius 2.1 cm
    # Draw as two rectangles (top and bottom halves of annulus cross-section)
    coll_z = -52.5
    coll_dz = 5.0
    inner_r = 2.1
    outer_r = 12.0

    # Top block
    top = patches.FancyBboxPatch(
        (coll_z, inner_r), coll_dz, outer_r - inner_r,
        boxstyle="round,pad=0.1",
        facecolor="#ccaa44", edgecolor="#886600", linewidth=2)
    ax.add_patch(top)
    # Bottom block
    bot = patches.FancyBboxPatch(
        (coll_z, -outer_r), coll_dz, outer_r - inner_r,
        boxstyle="round,pad=0.1",
        facecolor="#ccaa44", edgecolor="#886600", linewidth=2)
    ax.add_patch(bot)
    ax.text(-50, outer_r + 1.5, "Brass Collimator",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="#886600")
    ax.text(-50, outer_r + 0.3, f"Aperture: {2*inner_r:.1f} cm dia, "
            f"Thickness: {coll_dz:.0f} cm",
            ha="center", va="bottom", fontsize=9, color="#886600")

    # ---- Beam source ----
    source_z = -100
    ax.plot(source_z, 0, "r^", markersize=12, zorder=5)
    ax.text(source_z, 2.5, "Proton Source\n200 MeV",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="red")

    # ---- Beam divergence cone ----
    # Angular cutoff 0.09967 rad ≈ 5.7°
    ang = 0.09967  # rad
    # Draw cone from source to collimator front face
    z_coll_front = coll_z
    z_phantom = 0
    y_at_coll = abs(source_z - z_coll_front) * np.tan(ang)
    y_at_phantom = abs(source_z - z_phantom) * np.tan(ang)

    # Full divergence cone (faint dashed — uncollimated envelope)
    cone_z = [source_z, z_phantom]
    ax.fill_between(cone_z, [-0, -y_at_phantom], [0, y_at_phantom],
                    color="red", alpha=0.05, zorder=1)
    ax.plot([source_z, z_phantom], [0, y_at_phantom], "r--",
            linewidth=1.0, alpha=0.5, dashes=(6, 4))
    ax.plot([source_z, z_phantom], [0, -y_at_phantom], "r--",
            linewidth=1.0, alpha=0.5, dashes=(6, 4))

    # Collimated beam (solid — through aperture to phantom)
    # Ray from source through aperture edge (r = 2.1 cm at collimator back face Z=-47.5)
    coll_back_z = coll_z + coll_dz
    # Angle of limiting ray: atan(inner_r / |source_z - coll_back_z|)
    dist_to_back = abs(source_z - coll_back_z)
    beam_ang = np.arctan(inner_r / dist_to_back)
    y_at_phantom_coll = abs(source_z - z_phantom) * np.tan(beam_ang)

    ax.fill_between([source_z, z_phantom],
                    [0, -y_at_phantom_coll], [0, y_at_phantom_coll],
                    color="red", alpha=0.12, zorder=2)
    ax.plot([source_z, z_phantom], [0, y_at_phantom_coll], "r-",
            linewidth=1.8, alpha=0.8)
    ax.plot([source_z, z_phantom], [0, -y_at_phantom_coll], "r-",
            linewidth=1.8, alpha=0.8)

    # ---- Dimension annotations ----
    # Source to phantom distance
    ax.annotate("", xy=(0, -15), xytext=(source_z, -15),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(-50, -16.2, "100 cm", ha="center", va="top", fontsize=10)

    # Collimator centre to phantom surface (50 cm)
    ax.annotate("", xy=(0, -12.5), xytext=(-50, -12.5),
                arrowprops=dict(arrowstyle="<->", color="#886600", lw=1.2))
    ax.text(-25, -13.5, "50 cm to phantom surface",
            ha="center", va="top", fontsize=9, color="#886600")

    # Beam diameter at phantom
    ax.annotate("", xy=(-3, y_at_phantom_coll), xytext=(-3, -y_at_phantom_coll),
                arrowprops=dict(arrowstyle="<->", color="darkred", lw=1.5))
    ax.text(-6, 0, f"~{2*y_at_phantom_coll:.1f} cm\nbeam dia.",
            ha="center", va="center", fontsize=9, color="darkred",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # ---- Axis setup ----
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Z (cm) — beam direction →", fontsize=13)
    ax.set_ylabel("Y (cm)", fontsize=13)
    ax.set_title("Geometry Used for Neutron-Production Study", fontsize=15)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.15)
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)

    # Legend distinguishing dashed (uncollimated) vs solid (collimated) beam
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", linewidth=1.0, linestyle="--",
               dashes=(6, 4), alpha=0.5, label="Uncollimated beam envelope"),
        Line2D([0], [0], color="red", linewidth=1.8, alpha=0.8,
               label="Collimated beam after aperture"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              frameon=True, fancybox=True, framealpha=0.9)

    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
