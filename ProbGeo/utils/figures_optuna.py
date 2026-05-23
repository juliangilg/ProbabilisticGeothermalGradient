import os
from pathlib import Path

import numpy as np
import pandas as pd

from optuna.visualization._contour import _get_contour_info
from optuna.visualization.matplotlib import _contour
from optuna.visualization.matplotlib._contour import (
    _filter_missing_values,
    _calculate_axis_data,
    _calculate_griddata,
)


class OptunaContourExporter:
    """
    Exporta la información interna de optuna.visualization.matplotlib.plot_contour
    en archivos compatibles con PGFPlots/LaTeX.

    Parameters
    ----------
    params : list[str]
        Lista con exactamente dos parámetros, por ejemplo:
        ["lr", "num_inducing"].

    target_name : str
        Nombre del target u objetivo, por ejemplo:
        "Objective Value", "RMSE", "Validation Loss".

    parent_dir : str or pathlib.Path
        Carpeta padre donde se guardarán los archivos generados.

    contour_point_num : int, optional
        Número de puntos usados por Optuna para construir la malla del contour.

    x_latex_name : str, optional
        Etiqueta LaTeX para el eje x.

    y_latex_name : str, optional
        Etiqueta LaTeX para el eje y.

    z_latex_name : str, optional
        Etiqueta LaTeX para la variable objetivo.

    file_prefix : str, optional
        Prefijo para los archivos generados.
    """

    def __init__(
        self,
        params,
        target_name,
        parent_dir,
        contour_point_num=70,
        x_latex_name=None,
        y_latex_name=None,
        z_latex_name=None,
        file_prefix="optuna_contour",
    ):
        if not isinstance(params, (list, tuple)) or len(params) != 2:
            raise ValueError("params debe ser una lista o tupla con exactamente dos parámetros.")

        self.params = list(params)
        self.target_name = target_name
        self.parent_dir = Path(parent_dir)
        self.contour_point_num = contour_point_num
        self.file_prefix = file_prefix

        self.x_latex_name = x_latex_name if x_latex_name is not None else self.params[0]
        self.y_latex_name = y_latex_name if y_latex_name is not None else self.params[1]
        self.z_latex_name = z_latex_name if z_latex_name is not None else target_name

        self.parent_dir.mkdir(parents=True, exist_ok=True)

        self.grid_file = self.parent_dir / f"{self.file_prefix}_grid.csv"
        self.grid_block_file = self.parent_dir / f"{self.file_prefix}_grid_block.dat"
        self.trials_file = self.parent_dir / f"{self.file_prefix}_trials.csv"
        self.feasible_file = self.parent_dir / f"{self.file_prefix}_trials_feasible.csv"
        self.infeasible_file = self.parent_dir / f"{self.file_prefix}_trials_infeasible.csv"
        self.metadata_file = self.parent_dir / f"{self.file_prefix}_metadata.tex"

    def export(self, study, target=None):
        """
        Ejecuta la exportación completa.

        Parameters
        ----------
        study : optuna.study.Study
            Estudio de Optuna ya entrenado.

        target : callable or None
            Función target usada por Optuna. Si se deja en None, usa el valor objetivo
            principal del estudio.

        Returns
        -------
        dict
            Diccionario con rutas de archivos generados y resumen de la malla.
        """

        _contour.CONTOUR_POINT_NUM = self.contour_point_num

        info = _get_contour_info(
            study=study,
            params=self.params,
            target=target,
            target_name=self.target_name,
        )

        sub_info = info.sub_plot_infos[0][0]

        xaxis = sub_info.xaxis
        yaxis = sub_info.yaxis

        x_values, y_values = _filter_missing_values(xaxis, yaxis)

        xi, x_cat_labels, x_cat_pos, _ = _calculate_axis_data(xaxis, x_values)
        yi, y_cat_labels, y_cat_pos, _ = _calculate_axis_data(yaxis, y_values)

        zi, feasible_values, infeasible_values = _calculate_griddata(sub_info)

        self._validate_grid(zi, xi, yi)

        grid_df = self._export_grid_csv(xi, yi, zi)
        self._export_grid_block(xi, yi, zi)

        feasible_df = self._export_trials(
            feasible_values,
            self.feasible_file,
            status="feasible",
        )

        infeasible_df = self._export_trials(
            infeasible_values,
            self.infeasible_file,
            status="infeasible",
        )

        trials_df = pd.concat(
            [feasible_df, infeasible_df],
            axis=0,
            ignore_index=True,
        )

        trials_df.to_csv(self.trials_file, index=False)

        self._export_metadata(xi, yi)

        return {
            "grid_file": str(self.grid_file),
            "grid_block_file": str(self.grid_block_file),
            "trials_file": str(self.trials_file),
            "feasible_file": str(self.feasible_file),
            "infeasible_file": str(self.infeasible_file),
            "metadata_file": str(self.metadata_file),
            "zi_shape": zi.shape,
            "mesh_rows": len(yi),
            "mesh_cols": len(xi),
            "x_axis": {
                "name": xaxis.name,
                "is_log": xaxis.is_log,
                "range": xaxis.range,
            },
            "y_axis": {
                "name": yaxis.name,
                "is_log": yaxis.is_log,
                "range": yaxis.range,
            },
            "num_grid_points": len(grid_df),
            "num_feasible_trials": len(feasible_df),
            "num_infeasible_trials": len(infeasible_df),
        }

    def _validate_grid(self, zi, xi, yi):
        if zi.size == 0:
            raise ValueError(
                "No se pudo construir la malla zi. Revisa que ambos parámetros tengan "
                "al menos dos valores distintos y que coexistan en los trials."
            )

        if zi.shape != (len(yi), len(xi)):
            raise ValueError(
                f"Dimensiones inconsistentes: zi.shape={zi.shape}, "
                f"len(yi)={len(yi)}, len(xi)={len(xi)}"
            )

    def _export_grid_csv(self, xi, yi, zi):
        grid_rows = []

        for j, y in enumerate(yi):
            for i, x in enumerate(xi):
                z = zi[j, i]

                if np.isfinite(z):
                    grid_rows.append(
                        {
                            self.params[0]: float(x),
                            self.params[1]: float(y),
                            "objective": float(z),
                        }
                    )

        grid_df = pd.DataFrame(grid_rows)
        grid_df.to_csv(self.grid_file, index=False)

        return grid_df

    def _export_grid_block(self, xi, yi, zi):
        with open(self.grid_block_file, "w") as f:
            f.write(f"{self.params[0]} {self.params[1]} objective\n")

            for j, y in enumerate(yi):
                for i, x in enumerate(xi):
                    z = zi[j, i]

                    if np.isfinite(z):
                        f.write(
                            f"{float(x):.12g} "
                            f"{float(y):.12g} "
                            f"{float(z):.12g}\n"
                        )
                    else:
                        f.write(
                            f"{float(x):.12g} "
                            f"{float(y):.12g} "
                            "nan\n"
                        )

                f.write("\n")

    def _export_trials(self, values, filename, status):
        rows = []

        xs = getattr(values, "x", [])
        ys = getattr(values, "y", [])

        for x, y in zip(xs, ys):
            if np.isfinite(float(x)) and np.isfinite(float(y)):
                rows.append(
                    {
                        self.params[0]: float(x),
                        self.params[1]: float(y),
                        "status": status,
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        return df

    def _export_metadata(self, xi, yi):
        with open(self.metadata_file, "w") as f:
            f.write("% Archivo generado automáticamente desde Python\n")
            f.write(f"\\newcommand{{\\ContourRows}}{{{len(yi)}}}\n")
            f.write(f"\\newcommand{{\\ContourCols}}{{{len(xi)}}}\n")
            f.write(f"\\newcommand{{\\ContourXMin}}{{{float(np.min(xi)):.12g}}}\n")
            f.write(f"\\newcommand{{\\ContourXMax}}{{{float(np.max(xi)):.12g}}}\n")
            f.write(f"\\newcommand{{\\ContourYMin}}{{{float(np.min(yi)):.12g}}}\n")
            f.write(f"\\newcommand{{\\ContourYMax}}{{{float(np.max(yi)):.12g}}}\n")
            f.write(f"\\newcommand{{\\ContourXLabel}}{{{self.x_latex_name}}}\n")
            f.write(f"\\newcommand{{\\ContourYLabel}}{{{self.y_latex_name}}}\n")
            f.write(f"\\newcommand{{\\ContourZLabel}}{{{self.z_latex_name}}}\n")
