#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт визуализации результатов расчёта квазиодномерной модели солнечного ветра (AWSoM-1D)
с учётом переходного слоя (теплопроводность + радиационные потери).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_ini(filename="param.ini"):
    params = {
        "output_profile": "solution_1d.csv",
        "output_history": "convergence_history.csv",
        "input_mode": "dimensionless",
        "enable_tr": "1"
    }
    if not os.path.exists(filename):
        return params
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].split(";")[0].strip()
            if not line or line.startswith("["):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                params[k.strip()] = v.strip()
    return params

def main():
    ini_file = sys.argv[1] if len(sys.argv) > 1 else "param.ini"
    P = parse_ini(ini_file)

    prof_file = P.get("output_profile", "solution_1d.csv")
    hist_file = P.get("output_history", "convergence_history.csv")

    if not os.path.exists(prof_file):
        print(f"[Ошибка] Файл профиля '{prof_file}' не найден. Сначала запустите солвер ./awsom_1d")
        sys.exit(1)

    df = pd.read_csv(prof_file)
    has_hist = os.path.exists(hist_file)
    if has_hist:
        df_hist = pd.read_csv(hist_file)

    r = df["r"].values
    u = df["u"].values
    cs = df["cs"].values
    VA = df["VA"].values
    rho = df["rho"].values
    p = df["p"].values
    T = df["T"].values
    wp = df["wp"].values
    wm = df["wm"].values
    Qw = df["Qw"].values
    Qrad = df["Qrad"].values if "Qrad" in df.columns else np.zeros_like(r)
    q_cond = df["q_cond"].values if "q_cond" in df.columns else np.zeros_like(r)
    m_flux = df["mass_flux"].values
    bern = df["Bernoulli"].values

    idx_s = np.where(np.diff(np.sign(u - cs)) > 0)[0]
    r_sonic = r[idx_s[0]] if len(idx_s) > 0 else None

    idx_a = np.where(np.diff(np.sign(u - VA)) > 0)[0]
    r_alfven = r[idx_a[0]] if len(idx_a) > 0 else None

    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.8
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Квазиодномерная модель солнечного ветра с переходным слоем и альфвеновским нагревом (AWSoM-1D)",
                 fontsize=13.5, fontweight="bold")

    # 1. Скорости: u(r), c_s(r), V_A(r) и критические точки
    ax = axes[0, 0]
    ax.plot(r, df["u_kms"], label=r"Скорость ветра $u$", color="navy")
    ax.plot(r, df["cs_kms"], label=r"Скорость звука $c_s$", color="darkorange", linestyle="--")
    ax.plot(r, df["VA_kms"], label=r"Скорость Альфвена $V_A$", color="crimson", linestyle="-.")
    if r_sonic is not None:
        u_s_kms = df["u_kms"].values[idx_s[0]]
        ax.plot(r_sonic, u_s_kms, "o", color="darkorange", markersize=7, label=f"Звуковая точка ($r_s={r_sonic:.2f}\\,R_\\odot$)")
    if r_alfven is not None:
        u_a_kms = df["u_kms"].values[idx_a[0]]
        ax.plot(r_alfven, u_a_kms, "s", color="crimson", markersize=7, label=f"Альфвеновская точка ($r_A={r_alfven:.2f}\\,R_\\odot$)")
    ax.set_xlabel(r"$r / R_\odot$")
    ax.set_ylabel("Скорость [км/с]")
    ax.set_yscale("log")
    ax.set_title("Профили скоростей и особые точки")
    ax.legend(loc="best", fontsize=8.5)

    # 2. Структура переходного слоя и короны: Температура T(r) и плотность rho(r)
    ax = axes[0, 1]
    ax.plot(r - 1.0, df["T_K"] * 1e-6, color="firebrick", label=r"Температура $T$ [МК]")
    ax.set_xscale("log")
    ax.set_xlabel(r"Высота над хромосферой $(r - R_\odot) / R_\odot$")
    ax.set_ylabel(r"Температура $T$ [$10^6$ К]", color="firebrick")
    ax.set_title("Формирование переходного слоя и короны")
    ax2 = ax.twinx()
    ax2.plot(r - 1.0, rho, color="teal", linestyle="--", label=r"Плотность $\hat{\rho}$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"Безразмерная плотность $\hat{\rho} = \rho / \rho_0$", color="teal")
    ax.legend(loc="upper left", fontsize=8.5)
    ax2.legend(loc="lower left", fontsize=8.5)

    # 3. Плотность плазмы и баланс давлений
    ax = axes[0, 2]
    ax.plot(r, rho, label=r"Плотность $\hat{\rho}(r)$", color="teal")
    ax.plot(r, p, label=r"Тепловое давление $\hat{p}(r)$", color="darkorange", linestyle="--")
    ax.plot(r, 0.5 * (wp + wm), label=r"Волновое давление $\hat{P}_A$", color="purple", linestyle="-.")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r / R_\odot$")
    ax.set_ylabel("Безразмерные величины")
    ax.set_title("Плотность и баланс давлений")
    ax.legend(loc="best", fontsize=8.5)

    # 4. Энергетика: волны w_+, w_-, нагрев Q_w и радиационные потери Q_rad
    ax = axes[1, 0]
    ax.plot(r, wp, label=r"Выходящие волны $\hat{w}_+$", color="royalblue")
    ax.plot(r, wm, label=r"Отражённые волны $\hat{w}_-$", color="mediumvioletred", linestyle="--")
    ax.plot(r, Qw, label=r"Турбулентный нагрев $\hat{Q}_w$", color="seagreen", linestyle=":")
    if np.any(Qrad > 0):
        ax.plot(r, Qrad + 1e-20, label=r"Радиационные потери $\hat{Q}_{\rm rad}$", color="darkgoldenrod", linestyle="-.")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-9)
    ax.set_xlabel(r"$r / R_\odot$")
    ax.set_ylabel("Безразмерная плотность энергии / источники")
    ax.set_title("Турбулентный нагрев и радиационное охлаждение")
    ax.legend(loc="best", fontsize=8)

    # 5. Контроль сохранения первых интегралов (массовый расход и обобщённый интеграл Бернулли)
    ax = axes[1, 1]
    m_rel = (m_flux - np.mean(m_flux)) / np.mean(m_flux) * 100.0
    ax.plot(r, m_flux, label=r"Массовый расход $\dot{m} = A\rho u$", color="darkgreen")
    ax.plot(r, bern, label=r"Интеграл полной энергии $\varepsilon^{\rm tot}$", color="indigo", linestyle="--")
    ax.set_xlabel(r"$r / R_\odot$")
    ax.set_ylabel("Безразмерные интегралы")
    ax.set_title(f"Контроль первых интегралов (разброс $\\dot{{m}}$: {np.std(m_rel):.2f}%)")
    ax.legend(loc="best", fontsize=8.5)

    # 6. История сходимости метода установления
    ax = axes[1, 2]
    if has_hist and len(df_hist) > 1:
        ax.plot(df_hist["time"], df_hist["res_u"], label=r"$L_2$-невязка $\partial u / \partial t$", color="navy")
        ax.plot(df_hist["time"], df_hist["res_rho"], label=r"$L_2$-невязка $\rho^{-1}\partial \rho / \partial t$", color="darkorange", linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel(r"Безразмерное время $t$")
        ax.set_ylabel("Норма невязки")
        ax.set_title("Сходимость метода установления")
        ax.legend(loc="best", fontsize=8.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_img = "solution_plot.png"
    plt.savefig(out_img, dpi=200)
    print(f"График успешно сохранён в файл: {out_img}")

if __name__ == "__main__":
    main()