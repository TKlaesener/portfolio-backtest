import numpy as np


def cagr(retornos_mensais):
    curva = (1 + retornos_mensais).cumprod()
    n_anos = len(retornos_mensais) / 12
    return curva.iloc[-1] ** (1 / n_anos) - 1


def vol_anualizada(retornos_mensais):
    return retornos_mensais.std() * np.sqrt(12)


def max_drawdown(retornos_mensais):
    curva = (1 + retornos_mensais).cumprod()
    pico = curva.cummax()
    drawdown = curva / pico - 1
    return drawdown.min()


def sharpe(retornos_mensais):
    if retornos_mensais.std() == 0:
        return np.nan
    return (retornos_mensais.mean() / retornos_mensais.std()) * np.sqrt(12)