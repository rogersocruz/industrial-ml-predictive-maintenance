"""
Smoke test of project dependencies.
Run with:  uv run python main.py

"""
import sys


def check(nome, fn):
    try:
        info = fn()
        print(f"[OK]    {nome:13} {info}")
        return True
    except Exception as e:
        print(f"[FALHA] {nome:13} {type(e).__name__}: {e}")
        return False


print(f"Python {sys.version.split()[0]}\n")
results = []


# --- version + functional smoke test of each lib ---
def t_numpy():
    import numpy as np
    a = np.arange(10)
    assert a.sum() == 45
    return f"v{np.__version__}  (array ok)"


def t_pandas():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert df["a"].mean() == 2.0
    return f"v{pd.__version__}  (dataframe ok)"


def t_sklearn():
    import sklearn, numpy as np
    from sklearn.ensemble import RandomForestRegressor
    X, y = np.random.rand(50, 4), np.random.rand(50)
    RandomForestRegressor(n_estimators=5).fit(X, y).predict(X[:1])
    return f"v{sklearn.__version__}  (fit/predict ok)"


def t_lightgbm():
    import lightgbm as lgb, numpy as np
    X, y = np.random.rand(100, 4), np.random.rand(100)
    lgb.LGBMRegressor(n_estimators=5, verbose=-1).fit(X, y).predict(X[:1])
    return f"v{lgb.__version__}  (fit/predict ok)"


def t_mlflow():
    import mlflow
    return f"v{mlflow.__version__}  (import ok)"


for nome, fn in [
    ("numpy", t_numpy),
    ("pandas", t_pandas),
    ("scikit-learn", t_sklearn),
    ("lightgbm", t_lightgbm),
    ("mlflow", t_mlflow),
]:
    results.append(check(nome, fn))


# --- torch: smoke CPU ---
def t_torch():
    import torch
    x = torch.randn(3, 3)
    (x @ x).sum().item()
    return f"v{torch.__version__}  (tensor ok)"


print()
results.append(check("torch", t_torch))


# --- GPU diagnostic
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"\n  CUDA disponível : {cuda}")
    print(f"  CUDA do wheel   : {torch.version.cuda}")   # cu126 -> '12.6' ; None = wheel CPU
    if cuda:
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        print(f"  device count    : {torch.cuda.device_count()}")
        g = torch.randn(2000, 2000, device="cuda")
        (g @ g).sum().item()                              # matmul real na GPU
        torch.cuda.synchronize()
        print(f"  matmul na GPU   : ok")
    else:
        print("  -> rodando em CPU.")
        print("     Se esperava GPU: você sincronizou com --extra cpu,")
        print("     ou o driver é antigo demais pra versão CUDA do wheel.")
except Exception as e:
    print(f"  diagnóstico GPU falhou: {type(e).__name__}: {e}")


# --- resume ---
ok, total = sum(results), len(results)
print(f"\n{'=' * 42}")
print(f"{ok}/{total} dependências OK")
sys.exit(0 if ok == total else 1)