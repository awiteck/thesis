from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from pmdarima import auto_arima


def adf_test():
    print("Loading data...")
    df_train = pd.read_csv("./data/final_tables/erco/erco_train.csv")[
        "Normalized Demand"
    ]
    print("Loaded data...")
    # Perform the ADF test
    adf_result = adfuller(df_train)

    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print("\t%s: %.3f" % (key, value))


def sarima_fit(hyper_opt=False):
    # p, m = 24
    # if hyper_opt:
    #     # df_train = pd.read_csv("./data/final_tables/sarimax/train_hyper.csv")
    #     # df_test = pd.read_csv("./data/final_tables/sarimax/test_hyper.csv")
    #     # # df_train = pd.read_csv("/scratch/gpfs/awiteck/data/train_hyper.csv")
    #     # df_test = pd.read_csv("/scratch/gpfs/awiteck/data/test_hyper.csv")
    # else:
    print("Loading data...")
    df_train = pd.read_csv("./data/final_tables/erco/erco_train.csv")[
        "Normalized Demand"
    ]
    df_test = pd.read_csv("./data/final_tables/erco/erco_test.csv")["Normalized Demand"]

    print("Loaded data...")
    # Assuming df_train['loads'] is your target series

    p = 8
    q = 2
    d = 0

    # Fit ARIMA model
    model = ARIMA(df_train, order=(p, d, q))
    model_fit = model.fit()

    # Model summary

    print(model_fit.summary())


sarima_fit()
# sarima_fit()
# df_train = pd.read_csv("/scratch/gpfs/awiteck/data/train.csv")
# df_test = pd.read_csv("/scratch/gpfs/awiteck/data/test.csv")
