from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

def precisions(train_data: pd.DataFrame) -> None:
    y_true = train_data["price"]
    y_predict = #ou et comment recuperer les prix predits

    r2 = r2_score(y_true, y_predict)
    mse = mean_square_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)

    print(f"R2 score : {r2}")
    print(f"MSE : {mse}")
    print(f"MAE : {mae}")
