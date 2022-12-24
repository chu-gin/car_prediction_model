import dill
import json
import os
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '../')


def predict():
    path_model = [name for name in os.listdir(f'{path}/data/models/') if os.path.isfile(os.path.join(f'{path}/data/models/', name))]
    path_num = len(path_model)
    with open(f'{path}/data/models/'+path_model[path_num-1], 'rb') as file:
        model = dill.load(file)

    files_test = [name for name in os.listdir(f'{path}/data/test/') if os.path.isfile(os.path.join(f'{path}/data/test/', name))]

    list_test = list()
    for test in files_test:
        with open(f'{path}/data/test/' + test, 'rb') as file:
            data_test = json.load(file)
            df_test = pd.DataFrame([data_test])
            y = model.predict(df_test)
            list_test.append([int(df_test.id), int(df_test.price), str(y[0])])

    df_test = pd.DataFrame(list_test, columns=['id', 'price', 'price_category'])
    df_test.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
