import pandas as pd
import numpy as np


def NBC(train_df, test_df):
    # remove = ['latitude', 'longitude', 'reviewCount', 'checkins']
    # train_df = pd.read_csv(train).drop(columns=remove).to_numpy().astype(str)
    # test_df = pd.read_csv(test).drop(columns=remove).to_numpy().astype(str)
    y = train_df.T[0]
    x_train = train_df.T[1:].T
    classes = np.unique(y)
    n, d, C = x_train.shape[0], x_train.shape[1], len(classes)
    alpha = 1
    class_probs = {}
    for c in classes:
        class_probs[c] = (np.array(y == c).sum() + alpha) / (n + C * alpha)

    possible_values = []
    for i in x_train.T:
        unique = np.unique((i.astype(str)))
        if 'nan' not in unique:
            unique = np.append(unique, 'nan')
        possible_values.append(unique)
    feature_probs = {(j, c): {v: 0 for v in possible_values[j]} for c in classes for j in range(d)}
    for j in range(d):
        for c in classes:
            in_class_c = x_train[y == c, j]
            for x in possible_values[j]:
                numerator = sum(in_class_c == x) + alpha
                denominator = len(in_class_c) + len(possible_values[j]) * alpha
                feature_probs[j, c][x] = numerator / denominator

    x_test = test_df.T[1:].T
    test_size = len(x_test)
    y_pred = []
    sq_loss = 0
    n_sq = 0
    for i in range(test_size):
        posterior_prob = {c: 0 for c in classes}
        y_max = classes[0]
        for c in classes:
            posterior_prob[c] = class_probs[c]
            for j in range(d):
                x = x_test[i, j]
                if x not in feature_probs[j, c]:
                    x = 'nan'
                posterior_prob[c] *= feature_probs[j, c][x]
            if posterior_prob[c] >= posterior_prob[y_max]:
                y_max = c
        posterior = posterior_prob[y_max]
        total = sum(posterior_prob.values())
        sq_loss += (1 - (posterior / total)) ** 2
        n_sq += 1
        y_pred.append(y_max)

    y_test = test_df.T[0]
    zero_one_loss = 0
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            zero_one_loss += 1
    zero_one_loss /= len(y_test)
    sq_loss /= n_sq
    # print(f'ZERO-ONE LOSS={zero_one_loss}')
    # print(f'SQUARED LOSS={sq_loss}')
    return zero_one_loss, sq_loss


def k_fold(file):
    remove = ['latitude', 'longitude', 'reviewCount', 'checkins']
    yelp = pd.read_csv(file).drop(columns=remove).to_numpy().astype(str)
    z_o_l = []
    sq_l = []
    percent = [0.01, 0.1, 0.5]
    for k in percent:
        zol = []
        sql = []
        for i in range(10):
            np.random.shuffle(yelp)
            num_train = int(k * len(yelp))
            train = yelp[:num_train]
            test = yelp[num_train:]
            zero_one_loss, sq_loss = NBC(train, test)
            zol.append(zero_one_loss)
            sql.append(sq_loss)
        z_o_l.append(sum(zol) / len(zol))
        sq_l.append(sum(sql) / len(sql))
    print(f'Mean Zero-One Losses Across Each Training Size:')
    for i in range(len(z_o_l)):
        print(f'  {percent[i] * 100}%:  {z_o_l[i]}')
    print(f'Mean Squared Losses Across Each Training Size:')
    for i in range(len(sq_l)):
        print(f'  {percent[i] * 100}%:  {sq_l[i]}')


k_fold(input("Full CSV: "))
