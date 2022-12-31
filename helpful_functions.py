import sdv
import numpy as np
from sdv.tabular import TVAE
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN
from sdv.tabular import GaussianCopula
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from sklearn.metrics import roc_curve, auc


def generate_data(original_data, model_name, num_rows):
    if model_name == 'TVAE':
        model = TVAE()
        model.fit(original_data)
        synt_data = model.sample(num_rows)
    elif model_name == 'CTGAN':
        model = CTGAN()
        model.fit(original_data)
        synt_data = model.sample(num_rows)
    elif model_name == 'CopulaGan':
        model = CopulaGAN()
        model.fit(original_data)
        synt_data = model.sample(num_rows)
    elif model_name == 'GaussianCopula':
        model = GaussianCopula()
        model.fit(original_data)
        synt_data = model.sample(num_rows=num_rows)
    return synt_data


def evaluate_forest(original, generated, target_column):
    synt_data = generated.drop([target_column], axis=1).to_numpy()
    synt_values = generated[target_column].to_numpy()
    orig_data = original.drop([target_column], axis=1).to_numpy()
    orig_values = original[target_column].to_numpy()

    orig_x_train, orig_x_test, orig_y_train, orig_y_test = train_test_split(orig_data, orig_values)
    synt_x_train, synt_x_test, synt_y_train, synt_y_test = train_test_split(synt_data, synt_values)

    # classification
    orig_model = RandomForestClassifier()
    synt_model = RandomForestClassifier()

    orig_model.fit(orig_x_train, orig_y_train)
    synt_model.fit(synt_x_train, synt_y_train)

    r1 = orig_model.score(orig_x_test, orig_y_test)
    r2 = orig_model.score(synt_x_test, synt_y_test)
    r3 = synt_model.score(synt_x_test, synt_y_test)
    r4 = synt_model.score(orig_x_test, orig_y_test)

    return r1, r2, r3, r4


def tsne_visualization(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    res = tsne.fit_transform(X)

    fig = px.scatter(None, x=res[:, 0], y=res[:, 1],
                     labels={
                         "x": "Dimension 1",
                         "y": "Dimension 2",
                     },
                     opacity=1, color=y.astype(str))

    fig.update_layout(dict(plot_bgcolor='white'))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                     showline=True, linewidth=1, linecolor='black')

    fig.update_layout(title_text="t-SNE")

    fig.update_traces(marker=dict(size=3))
    fig.show()


def encode_columns(data, columns):
    # Encode Categorical Columns
    le = LabelEncoder()
    data[columns] = data[columns].apply(le.fit_transform)
    return data


def plot_corr(data):
    # Plot
    plt.figure(figsize=(24, 12), dpi=80)
    sns.heatmap(data.corr(), xticklabels=data.corr().columns, yticklabels=data.corr().columns, cmap='RdYlGn', center=0,
                annot=True)

    # Decorations
    plt.title('Correlogram of  Data Set', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def distance_beetween_two_matrix(original_data, synthetic_data):
    m = original_data.to_numpy() - synthetic_data.to_numpy()
    return np.linalg.norm(m) / m.shape[0] ** 2


def evaluate(original_data, model_name, num_rows, target_column, categorial_cols):
    r1_result = []
    r2_result = []
    r3_result = []
    r4_result = []
    for _ in range(5):
        #model = CTGAN()
        # model.fit(original_data)
        synthetic_data = generate_data(original_data, model_name, num_rows)
        encoded_synt_adult = encode_columns(synthetic_data, categorial_cols)
        r1, r2, r3, r4 = evaluate_forest(original_data, encoded_synt_adult, target_column)
        r1_result.append(r1)
        r2_result.append(r2)
        r3_result.append(r3)
        r4_result.append(r4)
        a1 = np.array(r1_result)
        a2 = np.array(r2_result)
        a3 = np.array(r3_result)
        a4 = np.array(r4_result)
        aa = a1.mean()
        ab = a2.mean()
        ac = a3.mean()
        ad = a4.mean()
        with open('result.txt', 'w') as f:
            f.write('%f\n' % aa)
            f.write('%f\n' % ab)
            f.write('%f\n' % ac)
            f.write('%f\n' % ad)
    return np.array(r1_result), np.array(r2_result), np.array(r3_result), np.array(r4_result)


def test(original_data, model, num_rows):
    return generate_data(original_data, model, num_rows)
