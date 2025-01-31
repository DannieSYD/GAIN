import pandas as pd
import warnings
import umap
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Tuple, Union, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from openpyxl import load_workbook
warnings.filterwarnings('ignore')


def load_data(layer: int, directory: str = './csv1', emb_type: str = 'x_conf') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from CSV files for a given layer.

    Args:
        layer (int): Layer index to load data for.
        directory (str): Directory path where the CSV files are located.
        emb_type (str): Type of embedding to load.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of node embeddings, node indices, and batch indices.
    """
    node_emb = pd.read_csv(f'{directory}/{emb_type}_l{layer}.csv').to_numpy()
    node_idx = pd.read_csv(f'{directory}/conf_node_idx_l{layer}.csv').to_numpy()
    batch = pd.read_csv(f'{directory}/batch_l{layer}.csv').to_numpy()

    if emb_type == 'x_topo':
        unique_nodes = np.array([node_idx[0]])
        counts = []
        count = 1
        for i in range(1, len(node_idx)):
            if node_idx[i] == node_idx[i - 1]:
                count += 1
            else:
                counts.append(count)
                unique_nodes = np.append(unique_nodes, node_idx[i])
                count = 1
        counts.append(count)
        counts = np.array(counts)
        reversed_idx = np.repeat(unique_nodes, counts)
        node_emb = node_emb[reversed_idx]

    return node_emb, node_idx, batch


def process_indices(node_idx: np.ndarray, batch: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Process node indices and batch data to extract molecule and conformation indices.

    Args:
        node_idx (np.ndarray): Node index array.
        batch (np.ndarray): Batch array.

    Returns:
        Tuple[List[int], List[int]]: Lists of molecule indices and conformation indices.
    """
    mol_idx = []
    conf_idx = []
    num_mol = 0
    num_conf = 0

    for j in range(len(node_idx) - 1):
        if (node_idx[j + 1][0] - node_idx[j][0] == 1) & (batch[j + 1][0] - batch[j][0] == 0):
            mol_idx.append(num_mol)
            conf_idx.append(num_conf)
        elif (node_idx[j + 1][0] - node_idx[j][0] < 0) & (batch[j + 1][0] - batch[j][0] == 1):
            mol_idx.append(num_mol)
            conf_idx.append(num_conf)
            num_conf += 1
        elif (node_idx[j + 1][0] - node_idx[j][0] == 1) & (batch[j + 1][0] - batch[j][0] == 1):
            mol_idx.append(num_mol)
            conf_idx.append(num_conf)
            num_mol += 1
            num_conf = 0

    mol_idx.append(num_mol)
    conf_idx.append(num_conf)

    return mol_idx, conf_idx


def get_selected_indices(mol_idx: List[int], conf_idx: List[int], molecules: List[int]) -> List[int]:
    """
    Get selected indices based on user-specified molecules.

    Args:
        mol_idx (List[int]): List of molecule indices.
        conf_idx (List[int]): List of conformation indices.
        molecules (List[int]): List of molecules to select.

    Returns:
        List[int]: List of selected conformation indices or molecule indices based on the selection.
    """
    selected_conf_idx = []
    selected_mol_idx = []
    for mol in molecules:
        for i, idx in enumerate(mol_idx):
            if idx == mol:
                selected_conf_idx.append(conf_idx[i])
                selected_mol_idx.append(mol_idx[i])

    return selected_conf_idx, selected_mol_idx


def evaluate_embeddings(embeddings: np.ndarray, labels: np.ndarray, emb_type: str, layer: int,
                        classifier_type: str = 'random_forest') -> Dict[str, Union[str, int, float]]:
    """
    Evaluate the embeddings using a specified classifier and return performance metrics.

    Args:
        embeddings (np.ndarray): The embedding data.
        labels (np.ndarray): The labels corresponding to the embeddings.
        emb_type (str): Type of embedding to evaluate.
        layer (int): Layer index to evaluate on.
        classifier_type (str): The type of classifier to use ('random_forest', 'logistic_regression', 'knn', 'decision_tree').

    Returns:
        Dict[str, Union[str, int, float]]: A dictionary containing evaluation metrics.
    """
    # Select the classifier based on the provided type
    if classifier_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=42)
    elif classifier_type == 'logistic_regression':
        classifier = LogisticRegression(random_state=42, max_iter=1000)
    elif classifier_type == 'knn':
        classifier = KNeighborsClassifier()
    elif classifier_type == 'decision_tree':
        classifier = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # Shuffle the data
    indices = np.random.permutation(len(embeddings))
    embeddings_shuffled = embeddings[indices]
    labels_shuffled = labels[indices]

    # Split the data into training and testing sets
    train_size = int(0.8 * len(embeddings))
    X_train, X_test = embeddings_shuffled[:train_size], embeddings_shuffled[train_size:]
    y_train, y_test = labels_shuffled[:train_size], labels_shuffled[train_size:]

    # Fit the classifier
    classifier.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = classifier.predict(X_test)
    metrics = {
        'embedding_type': emb_type,
        'layer': layer,
        'classifier_type': classifier_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    return metrics


def save_metrics_to_excel(metrics_list: List[Dict[str, Union[str, int, float]]],
                          filename: str = 'embedding_evaluation.xlsx'):
    """
    Save the evaluation metrics to an Excel file.

    Args:
        metrics_list (List[Dict[str, Union[str, int, float]]]): List of dictionaries containing evaluation metrics.
        filename (str): Name of the Excel file to save the metrics.
    """
    # Convert the list of metrics dictionaries to a DataFrame
    df = pd.DataFrame(metrics_list)

    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False)

    print(f"Evaluation metrics saved to {filename}")


def plot_embeddings(embs_2d: Dict[str, np.ndarray], selected_conf_idx: List[int], selected_mol_idx: List[int],
                    molecules: List[int], plot_type: str = 'single'):
    """
    Plot the 2D UMAP embeddings.

    Args:
        embs_2d (Dict[str, np.ndarray]): Dictionary of 2D embeddings for each layer.
        selected_conf_idx (List[int]): List of selected conformation indices.
        selected_mol_idx (List[int]): List of selected molecule indices.
        molecules (List[int]): List of molecules selected for visualization.
        plot_type (str): Type of plot to generate ('single' for a single molecule, 'multiple' for multiple molecules).
    """
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    markers = ['o', 's', 'D', '^', 'v', 'x']

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(len(selected_conf_idx)):
        label = selected_conf_idx[i] if plot_type == 'single' else (selected_mol_idx[i]-min(selected_mol_idx))
        axs[0].scatter(embs_2d["layer_0"][i, 0], embs_2d["layer_0"][i, 1], linewidths=0.3,
                       c=colors[label % len(colors)], marker=markers[label % len(markers)], alpha=0.5)
        axs[1].scatter(embs_2d["layer_1"][i, 0], embs_2d["layer_1"][i, 1], linewidths=0.3,
                       c=colors[label % len(colors)], marker=markers[label % len(markers)], alpha=0.5)
        axs[2].scatter(embs_2d["layer_2"][i, 0], embs_2d["layer_2"][i, 1], linewidths=0.3,
                       c=colors[label % len(colors)], marker=markers[label % len(markers)], alpha=0.5)
        axs[3].scatter(embs_2d["layer_3"][i, 0], embs_2d["layer_3"][i, 1], linewidths=0.3,
                       c=colors[label % len(colors)], marker=markers[label % len(markers)], alpha=0.5)

    title = 'UMAP Graph Embeddings Visualization (Single Molecule)' if plot_type == 'single' \
        else 'UMAP Graph Embeddings Visualization (Multiple Molecules)'
    plt.suptitle(title)
    axs[0].set_xlabel('Component 1')
    axs[0].set_ylabel('Component 2')

    plt.tight_layout()
    plt.legend(title='Conformers', loc='upper right')
    plt.savefig(f'./figs/embs_visualization_{plot_type}_{len(molecules)}.png', dpi=600)
    plt.show()


def main(layers: int = 4, directory: str = './csv1', emb_type: str = 'x_conf', molecules: Union[int, List[int]] = 0,
         classifier_type: str = 'random_forest'):
    # Ensure molecules is a list
    if isinstance(molecules, int):
        molecules = [molecules]

    # Load and process data
    node_embs = {}
    embs_2d = {}
    evaluation_results = []

    for i in range(layers):
        node_emb, node_idx, batch = load_data(layer=i, directory=directory, emb_type=emb_type)
        mol_idx, conf_idx = process_indices(node_idx, batch)

        # Get selected indices and embeddings based on user input
        selected_conf_idx, selected_mol_idx = get_selected_indices(mol_idx, conf_idx, molecules)
        node_embs[f"layer_{i}"] = node_emb[0:len(selected_conf_idx)]

        # Perform UMAP reduction
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embs_2d[f"layer_{i}"] = umap_reducer.fit_transform(node_embs[f"layer_{i}"])

        # Evaluate the quality of embeddings
        evaluation_metrics = evaluate_embeddings(node_embs[f"layer_{i}"], selected_mol_idx, emb_type, i, classifier_type)
        evaluation_results.append(evaluation_metrics)

    # Save the evaluation metrics to an Excel file, appending to the existing file if it exists
    save_metrics_to_excel(evaluation_results, filename='embedding_evaluation.xlsx')

    # Determine plot type based on the number of molecules
    plot_type = 'single' if len(molecules) == 1 else 'multiple'
    plot_embeddings(embs_2d, selected_conf_idx, selected_mol_idx, molecules, plot_type=plot_type)


# Execute the main function
if __name__ == "__main__":
    main(layers=1, directory='./csv/dimenet-ds-1', emb_type='x_conf', molecules=[0, 1, 2, 3, 4, 5],
         classifier_type='logistic_regression')
