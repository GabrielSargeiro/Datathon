import os
import gc
import pickle
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.data import Dataset

os.environ["OMP_NUM_THREADS"] = "1"  # Evitar paralelismo interno

def load_data(model_path):
    """Carrega apenas os dados essenciais do model.pkl gerado no treinamento TF-IDF."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        return {
            'interactions': data['train_df'][['userId', 'history_list']],
            'item_features': data['tfidf_matrix'],
            'item_mapping': data['article_id_to_idx']
        }

def process_data(data, chunk_size=5000):
    """Geração de chunks leve usando iteradores para interações."""
    items = set(data['item_mapping'].keys())
    for i in range(0, len(data['interactions']), chunk_size):
        chunk = data['interactions'].iloc[i:i + chunk_size]
        # Gerador de interações
        interactions = (
            (str(row.userId), str(item))
            for row in chunk.itertuples()
            for item in row.history_list
            if item in items
        )
        yield interactions
        del chunk
        gc.collect()

def train_model(data):
    """Treinamento simplificado com monitoramento de memória."""
    dataset = Dataset()
    dataset.fit(
        users=(str(u) for u in data['interactions'].userId.unique()),
        items=data['item_mapping'].keys()
    )

    model = LightFM(
        loss='warp',
        learning_rate=0.03,
        item_alpha=1e-6,
        random_state=42
    )

    # Treinamento incremental
    for chunk_idx, interactions in enumerate(process_data(data)):
        print(f"Processando chunk {chunk_idx + 1}")
        partial_matrix, _ = dataset.build_interactions(interactions)
        model.fit_partial(
            partial_matrix,
            item_features=data['item_features'],
            epochs=1,
            num_threads=1
        )
        del partial_matrix
        gc.collect()

    return model

def main():
    try:
        print("Iniciando pipeline...")

        data = load_data("../app/artifacts/model.pkl")

        dataset = Dataset()
        dataset.fit(
            users=(str(u) for u in data['interactions'].userId.unique()),
            items=data['item_mapping'].keys()
        )

        model = LightFM(loss='warp', learning_rate=0.03, item_alpha=1e-6, random_state=42)

        chunk_idx = 0
        for interactions in process_data(data):
            chunk_idx += 1
            print(f"Processando chunk {chunk_idx}")
            partial_matrix, _ = dataset.build_interactions(interactions)
            model.fit_partial(
                partial_matrix,
                item_features=data['item_features'],
                epochs=1,
                num_threads=1
            )
            del partial_matrix
            gc.collect()

        with open("../app/artifacts/model.pkl", 'rb') as f:
            model_pickle = pickle.load(f)
        itens_df = model_pickle.get('itens_df')

        _, dataset_item_mapping, _, _ = dataset.mapping()
        n_features = data['item_features'].shape[1]
        new_item_features_rows = []
        for item, idx in sorted(dataset_item_mapping.items(), key=lambda x: x[1]):
            if item in data['item_mapping']:
                original_idx = data['item_mapping'][item]
                new_item_features_rows.append(data['item_features'][original_idx])
            else:
                new_item_features_rows.append(sp.csr_matrix((1, n_features)))
        from scipy.sparse import vstack
        item_features_reordered = vstack(new_item_features_rows)

        lightfm_data = {
            'lightfm_dataset': dataset,
            'lightfm_model': model,
            'item_features': item_features_reordered,
            'item_mapping': data['item_mapping'],
            'itens_df': itens_df
        }

        with open("../app/artifacts/model_lightfm.pkl", 'wb') as f:
            pickle.dump(lightfm_data, f, protocol=4)

        print("Treinamento concluído com sucesso!")

    except Exception as e:
        print(f"Erro: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
