import pandas as pd
import numpy as np
import lightgbm as lgb
import re
import os
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

SEED = 993

def set_all_seeds(seed=SEED):
    """Установка сидов для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['LIGHTGBM_SEED'] = str(seed)

def load_data():
    """Загрузка данных"""
    print("Загрузка данных...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

def preprocess_data(train_df, test_df):
    print("\nПредобработка данных...")
    def fast_preprocess(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    for df in [train_df, test_df]:
        df['query_clean'] = df['query'].apply(fast_preprocess)
        df['title_clean'] = df['product_title'].apply(fast_preprocess)
        df['desc_clean'] = df['product_description'].apply(fast_preprocess)
        df['bullets_clean'] = df['product_bullet_point'].fillna('').apply(fast_preprocess)
        df['product_text'] = df['title_clean'] + ' ' + df['desc_clean'] + ' ' + df['bullets_clean']
    return train_df, test_df

def create_features(df, train_df_ref=None, is_train=True):
    features = pd.DataFrame(index=df.index)
    features['query_len'] = df['query_clean'].str.len()
    features['title_len'] = df['title_clean'].str.len()
    features['desc_len'] = df['desc_clean'].str.len()
    features['query_words'] = df['query_clean'].str.split().str.len()
    features['title_words'] = df['title_clean'].str.split().str.len()
    features['desc_words'] = df['desc_clean'].str.split().str.len()

    def enhanced_jaccard(row, n=10):
        q_words = str(row['query_clean']).split()[:n]
        t_words = str(row['title_clean']).split()[:n]

        if not q_words or not t_words:
            return 0
        q_set = set(q_words)
        t_set = set(t_words)
        intersection = len(q_set & t_set)
        union = len(q_set) + len(t_set) - intersection
        return intersection / union if union > 0 else 0
    features['jaccard_5'] = df.apply(lambda x: enhanced_jaccard(x, 5), axis=1)
    features['jaccard_10'] = df.apply(lambda x: enhanced_jaccard(x, 10), axis=1)
    features['jaccard_15'] = df.apply(lambda x: enhanced_jaccard(x, 15), axis=1)

    def enhanced_coverage(row):
        q_words = set(str(row['query_clean']).split())
        t_words = set(str(row['title_clean']).split())

        if not q_words:
            return 0

        intersection = len(q_words & t_words)
        return intersection / len(q_words)
    features['coverage'] = df.apply(enhanced_coverage, axis=1)

    def reverse_coverage(row):
        q_words = set(str(row['query_clean']).split())
        t_words = set(str(row['title_clean']).split())

        if not t_words:
            return 0

        intersection = len(q_words & t_words)
        return intersection / len(t_words)

    features['reverse_coverage'] = df.apply(reverse_coverage, axis=1)

    def common_words_metrics(row):
        q_words = set(str(row['query_clean']).split())
        t_words = set(str(row['title_clean']).split())
        intersection = len(q_words & t_words)
        return pd.Series({
            'common_words': intersection,
            'common_ratio': intersection / (len(q_words) + 1e-10),
            'dice_coeff': 2 * intersection / (len(q_words) + len(t_words) + 1e-10)
        })
    common_metrics = df.apply(common_words_metrics, axis=1)
    features = pd.concat([features, common_metrics], axis=1)


    features['exact_match'] = df.apply(
        lambda row: 1 if str(row['query_clean']).strip() in str(row['title_clean']) else 0, 
        axis=1
    )

    features['title_starts_with_query'] = df.apply(
        lambda row: 1 if str(row['title_clean']).startswith(str(row['query_clean']).strip()) else 0,
        axis=1
    )
    

    ecommerce_keywords = ['new', 'best', '2024', '2023', 'sale', 'cheap', 'price',
                         'professional', 'quality', 'original', 'free', 'shipping']
    
    for keyword in ecommerce_keywords[:10]:
        features[f'query_{keyword}'] = df['query_clean'].str.contains(keyword, na=False).astype(int)
        features[f'title_{keyword}'] = df['title_clean'].str.contains(keyword, na=False).astype(int)

    features['len_ratio'] = features['query_len'] / (features['title_len'] + 1)
    features['words_ratio'] = features['query_words'] / (features['title_words'] + 1)
    features['len_diff'] = np.abs(features['query_len'] - features['title_len'])
    features['words_diff'] = np.abs(features['query_words'] - features['title_words'])

    features['query_has_digits'] = df['query_clean'].str.contains(r'\d', na=False).astype(int)
    features['title_has_digits'] = df['title_clean'].str.contains(r'\d', na=False).astype(int)

    if 'product_brand' in df.columns:
        features['has_brand'] = df['product_brand'].notna().astype(int)
        if is_train:
            brand_counts = df['product_brand'].fillna('').value_counts()
        else:
            brand_counts = train_df_ref['product_brand'].fillna('').value_counts()
        features['brand_freq'] = df['product_brand'].fillna('').map(brand_counts).fillna(1)
    
    if 'product_color' in df.columns:
        features['has_color'] = df['product_color'].notna().astype(int)
    
    if 'product_locale' in df.columns:
        features['locale_us'] = (df['product_locale'] == 'en_US').astype(int)

    temp_df = df.copy()
    temp_df['title_len_temp'] = features['title_len']
    temp_df['title_words_temp'] = features['title_words']
    
    if 'query_id' in temp_df.columns:
        features['query_group_size'] = temp_df.groupby('query_id')['query_id'].transform('count')
        features['title_len_mean_in_group'] = temp_df.groupby('query_id')['title_len_temp'].transform('mean')
        features['title_words_mean_in_group'] = temp_df.groupby('query_id')['title_words_temp'].transform('mean')
        features['title_len_rank_pct'] = temp_df.groupby('query_id')['title_len_temp'].rank(pct=True)

    features['jaccard_x_coverage'] = features['jaccard_10'] * features['coverage']
    features['coverage_x_len'] = features['coverage'] * features['title_len']

    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    return features

def create_tfidf_features(train_df, test_df):



    title_vec = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    all_titles = pd.concat([train_df['title_clean'], test_df['title_clean']])
    title_vec.fit(all_titles)

    train_title_vec = title_vec.transform(train_df['title_clean'])
    test_title_vec = title_vec.transform(test_df['title_clean'])

    svd = TruncatedSVD(n_components=80, random_state=SEED, n_iter=5)
    train_title_svd = svd.fit_transform(train_title_vec)
    test_title_svd = svd.transform(test_title_vec)

    query_vec = CountVectorizer(
        max_features=300,
        ngram_range=(1, 3),
        binary=True,
        min_df=2
    )
    
    all_queries = pd.concat([train_df['query_clean'], test_df['query_clean']])
    query_vec.fit(all_queries)
    
    train_query_vec = query_vec.transform(train_df['query_clean'])
    test_query_vec = query_vec.transform(test_df['query_clean'])
    
    svd_query = TruncatedSVD(n_components=50, random_state=SEED, n_iter=5)
    train_query_svd = svd_query.fit_transform(train_query_vec)
    test_query_svd = svd_query.transform(test_query_vec)

    def safe_cosine_similarity(query_svd, title_svd):
        n_samples = query_svd.shape[0]
        similarities = np.zeros(n_samples)
        
        for i in range(n_samples):
            query_vec_i = query_svd[i]
            title_vec_i = title_svd[i]
            
            query_norm = np.linalg.norm(query_vec_i)
            title_norm = np.linalg.norm(title_vec_i)
            
            if query_norm > 0 and title_norm > 0:
                similarities[i] = np.dot(query_vec_i, title_vec_i) / (query_norm * title_norm)
            else:
                similarities[i] = 0
        
        return similarities.reshape(-1, 1)
    
    print("Вычисление косинусного сходства...")
    train_cosine = safe_cosine_similarity(train_query_svd, train_title_svd[:, :50])
    test_cosine = safe_cosine_similarity(test_query_svd, test_title_svd[:, :50])

    train_unique_words = (train_query_vec != 0).sum(axis=1).A.astype(np.float32)
    test_unique_words = (test_query_vec != 0).sum(axis=1).A.astype(np.float32)
    
    return (train_query_svd, train_title_svd, train_cosine, train_unique_words,
            test_query_svd, test_title_svd, test_cosine, test_unique_words)

def combine_features(manual_features, query_svd, title_svd, cosine_sim, unique_words):

    features = np.hstack([
        manual_features.values.astype(np.float32),
        query_svd.astype(np.float32),
        title_svd.astype(np.float32),
        cosine_sim.astype(np.float32),
        unique_words,
        np.log1p(unique_words)
    ])
    return features

def train_model(X_train, y_train, groups, X_test):


    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'boosting_type': 'gbdt',
        'num_leaves': 511,
        'max_depth': 9,
        'learning_rate': 0.025,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'max_bin': 511,
        'verbosity': -1,
        'num_threads': 8,
        'seed': SEED,
        'deterministic': True,
    }

    def calculate_ndcg(y_true, y_pred, query_ids, k=10):

        df = pd.DataFrame({
            'query_id': query_ids,
            'true': y_true,
            'pred': y_pred
        })
        
        ndcg_sum = 0
        query_count = 0
        
        for qid in df['query_id'].unique():
            q_data = df[df['query_id'] == qid]
            if len(q_data) < 2:
                continue
                
            sorted_idx = np.argsort(-q_data['pred'].values)
            sorted_true = q_data['true'].values[sorted_idx]
            
            ideal_idx = np.argsort(-q_data['true'].values)
            ideal_true = q_data['true'].values[ideal_idx]
            
            dcg = 0
            idcg = 0
            for i in range(min(len(sorted_true), k)):
                dcg += (2**sorted_true[i] - 1) / np.log2(i + 2)
                idcg += (2**ideal_true[i] - 1) / np.log2(i + 2)
            
            if idcg > 0:
                ndcg_sum += dcg / idcg
                query_count += 1
        
        return ndcg_sum / query_count if query_count > 0 else 0
    

    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    models = []
    oof_predictions = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        print(f"\nFold {fold + 1}/5")
        
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        groups_tr = groups[train_idx]
        

        unique_groups_tr = np.unique(groups_tr)
        group_sizes_tr = np.array([np.sum(groups_tr == g) for g in unique_groups_tr])

        train_data = lgb.Dataset(X_tr, label=y_tr, group=group_sizes_tr)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1500,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(200)
            ]
        )

        y_val_pred = model.predict(X_val)
        oof_predictions[val_idx] = y_val_pred

        ndcg_score = calculate_ndcg(y_val, y_val_pred, groups[val_idx], k=10)
        cv_scores.append(ndcg_score)
        models.append(model)
        
        print(f"Fold {fold + 1} - nDCG@10: {ndcg_score:.5f}")
        print(f"Best iteration: {model.best_iteration}")

    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ")
    print(f"{'='*60}")
    
    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: nDCG@10 = {score:.5f}")
    
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"\nСредний nDCG@10: {mean_cv:.5f} (+/- {std_cv:.5f})")
    
    # OOF оценка
    oof_ndcg = calculate_ndcg(y_train, oof_predictions, groups, k=10)
    print(f"Out-of-Fold nDCG@10: {oof_ndcg:.5f}")


    unique_groups_all = np.unique(groups)
    group_sizes_all = np.array([np.sum(groups == g) for g in unique_groups_all])
    
    train_data_all = lgb.Dataset(X_train, label=y_train, group=group_sizes_all)
    
    best_iterations = [model.best_iteration for model in models]
    median_iterations = int(np.median(best_iterations))
    final_num_round = min(median_iterations + 100, 2000)
 
    
    final_model = lgb.train(
        params,
        train_data_all,
        num_boost_round=final_num_round,
        callbacks=[lgb.log_evaluation(100)]
    )


    cv_weights = [score/sum(cv_scores) for score in cv_scores]
    test_preds_ensemble = np.zeros(len(X_test))

    for model, weight in zip(models, cv_weights):
        test_preds_ensemble += model.predict(X_test) * weight

    test_preds_final = final_model.predict(X_test)

    test_preds_combined = 0.7 * test_preds_ensemble + 0.3 * test_preds_final

    return test_preds_combined, mean_cv, oof_ndcg

def create_submission(predictions, test_df, submission_path='results/submission.csv'):

    import os

    os.makedirs('results', exist_ok=True)

    test_temp = test_df.copy()
    test_temp['prediction'] = predictions

    def extreme_normalization(group):

        values = group.values.copy()
        n = len(values)

        if n <= 1:
            return values

        min_val, max_val = values.min(), values.max()

        if max_val - min_val > 1e-10:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values)

        sorted_indices = np.argsort(-normalized)
        rank_bonus = np.linspace(0.05, 0, n)

        for i, idx in enumerate(sorted_indices):
            normalized[idx] += rank_bonus[i]

        if normalized.std() < 0.02:
            noise = np.linspace(0, 0.03, n)
            np.random.seed(SEED)
            noise = noise + np.random.randn(n) * 0.005
            normalized = normalized + noise

        exp_values = np.exp(normalized * 2)
        normalized = exp_values / exp_values.sum()

        return np.clip(normalized, 0, 1)

    test_temp['prediction_norm'] = test_temp.groupby('query_id')['prediction'].transform(extreme_normalization)
    final_predictions = test_temp['prediction_norm'].values

    final_predictions = (final_predictions - final_predictions.min()) / \
                        (final_predictions.max() - final_predictions.min() + 1e-10)

    submission_df = pd.DataFrame({
        'id': test_df['id'].values,
        'prediction': final_predictions
    })

    submission_df = submission_df.sort_values('id')
    submission_df.to_csv(submission_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUBMISSION СОХРАНЕН: {submission_path}")
    print(f"{'='*60}")


    return submission_df, submission_path

def main():
    """
    Главная функция программы
    """
    print("=" * 60)
    print("Запуск решения соревнования по ранжированию")
    print("=" * 60)

    try:
        set_all_seeds(SEED)
        print(f"Установлен глобальный seed: {SEED}")

        train_df, test_df = load_data()

        train_df, test_df = preprocess_data(train_df, test_df)


        train_manual = create_features(train_df, train_df_ref=train_df, is_train=True)
        test_manual = create_features(test_df, train_df_ref=train_df, is_train=False)
        print(f"Создано {train_manual.shape[1]} супер-признаков")

        (train_query_svd, train_title_svd, train_cosine, train_unique_words,
         test_query_svd, test_title_svd, test_cosine, test_unique_words) = create_tfidf_features(train_df, test_df)

        print("\nКомбинирование всех признаков...")
        X_train = combine_features(train_manual, train_query_svd, train_title_svd,
                                 train_cosine, train_unique_words)
        X_test = combine_features(test_manual, test_query_svd, test_title_svd,
                                test_cosine, test_unique_words)

        print(f"Общая размерность: X_train {X_train.shape}, X_test {X_test.shape}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_train = train_df['relevance'].values
        groups = train_df['query_id'].values
        test_ids = test_df['id'].values

        predictions, mean_cv, oof_ndcg = train_model(X_train_scaled, y_train, groups, X_test_scaled)

        submission_df, submission_path = create_submission(predictions, test_df)


    except Exception as e:
        print(f"Ошибка в выполнении: {e}")
        raise

    print("=" * 60)
    print("Выполнение завершено успешно!")
    print("=" * 60)

    return submission_df

if __name__ == "__main__":
    main()