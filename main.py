import copy
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

ACTION_NAMES = ['Mens E-Mail', 'Womens E-Mail', 'No E-Mail']
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_NAMES)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}
FEATURE_NUMERIC = ['recency', 'history', 'mens', 'womens', 'newbie']
FEATURE_CATEGORICAL = ['zip_code', 'channel', 'history_segment']

TARGET_COL = 'visit'
ACTION_COL = 'segment'
LOGGING_PROPENSITY = 1.0 / 3.0

RANDOM_STATE = None
SUBMISSION_NAME = 'submission.csv'


class RobustFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.fitted_features: list[str] = []

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        Xp = X.copy()
        self.fitted_features = list(X.columns)

        num = [c for c in FEATURE_NUMERIC if c in Xp.columns]
        if num:
            Xp[num] = Xp[num].fillna(0)

        cat = [c for c in FEATURE_CATEGORICAL if c in Xp.columns]
        for c in cat:
            Xp[c] = Xp[c].astype(str).fillna('missing')
            le = LabelEncoder()
            Xp[c] = le.fit_transform(Xp[c])
            self.label_encoders[c] = le

        X_arr = self.scaler.fit_transform(Xp)
        return X_arr

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xp = X.copy()

        num = [c for c in FEATURE_NUMERIC if c in Xp.columns]
        if num:
            Xp[num] = Xp[num].fillna(0)

        for c, le in self.label_encoders.items():
            if c in Xp.columns:
                Xp[c] = Xp[c].astype(str).fillna('missing')

                unk = ~Xp[c].isin(le.classes_)
                if np.any(unk):
                    Xp.loc[unk, c] = le.classes_[0]
                Xp[c] = le.transform(Xp[c])

        for c in set(self.fitted_features) - set(Xp.columns):
            Xp[c] = 0

        Xp = Xp.drop(columns=[c for c in Xp.columns if c not in self.fitted_features], errors='ignore')
        Xp = Xp[self.fitted_features]

        return self.scaler.transform(Xp)


class ContextualBanditAlgorithm:
    def __init__(self, n_arms: int, n_features: Optional[int] = None):
        self.n_arms = n_arms
        self.n_features = n_features

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        raise NotImplementedError

    def update(self, arm: int, context: Optional[np.ndarray] = None, reward: Optional[float] = None):
        raise NotImplementedError

    def get_action_scores(self, context: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return None

    def get_action_probabilities(self, context: Optional[np.ndarray] = None) -> np.ndarray:
        scores = self.get_action_scores(context)
        if scores is None:
            return np.full(self.n_arms, 1.0 / self.n_arms)

        if np.any(np.isinf(scores)):
            probs = np.zeros(self.n_arms)
            idx = np.where(np.isinf(scores))[0]
            if len(idx) > 0:
                probs[idx] = 1.0 / len(idx)
            return probs

        finite = np.isfinite(scores)
        max_score = np.max(scores[finite]) if np.any(finite) else 0.0
        exps = np.exp(scores - max_score)
        s = np.sum(exps)
        return exps / s if s > 0 else np.full(self.n_arms, 1.0 / self.n_arms)


class LinUCB(ContextualBanditAlgorithm):
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        super().__init__(n_arms, n_features)
        self.alpha = alpha
        self.A = [alpha * np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    def get_action_scores(self, context: np.ndarray) -> np.ndarray:
        x = context.reshape(-1, 1)
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            try:
                Ainv = np.linalg.inv(self.A[a])
                theta = Ainv @ self.b[a]
                explo = self.alpha * np.sqrt((x.T @ Ainv @ x).item())
                scores[a] = (theta.T @ x).item() + explo
            except np.linalg.LinAlgError:
                scores[a] = 0.0
        return scores

    def select_arm(self, context: np.ndarray) -> int:
        return int(np.argmax(self.get_action_scores(context)))

    def update(self, arm: int, context: np.ndarray, reward: float):
        x = context.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x

class EpsilonGreedy(ContextualBanditAlgorithm):
    def __init__(self, n_arms: int, n_features: int, epsilon: float = 0.1):
        super().__init__(n_arms, n_features)
        self.epsilon = epsilon
        self.weights = np.zeros((n_arms, n_features))

    def get_action_probabilities(self, context: np.ndarray) -> np.ndarray:
        preds = self.weights @ context
        m = np.max(preds)
        greedy = np.where(preds == m)[0]
        p = np.full(self.n_arms, self.epsilon / self.n_arms)
        p[greedy] += (1.0 - self.epsilon) / len(greedy)
        return p

    def select_arm(self, context: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.weights @ context))

    def update(self, arm: int, context: np.ndarray, reward: float, lr: float = 0.01):
        pred = self.weights[arm] @ context
        err = reward - pred
        self.weights[arm] += lr * err * context

class ThompsonSampling(ContextualBanditAlgorithm):
    def __init__(self, n_arms: int):
        super().__init__(n_arms, None)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def get_action_probabilities(self, context: Optional[np.ndarray] = None, num_samples: int = 100) -> np.ndarray:
        samples = np.random.beta(self.alpha[np.newaxis, :], self.beta[np.newaxis, :], size=(num_samples, self.n_arms))
        chosen = np.argmax(samples, axis=1)
        counts = np.bincount(chosen, minlength=self.n_arms)
        return counts / num_samples

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        return int(np.argmax(np.random.beta(self.alpha, self.beta)))

    def update(self, arm: int, context: Optional[np.ndarray] = None, reward: Optional[float] = None):
        if reward is not None:
            self.alpha[arm] += reward
            self.beta[arm] += 1 - reward

class UCB(ContextualBanditAlgorithm):
    def __init__(self, n_arms: int):
        super().__init__(n_arms, None)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def get_action_scores(self, context: Optional[np.ndarray] = None) -> np.ndarray:
        if self.t < self.n_arms:
            s = np.full(self.n_arms, -np.inf)
            s[self.t] = np.inf
            return s
        safe = np.where(self.counts == 0, 1e-8, self.counts)
        return self.values + np.sqrt(2 * np.log(self.t) / safe)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        if self.t < self.n_arms:
            return self.t
        return int(np.argmax(self.get_action_scores(context)))

    def update(self, arm: int, context: Optional[np.ndarray] = None, reward: Optional[float] = None):
        self.t += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * (reward if reward is not None else 0.0)


class BanditPolicy:
    def __init__(self, bandit: ContextualBanditAlgorithm, stochastic_eps: float = 0.05):
        self.bandit = bandit
        self.stochastic_eps = float(stochastic_eps)
        self.n_arms = bandit.n_arms
        self._uniform = np.full(self.n_arms, 1.0 / self.n_arms)

    def _scores_to_probs(self, scores: np.ndarray) -> np.ndarray:
        if np.all(scores == -np.inf):
            return self._uniform
        if np.any(np.isinf(scores)):
            p = np.zeros(self.n_arms)
            idx = np.where(np.isinf(scores))[0]
            if len(idx) > 0:
                p[idx] = 1.0 / len(idx)
            return p
        maxs = np.max(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else 0.0
        exps = np.exp(scores - maxs)
        s = np.sum(exps)
        return exps / s if s > 0 else self._uniform

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        P = np.zeros((n, self.n_arms))

        is_contextual = isinstance(self.bandit, (LinUCB, EpsilonGreedy))
        if not is_contextual:
            if hasattr(self.bandit, 'get_action_probabilities'):
                base = self.bandit.get_action_probabilities(None)
            elif hasattr(self.bandit, 'get_action_scores'):
                base = self._scores_to_probs(self.bandit.get_action_scores(None))
            else:
                base = self._uniform
            P[:] = base
        else:
            for i in range(n):
                x = X[i]
                if hasattr(self.bandit, 'get_action_probabilities'):
                    p = self.bandit.get_action_probabilities(x)
                elif hasattr(self.bandit, 'get_action_scores'):
                    p = self._scores_to_probs(self.bandit.get_action_scores(x))
                else:
                    p = np.zeros(self.n_arms); p[self.bandit.select_arm(x)] = 1.0
                P[i] = p

        if self.stochastic_eps > 0:
            P = (1 - self.stochastic_eps) * P + self.stochastic_eps * self._uniform[None, :]
            P = P / np.sum(P, axis=1, keepdims=True)
        return P


def train_bandit_on_logs(bandit: ContextualBanditAlgorithm,
                         X: np.ndarray,
                         actions: np.ndarray,
                         rewards: np.ndarray) -> ContextualBanditAlgorithm:
    for i in range(len(X)):
        bandit.update(int(actions[i]), X[i], float(rewards[i]))
    return bandit

def estimate_ips(actions: np.ndarray,
                 rewards: np.ndarray,
                 target_probs: np.ndarray,
                 logging_mu: float = LOGGING_PROPENSITY) -> float:
    idx = np.arange(len(actions))
    denom = np.clip(logging_mu * np.ones_like(actions, dtype=float), 1e-6, 1.0)
    w = target_probs[idx, actions] / denom
    return float(np.mean(w * rewards))

def estimate_snips(actions: np.ndarray,
                   rewards: np.ndarray,
                   target_probs: np.ndarray,
                   logging_mu: float = LOGGING_PROPENSITY) -> float:
    idx = np.arange(len(actions))
    denom = np.clip(logging_mu * np.ones_like(actions, dtype=float), 1e-6, 1.0)
    w = target_probs[idx, actions] / denom
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    num = np.sum(w * rewards)
    den = np.sum(w) + 1e-8
    return float(num / den)

def best_static_benchmark(actions: np.ndarray,
                          rewards: np.ndarray,
                          logging_mu: float = LOGGING_PROPENSITY,
                          n_arms: int = 3) -> Tuple[int, float]:
    best_a, best_val = 0, -1.0
    inv_mu = 1.0 / logging_mu
    for a in range(n_arms):
        mask = (actions == a)
        if np.sum(mask) == 0:
            val = 0.0
        else:
            num = np.sum(rewards[mask] * inv_mu)
            den = np.sum(np.ones(np.sum(mask)) * inv_mu)
            val = float(num / den) if den > 0 else 0.0
        if val > best_val:
            best_val = val
            best_a = a
    return best_a, best_val


class RewardScorerPolicy:
    def __init__(self, n_arms: int, temperature: float = 1.0):
        self.n_arms = n_arms
        self.temperature = temperature
        self.models: Dict[int, LogisticRegression] = {}

    def fit(self, X: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        for a in range(self.n_arms):
            mask = (actions == a)
            if np.sum(mask) < 5:
                self.models[a] = None
                continue
            clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)
            clf.fit(X[mask], rewards[mask])
            self.models[a] = clf

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(X), self.n_arms))
        for a in range(self.n_arms):
            if self.models.get(a) is None:
                scores[:, a] = 0.0
            else:
                scores[:, a] = self.models[a].predict_proba(X)[:, 1]

        t = max(self.temperature, 1e-6)
        s = scores / t
        s -= np.max(s, axis=1, keepdims=True)
        exps = np.exp(s)
        P = exps / np.sum(exps, axis=1, keepdims=True)
        return P

def evaluate_bandit_algorithms(X_train, a_train, r_train,
                               X_val, a_val, r_val):
    n_features = X_train.shape[1]
    n_arms = len(ACTION_NAMES)

    algos = {
        'LinUCB_0.1': LinUCB(n_arms=n_arms, n_features=n_features, alpha=0.1),
        'LinUCB_1.0': LinUCB(n_arms=n_arms, n_features=n_features, alpha=1.0),
        'EpsGreedy_0.1': EpsilonGreedy(n_arms=n_arms, n_features=n_features, epsilon=0.1),
        'EpsGreedy_0.2': EpsilonGreedy(n_arms=n_arms, n_features=n_features, epsilon=0.2),
        'Thompson': ThompsonSampling(n_arms=n_arms),
        'UCB': UCB(n_arms=n_arms),
    }

    results = {}

    best_static_arm, best_static_val = best_static_benchmark(a_val, r_val, LOGGING_PROPENSITY, n_arms)
    print(f"[Benchmark] Best static arm={ID_TO_ACTION[best_static_arm]} IPS={best_static_val:.6f}")

    for name, algo_template in algos.items():
        print(f"Training {name} on logs...")
        bandit = copy.deepcopy(algo_template)
        bandit = train_bandit_on_logs(bandit, X_train, a_train, r_train)
        policy = BanditPolicy(bandit, stochastic_eps=0.05)
        P = policy.predict_proba(X_val)

        ips = estimate_ips(a_val, r_val, P, LOGGING_PROPENSITY)
        snips = estimate_snips(a_val, r_val, P, LOGGING_PROPENSITY)
        score = snips - best_static_val

        results[name] = {
            'bandit': bandit,
            'policy': policy,
            'ips': ips,
            'snips': snips,
            'score': score
        }
        print(f"  {name}: IPS={ips:.6f} SNIPS={snips:.6f} FinalScore={score:.6f}")

    print("Training RewardScorerPolicy (logistic per-arm)...")
    scorer = RewardScorerPolicy(n_arms=n_arms, temperature=1.0)
    scorer.fit(X_train, a_train, r_train)
    P_scorer = scorer.predict_proba(X_val)
    ips_s = estimate_ips(a_val, r_val, P_scorer, LOGGING_PROPENSITY)
    snips_s = estimate_snips(a_val, r_val, P_scorer, LOGGING_PROPENSITY)
    score_s = snips_s - best_static_val
    results['RewardScorer_LogReg'] = {
        'bandit': None,
        'policy': scorer,
        'ips': ips_s,
        'snips': snips_s,
        'score': score_s
    }
    print(f"  RewardScorer_LogReg: IPS={ips_s:.6f} SNIPS={snips_s:.6f} FinalScore={score_s:.6f}")

    return results, (best_static_arm, best_static_val)

def contextual_bandit_pipeline():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    print("Pipeline start")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Train columns: {list(train.columns)}")

    for c in [ACTION_COL, TARGET_COL]:
        if c not in train.columns:
            raise ValueError(f"Missing required column in train: {c}")

    feature_cols = [c for c in FEATURE_NUMERIC + FEATURE_CATEGORICAL if c in train.columns]
    print(f"Using features (context only): {feature_cols}")

    train_clean = train[train[ACTION_COL].isin(ACTION_NAMES)].copy()
    if len(train_clean) == 0:
        raise ValueError("No valid actions found in training data")

    fe = RobustFeatureEngineer()
    X_all = fe.fit_transform(train_clean[feature_cols])
    X_test = fe.transform(test[feature_cols])

    actions_all = train_clean[ACTION_COL].map(ACTION_TO_ID).values.astype(int)
    rewards_all = train_clean[TARGET_COL].astype(float).values

    try:
        tr_idx, val_idx = train_test_split(
            np.arange(len(X_all)),
            test_size=0.2,
            stratify=actions_all if len(np.unique(actions_all)) > 1 else None
        )
    except Exception:
        tr_idx, val_idx = train_test_split(
            np.arange(len(X_all)),
            test_size=0.2,
        )

    X_tr, X_val = X_all[tr_idx], X_all[val_idx]
    a_tr, a_val = actions_all[tr_idx], actions_all[val_idx]
    r_tr, r_val = rewards_all[tr_idx], rewards_all[val_idx]
    print(f"Train/Val split: {len(X_tr)}/{len(X_val)}")

    results, best_static = evaluate_bandit_algorithms(X_tr, a_tr, r_tr, X_val, a_val, r_val)
    if not results:
        raise ValueError("No bandit algorithms evaluated.")

    best_name = max(results.keys(), key=lambda k: results[k]['snips'])
    best_res = results[best_name]
    print(f"Best policy on validation: {best_name} SNIPS={best_res['snips']:.6f} FinalScore={best_res['score']:.6f}")

    n_features = X_all.shape[1]; n_arms = len(ACTION_NAMES)
    if best_name.startswith('LinUCB'):
        alpha = float(best_name.split('_')[1])
        final_bandit = LinUCB(n_arms=n_arms, n_features=n_features, alpha=alpha)
        final_bandit = train_bandit_on_logs(final_bandit, X_all, actions_all, rewards_all)
        final_policy = BanditPolicy(final_bandit, stochastic_eps=0.05)
        test_probs = final_policy.predict_proba(X_test)
    elif best_name.startswith('EpsGreedy'):
        eps = float(best_name.split('_')[1])
        final_bandit = EpsilonGreedy(n_arms=n_arms, n_features=n_features, epsilon=eps)
        final_bandit = train_bandit_on_logs(final_bandit, X_all, actions_all, rewards_all)
        final_policy = BanditPolicy(final_bandit, stochastic_eps=0.05)
        test_probs = final_policy.predict_proba(X_test)
    elif best_name == 'Thompson':
        final_bandit = ThompsonSampling(n_arms=n_arms)
        final_bandit = train_bandit_on_logs(final_bandit, X_all, actions_all, rewards_all)
        final_policy = BanditPolicy(final_bandit, stochastic_eps=0.05)
        test_probs = final_policy.predict_proba(X_test)
    elif best_name == 'UCB':
        final_bandit = UCB(n_arms=n_arms)
        final_bandit = train_bandit_on_logs(final_bandit, X_all, actions_all, rewards_all)
        final_policy = BanditPolicy(final_bandit, stochastic_eps=0.05)
        test_probs = final_policy.predict_proba(X_test)
    else:
        scorer = RewardScorerPolicy(n_arms=n_arms, temperature=1.0)
        scorer.fit(X_all, actions_all, rewards_all)
        test_probs = scorer.predict_proba(X_test)

    submission = pd.DataFrame({
        'id': test['id'],
        'p_mens_email': test_probs[:, ACTION_TO_ID['Mens E-Mail']],
        'p_womens_email': test_probs[:, ACTION_TO_ID['Womens E-Mail']],
        'p_no_email': test_probs[:, ACTION_TO_ID['No E-Mail']],
    })

    prob_cols = ['p_mens_email', 'p_womens_email', 'p_no_email']
    submission[prob_cols] = np.clip(submission[prob_cols], 0.0, 1.0)
    sums = submission[prob_cols].sum(axis=1).replace(0, 1e-8)
    submission[prob_cols] = submission[prob_cols].div(sums, axis=0)

    sums_ok = np.allclose(submission[prob_cols].sum(axis=1).values, 1.0)
    rng_min, rng_max = submission[prob_cols].min().min(), submission[prob_cols].max().max()
    print(f"Probabilities sum to 1: {sums_ok}")
    print(f"Probability range: [{rng_min:.6f}, {rng_max:.6f}]")
    for c in prob_cols:
        print(f"  mean({c}) = {submission[c].mean():.6f}")

    submission.to_csv(SUBMISSION_NAME, index=False)
    print(f"Submission saved to: {SUBMISSION_NAME}")

    print("\n validation")
    for n, res in results.items():
        print(f"{n:22} | IPS={res['ips']:.6f} | SNIPS={res['snips']:.6f} | FinalScore={res['score']:.6f}")

    # Возвращаем для внешнего main
    return submission, best_name, results, best_static


def create_submission(submission_df: pd.DataFrame) -> str:
    """
    Создает файл results/submission.csv
    """
    import os
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission файл сохранен: {submission_path}")
    return submission_path


def main():
    """
    Главная функция программы
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    try:
        submission, best_algo, all_results, best_static = contextual_bandit_pipeline()
        print(f"\nPIPELINE COMPLETED. Best policy: {best_algo}")
        print(f"Best static arm: {best_static[0]} ({ID_TO_ACTION[best_static[0]]}), IPS={best_static[1]:.6f}")
    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise

    # Дополнительно сохраняем в требуемое место
    create_submission(submission)

    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()